"""Enhanced LatentMAS method with multi-path reasoning capabilities.

This module implements the Graph-Structured Multi-Path Latent Reasoning (GMLR)
enhancement to the original LatentMAS method, supporting multiple reasoning paths,
intelligent pruning, path merging, and adaptive branching.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import argparse
from vllm import SamplingParams
import numpy as np
from collections import Counter

from .latent_mas import LatentMASMethod
from .path_manager import PathState, PathManager
from .graph_structure import ReasoningGraph, ReasoningNode
from .scoring_metrics import EnsembleScorer, SelfConsistencyScorer, PerplexityScorer, VerificationScorer, \
    HiddenStateQualityScorer, LatentConsistencyScorer
from .pruning_strategies import TopKPruning, AdaptivePruning, DiversityAwarePruning, BudgetBasedPruning
from .path_merging import PathMerger, WeightedMergeStrategy, AverageMergeStrategy, PathSimilarityDetector
from .diversity_strategies import HybridDiversityStrategy, TemperatureDiversityStrategy, NoiseDiversityStrategy
from .latent_prm import LatentPRMDataCollector, PathTreeBuilder, PathScoreBackpropagator, PRMDataStorage, PRMScorer
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout

# Logger setup
logger = logging.getLogger(__name__)


class LatentMASMultiPathMethod(LatentMASMethod):
    """Enhanced LatentMAS method with multi-path reasoning capabilities.
    
    This class extends the original LatentMASMethod to support:
    - Multiple parallel reasoning paths
    - Training-free path scoring and evaluation
    - Intelligent pruning strategies
    - Path merging for efficiency
    - Adaptive branching based on uncertainty
    
    Attributes:
        num_paths: Number of parallel reasoning paths to maintain
        enable_branching: Whether to use adaptive branching
        enable_merging: Whether to merge similar paths
        pruning_strategy: Strategy for pruning low-quality paths
        scoring_weights: Weights for ensemble scorer components
        merge_threshold: Similarity threshold for path merging
        branch_threshold: Uncertainty threshold for branching
        diversity_strategy: Strategy for generating diverse paths
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        latent_consistency_metric: str = "cosine",
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        # Multi-path specific parameters
        num_paths: int = 5,
        num_parent_paths: int = 5,
        enable_branching: bool = True,
        enable_merging: bool = True,
        pruning_strategy: str = "adaptive",
        topk_k: int = 3,
        scoring_weights: Optional[Dict[str, float]] = None,
        merge_threshold: float = 0.9,
        branch_threshold: float = 0.5,
        diversity_strategy: str = "hybrid"
    ) -> None:
        """Initialize the multi-path LatentMAS method.
        
        Args:
            model: Model wrapper for inference
            latent_steps: Number of latent thinking steps per path
            judger_max_new_tokens: Max tokens for judger generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            generate_bs: Batch size for generation
            args: Additional arguments namespace
            num_paths: Number of parallel paths (default: 5)
            num_parent_paths: Number of top-scoring parent paths to use for next agent (default: 5)
            enable_branching: Enable adaptive branching (default: True)
            enable_merging: Enable path merging (default: True)
            pruning_strategy: Pruning strategy name (default: "adaptive")
            scoring_weights: Custom weights for ensemble scorer
            merge_threshold: Similarity threshold for merging (default: 0.9)
            branch_threshold: Uncertainty threshold for branching (default: 0.5)
            diversity_strategy: Diversity strategy name (default: "hybrid")
            latent_consistency_metric: Similarity metric for latent consistency 
                ("cosine", "euclidean", "l2", "kl_divergence", default: "cosine")
        """
        # Initialize parent class
        super().__init__(
            model=model,
            latent_steps=latent_steps,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            args=args,
        )
        
        # Multi-path configuration
        self.num_paths = num_paths
        self.num_parent_paths = num_parent_paths
        self.enable_branching = enable_branching
        self.enable_merging = enable_merging
        self.merge_threshold = merge_threshold
        self.branch_threshold = branch_threshold
        self.latent_consistency_metric = latent_consistency_metric
        self.topk_k = topk_k
        
        # Override method name
        self.method_name = 'latent_mas_multipath'
        
        logger.info(f"Initialized: num_paths={num_paths}, num_parent_paths={num_parent_paths}, "
                   f"pruning={pruning_strategy}, diversity={diversity_strategy}, "
                   f"branching={enable_branching}, merging={enable_merging}")
        if pruning_strategy == "topk":
            logger.info(f"  - topk_k: {topk_k} (number of paths to keep when using topk pruning)")
        logger.debug(f"  - merge_threshold: {merge_threshold}")
        logger.debug(f"  - branch_threshold: {branch_threshold}")
        
        # Initialize path manager
        self.path_manager = PathManager()
        logger.debug("[LatentMASMultiPathMethod] Initialized PathManager")
        
        # Initialize reasoning graph
        self.reasoning_graph = ReasoningGraph()
        logger.debug("[LatentMASMultiPathMethod] Initialized ReasoningGraph")
        
        # Initialize diversity strategy with baseline temperature
        self.diversity_strategy = self._create_diversity_strategy(diversity_strategy, temperature)
        logger.info(f"[LatentMASMultiPathMethod] Created diversity strategy: {diversity_strategy} "
                   f"with baseline temperature: {temperature}")
        
        # Initialize scoring metrics
        if scoring_weights is None:
            scoring_weights = {
                'latent_consistency': 0.4,
                'perplexity': 0.3,
                'verification': 0.2,
                'hidden_quality': 0.1,
            }
        
        # Create ensemble scorer and add individual scorers
        self.ensemble_scorer = EnsembleScorer(default_weights=scoring_weights)
        
        # Add scorers with their respective weights
        if 'perplexity' in scoring_weights:
            perplexity_scorer = PerplexityScorer(model=model)
            self.ensemble_scorer.add_scorer('perplexity', perplexity_scorer, scoring_weights['perplexity'])
        
        if 'hidden_quality' in scoring_weights:
            hidden_quality_scorer = HiddenStateQualityScorer()
            self.ensemble_scorer.add_scorer('hidden_quality', hidden_quality_scorer, scoring_weights['hidden_quality'])
        
        if 'self_consistency' in scoring_weights:
            # Text-based self-consistency (slower, requires decoding)
            self_consistency_scorer = SelfConsistencyScorer(model_wrapper=model)
            self.ensemble_scorer.add_scorer('self_consistency', self_consistency_scorer, scoring_weights['self_consistency'])
            logger.info("[LatentMASMultiPathMethod] Using text-based self-consistency (slower)")
        
        if 'latent_consistency' in scoring_weights:
            # Latent-based consistency (faster, no decoding required)
            # Note: This will be computed at the path group level
            logger.info(f"[LatentMASMultiPathMethod] Initializing LatentConsistencyScorer with metric: {self.latent_consistency_metric}")
            latent_consistency_scorer = LatentConsistencyScorer(
                similarity_metric=self.latent_consistency_metric,
                aggregation_method='mean',
                use_last_latent=False
            )
            self.latent_consistency_scorer = latent_consistency_scorer
            self.latent_consistency_weight = scoring_weights['latent_consistency']
            logger.info(f"[LatentMASMultiPathMethod] Using latent-based self-consistency with {self.latent_consistency_metric} metric (faster, no decoding)")
        else:
            self.latent_consistency_scorer = None
            self.latent_consistency_weight = 0.0
            logger.debug("[LatentMASMultiPathMethod] Latent consistency scorer not enabled in scoring weights")
        
        if 'verification' in scoring_weights:
            verification_scorer = VerificationScorer(model_wrapper=model)
            self.ensemble_scorer.add_scorer('verification', verification_scorer, scoring_weights['verification'])
        
        logger.info(f"[LatentMASMultiPathMethod] Initialized EnsembleScorer with weights: {scoring_weights}")
        
        # Initialize pruning strategy
        self.pruning_strategy = self._create_pruning_strategy(pruning_strategy, topk_k)
        logger.debug(f"[LatentMASMultiPathMethod] Created pruning strategy: {pruning_strategy}")
        if pruning_strategy == "topk":
            logger.info(f"[LatentMASMultiPathMethod] TopK pruning configured with k={topk_k}")
        
        # Initialize path merger
        similarity_detector = PathSimilarityDetector(cosine_threshold=merge_threshold)
        self.path_merger = PathMerger(
            similarity_detector=similarity_detector,
            merge_strategy=WeightedMergeStrategy(),
        )
        logger.debug(f"[LatentMASMultiPathMethod] Initialized PathMerger with threshold: {merge_threshold}")
        
        # Initialize PRM data collection components (disabled by default)
        self.prm_data_collector = None
        self.prm_tree_builder = None
        self.prm_backpropagator = None
        self.prm_data_storage = None
        self.prm_scorer = None
        self.collect_prm_data = False
        logger.debug("[LatentMASMultiPathMethod] PRM data collection initialized (disabled by default)")
    
    def enable_prm_data_collection(
        self,
        output_dir: str = "prm_data",
        disable_pruning: bool = True,
        disable_merging: bool = True
    ) -> None:
        """Enable PRM training data collection mode.
        
        In this mode:
        - All paths are collected (no pruning/merging by default)
        - Path relationships are tracked
        - Final answer correctness is recorded
        - Data is saved for PRM training
        
        Args:
            output_dir: Directory to save collected data (default: prm_data at project root)
            disable_pruning: Disable pruning to collect all paths
            disable_merging: Disable merging to collect all paths
        """
        logger.info("=" * 80)
        logger.info("[LatentMASMultiPathMethod] Enabling PRM data collection mode")
        logger.info("=" * 80)
        logger.info(f"[LatentMASMultiPathMethod] Output directory: {output_dir}")
        logger.debug(f"[LatentMASMultiPathMethod] Disable pruning: {disable_pruning}")
        logger.debug(f"[LatentMASMultiPathMethod] Disable merging: {disable_merging}")
        
        self.collect_prm_data = True
        
        # Initialize PRM components
        logger.info("[LatentMASMultiPathMethod] Initializing PRM data collection components")
        
        # NEW: Initialize PathScoreBackpropagator for per-batch PRM score computation
        self.prm_backpropagator = PathScoreBackpropagator()
        logger.info("[LatentMASMultiPathMethod] ✓ PathScoreBackpropagator initialized (per-batch scoring)")
        
        # Initialize LatentPRMDataCollector with backpropagator
        self.prm_data_collector = LatentPRMDataCollector(
            enabled=True,
            backpropagator=self.prm_backpropagator
        )
        logger.info("[LatentMASMultiPathMethod] ✓ LatentPRMDataCollector initialized with backpropagator")
        
        # Keep PathTreeBuilder for backward compatibility (legacy)
        self.prm_tree_builder = PathTreeBuilder()
        logger.debug("[LatentMASMultiPathMethod] ✓ PathTreeBuilder initialized (legacy, not used)")
        
        self.prm_data_storage = PRMDataStorage(output_dir=output_dir)
        logger.debug("[LatentMASMultiPathMethod] ✓ PRMDataStorage initialized")
        
        # Keep PRMScorer for backward compatibility (legacy)
        self.prm_scorer = PRMScorer(correct_score=1.0, incorrect_score=0.0)
        logger.debug("[LatentMASMultiPathMethod] ✓ PRMScorer initialized (legacy, not used)")
        
        # Optionally disable pruning and merging for maximum path diversity
        if disable_pruning:
            logger.info("[LatentMASMultiPathMethod] Disabling pruning for data collection")
            # We'll handle this in run_batch by skipping pruning logic
            self._prm_disable_pruning = True
        else:
            self._prm_disable_pruning = False
        
        if disable_merging:
            logger.info("[LatentMASMultiPathMethod] Disabling merging for data collection")
            self._prm_disable_merging = True
        else:
            self._prm_disable_merging = False
        
        logger.info("=" * 80)
        logger.info("[LatentMASMultiPathMethod] ✓ PRM data collection mode ENABLED")
        logger.info(f"[LatentMASMultiPathMethod] Configuration:")
        logger.info(f"  - Output directory: {output_dir}")
        logger.info(f"  - Disable pruning: {disable_pruning}")
        logger.info(f"  - Disable merging: {disable_merging}")
        logger.info("=" * 80)
    
    def _create_diversity_strategy(self, strategy_name: str, base_temperature: float):
        """Create diversity strategy based on name with baseline temperature.
        
        Args:
            strategy_name: Name of the diversity strategy
            base_temperature: Baseline temperature for generating temperature series
            
        Returns:
            Diversity strategy instance
        """
        logger.info(f"[LatentMASMultiPathMethod] Creating diversity strategy '{strategy_name}' "
                   f"with base_temperature={base_temperature}")
        
        if strategy_name == "temperature":
            return TemperatureDiversityStrategy(base_temperature=base_temperature)
        elif strategy_name == "noise":
            return NoiseDiversityStrategy(base_temperature=base_temperature)
        elif strategy_name == "hybrid":
            return HybridDiversityStrategy(base_temperature=base_temperature)
        else:
            logger.warning(f"[LatentMASMultiPathMethod] Unknown diversity strategy '{strategy_name}', "
                         f"using 'hybrid' as default with base_temperature={base_temperature}")
            return HybridDiversityStrategy(base_temperature=base_temperature)
    
    def _create_pruning_strategy(self, strategy_name: str, topk_k: int = 3):
        """Create pruning strategy based on name.
        
        Args:
            strategy_name: Name of the pruning strategy
            topk_k: Number of paths to keep when using topk pruning strategy (only effective when strategy_name=topk)
            
        Returns:
            Pruning strategy instance
        """
        if strategy_name == "topk":
            logger.info(f"[LatentMASMultiPathMethod] Creating TopKPruning with k={topk_k}")
            return TopKPruning(k=topk_k)
        elif strategy_name == "adaptive":
            logger.debug(f"[LatentMASMultiPathMethod] Creating AdaptivePruning with min_paths={max(2, self.num_paths // 2)}")
            return AdaptivePruning(min_paths=max(2, self.num_paths // 2))
        elif strategy_name == "diversity":
            logger.debug(f"[LatentMASMultiPathMethod] Creating DiversityAwarePruning with target_count={self.num_paths}")
            return DiversityAwarePruning(target_count=self.num_paths)
        elif strategy_name == "budget":
            logger.debug(f"[LatentMASMultiPathMethod] Creating BudgetBasedPruning with max_budget=1000000")
            return BudgetBasedPruning(max_budget=1000000)
        else:
            logger.warning(f"[LatentMASMultiPathMethod] Unknown pruning strategy '{strategy_name}', "
                         f"using 'adaptive' as default")
            return AdaptivePruning(min_paths=max(2, self.num_paths // 2))
    
    def _calculate_uncertainty(self, hidden_states: torch.Tensor) -> float:
        """Calculate uncertainty from hidden states using entropy.
        
        Args:
            hidden_states: Hidden states tensor [B, D]
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        logger.debug("[LatentMASMultiPathMethod._calculate_uncertainty] Calculating uncertainty")
        
        # Get logits from hidden states
        logits = self.model.model.lm_head(hidden_states)
        
        # Calculate entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        uncertainty = entropy.item()
        logger.debug(f"[LatentMASMultiPathMethod._calculate_uncertainty] Uncertainty: {uncertainty:.4f}")
        
        return uncertainty
    
    def _should_branch(self, paths: List[PathState], agent_idx: int, total_agents: int) -> bool:
        """Determine whether to branch based on uncertainty and configuration.
        
        Args:
            paths: Current active paths
            agent_idx: Current agent index
            total_agents: Total number of agents
            
        Returns:
            True if should branch, False otherwise
        """
        if not self.enable_branching:
            logger.debug("[LatentMASMultiPathMethod._should_branch] Branching disabled")
            return False
        
        # Don't branch on the last agent (judger)
        if agent_idx >= total_agents - 1:
            logger.debug("[LatentMASMultiPathMethod._should_branch] Last agent, no branching")
            return False
        
        # Calculate average uncertainty across paths
        uncertainties = []
        for path in paths:
            if path.hidden_states is not None:
                uncertainty = self._calculate_uncertainty(path.hidden_states)
                uncertainties.append(uncertainty)
        
        if not uncertainties:
            logger.debug("[LatentMASMultiPathMethod._should_branch] No valid hidden states")
            return False
        
        avg_uncertainty = np.mean(uncertainties)
        should_branch = avg_uncertainty > self.branch_threshold
        
        logger.debug(f"Avg uncertainty: {avg_uncertainty:.4f}, threshold: {self.branch_threshold:.4f}, branching: {should_branch}")
        
        return should_branch
    
    def _verify_path_answer(
        self,
        raw_answer: str,
        gold_answer: str,
        task: str,
        item: Dict
    ) -> tuple:
        """Verify a single path's answer against ground truth.
        
        Args:
            raw_answer: Raw generated answer text
            gold_answer: Ground truth answer
            task: Task name (gsm8k, aime2024, etc.)
            item: Full item dictionary (for code tasks)
            
        Returns:
            Tuple of (extracted_answer, is_correct)
        """
        from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
        
        logger.debug(f"[_verify_path_answer] Verifying answer for task: {task}")
        logger.debug(f"[_verify_path_answer] Raw answer length: {len(raw_answer)} chars")
        logger.debug(f"[_verify_path_answer] Gold answer: {gold_answer}")
        
        # Extract and verify based on task type
        if task in ['mbppplus', 'humanevalplus']:
            # Code generation tasks
            extracted = extract_markdown_python_block(raw_answer)
            
            if extracted is None:
                logger.debug(f"[_verify_path_answer] No Python code block found")
                return "", False
            
            # Execute code with test cases
            python_code_to_exe = extracted + "\n" + gold_answer
            is_correct, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
            
            if error_msg:
                logger.debug(f"[_verify_path_answer] Code execution error: {error_msg}")
            
            return extracted, is_correct
        
        elif task in ["aime2024", "aime2025"]:
            # Integer answer tasks
            extracted = normalize_answer(extract_gsm8k_answer(raw_answer))
            gold = str(gold_answer).strip()
            
            try:
                pred_int = int(extracted)
                gold_int = int(gold)
                is_correct = (pred_int == gold_int)
                logger.debug(f"[_verify_path_answer] Integer comparison: {pred_int} vs {gold_int} = {is_correct}")
                return extracted, is_correct
            except ValueError as e:
                logger.debug(f"[_verify_path_answer] ValueError in parsing: {e}")
                return extracted if extracted else "", False
        
        else:
            # Default: GSM8K-style tasks
            extracted = normalize_answer(extract_gsm8k_answer(raw_answer))
            gold = gold_answer if isinstance(gold_answer, str) else str(gold_answer)
            is_correct = (extracted == gold) if (extracted and gold) else False
            
            logger.debug(f"[_verify_path_answer] String comparison: '{extracted}' vs '{gold}' = {is_correct}")
            return extracted if extracted else "", is_correct
    
    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        """Run multi-path reasoning on a batch of items.
        
        This method implements the core multi-path reasoning logic:
        1. For each non-judger agent, generate diverse latent paths
        2. Score all paths using ensemble metrics
        3. Prune low-quality paths
        4. Optionally merge similar paths
        5. Optionally branch on high uncertainty
        6. Aggregate paths at judger stage
        
        Args:
            items: List of input items with 'question' field
            
        Returns:
            List of result dictionaries with predictions and traces
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        
        batch_size = len(items)
        logger.info(f"Processing batch of {batch_size} items with {self.num_paths} paths per item")
        
        # PRM data collection: Start collecting for each question in batch
        if self.collect_prm_data and self.prm_data_collector:
            logger.info("[PRM DataCollection] Starting data collection for batch")
            for batch_idx, item in enumerate(items):
                question_id = f"q_{batch_idx}"
                self.prm_data_collector.start_question(
                    question_id=question_id,
                    question=item["question"],
                    gold_answer=item.get("gold", "")
                )
        
        # CRITICAL: Clear PathManager before batch to ensure complete independence between batches
        # This prevents cross-batch accumulation of paths and GPU memory leaks
        if len(self.path_manager.paths) > 0:
            logger.warning(f"[Memory Leak Prevention] PathManager has {len(self.path_manager.paths)} paths before batch start - clearing now")
            self.path_manager.clear()
            
            # Force GPU cleanup after clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"[Memory Leak Prevention] Forced GPU cleanup after PathManager clear")
        
        # Reset reasoning graph to ensure clean state
        if self.reasoning_graph is not None and len(self.reasoning_graph.nodes) > 0:
            logger.warning(f"[Memory Leak Prevention] ReasoningGraph has {len(self.reasoning_graph.nodes)} nodes before batch start - clearing now")
            self.reasoning_graph.clear()
            self.reasoning_graph = ReasoningGraph()
        
        # Log GPU memory before batch processing
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU Memory] Before batch (after pre-cleanup): allocated={gpu_mem_allocated:.2f}GB, reserved={gpu_mem_reserved:.2f}GB")
            logger.info(f"[PathManager] Paths in manager after pre-cleanup: {len(self.path_manager.paths)}")
            logger.debug(f"[PathManager] Active paths: {len(self.path_manager.active_paths)}")
        
        # Initialize paths for each item in batch
        batch_paths: List[List[PathState]] = [[] for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        
        # Process each agent
        for agent_idx, agent in enumerate(self.agents):
            logger.info("=" * 80)
            logger.info(f"Agent {agent_idx + 1}/{len(self.agents)}: {agent.name} ({agent.role})")
            logger.info("=" * 80)
            
            # Build messages for this agent
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            
            # Log agent prompts
            logger.info(f"[{agent.name}] Preparing prompts for {len(batch_messages)} items")
            for batch_idx, messages in enumerate(batch_messages):
                logger.debug(f"[{agent.name}] Item {batch_idx + 1} messages: {messages}")
                # Log the full prompt that will be sent to the model
                prompt_preview = str(messages)[:500] if len(str(messages)) > 500 else str(messages)
                logger.info(f"[{agent.name}] Item: {batch_idx + 1}, Prompt preview: {prompt_preview}...")
            
            # Prepare input
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            
            if agent.role != "judger":
                # Non-judger agent: multi-path latent reasoning
                logger.info(f"[{agent.name}] Starting multi-path latent reasoning")
                logger.debug(f"[{agent.name}] Configuration: {self.num_paths} paths, {self.latent_steps} latent steps per path")
                logger.debug(f"[{agent.name}] Pruning strategy: {self.pruning_strategy.__class__.__name__}")
                logger.debug(f"[{agent.name}] Diversity strategy: {self.diversity_strategy.__class__.__name__}")
                
                # Wrap prompts with <think> tag if needed
                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                    logger.debug(f"[{agent.name}] Added <think> tag to prompts")
                else:
                    wrapped_prompts = prompts
                
                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                
                # Process each item in batch
                for batch_idx in range(batch_size):
                    logger.info(f"[{agent.name}] Processing item {batch_idx + 1}/{batch_size}")
                    logger.info(f"[{agent.name}] Question: {items[batch_idx]['question']}")
                    
                    # Log GPU memory at start of item processing
                    if torch.cuda.is_available():
                        gpu_mem_item_start = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_reserved_start = torch.cuda.memory_reserved() / 1024**3
                        logger.debug(f"[GPU Memory] Item {batch_idx + 1} start: allocated={gpu_mem_item_start:.2f}GB, reserved={gpu_mem_reserved_start:.2f}GB")
                    
                    # Get input for this item
                    item_ids = wrapped_ids[batch_idx:batch_idx+1]
                    item_mask = wrapped_mask[batch_idx:batch_idx+1]
                    
                    # Select parent paths from previous agent (if any)
                    parent_paths = []
                    if batch_paths[batch_idx]:
                        # Select top-k parent paths based on scores
                        sorted_paths = sorted(batch_paths[batch_idx], key=lambda p: p.score, reverse=True)
                        num_parents = min(self.num_parent_paths, len(sorted_paths))
                        parent_paths = sorted_paths[:num_parents]
                        
                        logger.info(f"[{agent.name}] Selected {num_parents} parent paths from previous agent")
                        logger.debug(f"[{agent.name}] Parent path IDs and scores: "
                                   f"{[(p.path_id, f'{p.score:.4f}') for p in parent_paths]}")
                    else:
                        logger.debug(f"[{agent.name}] No parent paths available (first agent in chain)")
                    
                    # Determine if we should branch
                    should_branch = self._should_branch(
                        batch_paths[batch_idx],
                        agent_idx,
                        len(self.agents)
                    )
                    
                    # Generate paths distributed across parent paths
                    if torch.cuda.is_available():
                        gpu_mem_before_gen = torch.cuda.memory_allocated() / 1024**3
                        logger.info(f"[{agent.name}] GPU memory before path generation: {gpu_mem_before_gen:.2f}GB")
                    
                    new_paths = []
                    
                    if not parent_paths:
                        # First agent: generate all paths from scratch
                        logger.info(f"[{agent.name}] Generating {self.num_paths} diverse reasoning paths (no parent)")
                        logger.debug(f"[{agent.name}] Each path will perform {self.latent_steps} latent thinking steps")
                        
                        path_dicts = self.model.generate_diverse_latent_paths(
                            input_ids=item_ids,
                            attention_mask=item_mask,
                            num_paths=self.num_paths,
                            latent_steps=self.latent_steps,
                            diversity_strategy=self.diversity_strategy,
                            past_key_values=None,
                        )
                        
                        # Convert to PathState objects
                        for path_dict in path_dicts:
                            path_id = self.path_manager.next_path_id
                            self.path_manager.next_path_id += 1
                            
                            # Clean metadata
                            clean_metadata = {}
                            for k, v in path_dict.get('metadata', {}).items():
                                if not isinstance(v, torch.Tensor):
                                    clean_metadata[k] = v
                            
                            clean_metadata['agent_name'] = agent.name
                            clean_metadata['agent_idx'] = agent_idx
                            clean_metadata['batch_idx'] = batch_idx
                            clean_metadata['parent_path_id'] = None
                            
                            path_state = PathState(
                                path_id=path_id,
                                latent_history=path_dict['latent_history'],
                                hidden_states=path_dict['hidden_states'],
                                kv_cache=path_dict['kv_cache'],
                                metadata=clean_metadata,
                            )
                            new_paths.append(path_state)
                            self.path_manager.paths[path_id] = path_state
                            logger.debug(f"[{agent.name}] Created path {path_id} with parent=None")
                        
                        del path_dicts
                    else:
                        # Subsequent agents: distribute path generation across parent paths
                        num_parents = len(parent_paths)
                        paths_per_parent = self.num_paths // num_parents
                        remaining_paths = self.num_paths % num_parents
                        
                        logger.info(f"[{agent.name}] Distributing {self.num_paths} paths across {num_parents} parents")
                        logger.info(f"[{agent.name}] Base paths per parent: {paths_per_parent}, "
                                   f"extra paths: {remaining_paths}")
                        
                        for parent_idx, parent_path in enumerate(parent_paths):
                            # Calculate number of paths to generate from this parent
                            num_paths_for_parent = paths_per_parent
                            if parent_idx < remaining_paths:
                                num_paths_for_parent += 1
                            
                            if num_paths_for_parent == 0:
                                continue
                            
                            logger.info(f"[{agent.name}] Generating {num_paths_for_parent} paths from "
                                       f"parent {parent_path.path_id} (score: {parent_path.score:.4f})")
                            
                            # Generate paths from this parent
                            if should_branch and parent_path.kv_cache is not None:
                                # Branch from existing path
                                logger.debug(f"[{agent.name}] Branching from parent {parent_path.path_id}")
                                path_dicts = self.model.generate_latent_with_branching(
                                    input_ids=item_ids,
                                    attention_mask=item_mask,
                                    num_branches=num_paths_for_parent,
                                    latent_steps=self.latent_steps,
                                    diversity_strategy=self.diversity_strategy,
                                    past_key_values=parent_path.kv_cache,
                                )
                            else:
                                # Generate diverse paths from parent
                                logger.debug(f"[{agent.name}] Generating diverse paths from parent {parent_path.path_id}")
                                path_dicts = self.model.generate_diverse_latent_paths(
                                    input_ids=item_ids,
                                    attention_mask=item_mask,
                                    num_paths=num_paths_for_parent,
                                    latent_steps=self.latent_steps,
                                    diversity_strategy=self.diversity_strategy,
                                    past_key_values=parent_path.kv_cache,
                                )
                            
                            # Convert to PathState objects with parent tracking
                            for path_dict in path_dicts:
                                path_id = self.path_manager.next_path_id
                                self.path_manager.next_path_id += 1
                                
                                # Clean metadata
                                clean_metadata = {}
                                for k, v in path_dict.get('metadata', {}).items():
                                    if not isinstance(v, torch.Tensor):
                                        clean_metadata[k] = v
                                
                                clean_metadata['agent_name'] = agent.name
                                clean_metadata['agent_idx'] = agent_idx
                                clean_metadata['batch_idx'] = batch_idx
                                clean_metadata['parent_path_id'] = parent_path.path_id
                                
                                path_state = PathState(
                                    path_id=path_id,
                                    latent_history=path_dict['latent_history'],
                                    hidden_states=path_dict['hidden_states'],
                                    kv_cache=path_dict['kv_cache'],
                                    metadata=clean_metadata,
                                )
                                new_paths.append(path_state)
                                self.path_manager.paths[path_id] = path_state
                                logger.debug(f"[{agent.name}] Created path {path_id} with parent={parent_path.path_id}")
                            
                            del path_dicts
                        
                        logger.info(f"[{agent.name}] Generated total of {len(new_paths)} paths from {num_parents} parents")
                    
                    if torch.cuda.is_available():
                        gpu_mem_after_gen = torch.cuda.memory_allocated() / 1024**3
                        mem_increase = gpu_mem_after_gen - gpu_mem_before_gen
                        logger.info(f"[{agent.name}] GPU memory after path generation: {gpu_mem_after_gen:.2f}GB "
                                   f"(+{mem_increase:.2f}GB for {len(new_paths)} paths)")
                    
                    # Score all paths
                    if torch.cuda.is_available():
                        gpu_mem_before_score = torch.cuda.memory_allocated() / 1024**3
                        logger.debug(f"[{agent.name}] GPU memory before scoring: {gpu_mem_before_score:.2f}GB")
                    
                    logger.info(f"[{agent.name}] Scoring {len(new_paths)} generated paths")
                    
                    # Compute individual latent consistency scores for each path if enabled
                    individual_consistency_scores = None
                    if self.latent_consistency_scorer is not None and len(new_paths) > 1:
                        logger.info(f"[{agent.name}] Computing individual latent consistency for {len(new_paths)} paths")
                        individual_consistency_scores = self.latent_consistency_scorer.score_individual_paths(new_paths)
                        logger.info(f"[{agent.name}] Individual consistency scores: {individual_consistency_scores} "
                                    f"(min={min(individual_consistency_scores):.4f}, "
                                    f"max={max(individual_consistency_scores):.4f}, "
                                    f"mean={np.mean(individual_consistency_scores):.4f})")
                    
                    # Score each path individually
                    for path_idx, path in enumerate(new_paths):
                        # Get base score from ensemble scorer (without latent_consistency)
                        base_score = self.ensemble_scorer.score(
                            path_state=path,
                            question=items[batch_idx]["question"],
                        )
                        
                        # Add individual latent consistency score if enabled
                        if individual_consistency_scores is not None:
                            path_consistency = individual_consistency_scores[path_idx]
                            
                            # Compute weighted combination
                            # Normalize weights: ensemble_weight + latent_consistency_weight = 1.0
                            ensemble_weight = 1.0 - self.latent_consistency_weight
                            final_score = (base_score * ensemble_weight + 
                                         path_consistency * self.latent_consistency_weight)
                            
                            path.metadata['base_score'] = base_score
                            path.metadata['latent_consistency'] = path_consistency
                            logger.debug(f"[{agent.name}] Path {path.path_id}: base={base_score:.4f}, "
                                      f"consistency={path_consistency:.4f}, final={final_score:.4f}")
                        else:
                            final_score = base_score
                            logger.debug(f"[{agent.name}] Path {path.path_id} score: {final_score:.4f}")
                        
                        path.update_state(score=final_score)
                        logger.debug(f"[{agent.name}] Path {path.path_id} metadata: {path.metadata}")
                        
                        # PRM data collection: Record this path
                        if self.collect_prm_data and self.prm_data_collector:
                            # Determine parent path ID
                            parent_path_id = path.metadata.get('parent_path_id', None)
                            
                            # Record the path
                            self.prm_data_collector.record_path(
                                path_id=path.path_id,
                                agent_name=agent.name,
                                agent_idx=agent_idx,
                                parent_path_id=parent_path_id,
                                latent_history=path.latent_history,
                                hidden_states=path.hidden_states,
                                score=final_score,
                                metadata=path.metadata.copy() if path.metadata else {}
                            )
                            logger.debug(f"[PRM DataCollection] Recorded path {path.path_id} "
                                       f"for agent {agent.name} (parent: {parent_path_id})")
                    
                    # Clean up consistency scores list after scoring
                    if individual_consistency_scores is not None:
                        del individual_consistency_scores
                        logger.debug(f"[{agent.name}] Cleaned up consistency scores list")
                    
                    # Clean up any temporary tensors in path metadata to prevent references
                    for path in new_paths:
                        if path.metadata:
                            # Remove any tensor references from metadata
                            keys_to_remove = [k for k, v in path.metadata.items() if isinstance(v, torch.Tensor)]
                            for k in keys_to_remove:
                                del path.metadata[k]
                                logger.debug(f"[{agent.name}] Removed tensor reference '{k}' from path {path.path_id} metadata")
                    
                    # Immediate GPU cleanup after scoring (scoring creates many temporary tensors)
                    if torch.cuda.is_available():
                        gpu_mem_after_score = torch.cuda.memory_allocated() / 1024**3
                        mem_increase_score = gpu_mem_after_score - gpu_mem_before_score
                        logger.info(f"[{agent.name}] GPU memory after scoring: {gpu_mem_after_score:.2f}GB "
                                   f"(+{mem_increase_score:.2f}GB during scoring)")
                        
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Force synchronization to ensure cleanup
                        gpu_mem_after_cleanup = torch.cuda.memory_allocated() / 1024**3
                        mem_freed = gpu_mem_after_score - gpu_mem_after_cleanup
                        logger.debug(f"[GPU Memory] After scoring cleanup: {gpu_mem_after_cleanup:.2f}GB "
                                    f"(freed {mem_freed:.2f}GB)")
                    
                    # Log all path scores before pruning (in path_id order for clarity)
                    all_scores = [(p.path_id, p.score) for p in sorted(new_paths, key=lambda x: x.path_id)]
                    logger.info(f"[{agent.name}] All path scores before pruning (by path_id): "
                              f"{[f'({pid}:{score:.4f})' for pid, score in all_scores]}")
                    
                    # Detect if current agent is Refiner (倒数第二个 agent)
                    is_refiner = (agent.role == "refiner")
                    is_judger = (agent.role == "judger")
                    
                    if is_refiner:
                        logger.info(f"[{agent.name}] Detected Refiner stage (agent {agent_idx + 1}/{len(self.agents)})")
                        logger.info(f"[{agent.name}] Will force pruning to keep ONLY 1 path for final Judger")
                    
                    # Prune low-quality paths (skip if in PRM data collection mode)
                    if self.collect_prm_data and self._prm_disable_pruning:
                        logger.info(f"[{agent.name}] PRM data collection mode: SKIPPING pruning to collect all paths")
                        pruned_paths = new_paths
                    else:
                        if torch.cuda.is_available():
                            gpu_mem_before_prune = torch.cuda.memory_allocated() / 1024**3
                            logger.debug(f"[{agent.name}] GPU memory before pruning: {gpu_mem_before_prune:.2f}GB")
                        
                        logger.info(f"[{agent.name}] Pruning low-quality paths using {self.pruning_strategy.__class__.__name__}")
                        logger.debug(f"[{agent.name}] Pre-pruning path scores: {[f'{p.path_id}:{p.score:.4f}' for p in new_paths]}")
                        
                        # For Refiner, force keep_count=1 to ensure only 1 path for Judger
                        if is_refiner:
                            logger.info(f"[{agent.name}] Forcing pruning to keep exactly 1 path (Refiner strategy)")
                            pruned_paths = self.pruning_strategy.prune(
                                paths=new_paths,
                                current_step=agent_idx,
                                total_steps=len(self.agents),
                                force_keep_count=1,  # Force Refiner to keep only 1 path
                            )
                        else:
                            # Pass k parameter when using topk strategy
                            prune_kwargs = {
                                "paths": new_paths,
                                "current_step": agent_idx,
                                "total_steps": len(self.agents),
                            }
                            if isinstance(self.pruning_strategy, TopKPruning):
                                prune_kwargs["k"] = self.topk_k
                                logger.debug(f"[{agent.name}] Using topk pruning with k={self.topk_k}")
                            pruned_paths = self.pruning_strategy.prune(**prune_kwargs)
                        
                        logger.info(f"[{agent.name}] Pruning complete: kept {len(pruned_paths)}/{len(new_paths)} paths")
                        kept_paths_info = [(p.path_id, p.score) for p in pruned_paths]
                        logger.info(f"[{agent.name}] Kept paths (sorted by score desc): "
                                  f"{[f'Path{pid}={score:.4f}' for pid, score in kept_paths_info]}")
                    
                    # Merge similar paths if enabled
                    # Agent role-aware merge timing strategy:
                    # - Planner: Skip merge to preserve diversity for exploration
                    # - Critic: Allow merge to reduce redundancy after evaluation
                    # - Refiner: Skip merge (already pruned to 1 path)
                    # - Judger: No merge needed (final stage)
                    
                    is_planner = (agent.role == "planner")
                    is_critic = (agent.role == "critic")
                    
                    # Determine if we should attempt merging for this agent
                    # Skip merging if in PRM data collection mode
                    should_attempt_merge = (
                        self.enable_merging and 
                        len(pruned_paths) > 1 and 
                        not is_refiner and 
                        not is_planner and  # Skip Planner to preserve diversity
                        is_critic and  # Only merge at Critic stage
                        not (self.collect_prm_data and self._prm_disable_merging)  # Skip if PRM mode
                    )
                    
                    if should_attempt_merge:
                        logger.info(f"[{agent.name}] Attempting to merge similar paths (threshold: {self.merge_threshold})")
                        logger.info(f"[{agent.name}] Agent role '{agent.role}' allows merging at this stage")
                        logger.debug(f"[{agent.name}] Pre-merge path count: {len(pruned_paths)}")
                        
                        # Log pre-merge path details
                        pre_merge_info = [(p.path_id, p.score, p.metadata.get('latent_consistency', None)) 
                                         for p in pruned_paths]
                        pre_merge_strs = []
                        for pid, score, cons in pre_merge_info:
                            if cons is not None:
                                pre_merge_strs.append(f"Path{pid}(score={score:.4f}, cons={cons:.4f})")
                            else:
                                pre_merge_strs.append(f"Path{pid}(score={score:.4f}, cons=N/A)")
                        logger.info(f"[{agent.name}] Pre-merge paths: " + ", ".join(pre_merge_strs))
                        
                        merged_paths = self.path_merger.merge_similar_paths(
                            paths=pruned_paths,
                            path_manager=self.path_manager,
                            model_lm_head=self.model.model.lm_head,
                            use_kl=False,
                            min_group_size=2
                        )
                        logger.info(f"[{agent.name}] Merging complete: reduced to {len(merged_paths)} paths")
                        
                        if len(merged_paths) < len(pruned_paths):
                            num_merged = len(pruned_paths) - len(merged_paths)
                            logger.info(f"[{agent.name}] Successfully merged {num_merged} similar paths")
                            
                            # Log post-merge path details
                            post_merge_info = [(p.path_id, p.score, p.metadata.get('latent_consistency', None)) 
                                             for p in merged_paths]
                            post_merge_strs = []
                            for pid, score, cons in post_merge_info:
                                if cons is not None:
                                    post_merge_strs.append(f"Path{pid}(score={score:.4f}, cons={cons:.4f})")
                                else:
                                    post_merge_strs.append(f"Path{pid}(score={score:.4f}, cons=N/A)")
                            logger.info(f"[{agent.name}] Post-merge paths: " + ", ".join(post_merge_strs))
                        else:
                            logger.info(f"[{agent.name}] No paths were merged (no suitable merge candidates found)")
                        
                        batch_paths[batch_idx] = merged_paths
                        
                        # Immediate GPU cleanup after merging (merging creates similarity matrices)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug(f"[{agent.name}] GPU cache cleaned after merge operation")
                    else:
                        # Log reason for skipping merge
                        if is_refiner:
                            logger.info(f"[{agent.name}] Skipping merge for Refiner (already has {len(pruned_paths)} path)")
                        elif is_planner:
                            logger.info(f"[{agent.name}] Skipping merge for Planner (preserving diversity for exploration)")
                        elif not is_critic:
                            logger.info(f"[{agent.name}] Skipping merge for agent role '{agent.role}' (merge only at Critic stage)")
                        elif not self.enable_merging:
                            logger.debug(f"[{agent.name}] Path merging disabled globally")
                        else:
                            logger.debug(f"[{agent.name}] Only {len(pruned_paths)} path(s), skipping merge")
                        
                        batch_paths[batch_idx] = pruned_paths
                    
                    # Record agent trace
                    final_path_scores = [p.score for p in batch_paths[batch_idx]]
                    final_paths_info = [(p.path_id, p.score) for p in batch_paths[batch_idx]]
                    logger.info(f"[{agent.name}] Final state for item {batch_idx + 1}: {len(batch_paths[batch_idx])} active paths")
                    logger.info(f"[{agent.name}] Final paths (sorted by score desc): "
                              f"{[f'Path{pid}={score:.4f}' for pid, score in final_paths_info]}")
                    logger.debug(f"[{agent.name}] Agent completed processing for item {batch_idx + 1}")
                    
                    agent_traces[batch_idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[batch_idx],
                        "latent_steps": self.latent_steps,
                        "num_paths": len(batch_paths[batch_idx]),
                        "path_scores": final_path_scores,
                        "output": "",
                    })
                    
                    # Clean up temporary scoring data and other item-specific variables
                    del final_path_scores, final_paths_info
                    
                    # Force GPU cache cleanup after each item to prevent cross-item accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.debug(f"[{agent.name}] GPU cache cleaned after processing item {batch_idx + 1}")
                    
                    # Clean up pruned paths to free GPU memory immediately
                    # Keep only the paths that survived pruning/merging
                    kept_path_ids = {p.path_id for p in batch_paths[batch_idx]}
                    paths_to_remove_ids = [p.path_id for p in new_paths if p.path_id not in kept_path_ids]
                    
                    if paths_to_remove_ids:
                        logger.debug(f"[Memory Cleanup] Cleaning up {len(paths_to_remove_ids)} pruned paths for item {batch_idx + 1}")
                        
                        # Explicitly delete tensors from pruned paths before removing from PathManager
                        for path_id in paths_to_remove_ids:
                            path = self.path_manager.paths.get(path_id)
                            if path:
                                # Delete latent history tensors
                                if path.latent_history:
                                    for tensor in path.latent_history:
                                        if tensor is not None:
                                            del tensor
                                    path.latent_history.clear()
                                
                                # Delete hidden states
                                if path.hidden_states is not None:
                                    del path.hidden_states
                                    path.hidden_states = None
                                
                                # Delete KV cache
                                if path.kv_cache is not None:
                                    del path.kv_cache
                                    path.kv_cache = None
                        
                        # Remove from PathManager
                        removed_count = self.path_manager.clear_paths(paths_to_remove_ids)
                        logger.info(f"[Memory Cleanup] Freed GPU memory from {removed_count} pruned paths for item {batch_idx + 1}")
                        logger.debug(f"[PathManager] Remaining paths in manager: {len(self.path_manager.paths)}")
                        
                        # Log GPU memory after cleanup
                        if torch.cuda.is_available():
                            gpu_mem_after_prune_cleanup = torch.cuda.memory_allocated() / 1024**3
                            mem_freed_prune = gpu_mem_before_prune - gpu_mem_after_prune_cleanup
                            logger.info(f"[{agent.name}] GPU memory after pruning cleanup: {gpu_mem_after_prune_cleanup:.2f}GB "
                                       f"(freed {mem_freed_prune:.2f}GB by removing {len(paths_to_remove_ids)} paths)")
                    
                    # Log GPU memory at end of item processing
                    if torch.cuda.is_available():
                        gpu_mem_item_end = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_reserved_end = torch.cuda.memory_reserved() / 1024**3
                        gpu_mem_item_delta = gpu_mem_item_end - gpu_mem_item_start
                        logger.info(f"[GPU Memory] Item {batch_idx + 1} end: allocated={gpu_mem_item_end:.2f}GB, reserved={gpu_mem_reserved_end:.2f}GB")
                        logger.info(f"[GPU Memory] Item {batch_idx + 1} delta: {gpu_mem_item_delta:+.3f}GB")
                        
                        # Warning if memory increased significantly
                        if gpu_mem_item_delta > 0.5:
                            logger.warning(f"[GPU Memory] Item {batch_idx + 1} caused significant memory increase: {gpu_mem_item_delta:.2f}GB - potential memory leak!")
                
                # Agent-level GPU memory cleanup (after processing all items for this agent)
                logger.info(f"[Memory Cleanup] Agent {agent_idx + 1}/{len(self.agents)} ({agent.name}) processing complete")
                if torch.cuda.is_available():
                    logger.debug(f"[Memory Cleanup] Forcing GPU cache cleanup after agent {agent.name}")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gpu_mem_after_agent = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"[GPU Memory] After agent {agent.name}: allocated={gpu_mem_after_agent:.2f}GB")
            
            else:
                # Judger agent: aggregate multiple paths OR verify individual paths (PRM mode)
                if self.collect_prm_data:
                    logger.info(f"[{agent.name}] PRM Data Collection Mode: Decoding and verifying ALL paths individually")
                    logger.info(f"[{agent.name}] Will NOT aggregate - each path gets individual correctness score")
                else:
                    logger.info(f"[{agent.name}] Starting final answer aggregation")
                    logger.debug(f"[{agent.name}] Will aggregate paths from all previous agents")
                
                # Wrap prompts with <think> tag if needed
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                
                # Process each item in batch
                for batch_idx in range(batch_size):
                    item_paths = batch_paths[batch_idx]
                    
                    # Log GPU memory before judger processing
                    if torch.cuda.is_available():
                        gpu_mem_judger_start = torch.cuda.memory_allocated() / 1024**3
                        logger.debug(f"[GPU Memory] Judger item {batch_idx + 1} start: {gpu_mem_judger_start:.2f}GB")
                    
                    if self.collect_prm_data:
                        logger.info(f"[{agent.name}] PRM Mode: Verifying {len(item_paths)} paths individually for item {batch_idx + 1}/{batch_size}")
                    else:
                        logger.info(f"[{agent.name}] Aggregating paths for item {batch_idx + 1}/{batch_size}")
                    logger.info(f"[{agent.name}] Available paths: {len(item_paths)}")
                    
                    if not item_paths:
                        logger.warning(f"[{agent.name}] No paths available for item {batch_idx}, using standard generation")
                        # Fallback to standard generation
                        generated_batch, _ = self.model.generate_text_batch(
                            judger_ids[batch_idx:batch_idx+1],
                            judger_mask[batch_idx:batch_idx+1],
                            max_new_tokens=self.judger_max_new_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            past_key_values=None,
                        )
                        final_texts[batch_idx] = generated_batch[0].strip()
                    else:
                        # Generate answers from all paths
                        logger.info(f"[{agent.name}] Generating answers from {len(item_paths)} paths")
                        
                        path_answers = []
                        path_scores = []
                        
                        for path_idx, path in enumerate(item_paths):
                            logger.debug(f"[{agent.name}] Generating answer from path {path.path_id} (score: {path.score:.4f})")
                            # Generate answer from this path
                            generated_batch, new_kv_cache = self.model.generate_text_batch(
                                judger_ids[batch_idx:batch_idx+1],
                                judger_mask[batch_idx:batch_idx+1],
                                max_new_tokens=self.judger_max_new_tokens,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                past_key_values=path.kv_cache if self.latent_steps > 0 else None,
                            )
                            answer = generated_batch[0].strip()
                            path_answers.append(answer)
                            path_scores.append(path.score)

                            if len(answer) <= 200:
                                answer_display = answer
                            else:
                                answer_display = f"{answer[:100]}...{answer[-100:]}"

                            logger.info(f"[{agent.name}] Path {path.path_id} generated answer: {answer_display}")
                            logger.debug(f"[{agent.name}] Path {path.path_id} answer length: {len(answer)} chars")
                            
                            # Immediately clean up the new KV cache from generation to prevent accumulation
                            # We don't need it after getting the answer
                            if new_kv_cache is not None:
                                del new_kv_cache
                        
                        # PRM Mode: Verify each path individually and store results
                        if self.collect_prm_data:
                            logger.info(f"[{agent.name}] PRM Mode: Verifying individual path correctness")
                            logger.info("=" * 80)
                            logger.info(f"[{agent.name}] INDIVIDUAL PATH VERIFICATION REPORT")
                            logger.info("=" * 80)
                            
                            gold_answer = items[batch_idx].get("gold", "")
                            
                            # CRITICAL FIX: Create Judger layer path records
                            # Previously, is_correct was stored in parent path metadata
                            # Now we create new Judger paths as children of parent paths
                            judger_paths_created = []
                            judger_results = []  # Store (extracted_answer, is_correct) for each judger path
                            
                            for path_idx, (parent_path, raw_answer) in enumerate(zip(item_paths, path_answers)):
                                # Extract and normalize answer based on task type
                                extracted_answer, is_path_correct = self._verify_path_answer(
                                    raw_answer=raw_answer,
                                    gold_answer=gold_answer,
                                    task=self.task,
                                    item=items[batch_idx]
                                )
                                
                                # Store result for later use
                                judger_results.append((extracted_answer, is_path_correct))
                                
                                # Create a new Judger layer path as child of parent_path
                                # Generate unique path ID for this Judger path
                                judger_path_id = self.path_manager.next_path_id
                                self.path_manager.next_path_id += 1
                                
                                # Judger paths don't generate new latent vectors, 
                                # they just decode from parent's latent state
                                # So we use parent's hidden states and empty latent history
                                judger_latent_history = []  # Judger doesn't add new latent steps
                                judger_hidden_states = parent_path.hidden_states  # Use parent's final state
                                
                                # Compute a simple score for the Judger path
                                # Use parent's score as base (could be enhanced later)
                                judger_score = parent_path.score
                                
                                # Create metadata for Judger path
                                judger_metadata = {
                                    'decoded_answer': raw_answer,
                                    'extracted_answer': extracted_answer,
                                    'is_correct': is_path_correct,
                                    'parent_path_id': parent_path.path_id,
                                    'gold_answer': gold_answer,
                                }
                                
                                # Record the Judger path to data collector
                                if self.prm_data_collector:
                                    self.prm_data_collector.record_path(
                                        path_id=judger_path_id,
                                        agent_name=agent.name,
                                        agent_idx=agent_idx,
                                        parent_path_id=parent_path.path_id,
                                        latent_history=judger_latent_history,
                                        hidden_states=judger_hidden_states,
                                        score=judger_score,
                                        metadata=judger_metadata
                                    )
                                    judger_paths_created.append(judger_path_id)
                                    logger.debug(f"[{agent.name}] Created Judger path {judger_path_id} "
                                               f"as child of path {parent_path.path_id}")
                                
                                # Log individual path verification
                                logger.info(f"[{agent.name}] Path {parent_path.path_id} → Judger path {judger_path_id} "
                                          f"(#{path_idx + 1}/{len(item_paths)}):")
                                logger.info(f"  - Raw answer: {raw_answer[:100]}{'...' if len(raw_answer) > 100 else ''}")
                                logger.info(f"  - Extracted: {extracted_answer}")
                                logger.info(f"  - Gold: {gold_answer}")
                                logger.info(f"  - Result: {'✓ CORRECT' if is_path_correct else '✗ INCORRECT'}")
                                logger.debug(f"  - Parent score: {parent_path.score:.4f}")
                            
                            logger.info(f"[{agent.name}] Created {len(judger_paths_created)} Judger layer paths")
                            logger.debug(f"[{agent.name}] Judger path IDs: {judger_paths_created}")
                            
                            logger.info("=" * 80)
                            
                            # Count correct paths using judger_results
                            num_correct = sum(1 for (_, is_correct) in judger_results if is_correct)
                            logger.info(f"[{agent.name}] Path verification summary: {num_correct}/{len(judger_results)} paths correct")
                            logger.info(f"[{agent.name}] Individual path accuracy: {num_correct / len(judger_results) * 100:.1f}%")
                            
                            # For final answer in PRM mode, use majority voting (but don't use it for scoring)
                            # This is just for reporting purposes
                            correct_answers = [extracted_answer for (extracted_answer, is_correct) in judger_results if is_correct]
                            if correct_answers:
                                # Use the most common correct answer
                                from collections import Counter
                                answer_counts = Counter(correct_answers)
                                final_texts[batch_idx] = answer_counts.most_common(1)[0][0]
                                logger.info(f"[{agent.name}] Using most common correct answer for reporting: {final_texts[batch_idx]}")
                            else:
                                # No correct paths, use the answer from highest-scored path
                                best_path_idx = path_scores.index(max(path_scores))
                                final_texts[batch_idx] = judger_results[best_path_idx][0]  # Use extracted_answer from judger_results
                                logger.warning(f"[{agent.name}] No correct paths found, using answer from best-scored path")
                        
                        else:
                            # Normal inference mode: Voting to aggregate
                            logger.info(f"[{agent.name}] Performing weighted voting across {len(path_answers)} answers")
                            answer_votes = {}
                            for answer, score in zip(path_answers, path_scores):
                                if answer not in answer_votes:
                                    answer_votes[answer] = 0.0
                                answer_votes[answer] += score
                            
                            logger.debug(f"[{agent.name}] Vote distribution: {answer_votes}")
                            
                            # Select answer with highest weighted vote
                            best_answer = max(answer_votes.items(), key=lambda x: x[1])[0]
                            final_texts[batch_idx] = best_answer

                            if len(best_answer) <= 200:
                                best_answer_display = best_answer
                            else:
                                best_answer_display = f"{best_answer[:100]}...{best_answer[-100:]}"
                            logger.info(f"[{agent.name}] Selected final answer: {best_answer_display}")
                            logger.info(f"[{agent.name}] Winning vote weight: {answer_votes[best_answer]:.4f}")

                    # Log GPU memory after judger processing
                    if torch.cuda.is_available():
                        gpu_mem_judger_end = torch.cuda.memory_allocated() / 1024**3
                        gpu_mem_judger_delta = gpu_mem_judger_end - gpu_mem_judger_start
                        logger.debug(f"[GPU Memory] Judger item {batch_idx + 1} end: {gpu_mem_judger_end:.2f}GB (delta: {gpu_mem_judger_delta:+.3f}GB)")
                    
                    # Record judger trace
                    logger.debug(f"[{agent.name}] Recording trace for item {batch_idx + 1}")
                    agent_traces[batch_idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[batch_idx],
                        "num_paths_aggregated": len(item_paths),
                        "output": final_texts[batch_idx],
                    })
        
        # Prepare results
        logger.info("=" * 80)
        logger.info("Preparing final results")
        logger.info("=" * 80)
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            logger.debug(f"Processing result for item {idx + 1}/{len(items)}")
            
            # Special handling for cot_fact_wiki task: skip extraction and evaluation
            if self.task == "cot_fact_wiki":
                logger.info(f"Item {idx + 1}: Task is cot_fact_wiki, saving raw judger output directly")
                logger.debug(f"Item {idx + 1}: Raw judger output length: {len(final_text)} characters")
                
                # For cot_fact_wiki, save the raw judger output directly without extraction
                pred = final_text
                gold = item.get("gold", "")
                ok = None  # No evaluation for this task
                error_msg = None
                
                logger.info(f"Item {idx + 1}: Skipping answer extraction and correctness checking for cot_fact_wiki task")
                logger.debug(f"Item {idx + 1}: Raw output preview: {final_text[:200]}...")
            
            else:
                # Extract prediction based on task for other tasks
                logger.info(f"Item {idx + 1}: Extracting answer from final text")
                logger.debug(f"Item {idx + 1}: Final text: {final_text}")
                
                if self.task in ['mbppplus', 'humanevalplus']:
                    pred = extract_markdown_python_block(final_text)
                    gold = item.get("gold", "")
                    
                    if pred is None:
                        ok = False
                        error_msg = "python error: No python code block found"
                        logger.warning(f"Item {idx + 1}: No Python code block found in output")
                    else:
                        python_code_to_exe = pred + "\n" + gold
                        ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                    
                    logger.info(f'Item {idx + 1}: Evaluation result: {"CORRECT" if ok else "INCORRECT"}')
                    if error_msg:
                        logger.debug(f'Item {idx + 1}: Error: {error_msg}')
                
                elif self.task in ["aime2024", "aime2025"]:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                    gold = str(item.get("gold", "")).strip()
                    try:
                        pred_int = int(pred)
                        gold_int = int(gold)
                        ok = (pred_int == gold_int)
                        error_msg = None
                    except ValueError:
                        ok = False
                        error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'
                    
                    logger.info(f'Item {idx + 1}: Evaluation result: {"CORRECT" if ok else "INCORRECT"} (pred={pred}, gold={gold})')
                    if error_msg:
                        logger.debug(f'Item {idx + 1}: {error_msg}')
                
                else:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                    gold = item.get("gold", "")
                    ok = (pred == gold) if (pred and gold) else False
                    error_msg = None
                    
                    logger.info(f'Item {idx + 1}: Evaluation result: {"CORRECT" if ok else "INCORRECT"} (pred={pred}, gold={gold})')
            
            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
                "num_paths_used": len(batch_paths[idx]) if batch_paths[idx] else 0,
            })
            
            # PRM data collection: Finish collecting for this question
            # Skip PRM data collection for cot_fact_wiki task (no correctness evaluation)
            if self.collect_prm_data and self.prm_data_collector and self.task != "cot_fact_wiki":
                logger.info(f"[PRM DataCollection] Finishing data collection for item {idx + 1}")
                logger.debug(f"[PRM DataCollection] Final answer: {pred}, Correct: {ok}")
                self.prm_data_collector.finish_question(
                    final_answer=pred if pred else "",
                    is_correct=ok
                )
                logger.info(f"[PRM DataCollection] Data collection finished for item {idx + 1}")
        
        # Calculate accuracy (skip None values for cot_fact_wiki task)
        correct_count = sum(1 for r in results if r['correct'] is True)
        evaluated_count = sum(1 for r in results if r['correct'] is not None)
        
        if self.task == "cot_fact_wiki":
            logger.info(f"Batch complete: {len(results)} items processed (no evaluation for cot_fact_wiki task)")
        else:
            logger.info(f"Batch complete: accuracy={correct_count}/{evaluated_count}")
        
        # Clean up all paths from this batch to free GPU memory
        logger.info("=" * 80)
        logger.info("[Memory Cleanup] Starting comprehensive batch cleanup")
        logger.info("=" * 80)
        paths_before_cleanup = len(self.path_manager.paths)
        logger.info(f"[Memory Cleanup] Paths in PathManager before cleanup: {paths_before_cleanup}")
        
        # Collect all path IDs from this batch
        batch_path_ids = []
        for item_paths in batch_paths:
            for path in item_paths:
                batch_path_ids.append(path.path_id)
        
        logger.info(f"[Memory Cleanup] Collected {len(batch_path_ids)} path IDs from batch")
        
        # Explicitly delete all tensors from all batch paths before removing from PathManager
        logger.debug("[Memory Cleanup] Explicitly deleting tensors from all batch paths")
        tensor_count = 0
        for path_id in batch_path_ids:
            path = self.path_manager.paths.get(path_id)
            if path:
                # Delete latent history tensors
                if path.latent_history:
                    tensor_count += len(path.latent_history)
                    for tensor in path.latent_history:
                        if tensor is not None:
                            del tensor
                    path.latent_history.clear()
                
                # Delete hidden states
                if path.hidden_states is not None:
                    tensor_count += 1
                    del path.hidden_states
                    path.hidden_states = None
                
                # Delete KV cache (contains multiple layer tensors)
                if path.kv_cache is not None:
                    tensor_count += 1
                    del path.kv_cache
                    path.kv_cache = None
        
        logger.info(f"[Memory Cleanup] Deleted {tensor_count} tensor references from {len(batch_path_ids)} paths")
        
        # Remove paths from path manager
        if batch_path_ids:
            cleared_count = self.path_manager.clear_paths(batch_path_ids)
            logger.info(f"[Memory Cleanup] Cleared {cleared_count} paths from PathManager")
        
        # Clear batch_paths list
        for item_paths in batch_paths:
            item_paths.clear()
        batch_paths.clear()
        logger.debug("[Memory Cleanup] Cleared batch_paths list")
        
        paths_after_cleanup = len(self.path_manager.paths)
        logger.info(f"[PathManager] Remaining paths after cleanup: {paths_after_cleanup}")
        
        # Clear reasoning graph if it exists
        if self.reasoning_graph is not None:
            nodes_count = len(self.reasoning_graph.nodes)
            if nodes_count > 0:
                logger.info(f"[Memory Cleanup] Clearing reasoning graph with {nodes_count} nodes")
                self.reasoning_graph.clear()
                # Recreate empty graph for next batch
                self.reasoning_graph = ReasoningGraph()
                logger.info(f"[Memory Cleanup] Reasoning graph cleared and recreated")
            else:
                logger.debug("[Memory Cleanup] Reasoning graph is empty, no cleanup needed")
        
        # Force GPU cache cleanup and synchronization
        if torch.cuda.is_available():
            logger.debug("[GPU Memory] Forcing CUDA cache cleanup")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_freed = (gpu_mem_allocated - gpu_mem_reserved) if gpu_mem_reserved > 0 else 0
            
            logger.info(f"[GPU Memory] After batch cleanup:")
            logger.info(f"[GPU Memory]   - Allocated: {gpu_mem_allocated:.2f}GB")
            logger.info(f"[GPU Memory]   - Reserved: {gpu_mem_reserved:.2f}GB")
            logger.info(f"[GPU Memory]   - Memory freed: {abs(gpu_mem_freed):.2f}GB")
        
        logger.info("=" * 80)
        logger.info("[Memory Cleanup] Batch cleanup complete")
        logger.info("=" * 80)
        
        return results
    
    @torch.no_grad()
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        """Run multi-path reasoning on a batch using vLLM backend.
        
        This method implements multi-path reasoning with vLLM for efficient inference.
        It handles embedding concatenation for multiple paths and uses vLLM's
        batch processing capabilities.
        
        Args:
            items: List of input items with 'question' field
            
        Returns:
            List of result dictionaries with predictions and traces
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        
        batch_size = len(items)
        logger.info(f"Processing batch of {batch_size} items with {self.num_paths} paths per item (vLLM)")
        
        # CRITICAL: Clear PathManager before batch to ensure complete independence between batches
        # This prevents cross-batch accumulation of paths and GPU memory leaks
        if len(self.path_manager.paths) > 0:
            logger.warning(f"[Memory Leak Prevention][vLLM] PathManager has {len(self.path_manager.paths)} paths before batch start - clearing now")
            self.path_manager.clear()
            
            # Force GPU cleanup after clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"[Memory Leak Prevention][vLLM] Forced GPU cleanup after PathManager clear")
        
        # Reset reasoning graph to ensure clean state
        if self.reasoning_graph is not None and len(self.reasoning_graph.nodes) > 0:
            logger.warning(f"[Memory Leak Prevention][vLLM] ReasoningGraph has {len(self.reasoning_graph.nodes)} nodes before batch start - clearing now")
            self.reasoning_graph.clear()
            self.reasoning_graph = ReasoningGraph()
        
        # Log GPU memory before batch processing
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU Memory][vLLM] Before batch (after pre-cleanup): allocated={gpu_mem_allocated:.2f}GB, reserved={gpu_mem_reserved:.2f}GB")
            logger.debug(f"[PathManager][vLLM] Paths in manager after pre-cleanup: {len(self.path_manager.paths)}")
        
        # Initialize paths for each item in batch
        batch_paths: List[List[PathState]] = [[] for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        
        # Track embeddings for vLLM
        embedding_records: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        
        # Process each agent
        for agent_idx, agent in enumerate(self.agents):
            logger.info(f"Agent {agent_idx + 1}/{len(self.agents)}: {agent.name} ({agent.role}) [vLLM]")
            
            # Build messages for this agent
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args
                    )
                    for item in items
                ]
            
            # Prepare input
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            
            if agent.role != "judger":
                # Non-judger agent: multi-path latent reasoning
                logger.debug(f"Generating multi-path latent reasoning for agent {agent.name} [vLLM]")
                
                # Wrap prompts with <think> tag if needed
                if self.args.think:
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts
                
                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                
                # Process each item in batch
                for batch_idx in range(batch_size):
                    logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Processing item {batch_idx + 1}/{batch_size}")
                    
                    # Get input for this item
                    item_ids = wrapped_ids[batch_idx:batch_idx+1]
                    item_mask = wrapped_mask[batch_idx:batch_idx+1]
                    
                    # Get past KV cache from previous paths (if any)
                    past_kv = None
                    if batch_paths[batch_idx]:
                        # Use KV cache from the best path so far
                        best_path = max(batch_paths[batch_idx], key=lambda p: p.score)
                        past_kv = best_path.kv_cache
                        logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Using KV cache from path "
                                   f"{best_path.path_id} (score: {best_path.score:.4f})")
                    
                    # Determine if we should branch
                    should_branch = self._should_branch(
                        batch_paths[batch_idx],
                        agent_idx,
                        len(self.agents)
                    )
                    
                    # Generate paths and collect embeddings
                    path_embeddings = []
                    
                    if should_branch and past_kv is not None:
                        # Branch from existing paths
                        logger.debug(f"Branching from existing paths [vLLM]")
                        
                        # For vLLM, we need to generate embeddings for each branch
                        for branch_idx in range(self.num_paths):
                            # Generate latent path with hidden states
                            past_kv_branch, hidden_embedding = self.model.generate_latent_batch_hidden_state(
                                input_ids=item_ids,
                                attention_mask=item_mask,
                                latent_steps=self.latent_steps,
                                past_key_values=past_kv,
                            )
                            
                            # Create path state
                            path_id = self.path_manager.next_path_id
                            self.path_manager.next_path_id += 1
                            path_state = PathState(
                                path_id=path_id,
                                latent_history=[],
                                hidden_states=None,
                                kv_cache=past_kv_branch,
                                metadata={'branch_idx': branch_idx},
                            )
                            
                            # Store embedding
                            path_embeddings.append(hidden_embedding)
                            
                            # Add to paths
                            batch_paths[batch_idx].append(path_state)
                            self.path_manager.paths[path_id] = path_state
                            logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Created branch path {path_id}")
                    else:
                        # Generate new diverse paths
                        logger.debug(f"Generating {self.num_paths} diverse paths [vLLM]")
                        
                        for path_idx in range(self.num_paths):
                            # Generate latent path with hidden states
                            past_kv_path, hidden_embedding = self.model.generate_latent_batch_hidden_state(
                                input_ids=item_ids,
                                attention_mask=item_mask,
                                latent_steps=self.latent_steps,
                                past_key_values=past_kv,
                            )
                            
                            # Create path state
                            path_id = self.path_manager.next_path_id
                            self.path_manager.next_path_id += 1
                            path_state = PathState(
                                path_id=path_id,
                                latent_history=[],
                                hidden_states=None,
                                kv_cache=past_kv_path,
                                metadata={'path_idx': path_idx},
                            )
                            
                            # Store embedding
                            path_embeddings.append(hidden_embedding)
                            
                            # Add to paths
                            batch_paths[batch_idx].append(path_state)
                            self.path_manager.paths[path_id] = path_state
                            logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Created path {path_id}")
                    
                    # For vLLM, we need to aggregate embeddings from multiple paths
                    # Strategy: Use the embedding from the best-scored path
                    # (Scoring will happen after this, so for now we use the first path)
                    if path_embeddings:
                        # For now, use average of all path embeddings
                        avg_embedding = torch.stack(path_embeddings).mean(dim=0)
                        
                        # Handle sequential_info_only or latent_only modes
                        if self.sequential_info_only or self.latent_only:
                            if self.latent_only and self.latent_steps > 0:
                                avg_embedding = avg_embedding[:, -self.latent_steps:, :]
                            embedding_records[batch_idx] = [avg_embedding]
                        else:
                            embedding_records[batch_idx].append(avg_embedding)
                        
                        logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Stored embeddings for item {batch_idx}")
                    
                    # Score paths (using simplified scoring for vLLM efficiency)
                    logger.debug(f"Scoring {len(batch_paths[batch_idx])} paths [vLLM]")
                    
                    # Compute individual latent consistency scores if enabled
                    individual_consistency_scores = None
                    if self.latent_consistency_scorer is not None and len(batch_paths[batch_idx]) > 1:
                        logger.debug(f"Computing individual latent consistency for {len(batch_paths[batch_idx])} paths [vLLM]")
                        individual_consistency_scores = self.latent_consistency_scorer.score_individual_paths(batch_paths[batch_idx])
                        logger.debug(f"Individual consistency scores: min={min(individual_consistency_scores):.4f}, "
                                   f"max={max(individual_consistency_scores):.4f} [vLLM]")
                    
                    # Score each path
                    for path_idx, path in enumerate(batch_paths[batch_idx]):
                        # Use simplified base scoring for vLLM
                        base_score = 0.5  # Default score
                        
                        # Add individual latent consistency if enabled
                        if individual_consistency_scores is not None:
                            path_consistency = individual_consistency_scores[path_idx]
                            ensemble_weight = 1.0 - self.latent_consistency_weight
                            final_score = (base_score * ensemble_weight + 
                                         path_consistency * self.latent_consistency_weight)
                            path.metadata['latent_consistency'] = path_consistency
                        else:
                            final_score = base_score
                        
                        path.update_state(score=final_score)
                        logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Path {path.path_id} score: {final_score:.4f}")
                    
                    # Clean up consistency scores (vLLM)
                    if individual_consistency_scores is not None:
                        del individual_consistency_scores
                        logger.debug(f"[vLLM] Cleaned up consistency scores list")
                    
                    # Immediate GPU cleanup after scoring (vLLM)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"[vLLM] GPU cache cleaned after scoring")
                    
                    # Detect if current agent is Refiner (倒数第二个 agent)
                    is_refiner = (agent_idx == len(self.agents) - 2)
                    
                    if is_refiner:
                        logger.info(f"[{agent.name}][vLLM] Detected Refiner stage (agent {agent_idx + 1}/{len(self.agents)})")
                        logger.info(f"[{agent.name}][vLLM] Will force pruning to keep ONLY 1 path for final Judger")
                    
                    # Prune low-quality paths
                    logger.debug(f"Pruning paths [vLLM]")
                    
                    # For Refiner, force keep_count=1
                    if is_refiner:
                        logger.info(f"[{agent.name}][vLLM] Forcing pruning to keep exactly 1 path (Refiner strategy)")
                        pruned_paths = self.pruning_strategy.prune(
                            paths=batch_paths[batch_idx],
                            current_step=agent_idx,
                            total_steps=len(self.agents),
                            force_keep_count=1,
                        )
                    else:
                        # Pass k parameter when using topk strategy
                        prune_kwargs = {
                            "paths": batch_paths[batch_idx],
                            "current_step": agent_idx,
                            "total_steps": len(self.agents),
                        }
                        if isinstance(self.pruning_strategy, TopKPruning):
                            prune_kwargs["k"] = self.topk_k
                            logger.debug(f"[{agent.name}][vLLM] Using topk pruning with k={self.topk_k}")
                        pruned_paths = self.pruning_strategy.prune(**prune_kwargs)
                    
                    logger.info(f"Pruning: kept {len(pruned_paths)}/{len(batch_paths[batch_idx])} paths [vLLM]")
                    
                    # Merge similar paths if enabled
                    # Agent role-aware merge timing strategy (vLLM version):
                    # - Planner: Skip merge to preserve diversity for exploration
                    # - Critic: Allow merge to reduce redundancy after evaluation
                    # - Refiner: Skip merge (already pruned to 1 path)
                    # - Judger: No merge needed (final stage)
                    
                    is_planner = (agent.role == "planner")
                    is_critic = (agent.role == "critic")
                    
                    # Determine if we should attempt merging for this agent
                    # Skip merging if in PRM data collection mode
                    should_attempt_merge = (
                        self.enable_merging and 
                        len(pruned_paths) > 1 and 
                        not is_refiner and 
                        not is_planner and  # Skip Planner to preserve diversity
                        is_critic and  # Only merge at Critic stage
                        not (self.collect_prm_data and self._prm_disable_merging)  # Skip if PRM mode
                    )
                    
                    if should_attempt_merge:
                        logger.info(f"[{agent.name}][vLLM] Attempting to merge similar paths (threshold: {self.merge_threshold})")
                        logger.info(f"[{agent.name}][vLLM] Agent role '{agent.role}' allows merging at this stage")
                        logger.debug(f"[{agent.name}][vLLM] Pre-merge path count: {len(pruned_paths)}")
                        
                        merged_paths = self.path_merger.merge_similar_paths(
                            paths=pruned_paths,
                            path_manager=self.path_manager,
                            model_lm_head=self.vllm_model.llm_engine.model_executor.driver_worker.model_runner.model.lm_head if hasattr(self, 'vllm_model') else None,
                            use_kl=False,
                            min_group_size=2
                        )
                        logger.info(f"[{agent.name}][vLLM] Merging complete: reduced to {len(merged_paths)} paths")
                        
                        if len(merged_paths) < len(pruned_paths):
                            num_merged = len(pruned_paths) - len(merged_paths)
                            logger.info(f"[{agent.name}][vLLM] Successfully merged {num_merged} similar paths")
                        else:
                            logger.info(f"[{agent.name}][vLLM] No paths were merged (no suitable merge candidates found)")
                        
                        batch_paths[batch_idx] = merged_paths
                        
                        # Immediate GPU cleanup after merging (vLLM)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug(f"[vLLM] GPU cache cleaned after merge operation")
                    else:
                        # Log reason for skipping merge
                        if is_refiner:
                            logger.info(f"[{agent.name}][vLLM] Skipping merge for Refiner (already has {len(pruned_paths)} path)")
                        elif is_planner:
                            logger.info(f"[{agent.name}][vLLM] Skipping merge for Planner (preserving diversity for exploration)")
                        elif not is_critic:
                            logger.info(f"[{agent.name}][vLLM] Skipping merge for agent role '{agent.role}' (merge only at Critic stage)")
                        elif not self.enable_merging:
                            logger.debug(f"[{agent.name}][vLLM] Path merging disabled globally")
                        else:
                            logger.debug(f"[{agent.name}][vLLM] Only {len(pruned_paths)} path(s), skipping merge")
                        
                        batch_paths[batch_idx] = pruned_paths
                    
                    # Record agent trace
                    agent_traces[batch_idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[batch_idx],
                        "latent_steps": self.latent_steps,
                        "num_paths": len(batch_paths[batch_idx]),
                        "path_scores": [p.score for p in batch_paths[batch_idx]],
                        "output": "",
                    })
                    
                    # Clean up temporary data
                    del wrapped_prompts
                    
                    # Clean up pruned paths to free GPU memory immediately (vLLM)
                    # In vLLM mode, paths are generated directly into batch_paths
                    # After pruning/merging, remove paths that didn't survive
                    kept_path_ids = {p.path_id for p in batch_paths[batch_idx]}
                    
                    # Find all paths created in this iteration that should be removed
                    all_created_path_ids = [p.path_id for p in self.path_manager.paths.values() 
                                           if p.path_id not in kept_path_ids and 
                                           p.metadata.get('batch_idx') == batch_idx]
                    
                    if all_created_path_ids:
                        logger.debug(f"[Memory Cleanup][vLLM] Cleaning up {len(all_created_path_ids)} pruned paths for item {batch_idx + 1}")
                        
                        # Explicitly delete tensors
                        for path_id in all_created_path_ids:
                            path = self.path_manager.paths.get(path_id)
                            if path:
                                if path.latent_history:
                                    for tensor in path.latent_history:
                                        if tensor is not None:
                                            del tensor
                                    path.latent_history.clear()
                                
                                if path.hidden_states is not None:
                                    del path.hidden_states
                                    path.hidden_states = None
                                
                                if path.kv_cache is not None:
                                    del path.kv_cache
                                    path.kv_cache = None
                        
                        # Remove from PathManager
                        removed_count = self.path_manager.clear_paths(all_created_path_ids)
                        logger.info(f"[Memory Cleanup][vLLM] Freed GPU memory from {removed_count} pruned paths for item {batch_idx + 1}")
                    else:
                        logger.debug(f"[Memory Cleanup][vLLM] Item {batch_idx + 1}: keeping all {len(batch_paths[batch_idx])} paths, no cleanup needed")
                    
                    # Clean up path embedding list
                    if path_embeddings:
                        del path_embeddings
                        logger.debug(f"[Memory Cleanup][vLLM] Cleaned up path embeddings for item {batch_idx + 1}")
            
            else:
                # Judger agent: aggregate multiple paths using vLLM
                logger.debug(f"Judger agent: aggregating paths [vLLM]")
                
                # Wrap prompts with <think> tag if needed
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                
                # Process each item in batch
                for batch_idx in range(batch_size):
                    # Get current prompt embedding
                    curr_prompt_emb = self.model.embedding_layer(judger_encoded[batch_idx:batch_idx+1]).squeeze(0).to(self.vllm_device)
                    
                    # Get past embeddings
                    if embedding_records[batch_idx]:
                        past_embedding = torch.cat(embedding_records[batch_idx], dim=1).to(self.vllm_device)
                    else:
                        logger.warning(f"[LatentMASMultiPathMethod.run_batch_vllm] No embeddings for item {batch_idx}")
                        past_embedding = None
                    
                    # Prepare full prompt embedding
                    if past_embedding is not None:
                        # Handle latent embedding insertion position
                        assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, \
                            "latent_embedding_position is only supported for Qwen models currently."
                        
                        # Find insertion point
                        prompt_text = judger_prompts[batch_idx]
                        idx = prompt_text.find("<|im_start|>user\n")
                        left = prompt_text[: idx + len("<|im_start|>user\n")]
                        insert_idx = len(self.model.tokenizer(left)['input_ids'])
                        
                        # Split and concatenate embeddings
                        left_emb = curr_prompt_emb[:insert_idx, :]
                        right_emb = curr_prompt_emb[insert_idx:, :]
                        whole_prompt_emb = torch.cat([left_emb, past_embedding[0], right_emb], dim=0)
                    else:
                        whole_prompt_emb = curr_prompt_emb
                    
                    # Generate using vLLM
                    prompt_embeds_dict = {
                        "prompt_embeds": whole_prompt_emb.unsqueeze(0)
                    }
                    
                    outputs = self.model.vllm_engine.generate(
                        [prompt_embeds_dict],
                        self.sampling_params,
                    )
                    
                    text_out = outputs[0].outputs[0].text.strip()
                    final_texts[batch_idx] = text_out
                    
                    logger.debug(f"Generated answer for item {batch_idx} [vLLM]")
                    
                    # Record judger trace
                    agent_traces[batch_idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": judger_prompts[batch_idx],
                        "num_paths_aggregated": len(batch_paths[batch_idx]),
                        "output": text_out,
                    })
                
                # Agent-level GPU memory cleanup (vLLM version)
                logger.info(f"[Memory Cleanup][vLLM] Agent {agent_idx + 1}/{len(self.agents)} ({agent.name}) processing complete")
                if torch.cuda.is_available():
                    logger.debug(f"[Memory Cleanup][vLLM] Forcing GPU cache cleanup after agent {agent.name}")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gpu_mem_after_agent = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"[GPU Memory][vLLM] After agent {agent.name}: allocated={gpu_mem_after_agent:.2f}GB")
        
        # Prepare results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            
            # Special handling for cot_fact_wiki task: skip extraction and evaluation
            if self.task == "cot_fact_wiki":
                logger.info(f"[LatentMASMultiPathMethod.run_batch_vllm] Item {idx + 1}: Task is cot_fact_wiki, saving raw judger output directly")
                logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Item {idx + 1}: Raw judger output length: {len(final_text)} characters")
                
                # For cot_fact_wiki, save the raw judger output directly without extraction
                pred = final_text
                gold = item.get("gold", "")
                ok = None  # No evaluation for this task
                error_msg = None
                
                logger.info(f"[LatentMASMultiPathMethod.run_batch_vllm] Item {idx + 1}: Skipping answer extraction and correctness checking for cot_fact_wiki task")
                logger.debug(f"[LatentMASMultiPathMethod.run_batch_vllm] Item {idx + 1}: Raw output preview: {final_text[:200]}...")
            
            # Extract prediction based on task for other tasks
            elif self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                
                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                logger.info(f'[LatentMASMultiPathMethod.run_batch_vllm] Item {idx}: correct={ok}')
                if error_msg:
                    logger.debug(f'[LatentMASMultiPathMethod.run_batch_vllm] Error: {error_msg}')
            
            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'
                
                logger.info(f'[LatentMASMultiPathMethod.run_batch_vllm] Item {idx}: correct={ok}, pred={pred}, gold={gold}')
            
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
                
                logger.info(f'[LatentMASMultiPathMethod.run_batch_vllm] Item {idx}: correct={ok}, pred={pred}, gold={gold}')
            
            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
                "num_paths_used": len(batch_paths[idx]) if batch_paths[idx] else 0,
            })
        
        # Calculate accuracy, handling None values for cot_fact_wiki task
        if self.task == "cot_fact_wiki":
            logger.info(f"Batch complete: {len(results)} items processed (no evaluation for cot_fact_wiki task) [vLLM]")
        else:
            correct_count = sum(1 for r in results if r['correct'])
            evaluated_count = sum(1 for r in results if r['correct'] is not None)
            logger.info(f"Batch complete: accuracy={correct_count}/{evaluated_count} [vLLM]")
        
        # Clean up all paths from this batch to free GPU memory
        logger.info("=" * 80)
        logger.info("[Memory Cleanup][vLLM] Starting comprehensive batch cleanup")
        logger.info("=" * 80)
        paths_before_cleanup = len(self.path_manager.paths)
        logger.info(f"[Memory Cleanup][vLLM] Paths in PathManager before cleanup: {paths_before_cleanup}")
        
        # Collect all path IDs from this batch
        batch_path_ids = []
        for item_paths in batch_paths:
            for path in item_paths:
                batch_path_ids.append(path.path_id)
        
        logger.info(f"[Memory Cleanup][vLLM] Collected {len(batch_path_ids)} path IDs from batch")
        
        # Explicitly delete all tensors from all batch paths
        logger.debug("[Memory Cleanup][vLLM] Explicitly deleting tensors from all batch paths")
        tensor_count = 0
        for path_id in batch_path_ids:
            path = self.path_manager.paths.get(path_id)
            if path:
                # Delete latent history tensors
                if path.latent_history:
                    tensor_count += len(path.latent_history)
                    for tensor in path.latent_history:
                        if tensor is not None:
                            del tensor
                    path.latent_history.clear()
                
                # Delete hidden states
                if path.hidden_states is not None:
                    tensor_count += 1
                    del path.hidden_states
                    path.hidden_states = None
                
                # Delete KV cache
                if path.kv_cache is not None:
                    tensor_count += 1
                    del path.kv_cache
                    path.kv_cache = None
        
        logger.info(f"[Memory Cleanup][vLLM] Deleted {tensor_count} tensor references from {len(batch_path_ids)} paths")
        
        # Remove paths from path manager
        if batch_path_ids:
            cleared_count = self.path_manager.clear_paths(batch_path_ids)
            logger.info(f"[Memory Cleanup][vLLM] Cleared {cleared_count} paths from PathManager")
        
        # Clean up embedding records (vLLM-specific)
        logger.debug("[Memory Cleanup][vLLM] Cleaning up embedding records")
        emb_count = 0
        for batch_idx in range(batch_size):
            if embedding_records[batch_idx]:
                emb_count += len(embedding_records[batch_idx])
                for emb in embedding_records[batch_idx]:
                    if emb is not None:
                        del emb
                embedding_records[batch_idx].clear()
        
        logger.info(f"[Memory Cleanup][vLLM] Deleted {emb_count} embedding records")
        embedding_records.clear()
        
        # Clear batch_paths list
        for item_paths in batch_paths:
            item_paths.clear()
        batch_paths.clear()
        logger.debug("[Memory Cleanup][vLLM] Cleared batch_paths list")
        
        paths_after_cleanup = len(self.path_manager.paths)
        logger.info(f"[PathManager][vLLM] Remaining paths after cleanup: {paths_after_cleanup}")
        
        # Clear reasoning graph if it exists
        if self.reasoning_graph is not None:
            nodes_count = len(self.reasoning_graph.nodes)
            if nodes_count > 0:
                logger.info(f"[Memory Cleanup][vLLM] Clearing reasoning graph with {nodes_count} nodes")
                self.reasoning_graph.clear()
                # Recreate empty graph for next batch
                self.reasoning_graph = ReasoningGraph()
                logger.info(f"[Memory Cleanup][vLLM] Reasoning graph cleared and recreated")
            else:
                logger.debug("[Memory Cleanup][vLLM] Reasoning graph is empty, no cleanup needed")
        
        # Force GPU cache cleanup and synchronization
        if torch.cuda.is_available():
            logger.debug("[GPU Memory][vLLM] Forcing CUDA cache cleanup")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_freed = (gpu_mem_allocated - gpu_mem_reserved) if gpu_mem_reserved > 0 else 0
            
            logger.info(f"[GPU Memory][vLLM] After batch cleanup:")
            logger.info(f"[GPU Memory][vLLM]   - Allocated: {gpu_mem_allocated:.2f}GB")
            logger.info(f"[GPU Memory][vLLM]   - Reserved: {gpu_mem_reserved:.2f}GB")
            logger.info(f"[GPU Memory][vLLM]   - Memory freed: {abs(gpu_mem_freed):.2f}GB")
        
        logger.info("=" * 80)
        logger.info("[Memory Cleanup][vLLM] Batch cleanup complete")
        logger.info("=" * 80)
        
        return results
    
    def run_item(self, item: Dict) -> Dict:
        """Run multi-path reasoning on a single item.
        
        Args:
            item: Input item with 'question' field
            
        Returns:
            Result dictionary with prediction and trace
        """
        return self.run_batch([item])[0]

