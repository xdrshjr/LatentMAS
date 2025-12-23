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
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        # Multi-path specific parameters
        num_paths: int = 5,
        enable_branching: bool = True,
        enable_merging: bool = True,
        pruning_strategy: str = "adaptive",
        scoring_weights: Optional[Dict[str, float]] = None,
        merge_threshold: float = 0.9,
        branch_threshold: float = 0.5,
        diversity_strategy: str = "hybrid",
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
            enable_branching: Enable adaptive branching (default: True)
            enable_merging: Enable path merging (default: True)
            pruning_strategy: Pruning strategy name (default: "adaptive")
            scoring_weights: Custom weights for ensemble scorer
            merge_threshold: Similarity threshold for merging (default: 0.9)
            branch_threshold: Uncertainty threshold for branching (default: 0.5)
            diversity_strategy: Diversity strategy name (default: "hybrid")
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
        self.enable_branching = enable_branching
        self.enable_merging = enable_merging
        self.merge_threshold = merge_threshold
        self.branch_threshold = branch_threshold
        
        # Override method name
        self.method_name = 'latent_mas_multipath'
        
        logger.info(f"Initialized: num_paths={num_paths}, pruning={pruning_strategy}, "
                   f"diversity={diversity_strategy}, branching={enable_branching}, merging={enable_merging}")
        logger.debug(f"  - merge_threshold: {merge_threshold}")
        logger.debug(f"  - branch_threshold: {branch_threshold}")
        
        # Initialize path manager
        self.path_manager = PathManager()
        logger.debug("[LatentMASMultiPathMethod] Initialized PathManager")
        
        # Initialize reasoning graph
        self.reasoning_graph = ReasoningGraph()
        logger.debug("[LatentMASMultiPathMethod] Initialized ReasoningGraph")
        
        # Initialize diversity strategy
        self.diversity_strategy = self._create_diversity_strategy(diversity_strategy)
        logger.debug(f"[LatentMASMultiPathMethod] Created diversity strategy: {diversity_strategy}")
        
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
            latent_consistency_scorer = LatentConsistencyScorer(
                similarity_metric='cosine',
                aggregation_method='mean',
                use_last_latent=True
            )
            self.latent_consistency_scorer = latent_consistency_scorer
            self.latent_consistency_weight = scoring_weights['latent_consistency']
            logger.info("[LatentMASMultiPathMethod] Using latent-based self-consistency (faster, no decoding)")
        else:
            self.latent_consistency_scorer = None
            self.latent_consistency_weight = 0.0
        
        if 'verification' in scoring_weights:
            verification_scorer = VerificationScorer(model_wrapper=model)
            self.ensemble_scorer.add_scorer('verification', verification_scorer, scoring_weights['verification'])
        
        logger.info(f"[LatentMASMultiPathMethod] Initialized EnsembleScorer with weights: {scoring_weights}")
        
        # Initialize pruning strategy
        self.pruning_strategy = self._create_pruning_strategy(pruning_strategy)
        logger.debug(f"[LatentMASMultiPathMethod] Created pruning strategy: {pruning_strategy}")
        
        # Initialize path merger
        similarity_detector = PathSimilarityDetector(cosine_threshold=merge_threshold)
        self.path_merger = PathMerger(
            similarity_detector=similarity_detector,
            merge_strategy=WeightedMergeStrategy(),
        )
        logger.debug(f"[LatentMASMultiPathMethod] Initialized PathMerger with threshold: {merge_threshold}")
    
    def _create_diversity_strategy(self, strategy_name: str):
        """Create diversity strategy based on name.
        
        Args:
            strategy_name: Name of the diversity strategy
            
        Returns:
            Diversity strategy instance
        """
        if strategy_name == "temperature":
            return TemperatureDiversityStrategy()
        elif strategy_name == "noise":
            return NoiseDiversityStrategy()
        elif strategy_name == "hybrid":
            return HybridDiversityStrategy()
        else:
            logger.warning(f"[LatentMASMultiPathMethod] Unknown diversity strategy '{strategy_name}', "
                         f"using 'hybrid' as default")
            return HybridDiversityStrategy()
    
    def _create_pruning_strategy(self, strategy_name: str):
        """Create pruning strategy based on name.
        
        Args:
            strategy_name: Name of the pruning strategy
            
        Returns:
            Pruning strategy instance
        """
        if strategy_name == "topk":
            return TopKPruning(k=self.num_paths)
        elif strategy_name == "adaptive":
            return AdaptivePruning(min_paths=max(2, self.num_paths // 2))
        elif strategy_name == "diversity":
            return DiversityAwarePruning(target_count=self.num_paths)
        elif strategy_name == "budget":
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
                logger.info(f"[{agent.name}] Item {batch_idx + 1} prompt preview: {prompt_preview}...")
            
            # Prepare input
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            
            if agent.role != "judger":
                # Non-judger agent: multi-path latent reasoning
                logger.info(f"[{agent.name}] Starting multi-path latent reasoning")
                logger.info(f"[{agent.name}] Configuration: {self.num_paths} paths, {self.latent_steps} latent steps per path")
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
                    
                    # Get input for this item
                    item_ids = wrapped_ids[batch_idx:batch_idx+1]
                    item_mask = wrapped_mask[batch_idx:batch_idx+1]
                    
                    # Get past KV cache from previous paths (if any)
                    past_kv = None
                    if batch_paths[batch_idx]:
                        # Use KV cache from the best path so far
                        best_path = max(batch_paths[batch_idx], key=lambda p: p.score)
                        past_kv = best_path.kv_cache
                        logger.debug(f"[LatentMASMultiPathMethod.run_batch] Using KV cache from path "
                                   f"{best_path.path_id} (score: {best_path.score:.4f})")
                    
                    # Determine if we should branch
                    should_branch = self._should_branch(
                        batch_paths[batch_idx],
                        agent_idx,
                        len(self.agents)
                    )   # 当路径中的不确定性大于阈值的时候，需要创建一个分支
                    
                    if should_branch and past_kv is not None:
                        # Branch from existing paths
                        logger.info(f"[{agent.name}] Branching from existing paths (uncertainty triggered)")
                        logger.debug(f"[{agent.name}] Creating {self.num_paths} branches with {self.latent_steps} steps each")
                        path_dicts = self.model.generate_latent_with_branching(
                            input_ids=item_ids,
                            attention_mask=item_mask,
                            num_branches=self.num_paths,
                            latent_steps=self.latent_steps,
                            diversity_strategy=self.diversity_strategy,
                            past_key_values=past_kv,
                        )
                    else:
                        # Generate new diverse paths
                        logger.info(f"[{agent.name}] Generating {self.num_paths} diverse reasoning paths")
                        logger.debug(f"[{agent.name}] Each path will perform {self.latent_steps} latent thinking steps")
                        path_dicts = self.model.generate_diverse_latent_paths(
                            input_ids=item_ids,
                            attention_mask=item_mask,
                            num_paths=self.num_paths,
                            latent_steps=self.latent_steps,
                            diversity_strategy=self.diversity_strategy,
                            past_key_values=past_kv,
                        )
                    
                    # Convert to PathState objects
                    new_paths = []
                    for path_dict in path_dicts:
                        path_id = self.path_manager.next_path_id
                        self.path_manager.next_path_id += 1
                        path_state = PathState(
                            path_id=path_id,
                            latent_history=path_dict['latent_history'],
                            hidden_states=path_dict['hidden_states'],
                            kv_cache=path_dict['kv_cache'],
                            metadata=path_dict.get('metadata', {}),
                        )
                        new_paths.append(path_state)
                        self.path_manager.paths[path_id] = path_state
                        logger.debug(f"[LatentMASMultiPathMethod.run_batch] Created path {path_id}")
                    
                    # Score all paths
                    logger.info(f"[{agent.name}] Scoring {len(new_paths)} generated paths")
                    
                    # Compute individual latent consistency scores for each path if enabled
                    individual_consistency_scores = None
                    if self.latent_consistency_scorer is not None and len(new_paths) > 1:
                        logger.info(f"[{agent.name}] Computing individual latent consistency for {len(new_paths)} paths")
                        individual_consistency_scores = self.latent_consistency_scorer.score_individual_paths(new_paths)
                        logger.info(f"[{agent.name}] Individual consistency scores: "
                                  f"min={min(individual_consistency_scores):.4f}, "
                                  f"max={max(individual_consistency_scores):.4f}, "
                                  f"mean={np.mean(individual_consistency_scores):.4f}")
                    
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
                            logger.info(f"[{agent.name}] Path {path.path_id}: base={base_score:.4f}, "
                                      f"consistency={path_consistency:.4f}, final={final_score:.4f}")
                        else:
                            final_score = base_score
                            logger.info(f"[{agent.name}] Path {path.path_id} score: {final_score:.4f}")
                        
                        path.update_state(score=final_score)
                        logger.debug(f"[{agent.name}] Path {path.path_id} metadata: {path.metadata}")
                    
                    # Prune low-quality paths
                    logger.info(f"[{agent.name}] Pruning low-quality paths using {self.pruning_strategy.__class__.__name__}")
                    logger.debug(f"[{agent.name}] Pre-pruning path scores: {[f'{p.path_id}:{p.score:.4f}' for p in new_paths]}")
                    pruned_paths = self.pruning_strategy.prune(
                        paths=new_paths,
                        current_step=agent_idx,
                        total_steps=len(self.agents),
                    )
                    logger.info(f"[{agent.name}] Pruning complete: kept {len(pruned_paths)}/{len(new_paths)} paths")
                    logger.info(f"[{agent.name}] Kept path IDs: {[p.path_id for p in pruned_paths]}")
                    logger.debug(f"[{agent.name}] Post-pruning path scores: {[f'{p.path_id}:{p.score:.4f}' for p in pruned_paths]}")
                    
                    # Merge similar paths if enabled
                    if self.enable_merging and len(pruned_paths) > 1:
                        logger.info(f"[{agent.name}] Attempting to merge similar paths (threshold: {self.merge_threshold})")
                        logger.debug(f"[{agent.name}] Pre-merge path count: {len(pruned_paths)}")
                        merged_paths = self.path_merger.merge_similar_paths(
                            paths=pruned_paths,
                            path_manager=self.path_manager,
                            model_lm_head=self.model.model.lm_head,
                            use_kl=False,
                            min_group_size=2
                        )
                        logger.info(f"[{agent.name}] Merging complete: reduced to {len(merged_paths)} paths")
                        if len(merged_paths) < len(pruned_paths):
                            logger.info(f"[{agent.name}] Successfully merged {len(pruned_paths) - len(merged_paths)} similar paths")
                        batch_paths[batch_idx] = merged_paths
                    else:
                        if not self.enable_merging:
                            logger.debug(f"[{agent.name}] Path merging disabled")
                        else:
                            logger.debug(f"[{agent.name}] Only {len(pruned_paths)} path(s), skipping merge")
                        batch_paths[batch_idx] = pruned_paths
                    
                    # Record agent trace
                    final_path_scores = [p.score for p in batch_paths[batch_idx]]
                    logger.info(f"[{agent.name}] Final state for item {batch_idx + 1}: {len(batch_paths[batch_idx])} active paths")
                    logger.info(f"[{agent.name}] Final path scores: {[f'{score:.4f}' for score in final_path_scores]}")
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
            
            else:
                # Judger agent: aggregate multiple paths
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
                        # Aggregate multiple paths using voting
                        logger.info(f"[{agent.name}] Generating answers from {len(item_paths)} paths")
                        
                        path_answers = []
                        path_scores = []
                        
                        for path_idx, path in enumerate(item_paths):
                            logger.debug(f"[{agent.name}] Generating answer from path {path.path_id} (score: {path.score:.4f})")
                            # Generate answer from this path
                            generated_batch, _ = self.model.generate_text_batch(
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
                            logger.info(f"[{agent.name}] Path {path.path_id} generated answer: {answer}")
                            logger.debug(f"[{agent.name}] Path {path.path_id} answer length: {len(answer)} chars")
                        
                        # Voting: weight by path scores
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
                        
                        logger.info(f"[{agent.name}] Selected final answer: {best_answer}")
                        logger.info(f"[{agent.name}] Winning vote weight: {answer_votes[best_answer]:.4f}")
                    
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
            
            # Extract prediction based on task
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
        
        logger.info(f"Batch complete: accuracy={sum(r['correct'] for r in results)}/{len(results)}")
        
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
                    
                    # Prune low-quality paths
                    logger.debug(f"Pruning paths [vLLM]")
                    pruned_paths = self.pruning_strategy.prune(
                        paths=batch_paths[batch_idx],
                        step=agent_idx,
                        total_steps=len(self.agents),
                    )
                    logger.info(f"Pruning: kept {len(pruned_paths)}/{len(batch_paths[batch_idx])} paths [vLLM]")
                    
                    # Merge similar paths if enabled
                    if self.enable_merging and len(pruned_paths) > 1:
                        logger.debug(f"Merging similar paths [vLLM]")
                        merged_paths = self.path_merger.merge_similar_paths(
                            paths=pruned_paths,
                            path_manager=self.path_manager,
                            model_lm_head=self.vllm_model.llm_engine.model_executor.driver_worker.model_runner.model.lm_head if hasattr(self, 'vllm_model') else None,
                            use_kl=False,
                            min_group_size=2
                        )
                        logger.info(f"Merging: reduced to {len(merged_paths)} paths [vLLM]")
                        batch_paths[batch_idx] = merged_paths
                    else:
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
        
        # Prepare results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            
            # Extract prediction based on task
            if self.task in ['mbppplus', 'humanevalplus']:
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
        
        logger.info(f"Batch complete: accuracy={sum(r['correct'] for r in results)}/{len(results)} [vLLM]")
        
        return results
    
    def run_item(self, item: Dict) -> Dict:
        """Run multi-path reasoning on a single item.
        
        Args:
            item: Input item with 'question' field
            
        Returns:
            Result dictionary with prediction and trace
        """
        return self.run_batch([item])[0]

