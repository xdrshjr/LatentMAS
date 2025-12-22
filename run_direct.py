"""
Direct run script with hardcoded questions and parameters.

This script allows running inference on a predefined set of questions
without loading from datasets. Questions and parameters are hardcoded
in the main function.

Supports all methods including the enhanced multi-path latent reasoning:
- baseline: Standard single-pass generation
- text_mas: Text-based multi-agent system
- latent_mas: Single-path latent reasoning
- latent_mas_multipath: Multi-path latent reasoning with pruning and merging

Multi-Path Configuration:
-------------------------
The latent_mas_multipath method explores multiple reasoning paths in parallel,
scoring and pruning them to find the best solution. Key parameters:

- num_paths: Number of parallel paths to explore (default: 5)
- latent_steps: Latent thinking steps per path (default: 10)
- pruning_strategy: How to prune paths ("topk", "adaptive", "diversity", "budget")
- diversity_strategy: How to generate diverse paths ("temperature", "noise", "hybrid")
- enable_branching: Adaptively branch on high uncertainty (default: True)
- enable_merging: Merge similar paths to save compute (default: True)
- merge_threshold: Similarity threshold for merging (0.0-1.0, default: 0.9)
- branch_threshold: Uncertainty threshold for branching (0.0-1.0, default: 0.5)

Configuration Presets:
---------------------
Instead of manually setting parameters, you can use presets:
- "conservative": Few paths, safe pruning (fast, lower quality)
- "balanced": Moderate paths and pruning (good trade-off)
- "aggressive": Many paths, aggressive pruning (slower, higher quality)
- "fast": Optimized for speed
- "quality": Optimized for accuracy

Usage Examples:
--------------
1. Use default multi-path settings:
   Set method="latent_mas_multipath" with default parameters

2. Use a preset configuration:
   Set config_preset="balanced"

3. Use a custom config file:
   Set config="path/to/config.json"

4. Customize individual parameters:
   Modify the args_dict parameters directly
"""

import argparse
import logging
from typing import Dict, List

from run import main
from utils import normalize_answer
from logging_config import setup_logging, create_log_file_path

# Logger will be configured in main_direct()
logger = logging.getLogger(__name__)


def create_question_dict(question: str, gold: str = None, solution: str = None) -> Dict:
    """Create a question dictionary in the expected format.
    
    Args:
        question: The question text.
        gold: Optional gold standard answer (for evaluation).
        solution: Optional solution text.
        
    Returns:
        Dictionary with question, gold, and solution fields.
    """
    result = {"question": question}
    if gold is not None:
        result["gold"] = normalize_answer(gold)
    if solution is not None:
        result["solution"] = solution
    return result


def main_direct():
    """Main function with hardcoded questions and parameters.
    
    Modify the questions list and args_dict below to customize
    the questions and parameters for your run.
    
    Example Configurations:
    ----------------------
    
    # Example 1: Fast multi-path (for quick testing)
    # method="latent_mas_multipath", num_paths=3, pruning_strategy="topk",
    # enable_branching=False, enable_merging=False
    
    # Example 2: Balanced multi-path (recommended default)
    # method="latent_mas_multipath", num_paths=5, pruning_strategy="adaptive",
    # enable_branching=True, enable_merging=True
    
    # Example 3: High-quality multi-path (for important tasks)
    # method="latent_mas_multipath", num_paths=10, pruning_strategy="diversity",
    # enable_branching=True, enable_merging=True
    
    # Example 4: Using a preset
    # method="latent_mas_multipath", config_preset="balanced"
    
    # Example 5: Single-path baseline (for comparison)
    # method="latent_mas", latent_steps=10
    """
    # Setup logging early (will be reconfigured in main() but we need it for initial messages)
    setup_logging(
        log_level="DEBUG",
        console_level="INFO",
        log_file=None,
        use_colors=True,
        progress_bar_mode=True
    )
    
    logger.info("=" * 80)
    logger.info("Starting direct run with hardcoded questions")
    logger.info("=" * 80)
    
    # ============================================================================
    # HARDCODED QUESTIONS - Modify this list to add/change questions
    # ============================================================================
    custom_questions: List[Dict] = [
        create_question_dict(
            question="Tom has 15 books. He gives 3 books to his friend and buys 7 new books. How many books does Tom have now?",
            gold="19",
            solution="19"
        ),
        create_question_dict(
            question="There are 30 students in a class. 12 students are boys. How many students are girls?",
            gold="18",
            solution="18"
        ),
        # Add more questions here as needed
        # create_question_dict(
        #     question="Your question here",
        #     gold="expected answer",
        #     solution="solution steps"
        # ),
    ]
    
    logger.info(f"Loaded {len(custom_questions)} hardcoded questions")
    for idx, q in enumerate(custom_questions, 1):
        logger.debug(f"Question {idx}: {q['question'][:50]}...")
    
    # ============================================================================
    # HARDCODED PARAMETERS - Modify this dictionary to change parameters
    # ============================================================================
    # Create a mock args object with hardcoded parameters
    # We'll use argparse.Namespace to create an object similar to parsed args
    args_dict = {
        # Core parameters
        "method": "latent_mas_multipath",  # Options: "baseline", "text_mas", "latent_mas", "latent_mas_multipath"
        "model_name": "/root/autodl-fs/models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",  # Options: "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"
        "max_samples": len(custom_questions),  # Number of questions to process
        "task": "gsm8k",  # Task name (not used in custom mode, but required)
        "prompt": "sequential",  # Options: "sequential", "hierarchical"
        "device": "cuda",  # Device to use
        "split": "test",  # Split name (not used in custom mode, but required)
        
        # Generation parameters
        "max_new_tokens": 8192,  # Maximum tokens to generate
        "latent_steps": 10,  # Number of latent steps (for latent_mas and latent_mas_multipath)
        "temperature": 0.6,  # Sampling temperature
        "top_p": 0.95,  # Top-p sampling parameter
        "generate_bs": 20,  # Batch size for generation
        
        # Method-specific parameters
        "text_mas_context_length": -1,  # Context length limit for text_mas
        "think": False,  # Add think token for LatentMAS
        "latent_space_realign": True,  # Latent space realignment
        
        # Multi-path specific parameters (for latent_mas_multipath)
        "num_paths": 5,  # Number of parallel reasoning paths (3-10 recommended, more=slower but potentially better)
        "enable_branching": True,  # Enable adaptive branching based on uncertainty
        "enable_merging": True,  # Enable path merging for efficiency (reduces redundant computation)
        "pruning_strategy": "adaptive",  # Options: "topk", "adaptive", "diversity", "budget"
                                         # - "topk": Keep top-k paths by score (simple, fast)
                                         # - "adaptive": Adjust pruning rate by progress (recommended)
                                         # - "diversity": Balance score and diversity (good for exploration)
                                         # - "budget": Prune based on computational budget
        "merge_threshold": 0.9,  # Similarity threshold for path merging (0.0-1.0)
                                 # Higher = only merge very similar paths, Lower = merge more aggressively
        "branch_threshold": 0.5,  # Uncertainty threshold for branching (0.0-1.0)
                                  # Higher = branch less often, Lower = branch more often
        "diversity_strategy": "hybrid",  # Options: "temperature", "noise", "hybrid"
                                        # - "temperature": Use different temperatures per path
                                        # - "noise": Add noise to hidden states
                                        # - "hybrid": Combine both strategies (recommended)
        
        # Configuration file support (optional, overrides above multi-path params if provided)
        "config": None,  # Path to JSON/YAML config file (e.g., "config_example.json")
                        # If provided, loads all multi-path settings from file
        "config_preset": None,  # Preset name: "conservative", "balanced", "aggressive", "fast", "quality"
                               # Presets provide pre-tuned configurations for different use cases:
                               # - "conservative": 3 paths, topk pruning (fastest)
                               # - "balanced": 5 paths, adaptive pruning (good default)
                               # - "aggressive": 10 paths, diversity pruning (highest quality)
                               # - "fast": Optimized for speed with minimal quality loss
                               # - "quality": Optimized for accuracy, slower
        
        # System parameters
        "seed": 42,  # Random seed
        "use_vllm": False,  # Use vLLM backend
        "enable_prefix_caching": False,  # Enable prefix caching
        "use_second_HF_model": False,  # Use second HF model
        "device2": "cuda:1",  # Second device
        "tensor_parallel_size": 1,  # Tensor parallel size
        "gpu_memory_utilization": 0.9,  # GPU memory utilization
        "log_level": "INFO",  # Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
    }
    
    logger.info("Hardcoded parameters:")
    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Convert dict to argparse.Namespace
    args = argparse.Namespace(**args_dict)
    
    # Auto-adjust some parameters based on method
    if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
        logger.info(f"Auto-enabled vLLM-specific settings for {args.method}")
    
    # Log multi-path configuration if using latent_mas_multipath
    if args.method == "latent_mas_multipath":
        logger.info("=" * 80)
        logger.info("Multi-Path Configuration:")
        logger.info(f"  Number of paths: {args.num_paths}")
        logger.info(f"  Latent steps per path: {args.latent_steps}")
        logger.info(f"  Pruning strategy: {args.pruning_strategy}")
        logger.info(f"  Diversity strategy: {args.diversity_strategy}")
        logger.info(f"  Adaptive branching: {args.enable_branching}")
        logger.info(f"  Path merging: {args.enable_merging}")
        logger.info(f"  Merge threshold: {args.merge_threshold}")
        logger.info(f"  Branch threshold: {args.branch_threshold}")
        if args.config:
            logger.info(f"  Config file: {args.config}")
        if args.config_preset:
            logger.info(f"  Config preset: {args.config_preset}")
        logger.info("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Starting inference...")
    logger.info("=" * 80)
    
    # Call the main function from run.py with custom questions and args
    try:
        main(custom_questions=custom_questions, args=args)
        logger.info("=" * 80)
        logger.info("Direct run completed successfully")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error during direct run: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main_direct()

