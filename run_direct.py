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
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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


def load_custom_questions_from_json(data_path: str) -> List[Dict]:
    """Load custom questions from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing custom questions.
                   The JSON file should contain an array of question objects,
                   each with at least a "question" field, and optionally
                   "gold" and "solution" fields.
    
    Returns:
        List of question dictionaries in the expected format.
    
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
        ValueError: If the JSON structure is invalid.
    """
    logger.info(f"Loading custom questions from JSON file: {data_path}")
    logger.debug(f"Resolving absolute path for: {data_path}")
    
    # Resolve the path (handles relative paths)
    json_path = Path(data_path).resolve()
    logger.debug(f"Resolved absolute path: {json_path}")
    
    # Check if file exists
    if not json_path.exists():
        error_msg = f"JSON file not found: {json_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.debug(f"JSON file exists, size: {json_path.stat().st_size} bytes")
    
    # Read and parse JSON file
    try:
        logger.debug("Reading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug("JSON file parsed successfully")
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in file {json_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error reading JSON file {json_path}: {e}"
        logger.error(error_msg)
        raise
    
    # Validate and process data
    logger.debug("Validating JSON structure...")
    
    # Handle both array format and object with "custom_questions" key
    if isinstance(data, list):
        questions = data
        logger.debug(f"JSON contains array with {len(questions)} items")
    elif isinstance(data, dict):
        if "custom_questions" in data:
            questions = data["custom_questions"]
            logger.debug(f"JSON contains object with 'custom_questions' key, {len(questions)} items")
        else:
            error_msg = f"JSON object must contain 'custom_questions' key or be an array. Found keys: {list(data.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        error_msg = f"JSON must be an array or an object with 'custom_questions' key. Got type: {type(data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate each question
    logger.debug("Validating question format...")
    validated_questions: List[Dict] = []
    for idx, item in enumerate(questions):
        if not isinstance(item, dict):
            logger.warning(f"Question {idx + 1} is not a dictionary, skipping")
            continue
        
        if "question" not in item:
            logger.warning(f"Question {idx + 1} missing 'question' field, skipping")
            continue
        
        # Normalize the question dict
        question_dict = {
            "question": item["question"]
        }
        
        if "gold" in item and item["gold"] is not None:
            question_dict["gold"] = normalize_answer(str(item["gold"]))
            logger.debug(f"Question {idx + 1} has gold answer: {question_dict['gold']}")
        
        if "solution" in item and item["solution"] is not None:
            question_dict["solution"] = str(item["solution"])
            logger.debug(f"Question {idx + 1} has solution")
        
        validated_questions.append(question_dict)
    
    logger.info(f"Successfully loaded {len(validated_questions)} questions from {json_path}")
    logger.debug(f"Questions loaded: {len(validated_questions)}/{len(questions)} (some may have been skipped)")
    
    if len(validated_questions) == 0:
        logger.warning("No valid questions found in JSON file")
    
    return validated_questions


def main_direct(data_path: Optional[str] = None):
    """Main function that loads questions from JSON file and runs inference.
    
    Args:
        data_path: Optional path to JSON file containing custom questions.
                   If not provided, will parse from command line arguments.
                   Default: "data/custom_questions.json"
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on custom questions from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/custom_questions.json",
        help="Path to JSON file containing custom questions (default: data/custom_questions.json)"
    )
    
    # Parse arguments
    args_parsed = parser.parse_args()
    
    # Use function parameter if provided, otherwise use command line argument
    if data_path is not None:
        final_data_path = data_path
        logger.debug(f"Using data_path from function parameter: {final_data_path}")
    else:
        final_data_path = args_parsed.data_path
        logger.debug(f"Using data_path from command line argument: {final_data_path}")
    
    # Setup logging early (will be reconfigured in main() but we need it for initial messages)
    setup_logging(
        log_level="DEBUG",
        console_level="INFO",
        log_file=None,
        use_colors=True,
        progress_bar_mode=True
    )
    
    logger.info("=" * 80)
    logger.info("Starting direct run with custom questions from JSON file")
    logger.info("=" * 80)
    
    # ============================================================================
    # LOAD QUESTIONS FROM JSON FILE
    # ============================================================================
    logger.info(f"Loading questions from: {final_data_path}")
    try:
        custom_questions: List[Dict] = load_custom_questions_from_json(final_data_path)
    except Exception as e:
        logger.error(f"Failed to load questions from JSON file: {e}")
        raise
    
    logger.info(f"Successfully loaded {len(custom_questions)} questions from JSON file")
    for idx, q in enumerate(custom_questions, 1):
        logger.debug(f"Question {idx}: {q['question'][:50]}...")
        if 'gold' in q:
            logger.debug(f"  Gold answer: {q['gold']}")
        if 'solution' in q:
            logger.debug(f"  Has solution: Yes")
    
    # ============================================================================
    # HARDCODED PARAMETERS - Modify this dictionary to change parameters
    # ============================================================================
    # Create a mock args object with hardcoded parameters
    # We'll use argparse.Namespace to create an object similar to parsed args
    # "/root/autodl-fs/models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    #
    args_dict = {
        # Core parameters
        "method": "latent_mas_multipath",  # Options: "baseline", "text_mas", "latent_mas", "latent_mas_multipath"
        "model_name": "Qwen/Qwen3-0.6B",  # Options: "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"
        "max_samples": len(custom_questions),  # Number of questions to process
        "task": "gsm8k",  # Task name (not used in custom mode, but required)
        "prompt": "sequential",  # Options: "sequential", "hierarchical"
        "device": "cuda",  # Device to use
        "split": "test",  # Split name (not used in custom mode, but required)
        
        # Generation parameters
        "max_new_tokens": 2048,  # Maximum tokens to generate
        "latent_steps": 5,  # Number of latent steps (for latent_mas and latent_mas_multipath)
        "temperature": 0.5,  # Baseline temperature, [base_temperature - 0.3, base_temperature + 0.3] for diversity)
        "top_p": 0.95,  # Top-p sampling parameter
        "generate_bs": 1,  # Batch size for generation
        
        # Method-specific parameters
        "text_mas_context_length": -1,  # Context length limit for text_mas
        "think": False,  # Add think token for LatentMAS
        "latent_space_realign": True,  # Latent space realignment
        
        # Multi-path specific parameters (for latent_mas_multipath)
        "num_paths": 20,  # Number of parallel reasoning paths (3-10 recommended, more=slower but potentially better)
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
        "diversity_strategy": "temperature",  # Options: "temperature", "noise", "hybrid"
                                        # - "temperature": Use different temperatures per path
                                        # - "noise": Add noise to hidden states
                                        # - "hybrid": Combine both strategies (recommended)
        "latent_consistency_metric": "cosine",  # Options: "cosine", "euclidean", "l2", "kl_divergence"
                                                # Similarity metric for latent consistency scoring
                                                # - "cosine": Cosine similarity (default, fast and effective)
                                                # - "euclidean": Euclidean distance
                                                # - "l2": L2 normalized distance
                                                # - "kl_divergence": KL divergence (treats vectors as distributions)
        
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
        
        # Visualization parameters
        "enable_visualization": False,  # Enable/disable visualization graph generation
                                       # Set to False to skip visualization and save time
    }
    
    logger.info("Hardcoded parameters:")
    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Log visualization setting prominently
    if args_dict.get('enable_visualization', True):
        logger.info("=" * 80)
        logger.info("Visualization generation: ENABLED")
        logger.info("Graphs will be generated in output/visualizations/")
        logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info("Visualization generation: DISABLED")
        logger.info("No visualization graphs will be generated")
        logger.info("=" * 80)
    
    # Convert dict to argparse.Namespace
    args = argparse.Namespace(**args_dict)
    
    # Auto-adjust some parameters based on method
    if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
        logger.info(f"Auto-enabled vLLM-specific settings for {args.method}")
    
    # Log temperature configuration
    logger.info("=" * 80)
    logger.info("[Temperature Configuration]")
    logger.info(f"  Baseline temperature: {args.temperature}")
    logger.info(f"  This baseline will be used to generate a series of temperatures")
    logger.info(f"  for diversity strategies in multi-path reasoning")
    logger.info(f"  Temperature range: [{args.temperature - 0.3:.2f}, {args.temperature + 0.3:.2f}]")
    logger.info("=" * 80)
    
    # Log multi-path configuration if using latent_mas_multipath
    if args.method == "latent_mas_multipath":
        logger.info("=" * 80)
        logger.info("Multi-Path Configuration:")
        logger.info(f"  Number of paths: {args.num_paths}")
        logger.info(f"  Latent steps per path: {args.latent_steps}")
        logger.info(f"  Pruning strategy: {args.pruning_strategy}")
        logger.info(f"  Diversity strategy: {args.diversity_strategy}")
        logger.info(f"  Latent consistency metric: {args.latent_consistency_metric}")
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

