"""
Direct run script with hardcoded questions and parameters.

This script allows running inference on a predefined set of questions
without loading from datasets. Questions and parameters are hardcoded
in the main function.
"""

import argparse
import logging
from typing import Dict, List

from run import main
from utils import normalize_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
    """
    logger.info("=" * 80)
    logger.info("Starting direct run with hardcoded questions")
    logger.info("=" * 80)
    
    # ============================================================================
    # HARDCODED QUESTIONS - Modify this list to add/change questions
    # ============================================================================
    custom_questions: List[Dict] = [
        create_question_dict(
            question="Janet has 16 candies. She eats 4 of them. How many candies does she have left?",
            gold="12",
            solution="12"
        ),
        create_question_dict(
            question="A store has 24 apples. They sell 8 apples in the morning and 6 apples in the afternoon. How many apples are left?",
            gold="10",
            solution="10"
        ),
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
        "method": "baseline",  # Options: "baseline", "text_mas", "latent_mas"
        "model_name": "/root/autodl-fs/models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",  # Options: "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"
        "max_samples": len(custom_questions),  # Number of questions to process
        "task": "gsm8k",  # Task name (not used in custom mode, but required)
        "prompt": "sequential",  # Options: "sequential", "hierarchical"
        "device": "cuda",  # Device to use
        "split": "test",  # Split name (not used in custom mode, but required)
        "max_new_tokens": 4096,  # Maximum tokens to generate
        "latent_steps": 0,  # Number of latent steps (for latent_mas)
        "temperature": 0.6,  # Sampling temperature
        "top_p": 0.95,  # Top-p sampling parameter
        "generate_bs": 20,  # Batch size for generation
        "text_mas_context_length": -1,  # Context length limit for text_mas
        "think": False,  # Add think token for LatentMAS
        "latent_space_realign": False,  # Latent space realignment
        "seed": 42,  # Random seed
        "use_vllm": False,  # Use vLLM backend
        "enable_prefix_caching": False,  # Enable prefix caching
        "use_second_HF_model": False,  # Use second HF model
        "device2": "cuda:1",  # Second device
        "tensor_parallel_size": 1,  # Tensor parallel size
        "gpu_memory_utilization": 0.9,  # GPU memory utilization
        "log_level": "INFO",  # Logging level
    }
    
    logger.info("Hardcoded parameters:")
    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Convert dict to argparse.Namespace
    args = argparse.Namespace(**args_dict)
    
    # Auto-adjust some parameters based on method
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
        logger.info("Auto-enabled vLLM-specific settings for latent_mas")
    
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

