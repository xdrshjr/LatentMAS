"""
Fact-Checking Inference Script
Supports fact-checking inference using check models with proper system prompts and answer extraction
"""

import torch
import logging
from transformers import pipeline
from typing import Optional, Dict, Any, Tuple
import argparse
import sys
import os

# Add parent directory to path for logging_config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import get_logger, setup_logging

# Setup logging
setup_logging(log_level="INFO", console_level="INFO")
logger = get_logger(__name__)

# System prompt for fact-checking
FACT_CHECK_SYSTEM_PROMPT = (
    "You are a helpful assistant. You need to determine whether the user's input "
    "constitutes factual content based on their input. Return the result enclosed "
    "within <fact></fact>, either <fact>False</fact> or <fact>True</fact>."
)


class FactCheckerInferencer:
    """Fact-checking model inference class using check models"""

    def __init__(
            self,
            check_model_name: str,
            device: Optional[int] = None,
            torch_dtype: Optional[torch.dtype] = None,
            trust_remote_code: bool = True
    ):
        """
        Initialize fact-checking inferencer

        Args:
            check_model_name: Check model name or path, e.g., "jdqqjr/Qwen2.5-0.5B-Instruct_8epoch_Fact_Checker"
            device: Device ID to use (None for auto, int for specific GPU)
            torch_dtype: Data type for model (None for auto, torch.float16, torch.bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        self.check_model_name = check_model_name
        self.device = device
        self.trust_remote_code = trust_remote_code

        logger.info(f"Loading check model: {check_model_name}")
        logger.debug(f"Device: {device}, torch_dtype: {torch_dtype}")

        # Prepare model loading arguments
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        # Determine device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        # Load check model using pipeline API
        try:
            self.checker = pipeline(
                "text-generation",
                model=check_model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                **model_kwargs
            )
            logger.info(f"Check model loaded successfully on device {device}")
        except Exception as e:
            logger.error(f"Failed to load check model: {e}")
            raise

    def extract_fact_result(self, response: str) -> str:
        """
        Extract fact-checking result from model response

        Args:
            response: Model response string

        Returns:
            Extracted result: "True", "False", or "Unknown"
        """
        logger.debug(f"Extracting fact result from response: {response[:200]}...")

        # Extract result from <fact>True</fact> or <fact>False</fact>
        if "<fact>True</fact>" in response:
            result = "True"
            logger.debug("Extracted result: True")
        elif "<fact>False</fact>" in response:
            result = "False"
            logger.debug("Extracted result: False")
        else:
            result = "Unknown"
            logger.warning(f"Could not extract fact result from response: {response[:200]}")

        return result

    def check_fact(
            self,
            question: str,
            answer: str,
            max_new_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.9
    ) -> Tuple[str, str]:
        """
        Check if an answer is factually correct for a given question

        Args:
            question: The question to check
            answer: The answer to verify
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Tuple of (result, raw_response) where:
            - result: "True", "False", or "Unknown"
            - raw_response: Raw model response
        """
        logger.info(f"Checking fact for question: {question[:100]}...")
        logger.debug(f"Answer to check: {answer[:200]}...")

        # Format prompt according to old_check.py pattern
        user_prompt = f"<question>{question}</question><answer>{answer}</answer>"

        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        logger.debug(f"System prompt: {FACT_CHECK_SYSTEM_PROMPT}")
        logger.debug(f"User prompt: {user_prompt[:200]}...")

        try:
            # Generate response using pipeline
            output = self.checker(
                messages,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

            # Extract generated text
            raw_response = output[0]["generated_text"]
            logger.debug(f"Raw response: {raw_response[:200]}...")

            # Extract fact result
            result = self.extract_fact_result(raw_response)

            logger.info(f"Fact-check result: {result}")
            return result, raw_response

        except Exception as e:
            logger.error(f"Error during fact-checking: {e}")
            return "Unknown", ""


def main():
    """
    Main function to run fact-checking inference with example data
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fact-Checking Inference Script"
    )

    # Model configuration
    parser.add_argument(
        "--check_model_name",
        type=str,
        default="jdqqjr/Qwen2.5-0.5B-Instruct_8epoch_Fact_Checker",
        help="Check model name or path"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device ID to use (None for auto, 0 for GPU 0, etc.)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", None],
        help="Data type for model (None for auto)"
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )

    args = parser.parse_args()

    # Convert torch_dtype string to dtype
    torch_dtype = None
    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    # Initialize inferencer
    logger.info("Initializing fact-checking inferencer...")
    inferencer = FactCheckerInferencer(
        check_model_name=args.check_model_name,
        device=args.device,
        torch_dtype=torch_dtype
    )

    # Example data for fact-checking
    example_data = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "Who wrote the novel '1984'?",
            "answer": "George Orwell wrote the novel '1984'."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter is the largest planet in our solar system."
        },
        {
            "question": "When did World War II end?",
            "answer": "World War II ended in 1945."
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "The chemical symbol for gold is Au."
        },
        {
            "question": "Where is xi'an city?",
            "answer": "The Xi'an is near the Africa."
        }
    ]

    logger.info("=" * 60)
    logger.info("Running fact-checking on example data")
    logger.info("=" * 60)

    # Run fact-checking on each example
    results = []
    for i, example in enumerate(example_data, 1):
        logger.info(f"\n--- Example {i}/{len(example_data)} ---")
        logger.info(f"Question: {example['question']}")
        logger.info(f"Answer: {example['answer']}")

        result, raw_response = inferencer.check_fact(
            question=example['question'],
            answer=example['answer'],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        logger.info(f"Fact-check Result: {result}")
        logger.debug(f"Raw Response: {raw_response}")

        results.append({
            "question": example['question'],
            "answer": example['answer'],
            "result": result,
            "raw_response": raw_response
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for i, result in enumerate(results, 1):
        logger.info(f"Example {i}: {result['result']} - {result['question'][:50]}...")

    true_count = sum(1 for r in results if r['result'] == 'True')
    false_count = sum(1 for r in results if r['result'] == 'False')
    unknown_count = sum(1 for r in results if r['result'] == 'Unknown')

    logger.info(f"\nTotal: {len(results)} examples")
    logger.info(f"True: {true_count}, False: {false_count}, Unknown: {unknown_count}")


if __name__ == "__main__":
    main()
