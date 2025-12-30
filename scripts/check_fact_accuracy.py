"""
Fact-Checking Accuracy Script
Loads a JSONL file, checks the factuality of prediction responses, and calculates overall accuracy.
Based on examples/fact-check-inference.py.
"""

import json
import os
import argparse
import sys
import logging
import torch
from transformers import pipeline
from typing import Optional, Dict, Any, Tuple

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
            check_model_name: Check model name or path
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
        # Extract result from <fact>True</fact> or <fact>False</fact>
        if "<fact>True</fact>" in response:
            result = "True"
        elif "<fact>False</fact>" in response:
            result = "False"
        else:
            result = "Unknown"
            logger.warning(f"Could not extract fact result from response: {response[:200]}...")

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
        
        # Format prompt
        user_prompt = f"<question>{question}</question><answer>{answer}</answer>"

        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

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
            
            # Extract fact result
            result = self.extract_fact_result(raw_response)

            logger.info(f"Fact-check result: {result}")
            logger.debug(f"Raw response: {raw_response[:200]}...")
            
            return result, raw_response

        except Exception as e:
            logger.error(f"Error during fact-checking: {e}")
            return "Unknown", ""


def load_jsonl(file_path: str) -> list:
    """Load JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} records from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        raise
    return data


def main():
    parser = argparse.ArgumentParser(description="Fact-Checking Accuracy Script")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="output/fact-res", help="Directory to save results")
    parser.add_argument("--check_model_name", type=str, default="jdqqjr/Qwen2.5-0.5B-Instruct_8epoch_Fact_Checker", help="Check model name or path")
    parser.add_argument("--device", type=int, default=None, help="Device ID")
    parser.add_argument("--torch_dtype", type=str, default=None, choices=["float16", "bfloat16", None], help="Model data type")
    
    args = parser.parse_args()

    # Convert torch_dtype
    torch_dtype = None
    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize inferencer
    logger.info("Initializing fact-checking inferencer...")
    try:
        inferencer = FactCheckerInferencer(
            check_model_name=args.check_model_name,
            device=args.device,
            torch_dtype=torch_dtype
        )
    except Exception as e:
        logger.error(f"Failed to initialize inferencer: {e}")
        return

    # Load data
    logger.info(f"Loading data from {args.input_file}")
    data = load_jsonl(args.input_file)

    if not data:
        logger.warning("No data found in input file.")
        return

    results_file = os.path.join(args.output_dir, f"{os.path.basename(args.input_file)}_checked.jsonl")
    summary_file = os.path.join(args.output_dir, f"{os.path.basename(args.input_file)}_summary.json")
    logger.info(f"Results will be saved to {results_file}")
    logger.info(f"Summary will be saved to {summary_file}")

    correct_count = 0
    false_count = 0
    unknown_count = 0
    total_count = 0
    
    # Open file for writing results incrementally
    with open(results_file, 'w', encoding='utf-8') as f_out:
        for i, record in enumerate(data, 1):
            question = record.get('question', '')
            prediction = record.get('prediction', '')
            
            if not question or not prediction:
                logger.warning(f"Skipping record {i}: Missing question or prediction")
                continue

            logger.info(f"Processing record {i}/{len(data)}")
            
            result, raw_response = inferencer.check_fact(question, prediction)
            
            # Update record with results
            record['fact_check_result'] = result
            record['fact_check_raw_response'] = raw_response
            
            # Write to file
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            f_out.flush()
            
            if result == 'True':
                correct_count += 1
            elif result == 'False':
                false_count += 1
            else:
                unknown_count += 1
            
            total_count += 1
            
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(data)} - Current Accuracy: {correct_count/total_count:.2%}")

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # Create summary dictionary
    summary = {
        "input_file": args.input_file,
        "check_model_name": args.check_model_name,
        "total_processed": total_count,
        "correct_count": correct_count,
        "false_count": false_count,
        "unknown_count": unknown_count,
        "accuracy": accuracy,
        "accuracy_percentage": f"{accuracy:.2%}"
    }

    # Save summary to file
    with open(summary_file, 'w', encoding='utf-8') as f_sum:
        json.dump(summary, f_sum, indent=4, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Final Summary")
    logger.info("=" * 60)
    logger.info(f"Input File: {args.input_file}")
    logger.info(f"Total Processed: {total_count}")
    logger.info(f"Correct (True): {correct_count}")
    logger.info(f"False: {false_count}")
    logger.info(f"Unknown: {unknown_count}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
