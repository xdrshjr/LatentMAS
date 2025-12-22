import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    """Evaluate predictions and calculate accuracy.
    
    Args:
        preds: List of prediction dictionaries, each containing a 'correct' field.
        
    Returns:
        Tuple of (accuracy, correct_count).
    """
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    logger.debug(f"Evaluation: {correct}/{total} correct, accuracy: {acc:.4f}")
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    """Process a batch of questions using the specified method.
    
    Args:
        method: The method instance (BaselineMethod, TextMASMethod, or LatentMASMethod).
        batch: List of question dictionaries to process.
        processed: Number of questions already processed.
        preds: List to accumulate predictions.
        progress: Progress bar object (can be None).
        max_samples: Maximum number of samples to process.
        args: Command line arguments.
        
    Returns:
        Tuple of (updated_processed_count, updated_predictions_list).
    """
    remaining = max_samples - processed
    if remaining <= 0:
        logger.debug(f"No remaining samples to process (processed: {processed}, max: {max_samples})")
        return processed, preds
    current_batch = batch[:remaining]
    logger.info(f"Processing batch of {len(current_batch)} questions (total processed: {processed})")
    
    try:
        if args.method == "latent_mas" and args.use_vllm: 
            logger.debug("Using vLLM backend for latent_mas method")
            results = method.run_batch_vllm(current_batch) 
        else:
            logger.debug(f"Using standard batch processing for {args.method} method")
            results = method.run_batch(current_batch)
    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        raise
    
    if len(results) > remaining:
        results = results[:remaining]
        logger.warning(f"Results truncated to {remaining} items")
    
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        logger.info(f"Processing problem #{problem_idx}")
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        logger.debug(f"Problem #{problem_idx} has {len(agents)} agents")
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        result_str = f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
        print(result_str)
        logger.info(f"Problem #{problem_idx} {result_str}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    logger.debug(f"Batch processing complete. Total processed: {processed}")
    return processed, preds


def run_custom_questions(
    method,
    custom_questions: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    """Run inference on a list of custom questions.
    
    Args:
        method: The method instance to use for inference.
        custom_questions: List of question dictionaries. Each dict should have at least a 'question' field.
                          Optional fields: 'gold', 'solution' for evaluation.
        args: Command line arguments.
        
    Returns:
        List of prediction dictionaries.
    """
    logger.info(f"Running custom questions mode with {len(custom_questions)} questions")
    logger.debug(f"Method: {args.method}, Model: {args.model_name}")
    
    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    max_samples = len(custom_questions)
    
    progress = tqdm(total=max_samples, desc="Processing custom questions")
    
    for item in custom_questions:
        if processed >= max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples,
                args,
            )
            batch = []
            if processed >= max_samples:
                break

    if batch and processed < max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=max_samples,
            args=args,
        )
    progress.close()
    
    logger.info(f"Completed processing {len(preds)} custom questions")
    return preds


def main(custom_questions: Optional[List[Dict]] = None, args: Optional[argparse.Namespace] = None):
    """Main function to run evaluation on datasets or custom questions.
    
    Args:
        custom_questions: Optional list of custom question dictionaries. If provided, 
                         will run on these questions instead of loading from dataset.
                         Each dict should have at least a 'question' field.
        args: Optional argparse.Namespace object with arguments. If None, will parse from command line.
    """
    if args is None:
        parser = argparse.ArgumentParser()

        # core args for experiments
        parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                            help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
        parser.add_argument("--model_name", type=str, required=True,
                            choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
                            help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
        parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
        parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                            help="Dataset/task to evaluate. Controls which loader is used.")
        parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")
        parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            help="Set the logging level.")

        # other args
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--max_new_tokens", type=int, default=4096)
        parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
        parser.add_argument("--temperature", type=float, default=0.6)
        parser.add_argument("--top_p", type=float, default=0.95)
        parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
        parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
        parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
        parser.add_argument("--latent_space_realign", action="store_true")
        parser.add_argument("--seed", type=int, default=42)

        # vLLM support
        parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
        parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
        parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
        parser.add_argument("--device2", type=str, default="cuda:1")
        parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

        args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    logger.info(f"Logging level set to {args.log_level}")
    
    # If custom_questions is provided, use it instead of dataset
    if custom_questions is not None:
        logger.info("Custom questions mode enabled")
        if args.max_samples == -1:
            args.max_samples = len(custom_questions)
        else:
            custom_questions = custom_questions[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} questions")
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
        logger.debug("Enabled vLLM-specific settings for latent_mas")
    
    logger.info(f"Initializing with method={args.method}, model={args.model_name}, seed={args.seed}")
    set_seed(args.seed)
    device = auto_device(args.device)
    logger.debug(f"Using device: {device}")
    
    logger.info("Loading model...")
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    logger.info("Model loaded successfully")
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    logger.info(f"Initializing {args.method} method...")
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    logger.info(f"Method {args.method} initialized successfully")

    # If custom questions provided, run on them directly
    if custom_questions is not None:
        preds = run_custom_questions(method, custom_questions, args)
    else:
        # dataset loading
        logger.info(f"Loading dataset: {args.task} (split: {args.split})")
        if args.task == "gsm8k":
            dataset_iter = load_gsm8k(split=args.split)
        elif args.task == "aime2024":
            dataset_iter = load_aime2024(split="train")
        elif args.task == "aime2025":
            dataset_iter = load_aime2025(split='train')
        elif args.task == "gpqa":
            dataset_iter = load_gpqa_diamond(split='test')
        elif args.task == "arc_easy":
            dataset_iter = load_arc_easy(split='test')
        elif args.task == "arc_challenge":
            dataset_iter = load_arc_challenge(split='test')
        elif args.task == "mbppplus":
            dataset_iter = load_mbppplus(split='test')
        elif args.task == "humanevalplus":
            dataset_iter = load_humanevalplus(split='test')
        elif args.task == "medqa":
            dataset_iter = load_medqa(split='test')
        else:
            raise ValueError(f'no {args.task} support')

        if args.max_samples == -1:
            dataset_iter = list(dataset_iter)  
            args.max_samples = len(dataset_iter)
            logger.info(f"Loaded all {args.max_samples} samples from dataset")
        else:
            logger.info(f"Will process up to {args.max_samples} samples")

        preds: List[Dict] = []
        processed = 0
        batch: List[Dict] = []

        progress = tqdm(total=args.max_samples, desc=f"Processing {args.task}")

        for item in dataset_iter:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, preds = process_batch(
                    method,
                    batch,
                    processed,
                    preds,
                    progress,
                    args.max_samples,
                    args,
                )
                batch = []
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples=args.max_samples,
                args=args,
            )
        progress.close()
        logger.info(f"Completed processing {len(preds)} samples from dataset")
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")

    acc, correct = evaluate(preds)
    logger.info(f"Final results: {correct}/{len(preds)} correct, accuracy: {acc:.4f}")
    
    # Load results in JSON format
    result_json = json.dumps(
        {
            "method": args.method,
            "model": args.model_name,
            "split": args.split if custom_questions is None else "custom",
            "seed": args.seed,
            "max_samples": args.max_samples,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time,4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4) if args.max_samples > 0 else 0,
        },
        ensure_ascii=False,
    )
    print(result_json)
    logger.debug(f"Result JSON: {result_json}")



if __name__ == "__main__":
    main()
