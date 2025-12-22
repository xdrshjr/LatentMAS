import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional

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
from methods.latent_mas_multipath import LatentMASMultiPathMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed, create_output_file_path, save_question_answer_record
from config import ConfigLoader, MultiPathConfig, list_presets, get_preset_description
from logging_config import setup_logging, create_log_file_path
from progress_utils import get_progress_manager, reset_progress_manager
import time

# Logger will be configured in main()
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
    max_samples: int,
    args: argparse.Namespace,
    progress_mgr=None,
    output_file: Optional[str] = None,
) -> Tuple[int, List[Dict]]:
    """Process a batch of questions using the specified method.
    
    Args:
        method: The method instance (BaselineMethod, TextMASMethod, or LatentMASMethod).
        batch: List of question dictionaries to process.
        processed: Number of questions already processed.
        preds: List to accumulate predictions.
        max_samples: Maximum number of samples to process.
        args: Command line arguments.
        progress_mgr: Progress manager for updating progress bar.
        output_file: Optional path to output file for saving question-answer records.
        
    Returns:
        Tuple of (updated_processed_count, updated_predictions_list).
    """
    remaining = max_samples - processed
    if remaining <= 0:
        logger.debug(f"No remaining samples to process (processed: {processed}, max: {max_samples})")
        return processed, preds
    current_batch = batch[:remaining]
    
    logger.info("=" * 80)
    logger.info(f"Processing batch of {len(current_batch)} questions (total processed: {processed}/{max_samples})")
    logger.info("=" * 80)
    
    # Log questions in the batch
    for batch_idx, item in enumerate(current_batch):
        logger.info(f"Batch item {batch_idx + 1}/{len(current_batch)}: {item.get('question', '')[:100]}...")
    
    try:
        if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm: 
            logger.info(f"Using vLLM backend for {args.method} method")
            results = method.run_batch_vllm(current_batch) 
        else:
            logger.info(f"Using standard batch processing for {args.method} method")
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
        
        # Log problem processing with detailed information
        logger.info("=" * 80)
        logger.info(f"COMPLETED PROBLEM #{problem_idx}")
        logger.info(f"  Question: {res.get('question', '')[:100]}...")
        logger.info(f"  Prediction: {res.get('prediction')}")
        logger.info(f"  Gold Answer: {res.get('gold')}")
        logger.info(f"  Result: {'✓ CORRECT' if res.get('correct') else '✗ INCORRECT'}")
        if 'num_paths_used' in res:
            logger.info(f"  Paths Used: {res.get('num_paths_used')}")
        logger.info("=" * 80)
        
        # Save question-answer record to output file
        if output_file:
            save_question_answer_record(
                output_file=output_file,
                problem_idx=problem_idx,
                question=res.get('question', ''),
                prediction=res.get('prediction'),
                gold=res.get('gold'),
                correct=res.get('correct', False),
                additional_info={
                    'method': args.method,
                    'model': args.model_name,
                }
            )
        
        # Show detailed agent trace information
        agents = res.get("agents", [])
        logger.info(f"Problem #{problem_idx} processed by {len(agents)} agents")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Problem #{problem_idx} Question: {res.get('question', '').strip()}")
            logger.debug(f"Problem #{problem_idx} Gold answer: {res.get('gold', '')}")
            
            for agent_idx, a in enumerate(agents):
                name = a.get("name", "Agent")
                role = a.get("role", "")
                logger.debug(f"  Agent {agent_idx + 1}/{len(agents)}: {name} ({role})")
                
                # Log agent input (prompt)
                agent_input = a.get("input", "")
                if agent_input:
                    logger.debug(f"    Input prompt: {agent_input[:200]}...")
                
                # Log agent processing details
                if "latent_steps" in a:
                    logger.debug(f"    Latent steps: {a.get('latent_steps')}")
                if "num_paths" in a:
                    logger.debug(f"    Number of paths: {a.get('num_paths')}")
                if "path_scores" in a:
                    scores = a.get('path_scores', [])
                    if scores:
                        logger.debug(f"    Path scores: {[f'{s:.4f}' for s in scores]}")
                
                # Log agent output
                agent_output = a.get("output", "").rstrip()
                if agent_output:
                    logger.debug(f"    Output: {agent_output}")
                else:
                    logger.debug(f"    Output: (no text output - latent reasoning only)")
        
        # Always show agent summary at INFO level
        for agent_idx, a in enumerate(agents):
            name = a.get("name", "Agent")
            role = a.get("role", "")
            if "num_paths" in a:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role}) - {a.get('num_paths')} paths")
            elif "num_paths_aggregated" in a:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role}) - aggregated {a.get('num_paths_aggregated')} paths")
            else:
                logger.info(f"  Agent {agent_idx + 1}: {name} ({role})")

    processed += len(results)
    
    # Update progress bar with current accuracy
    if progress_mgr is not None:
        correct = sum(1 for p in preds if p.get("correct", False))
        acc = correct / len(preds) if len(preds) > 0 else 0.0
        progress_mgr.update_main_progress(len(results))
        progress_mgr.set_main_postfix(acc=f"{acc:.2%}", correct=f"{correct}/{len(preds)}")
    
    logger.debug(f"Batch processing complete. Total processed: {processed}")
    return processed, preds


def run_custom_questions(
    method,
    custom_questions: List[Dict],
    args: argparse.Namespace,
    progress_mgr=None,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run inference on a list of custom questions.
    
    Args:
        method: The method instance to use for inference.
        custom_questions: List of question dictionaries. Each dict should have at least a 'question' field.
                          Optional fields: 'gold', 'solution' for evaluation.
        args: Command line arguments.
        progress_mgr: Progress manager for updating progress bar.
        output_file: Optional path to output file for saving question-answer records.
        
    Returns:
        List of prediction dictionaries.
    """
    logger.info(f"Running custom questions mode with {len(custom_questions)} questions")
    logger.debug(f"Method: {args.method}, Model: {args.model_name}")
    
    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    max_samples = len(custom_questions)
    
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
                max_samples,
                args,
                progress_mgr,
                output_file,
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
            max_samples=max_samples,
            args=args,
            progress_mgr=progress_mgr,
            output_file=output_file,
        )
    
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
        parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "latent_mas_multipath"], required=True,
                            help="Which multi-agent method to run: 'baseline', 'text_mas', 'latent_mas', or 'latent_mas_multipath'.")
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
        parser.add_argument("--latent_steps", type=int, default=None, help="Number of latent steps for LatentMAS method (default: uses method/config default)")
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
        
        # Multi-path specific arguments
        parser.add_argument("--num_paths", type=int, default=5, help="Number of parallel reasoning paths for latent_mas_multipath")
        parser.add_argument("--enable_branching", action="store_true", help="Enable adaptive branching in multi-path reasoning")
        parser.add_argument("--enable_merging", action="store_true", help="Enable path merging in multi-path reasoning")
        parser.add_argument("--pruning_strategy", type=str, choices=["topk", "adaptive", "diversity", "budget"], default="adaptive",
                            help="Pruning strategy for multi-path reasoning")
        parser.add_argument("--merge_threshold", type=float, default=0.9, help="Similarity threshold for path merging")
        parser.add_argument("--branch_threshold", type=float, default=0.5, help="Uncertainty threshold for adaptive branching")
        parser.add_argument("--diversity_strategy", type=str, choices=["temperature", "noise", "hybrid"], default="hybrid",
                            help="Diversity strategy for generating diverse paths")
        
        # Configuration file support
        parser.add_argument("--config", type=str, default=None, help="Path to configuration file (JSON or YAML)")
        parser.add_argument("--config_preset", type=str, default=None, 
                            choices=["conservative", "balanced", "aggressive", "fast", "quality"],
                            help="Use a preset configuration (conservative/balanced/aggressive/fast/quality)")
        parser.add_argument("--list_presets", action="store_true", help="List available configuration presets and exit")

        args = parser.parse_args()
        
        # Handle --list_presets
        if args.list_presets:
            print("Available configuration presets for latent_mas_multipath:\n")
            for preset_name in list_presets():
                print(f"  {preset_name}:")
                print(f"    {get_preset_description(preset_name)}")
                print()
            import sys
            sys.exit(0)
    
    # Setup enhanced logging system
    log_file = None
    if not hasattr(args, 'no_log_file') or not args.no_log_file:
        # Extract model short name for log file
        model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
        task_name = args.task if custom_questions is None else "custom"
        log_file = create_log_file_path(task_name, args.method, model_short)
    
    # Setup logging with progress bar support
    setup_logging(
        log_level="DEBUG",  # Log everything to file
        console_level=args.log_level,  # But only show specified level in console
        log_file=log_file,
        use_colors=True,
        progress_bar_mode=True
    )
    
    logger.info(f"Logging configured: console_level={args.log_level}, log_file={log_file}")
    
    # Create output file for question-answer records
    model_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    task_name = args.task if custom_questions is None else "custom"
    output_file = create_output_file_path(task_name, args.method, model_short)
    logger.info(f"Question-answer records will be saved to: {output_file}")
    
    # Initialize progress bar manager
    reset_progress_manager()
    progress_mgr = get_progress_manager()
    
    # Load configuration if using latent_mas_multipath
    multipath_config = None
    if args.method == "latent_mas_multipath":
        logger.info("[Configuration] Processing multi-path configuration")
        
        # Priority: config file > preset > command-line defaults
        if args.config:
            logger.info(f"[Configuration] Loading configuration from file: {args.config}")
            try:
                multipath_config = ConfigLoader.load_from_file(args.config)
                logger.info("[Configuration] Configuration file loaded successfully")
            except Exception as e:
                logger.error(f"[Configuration] Failed to load config file: {e}")
                raise
        elif args.config_preset:
            logger.info(f"[Configuration] Using preset configuration: {args.config_preset}")
            try:
                multipath_config = ConfigLoader.get_preset(args.config_preset)
                logger.info("[Configuration] Preset configuration loaded successfully")
            except Exception as e:
                logger.error(f"[Configuration] Failed to load preset: {e}")
                raise
        else:
            logger.debug("[Configuration] No config file or preset specified, using command-line arguments")
            # Create config from command-line arguments
            config_kwargs = {
                'num_paths': args.num_paths,
                'enable_branching': args.enable_branching,
                'enable_merging': args.enable_merging,
                'pruning_strategy': args.pruning_strategy,
                'merge_threshold': args.merge_threshold,
                'branch_threshold': args.branch_threshold,
                'diversity_strategy': args.diversity_strategy,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_new_tokens': args.max_new_tokens,
                'generate_bs': args.generate_bs,
            }
            # Only include latent_steps if explicitly provided (not None)
            if args.latent_steps is not None:
                config_kwargs['latent_steps'] = args.latent_steps
            multipath_config = MultiPathConfig(**config_kwargs)
        
        # Merge with command-line arguments (CLI overrides config file/preset)
        if args.config or args.config_preset:
            logger.debug("[Configuration] Merging configuration with command-line arguments")
            multipath_config = ConfigLoader.merge_with_args(multipath_config, args)
        
        # Update args with final config values
        args.num_paths = multipath_config.num_paths
        args.enable_branching = multipath_config.enable_branching
        args.enable_merging = multipath_config.enable_merging
        args.pruning_strategy = multipath_config.pruning_strategy
        args.merge_threshold = multipath_config.merge_threshold
        args.branch_threshold = multipath_config.branch_threshold
        args.diversity_strategy = multipath_config.diversity_strategy
        args.latent_steps = multipath_config.latent_steps
        args.temperature = multipath_config.temperature
        args.top_p = multipath_config.top_p
        args.max_new_tokens = multipath_config.max_new_tokens
        args.generate_bs = multipath_config.generate_bs
        
        logger.info(f"[Configuration] Final multi-path config: num_paths={args.num_paths}, "
                   f"pruning={args.pruning_strategy}, diversity={args.diversity_strategy}, "
                   f"branching={args.enable_branching}, merging={args.enable_merging}, "
                   f"latent_steps={args.latent_steps}")
    
    # If custom_questions is provided, use it instead of dataset
    if custom_questions is not None:
        logger.info("Custom questions mode enabled")
        if args.max_samples == -1:
            args.max_samples = len(custom_questions)
        else:
            custom_questions = custom_questions[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} questions")
    
    if args.method in ["latent_mas", "latent_mas_multipath"] and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
        logger.debug(f"Enabled vLLM-specific settings for {args.method}")
    
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
        # Use default latent_steps=10 if not specified
        latent_steps = args.latent_steps if args.latent_steps is not None else 10
        method = LatentMASMethod(
            model,
            latent_steps=latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == 'latent_mas_multipath':
        method = LatentMASMultiPathMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            num_paths=args.num_paths,
            enable_branching=args.enable_branching,
            enable_merging=args.enable_merging,
            pruning_strategy=args.pruning_strategy,
            merge_threshold=args.merge_threshold,
            branch_threshold=args.branch_threshold,
            diversity_strategy=args.diversity_strategy,
        )
    logger.info(f"Method {args.method} initialized successfully")

    # If custom questions provided, run on them directly
    if custom_questions is not None:
        # Create progress bar for custom questions
        progress_mgr.create_main_progress(
            total=args.max_samples,
            desc=f"Processing custom questions",
            unit="question"
        )
        preds = run_custom_questions(method, custom_questions, args, progress_mgr, output_file)
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

        # Create progress bar
        progress_mgr.create_main_progress(
            total=args.max_samples,
            desc=f"Processing {args.task}",
            unit="sample"
        )

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
                    args.max_samples,
                    args,
                    progress_mgr,
                    output_file,
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
                max_samples=args.max_samples,
                args=args,
                progress_mgr=progress_mgr,
                output_file=output_file,
            )
        logger.info(f"Completed processing {len(preds)} samples from dataset")
    
    # Close progress bar
    progress_mgr.close_all()
    
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
    
    # Print final results to console
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(result_json)
    print("="*80)
    logger.debug(f"Result JSON: {result_json}")



if __name__ == "__main__":
    main()
