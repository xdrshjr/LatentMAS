"""Multi-GPU orchestrator for parallel data collection.

This module coordinates multiple GPU processes to collect training data in parallel.
Each GPU processes a distinct subset of the dataset, and results are aggregated at the end.
"""

import argparse
import logging
import subprocess
import sys
import os
import time
import threading
import re
from pathlib import Path
from typing import List, Optional, TextIO, Dict
from datetime import datetime
from tqdm import tqdm

from utils import (
    get_gpu_data_range,
    validate_gpu_availability,
    aggregate_prm_data,
)
from logging_config import setup_logging

# Logger setup
logger = logging.getLogger(__name__)


def stream_output(
    pipe: TextIO,
    log_file: TextIO,
    gpu_id: int,
    stream_type: str,
    progress_bar: Optional[tqdm] = None
) -> None:
    """Stream output from subprocess pipe to both console and log file.
    
    This function runs in a separate thread to continuously read from a subprocess
    pipe and write the output to both the console (with GPU prefix) and a log file.
    It also parses progress markers and updates the progress bar.
    
    Args:
        pipe: Subprocess stdout or stderr pipe to read from
        log_file: File object to write logs to
        gpu_id: GPU ID for prefixing console output
        stream_type: Type of stream ('stdout' or 'stderr') for logging
        progress_bar: Optional tqdm progress bar to update with progress markers
    """
    # Regex pattern to match progress markers: [PROGRESS:5/100|acc:0.8500|correct:85/100]
    progress_pattern = re.compile(r'\[PROGRESS:(\d+)/(\d+)\|acc:([\d.]+)\|correct:(\d+)/(\d+)\]')
    
    try:
        for line in iter(pipe.readline, ''):
            if line:
                # Write to log file without prefix
                log_file.write(line)
                log_file.flush()
                
                # Remove trailing newline from line
                line_stripped = line.rstrip('\n\r')
                
                if line_stripped:
                    # Check if this is a progress marker
                    progress_match = progress_pattern.search(line_stripped)
                    
                    if progress_match and progress_bar is not None:
                        # Parse progress information
                        current = int(progress_match.group(1))
                        total = int(progress_match.group(2))
                        acc = float(progress_match.group(3))
                        correct = int(progress_match.group(4))
                        total_processed = int(progress_match.group(5))
                        
                        # Update progress bar
                        progress_bar.n = current
                        progress_bar.total = total
                        progress_bar.set_postfix(
                            acc=f"{acc:.2%}",
                            correct=f"{correct}/{total_processed}",
                            refresh=True
                        )
                        progress_bar.refresh()
                    else:
                        # Regular log line - write above progress bars using tqdm.write()
                        output_line = f"[GPU {gpu_id}] {line_stripped}"
                        
                        if progress_bar is not None:
                            # Use tqdm.write() to print above progress bars
                            tqdm.write(output_line, file=sys.stderr)
                        else:
                            # Fallback: direct output
                            if stream_type == 'stderr':
                                logger.info(output_line)
                            else:
                                print(output_line, flush=True)
    except Exception as e:
        logger.error(f"[Multi-GPU] Error streaming {stream_type} for GPU {gpu_id}: {e}")
    finally:
        pipe.close()


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """Parse GPU IDs from comma-separated string.
    
    Args:
        gpu_ids_str: Comma-separated GPU IDs (e.g., "0,1,2,3")
        
    Returns:
        List of GPU IDs as integers
        
    Raises:
        ValueError: If GPU IDs are invalid
    """
    try:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
        if not gpu_ids:
            raise ValueError("No GPU IDs provided")
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError("GPU IDs must be non-negative")
        logger.debug(f"[Multi-GPU] Parsed GPU IDs: {gpu_ids}")
        return gpu_ids
    except ValueError as e:
        logger.error(f"[Multi-GPU] Failed to parse GPU IDs from '{gpu_ids_str}': {e}")
        raise


def build_subprocess_command(
    base_args: argparse.Namespace,
    gpu_id: int,
    total_samples: int,
    num_gpus: int
) -> List[str]:
    """Build command for subprocess execution on a specific GPU.
    
    Args:
        base_args: Base command-line arguments
        gpu_id: GPU ID for this subprocess
        total_samples: Total number of samples in dataset
        num_gpus: Total number of GPUs
        
    Returns:
        Command as list of strings for subprocess.run()
    """
    # Calculate data range for this GPU
    start_idx, end_idx = get_gpu_data_range(total_samples, num_gpus, gpu_id)
    samples_for_gpu = end_idx - start_idx
    
    logger.info(f"[Multi-GPU] GPU {gpu_id}: Will process samples [{start_idx}:{end_idx}] ({samples_for_gpu} samples)")
    
    # Build command
    cmd = [
        sys.executable,  # Python interpreter
        "run.py",
        "--method", base_args.method,
        "--model_name", base_args.model_name,
        "--task", base_args.task,
        "--prompt", base_args.prompt,
        "--max_samples", str(samples_for_gpu),
        "--seed", str(base_args.seed + gpu_id),  # Different seed per GPU for diversity
        "--max_new_tokens", str(base_args.max_new_tokens),
        "--temperature", str(base_args.temperature),
        "--top_p", str(base_args.top_p),
        "--generate_bs", str(base_args.generate_bs),
        "--log_level", base_args.log_level,
        "--gpu_id", str(gpu_id),
        "--data_start_idx", str(start_idx),
        "--output_suffix", f"gpu_{gpu_id}",
    ]
    
    # Add optional arguments
    if base_args.latent_steps is not None:
        cmd.extend(["--latent_steps", str(base_args.latent_steps)])
    
    if base_args.num_paths is not None:
        cmd.extend(["--num_paths", str(base_args.num_paths)])
    
    if base_args.num_parent_paths is not None:
        cmd.extend(["--num_parent_paths", str(base_args.num_parent_paths)])
    
    if base_args.diversity_strategy:
        cmd.extend(["--diversity_strategy", base_args.diversity_strategy])
    
    if base_args.pruning_strategy:
        cmd.extend(["--pruning_strategy", base_args.pruning_strategy])
    
    if base_args.latent_consistency_metric:
        cmd.extend(["--latent_consistency_metric", base_args.latent_consistency_metric])
    
    if base_args.latent_space_realign:
        cmd.append("--latent_space_realign")
    
    if base_args.enable_branching:
        cmd.append("--enable_branching")
    
    if base_args.enable_merging:
        cmd.append("--enable_merging")
    
    if base_args.merge_threshold is not None:
        cmd.extend(["--merge_threshold", str(base_args.merge_threshold)])
    
    if base_args.branch_threshold is not None:
        cmd.extend(["--branch_threshold", str(base_args.branch_threshold)])
    
    # PRM data collection arguments
    if base_args.collect_prm_data:
        cmd.append("--collect_prm_data")
        cmd.extend(["--prm_output_dir", base_args.prm_output_dir])
        
        if base_args.prm_disable_pruning:
            cmd.append("--prm_disable_pruning")
        
        if base_args.prm_disable_merging:
            cmd.append("--prm_disable_merging")
    
    # Visualization
    if not base_args.enable_visualization:
        cmd.append("--disable_visualization")
    
    logger.debug(f"[Multi-GPU] GPU {gpu_id} command: {' '.join(cmd)}")
    
    return cmd


def launch_gpu_process(
    gpu_id: int,
    command: List[str],
    log_dir: Path,
    progress_bar: Optional[tqdm] = None
) -> tuple:
    """Launch a subprocess for a specific GPU with real-time log streaming.
    
    Args:
        gpu_id: GPU ID to use
        command: Command to execute
        log_dir: Directory for log files
        progress_bar: Optional tqdm progress bar for this GPU
        
    Returns:
        Tuple of (process, streaming_threads, log_files)
    """
    # Set CUDA_VISIBLE_DEVICES to isolate this GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create log files for stdout and stderr
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_dir / f"gpu_{gpu_id}_stdout.log"
    stderr_log_path = log_dir / f"gpu_{gpu_id}_stderr.log"
    
    # Use tqdm.write() to output above progress bars
    tqdm.write(f"[Multi-GPU] Launching process for GPU {gpu_id}", file=sys.stderr)
    tqdm.write(f"[Multi-GPU] GPU {gpu_id} CUDA_VISIBLE_DEVICES={gpu_id}", file=sys.stderr)
    tqdm.write(f"[Multi-GPU] GPU {gpu_id} stdout log: {stdout_log_path}", file=sys.stderr)
    tqdm.write(f"[Multi-GPU] GPU {gpu_id} stderr log: {stderr_log_path}", file=sys.stderr)
    logger.debug(f"[Multi-GPU] GPU {gpu_id} command: {' '.join(command)}")
    
    # Launch process with PIPE for real-time streaming
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    tqdm.write(f"[Multi-GPU] GPU {gpu_id} process started with PID {process.pid}", file=sys.stderr)
    
    # Open log files for writing
    stdout_log_file = open(stdout_log_path, 'w', buffering=1)
    stderr_log_file = open(stderr_log_path, 'w', buffering=1)
    
    # Start streaming threads for stdout and stderr
    # Pass progress_bar to stderr thread (progress markers are output to stderr)
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, stdout_log_file, gpu_id, 'stdout', None),
        daemon=True,
        name=f"GPU{gpu_id}-stdout"
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, stderr_log_file, gpu_id, 'stderr', progress_bar),
        daemon=True,
        name=f"GPU{gpu_id}-stderr"
    )
    
    stdout_thread.start()
    stderr_thread.start()
    
    logger.debug(f"[Multi-GPU] GPU {gpu_id} streaming threads started")
    
    # Return process and threads for monitoring
    streaming_threads = [stdout_thread, stderr_thread]
    log_files = [stdout_log_file, stderr_log_file]
    
    return process, streaming_threads, log_files


def monitor_processes(processes: List[tuple]) -> bool:
    """Monitor running processes and wait for completion.
    
    Args:
        processes: List of (gpu_id, process, streaming_threads, log_files, progress_bar) tuples
        
    Returns:
        True if all processes completed successfully, False otherwise
    """
    tqdm.write("=" * 80, file=sys.stderr)
    tqdm.write(f"[Multi-GPU] Monitoring {len(processes)} GPU processes", file=sys.stderr)
    tqdm.write(f"[Multi-GPU] Real-time logs from all GPUs will be displayed above progress bars", file=sys.stderr)
    tqdm.write("=" * 80, file=sys.stderr)
    
    start_time = time.time()
    all_success = True
    
    # Wait for all processes to complete
    for gpu_id, process, streaming_threads, log_files, progress_bar in processes:
        tqdm.write(f"[Multi-GPU] Waiting for GPU {gpu_id} process (PID {process.pid}) to complete...", file=sys.stderr)
        
        try:
            # Wait for process to complete
            return_code = process.wait()
            elapsed = time.time() - start_time
            
            # Wait for streaming threads to finish processing remaining output
            logger.debug(f"[Multi-GPU] Waiting for GPU {gpu_id} streaming threads to complete...")
            for thread in streaming_threads:
                thread.join(timeout=5.0)  # Wait up to 5 seconds for threads to finish
            
            # Close log files
            for log_file in log_files:
                try:
                    log_file.close()
                except Exception as e:
                    logger.warning(f"[Multi-GPU] Error closing log file for GPU {gpu_id}: {e}")
            
            # Close progress bar for this GPU
            if progress_bar is not None:
                progress_bar.close()
            
            if return_code == 0:
                tqdm.write(
                    f"[Multi-GPU] ✓ GPU {gpu_id} process completed successfully "
                    f"(elapsed: {elapsed:.2f}s)",
                    file=sys.stderr
                )
            else:
                tqdm.write(
                    f"[Multi-GPU] ✗ GPU {gpu_id} process failed with return code {return_code} "
                    f"(elapsed: {elapsed:.2f}s)",
                    file=sys.stderr
                )
                all_success = False
                
        except Exception as e:
            logger.error(f"[Multi-GPU] Error monitoring GPU {gpu_id} process: {e}", exc_info=True)
            all_success = False
    
    total_elapsed = time.time() - start_time
    
    tqdm.write("=" * 80, file=sys.stderr)
    if all_success:
        tqdm.write(f"[Multi-GPU] ✓ All GPU processes completed successfully (total time: {total_elapsed:.2f}s)", file=sys.stderr)
    else:
        tqdm.write(f"[Multi-GPU] ✗ Some GPU processes failed (total time: {total_elapsed:.2f}s)", file=sys.stderr)
    tqdm.write("=" * 80, file=sys.stderr)
    
    return all_success


def main():
    """Main function for multi-GPU orchestration."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU orchestrator for parallel data collection"
    )
    
    # Multi-GPU specific arguments
    parser.add_argument("--num_gpus", type=int, required=True,
                        help="Number of GPUs to use for parallel processing")
    parser.add_argument("--gpu_ids", type=str, required=True,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    
    # Core arguments (same as run.py)
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "text_mas", "latent_mas", "latent_mas_multipath"],
                        help="Method to use")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--task", type=str, default="gsm8k",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", 
                                "arc_challenge", "mbppplus", "humanevalplus", "medqa"],
                        help="Dataset/task to evaluate")
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"],
                        help="Multi-agent system architecture")
    parser.add_argument("--max_samples", type=int, required=True,
                        help="Total number of samples to process across all GPUs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (each GPU will use seed + gpu_id)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling parameter")
    parser.add_argument("--generate_bs", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    # Method-specific arguments
    parser.add_argument("--latent_steps", type=int, default=None,
                        help="Number of latent steps")
    parser.add_argument("--num_paths", type=int, default=5,
                        help="Number of reasoning paths")
    parser.add_argument("--num_parent_paths", type=int, default=5,
                        help="Number of parent paths to use")
    parser.add_argument("--diversity_strategy", type=str, default="hybrid",
                        choices=["temperature", "noise", "hybrid"],
                        help="Diversity strategy")
    parser.add_argument("--pruning_strategy", type=str, default="adaptive",
                        choices=["topk", "adaptive", "diversity", "budget"],
                        help="Pruning strategy")
    parser.add_argument("--latent_consistency_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "l2", "kl_divergence"],
                        help="Latent consistency metric")
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Enable latent space realignment")
    parser.add_argument("--enable_branching", action="store_true",
                        help="Enable adaptive branching")
    parser.add_argument("--enable_merging", action="store_true",
                        help="Enable path merging")
    parser.add_argument("--merge_threshold", type=float, default=None,
                        help="Similarity threshold for path merging")
    parser.add_argument("--branch_threshold", type=float, default=None,
                        help="Uncertainty threshold for branching")
    
    # PRM data collection arguments
    parser.add_argument("--collect_prm_data", action="store_true",
                        help="Enable PRM data collection mode")
    parser.add_argument("--prm_output_dir", type=str, default="prm_data",
                        help="Output directory for PRM data")
    parser.add_argument("--prm_disable_pruning", action="store_true",
                        help="Disable pruning in PRM data collection")
    parser.add_argument("--prm_disable_merging", action="store_true",
                        help="Disable merging in PRM data collection")
    
    # Visualization
    parser.add_argument("--enable_visualization", action="store_true", default=False,
                        help="Enable visualization (default: False for multi-GPU)")
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path('output') / 'multi_gpu_logs' / f"orchestrator_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        log_level="DEBUG",
        console_level=args.log_level,
        log_file=str(log_file),
        use_colors=True,
        progress_bar_mode=False
    )
    
    logger.info("=" * 80)
    logger.info("Multi-GPU Data Collection Orchestrator")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Total samples: {args.max_samples}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"GPU IDs: {args.gpu_ids}")
    logger.info("=" * 80)
    
    # Parse GPU IDs
    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except ValueError as e:
        logger.error(f"[Multi-GPU] Invalid GPU IDs: {e}")
        sys.exit(1)
    
    # Validate number of GPUs matches GPU IDs
    if len(gpu_ids) != args.num_gpus:
        logger.error(
            f"[Multi-GPU] Number of GPU IDs ({len(gpu_ids)}) does not match "
            f"num_gpus ({args.num_gpus})"
        )
        sys.exit(1)
    
    # Validate GPU availability
    logger.info("[Multi-GPU] Validating GPU availability...")
    if not validate_gpu_availability(gpu_ids):
        logger.error("[Multi-GPU] GPU validation failed")
        sys.exit(1)
    
    logger.info("[Multi-GPU] All GPUs validated successfully")
    
    # Calculate samples per GPU
    logger.info("[Multi-GPU] Calculating data distribution...")
    for gpu_id in gpu_ids:
        start_idx, end_idx = get_gpu_data_range(args.max_samples, args.num_gpus, gpu_id)
        logger.info(
            f"[Multi-GPU] GPU {gpu_id}: samples [{start_idx}:{end_idx}] "
            f"({end_idx - start_idx} samples)"
        )
    
    # Build commands for each GPU
    logger.info("[Multi-GPU] Building subprocess commands...")
    commands = []
    for gpu_id in gpu_ids:
        cmd = build_subprocess_command(args, gpu_id, args.max_samples, args.num_gpus)
        commands.append((gpu_id, cmd))
    
    # Create progress bars for each GPU
    logger.info("[Multi-GPU] Creating progress bars for each GPU...")
    progress_bars = {}
    samples_per_gpu = args.max_samples // args.num_gpus
    
    for idx, gpu_id in enumerate(gpu_ids):
        start_idx, end_idx = get_gpu_data_range(args.max_samples, args.num_gpus, gpu_id)
        samples_for_gpu = end_idx - start_idx
        
        # Create progress bar for this GPU
        # position: 0 for first GPU, 1 for second GPU, etc.
        progress_bar = tqdm(
            total=samples_for_gpu,
            desc=f"GPU {gpu_id}",
            position=idx,
            leave=True,
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            file=sys.stderr,
            dynamic_ncols=True
        )
        progress_bars[gpu_id] = progress_bar
    
    # Launch processes
    logger.info("=" * 80)
    logger.info("[Multi-GPU] Launching GPU processes...")
    logger.info("[Multi-GPU] Progress bars will be displayed below, logs will scroll above")
    logger.info("=" * 80)
    
    log_dir = Path('output') / 'multi_gpu_logs' / timestamp
    processes = []
    
    for gpu_id, cmd in commands:
        try:
            progress_bar = progress_bars.get(gpu_id)
            process, streaming_threads, log_files = launch_gpu_process(gpu_id, cmd, log_dir, progress_bar)
            processes.append((gpu_id, process, streaming_threads, log_files, progress_bar))
            time.sleep(2)  # Small delay between launches to avoid resource contention
        except Exception as e:
            logger.error(f"[Multi-GPU] Failed to launch process for GPU {gpu_id}: {e}", exc_info=True)
            # Close all progress bars
            for pb in progress_bars.values():
                pb.close()
            # Terminate any already-launched processes
            for _, p, _, log_fs, _ in processes:
                p.terminate()
                # Close log files
                for log_f in log_fs:
                    try:
                        log_f.close()
                    except:
                        pass
            sys.exit(1)
    
    # Monitor processes
    all_success = monitor_processes(processes)
    
    if not all_success:
        logger.error("[Multi-GPU] Some processes failed. Check individual GPU logs for details.")
        logger.error(f"[Multi-GPU] Log directory: {log_dir}")
        sys.exit(1)
    
    # Aggregate results if PRM data collection was enabled
    if args.collect_prm_data:
        logger.info("=" * 80)
        logger.info("[Multi-GPU] Aggregating PRM data from all GPUs...")
        logger.info("=" * 80)
        
        batch_name = f"{args.task}_{args.method}_{timestamp}"
        
        try:
            merged_file = aggregate_prm_data(
                output_dir=args.prm_output_dir,
                num_gpus=args.num_gpus,
                batch_name=batch_name
            )
            
            if merged_file:
                logger.info("=" * 80)
                logger.info("[Multi-GPU] Data aggregation completed successfully!")
                logger.info("=" * 80)
                logger.info(f"[Multi-GPU] Merged data file: {merged_file}")
                logger.info("=" * 80)
            else:
                logger.error("[Multi-GPU] Data aggregation failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"[Multi-GPU] Failed to aggregate data: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("[Multi-GPU] Multi-GPU data collection completed successfully!")
    logger.info("=" * 80)
    logger.info(f"[Multi-GPU] Total samples processed: {args.max_samples}")
    logger.info(f"[Multi-GPU] Number of GPUs used: {args.num_gpus}")
    logger.info(f"[Multi-GPU] Log directory: {log_dir}")
    if args.collect_prm_data:
        logger.info(f"[Multi-GPU] PRM data directory: {args.prm_output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

