import os
import random
import re
import time
import functools
import logging
import json
import csv
from typing import Optional, Callable, Any, Dict
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Logger setup for profiling
profiling_logger = logging.getLogger(__name__ + ".profiling")

# Logger for output operations
output_logger = logging.getLogger(__name__ + ".output")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)


# ============================================================================
# Performance Profiling Utilities
# ============================================================================

# Global profiling state
_PROFILING_ENABLED = False
_PROFILING_STATS = defaultdict(lambda: {
    'count': 0,
    'total_time': 0.0,
    'min_time': float('inf'),
    'max_time': 0.0,
    'total_memory': 0,
    'total_tokens': 0,
    'total_flops': 0,
})


def enable_profiling() -> None:
    """Enable performance profiling globally."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = True
    profiling_logger.info("[Profiling] Performance profiling enabled")


def disable_profiling() -> None:
    """Disable performance profiling globally."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = False
    profiling_logger.info("[Profiling] Performance profiling disabled")


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled.
    
    Returns:
        True if profiling is enabled
    """
    return _PROFILING_ENABLED


def reset_profiling_stats() -> None:
    """Reset all profiling statistics."""
    global _PROFILING_STATS
    _PROFILING_STATS.clear()
    profiling_logger.info("[Profiling] Reset all profiling statistics")


def get_profiling_stats() -> Dict[str, Dict[str, Any]]:
    """Get current profiling statistics.
    
    Returns:
        Dictionary mapping function names to their statistics
    """
    return dict(_PROFILING_STATS)


def print_profiling_report() -> None:
    """Print a formatted profiling report."""
    if not _PROFILING_STATS:
        profiling_logger.info("[Profiling] No profiling data available")
        return
    
    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING REPORT")
    print("=" * 80)
    
    # Sort by total time
    sorted_stats = sorted(
        _PROFILING_STATS.items(),
        key=lambda x: x[1]['total_time'],
        reverse=True
    )
    
    print(f"{'Function':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    print("-" * 80)
    
    for func_name, stats in sorted_stats:
        avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
        print(
            f"{func_name:<40} "
            f"{stats['count']:>8} "
            f"{stats['total_time']:>10.3f} "
            f"{avg_time * 1000:>10.2f} "
            f"{stats['min_time'] * 1000:>10.2f} "
            f"{stats['max_time'] * 1000:>10.2f}"
        )
    
    # Print memory and compute stats if available
    total_memory = sum(s['total_memory'] for s in _PROFILING_STATS.values())
    total_tokens = sum(s['total_tokens'] for s in _PROFILING_STATS.values())
    total_flops = sum(s['total_flops'] for s in _PROFILING_STATS.values())
    
    if total_memory > 0 or total_tokens > 0 or total_flops > 0:
        print("-" * 80)
        if total_memory > 0:
            print(f"Total Memory Allocated: {total_memory / (1024**3):.2f} GB")
        if total_tokens > 0:
            print(f"Total Tokens Processed: {total_tokens:,}")
        if total_flops > 0:
            print(f"Total FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    
    print("=" * 80 + "\n")
    
    profiling_logger.info("[Profiling] Printed profiling report")


def profile(
    func: Optional[Callable] = None,
    *,
    track_memory: bool = False,
    track_tokens: bool = False,
    track_flops: bool = False
) -> Callable:
    """Decorator for profiling function performance.
    
    This decorator tracks execution time and optionally memory usage,
    token counts, and FLOPs for the decorated function.
    
    Args:
        func: Function to profile (when used without arguments)
        track_memory: Whether to track GPU memory usage
        track_tokens: Whether to track token counts (from return value)
        track_flops: Whether to track FLOPs (from return value)
    
    Returns:
        Decorated function
    
    Example:
        @profile
        def my_function():
            pass
        
        @profile(track_memory=True)
        def my_gpu_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not _PROFILING_ENABLED:
                # Profiling disabled, just call function
                return f(*args, **kwargs)
            
            func_name = f"{f.__module__}.{f.__name__}"
            
            # Track memory before
            memory_before = 0
            if track_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # Time execution
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            
            # Track memory after
            memory_used = 0
            if track_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_used = max(0, memory_after - memory_before)
            
            # Extract token count from result if requested
            tokens_used = 0
            if track_tokens and isinstance(result, dict) and 'num_tokens' in result:
                tokens_used = result['num_tokens']
            
            # Extract FLOPs from result if requested
            flops_used = 0
            if track_flops and isinstance(result, dict) and 'flops' in result:
                flops_used = result['flops']
            
            # Update statistics
            stats = _PROFILING_STATS[func_name]
            stats['count'] += 1
            stats['total_time'] += elapsed
            stats['min_time'] = min(stats['min_time'], elapsed)
            stats['max_time'] = max(stats['max_time'], elapsed)
            stats['total_memory'] += memory_used
            stats['total_tokens'] += tokens_used
            stats['total_flops'] += flops_used
            
            profiling_logger.debug(
                f"[Profiling] {func_name}: {elapsed * 1000:.2f}ms"
                + (f", memory={memory_used / (1024**2):.2f}MB" if memory_used > 0 else "")
                + (f", tokens={tokens_used}" if tokens_used > 0 else "")
                + (f", flops={flops_used / 1e9:.2f}GFLOPs" if flops_used > 0 else "")
            )
            
            return result
        
        return wrapper
    
    # Handle both @profile and @profile(...) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


class PerformanceTimer:
    """Context manager for timing code blocks.
    
    Example:
        with PerformanceTimer("my_operation") as timer:
            # code to time
            pass
        print(f"Elapsed: {timer.elapsed}s")
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        """Initialize the performance timer.
        
        Args:
            name: Name of the operation being timed
            log_level: Logging level ('DEBUG', 'INFO', etc.)
        """
        self.name = name
        self.log_level = log_level.upper()
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        if _PROFILING_ENABLED:
            profiling_logger.debug(f"[PerformanceTimer] Started: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log result."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if _PROFILING_ENABLED:
            log_func = getattr(profiling_logger, self.log_level.lower(), profiling_logger.info)
            log_func(f"[PerformanceTimer] {self.name}: {self.elapsed * 1000:.2f}ms")


class MemoryTracker:
    """Context manager for tracking GPU memory usage.
    
    Example:
        with MemoryTracker("my_operation") as tracker:
            # code to track
            pass
        print(f"Memory used: {tracker.memory_used / (1024**2):.2f} MB")
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        """Initialize the memory tracker.
        
        Args:
            name: Name of the operation being tracked
            log_level: Logging level ('DEBUG', 'INFO', etc.)
        """
        self.name = name
        self.log_level = log_level.upper()
        self.memory_before = 0
        self.memory_after = 0
        self.memory_used = 0
        self.memory_peak = 0
    
    def __enter__(self):
        """Start tracking memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.memory_before = torch.cuda.memory_allocated()
            if _PROFILING_ENABLED:
                profiling_logger.debug(
                    f"[MemoryTracker] Started: {self.name}, "
                    f"initial={self.memory_before / (1024**2):.2f}MB"
                )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and log result."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_after = torch.cuda.memory_allocated()
            self.memory_peak = torch.cuda.max_memory_allocated()
            self.memory_used = max(0, self.memory_after - self.memory_before)
            
            if _PROFILING_ENABLED:
                log_func = getattr(profiling_logger, self.log_level.lower(), profiling_logger.info)
                log_func(
                    f"[MemoryTracker] {self.name}: "
                    f"used={self.memory_used / (1024**2):.2f}MB, "
                    f"peak={self.memory_peak / (1024**2):.2f}MB"
                )


class ComputeTracker:
    """Tracks computational costs (tokens, FLOPs) for operations.
    
    Example:
        tracker = ComputeTracker()
        tracker.add_tokens(100)
        tracker.add_flops(1e9)
        print(tracker.get_summary())
    """
    
    def __init__(self):
        """Initialize the compute tracker."""
        self.total_tokens = 0
        self.total_flops = 0
        self.operations = []
    
    def add_tokens(self, num_tokens: int, operation: str = "unknown") -> None:
        """Add token count.
        
        Args:
            num_tokens: Number of tokens processed
            operation: Name of the operation
        """
        self.total_tokens += num_tokens
        self.operations.append({
            'type': 'tokens',
            'count': num_tokens,
            'operation': operation
        })
        
        if _PROFILING_ENABLED:
            profiling_logger.debug(
                f"[ComputeTracker] Added {num_tokens} tokens for {operation}"
            )
    
    def add_flops(self, num_flops: int, operation: str = "unknown") -> None:
        """Add FLOP count.
        
        Args:
            num_flops: Number of FLOPs
            operation: Name of the operation
        """
        self.total_flops += num_flops
        self.operations.append({
            'type': 'flops',
            'count': num_flops,
            'operation': operation
        })
        
        if _PROFILING_ENABLED:
            profiling_logger.debug(
                f"[ComputeTracker] Added {num_flops / 1e9:.2f} GFLOPs for {operation}"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked computations.
        
        Returns:
            Dictionary with computation statistics
        """
        summary = {
            'total_tokens': self.total_tokens,
            'total_flops': self.total_flops,
            'total_gflops': self.total_flops / 1e9,
            'total_tflops': self.total_flops / 1e12,
            'num_operations': len(self.operations),
        }
        
        if _PROFILING_ENABLED:
            profiling_logger.info(
                f"[ComputeTracker] Summary: {self.total_tokens:,} tokens, "
                f"{summary['total_tflops']:.2f} TFLOPs"
            )
        
        return summary
    
    def reset(self) -> None:
        """Reset all tracked computations."""
        self.total_tokens = 0
        self.total_flops = 0
        self.operations.clear()
        
        if _PROFILING_ENABLED:
            profiling_logger.debug("[ComputeTracker] Reset all tracked computations")


# ============================================================================
# Question-Answer Output Utilities
# ============================================================================

def create_output_file_path(task: str, method: str, model_name: str) -> str:
    """Create a standardized output file path for question-answer records.
    
    Args:
        task: Task name (e.g., 'gsm8k', 'custom')
        method: Method name (e.g., 'latent_mas_multipath')
        model_name: Model name (e.g., 'Qwen3-4B')
        
    Returns:
        Path to output file in output/res/ directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = model_name.split('/')[-1].replace('-', '_').lower()
    output_dir = Path('output') / 'res'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{task}_{method}_{model_short}_{timestamp}.jsonl"
    output_path = str(output_dir / output_filename)
    
    output_logger.info(f"[Output] Created output file path: {output_path}")
    return output_path


def create_result_log_file_path(task: str, method: str) -> str:
    """Create a standardized result log file path for run summary.
    
    Args:
        task: Task name (e.g., 'gsm8k', 'custom')
        method: Method name (e.g., 'latent_mas_multipath')
        
    Returns:
        Path to result log file in output/simple_res/ directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output') / 'simple_res'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_filename = f"{method}_{task}_{timestamp}.log"
    log_path = str(output_dir / log_filename)
    
    output_logger.debug(f"[Output] Created result log file path: {log_path}")
    return log_path


def save_question_answer_record(
    output_file: str,
    problem_idx: int,
    question: str,
    prediction: Optional[str],
    gold: Optional[str],
    correct: Optional[bool],
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save a single question-answer record to the output file.
    
    This function appends a JSON record to the output file in JSONL format.
    Each line is a complete JSON object representing one question-answer pair.
    
    Args:
        output_file: Path to the output file
        problem_idx: Problem index/number
        question: The question text
        prediction: The predicted answer
        gold: The gold/correct answer
        correct: Whether the prediction was correct (None if not evaluated)
        additional_info: Optional dictionary with additional information to include
    """
    try:
        record = {
            'problem_idx': problem_idx,
            'timestamp': datetime.now().isoformat(),
            'question': question.strip() if question else "",
            'prediction': str(prediction) if prediction is not None else None,
            'gold': str(gold) if gold is not None else None,
            'correct': correct,  # Can be None for tasks without evaluation
        }
        
        # Add any additional information
        if additional_info:
            record.update(additional_info)
        
        # Append to file as JSON line
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        output_logger.debug(
            f"[Output] Saved record for problem #{problem_idx}: "
            f"Pred={prediction}, Gold={gold}, Correct={correct}"
        )
        
    except Exception as e:
        output_logger.error(f"[Output] Failed to save record for problem #{problem_idx}: {e}", exc_info=True)


def save_to_csv_results(
    run_params: Dict[str, Any],
    results: Dict[str, Any],
    timestamp: datetime
) -> None:
    """Save run parameters and results to a CSV file.
    
    This function records the complete run summary including all parameters
    and final results (accuracy, success rate, etc.) to a CSV file for
    later analysis and comparison across runs.
    
    The CSV file is stored in output/csv_res/results.csv and will be created
    if it doesn't exist, or appended to if it already exists.
    
    Args:
        run_params: Dictionary containing all run parameters (method, model, task, etc.)
        results: Dictionary containing final results (accuracy, correct count, timing, etc.)
        timestamp: Timestamp of the run
    """
    logger = logging.getLogger(__name__ + ".csv")
    
    try:
        # Create output directory
        csv_dir = Path('output') / 'csv_res'
        csv_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[CSV] Ensured CSV directory exists: {csv_dir}")
        
        # CSV file path
        csv_file = csv_dir / 'results.csv'
        file_exists = csv_file.exists()
        
        if file_exists:
            logger.info(f"[CSV] CSV file exists, will append to: {csv_file}")
        else:
            logger.info(f"[CSV] CSV file does not exist, will create new: {csv_file}")
        
        # Prepare row data
        # Core fields that should always be present
        row_data = {
            # Timestamp fields
            'timestamp': timestamp.isoformat(),
            'run_date': timestamp.strftime('%Y-%m-%d'),
            'run_time': timestamp.strftime('%H:%M:%S'),
            
            # Task and method info
            'task': run_params.get('task', 'unknown'),
            'method': run_params.get('method', 'unknown'),
            'model_name': run_params.get('model_name', 'unknown'),
            
            # Results
            'accuracy': results.get('accuracy', 0.0),
            'success_rate': results.get('success_rate', 0.0),
            'correct': results.get('correct', 0),
            'total': results.get('total', 0),
            'total_time_sec': results.get('total_time_sec', 0.0),
            'time_per_sample_sec': results.get('time_per_sample_sec', 0.0),
            
            # Core parameters
            'max_samples': run_params.get('max_samples', 0),
            'seed': run_params.get('seed', 42),
            'device': run_params.get('device', 'cuda'),
            'prompt': run_params.get('prompt', 'sequential'),
            'split': run_params.get('split', 'test'),
            
            # Generation parameters
            'max_new_tokens': run_params.get('max_new_tokens', 4096),
            'temperature': run_params.get('temperature', 0.7),
            'top_p': run_params.get('top_p', 0.95),
            'generate_bs': run_params.get('generate_bs', 1),
            
            # Method-specific parameters
            'latent_steps': run_params.get('latent_steps', None),
            'text_mas_context_length': run_params.get('text_mas_context_length', None),
            'think': run_params.get('think', False),
            'latent_space_realign': run_params.get('latent_space_realign', False),
            
            # vLLM parameters
            'use_vllm': run_params.get('use_vllm', False),
            'enable_prefix_caching': run_params.get('enable_prefix_caching', False),
            'use_second_HF_model': run_params.get('use_second_HF_model', False),
            'tensor_parallel_size': run_params.get('tensor_parallel_size', 1),
            'gpu_memory_utilization': run_params.get('gpu_memory_utilization', 0.9),
            
            # Multi-path specific parameters
            'num_paths': run_params.get('num_paths', None),
            'enable_branching': run_params.get('enable_branching', None),
            'enable_merging': run_params.get('enable_merging', None),
            'pruning_strategy': run_params.get('pruning_strategy', None),
            'merge_threshold': run_params.get('merge_threshold', None),
            'branch_threshold': run_params.get('branch_threshold', None),
            'diversity_strategy': run_params.get('diversity_strategy', None),
            'latent_consistency_metric': run_params.get('latent_consistency_metric', None),
            
            # Configuration
            'config': run_params.get('config', None),
            'config_preset': run_params.get('config_preset', None),
            'enable_visualization': run_params.get('enable_visualization', True),
            
            # Command parameters as JSON string for full record
            'command_params_json': json.dumps(run_params, ensure_ascii=False),
        }
        
        logger.debug(f"[CSV] Prepared row data with {len(row_data)} fields")
        
        # Define field order for CSV header
        fieldnames = [
            # Timestamp
            'timestamp', 'run_date', 'run_time',
            # Task and method
            'task', 'method', 'model_name',
            # Results
            'accuracy', 'success_rate', 'correct', 'total',
            'total_time_sec', 'time_per_sample_sec',
            # Core parameters
            'max_samples', 'seed', 'device', 'prompt', 'split',
            # Generation parameters
            'max_new_tokens', 'temperature', 'top_p', 'generate_bs',
            # Method-specific
            'latent_steps', 'text_mas_context_length', 'think', 'latent_space_realign',
            # vLLM
            'use_vllm', 'enable_prefix_caching', 'use_second_HF_model',
            'tensor_parallel_size', 'gpu_memory_utilization',
            # Multi-path
            'num_paths', 'enable_branching', 'enable_merging',
            'pruning_strategy', 'merge_threshold', 'branch_threshold',
            'diversity_strategy', 'latent_consistency_metric',
            # Configuration
            'config', 'config_preset', 'enable_visualization',
            # Full JSON record
            'command_params_json',
        ]
        
        logger.debug(f"[CSV] Using {len(fieldnames)} fields for CSV header")
        
        # Write to CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
                logger.info("[CSV] Wrote CSV header to new file")
            
            # Write data row
            writer.writerow(row_data)
            logger.info("[CSV] Wrote data row to CSV file")
        
        logger.info(f"[CSV] Successfully saved results to CSV: {csv_file}")
        logger.info(f"[CSV] Summary: {results.get('accuracy', 0.0):.4f} accuracy, "
                   f"{results.get('correct', 0)}/{results.get('total', 0)} correct, "
                   f"{results.get('total_time_sec', 0.0):.2f}s total time")
        
    except Exception as e:
        logger.error(f"[CSV] Failed to save results to CSV: {e}", exc_info=True)
        logger.debug(f"[CSV] Error details: {type(e).__name__}: {str(e)}")


# ============================================================================
# Multi-GPU Data Distribution Utilities
# ============================================================================

multi_gpu_logger = logging.getLogger(__name__ + ".multi_gpu")


def split_dataset_for_multi_gpu(
    dataset: list,
    num_gpus: int,
    gpu_id: int
) -> list:
    """Split dataset for multi-GPU parallel processing.
    
    This function divides the dataset evenly across multiple GPUs, ensuring
    each GPU processes a distinct subset of the data. The split is deterministic
    based on GPU ID.
    
    Args:
        dataset: Complete dataset as a list
        num_gpus: Total number of GPUs to distribute across
        gpu_id: Current GPU ID (0-indexed)
        
    Returns:
        Subset of dataset assigned to this GPU
        
    Example:
        # 100 samples, 4 GPUs
        # GPU 0: samples 0-24 (25 samples)
        # GPU 1: samples 25-49 (25 samples)
        # GPU 2: samples 50-74 (25 samples)
        # GPU 3: samples 75-99 (25 samples)
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    
    if gpu_id < 0 or gpu_id >= num_gpus:
        raise ValueError(f"gpu_id must be in range [0, {num_gpus-1}], got {gpu_id}")
    
    total_samples = len(dataset)
    
    if total_samples == 0:
        multi_gpu_logger.warning("[Multi-GPU] Empty dataset provided")
        return []
    
    # Calculate start and end indices for this GPU
    start_idx, end_idx = get_gpu_data_range(total_samples, num_gpus, gpu_id)
    
    # Extract subset
    gpu_dataset = dataset[start_idx:end_idx]
    
    multi_gpu_logger.info(
        f"[Multi-GPU] GPU {gpu_id}/{num_gpus-1}: Assigned samples [{start_idx}:{end_idx}] "
        f"({len(gpu_dataset)} samples out of {total_samples} total)"
    )
    multi_gpu_logger.debug(
        f"[Multi-GPU] GPU {gpu_id} data range: start={start_idx}, end={end_idx}, "
        f"count={len(gpu_dataset)}"
    )
    
    return gpu_dataset


def get_gpu_data_range(
    total_samples: int,
    num_gpus: int,
    gpu_id: int
) -> tuple:
    """Calculate the data range (start, end) for a specific GPU.
    
    This function computes the start and end indices for data distribution
    across multiple GPUs. It ensures balanced distribution with any remainder
    samples distributed to the first few GPUs.
    
    Args:
        total_samples: Total number of samples in the dataset
        num_gpus: Total number of GPUs
        gpu_id: Current GPU ID (0-indexed)
        
    Returns:
        Tuple of (start_index, end_index) for this GPU
        
    Example:
        >>> get_gpu_data_range(100, 4, 0)
        (0, 25)
        >>> get_gpu_data_range(100, 4, 3)
        (75, 100)
        >>> get_gpu_data_range(101, 4, 0)  # Uneven split
        (0, 26)  # First GPU gets extra sample
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    
    if gpu_id < 0 or gpu_id >= num_gpus:
        raise ValueError(f"gpu_id must be in range [0, {num_gpus-1}], got {gpu_id}")
    
    if total_samples <= 0:
        multi_gpu_logger.warning(
            f"[Multi-GPU] total_samples is {total_samples}, returning empty range"
        )
        return (0, 0)
    
    # Calculate base samples per GPU and remainder
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    
    # Distribute remainder to first few GPUs
    # GPUs 0 to (remainder-1) get one extra sample
    if gpu_id < remainder:
        start_idx = gpu_id * (samples_per_gpu + 1)
        end_idx = start_idx + samples_per_gpu + 1
    else:
        start_idx = remainder * (samples_per_gpu + 1) + (gpu_id - remainder) * samples_per_gpu
        end_idx = start_idx + samples_per_gpu
    
    multi_gpu_logger.debug(
        f"[Multi-GPU] Range calculation: total={total_samples}, num_gpus={num_gpus}, "
        f"gpu_id={gpu_id}, base_per_gpu={samples_per_gpu}, remainder={remainder}, "
        f"result=[{start_idx}:{end_idx}]"
    )
    
    return (start_idx, end_idx)


def validate_gpu_availability(gpu_ids: list) -> bool:
    """Validate that specified GPUs are available.
    
    Args:
        gpu_ids: List of GPU IDs to validate
        
    Returns:
        True if all GPUs are available, False otherwise
    """
    if not torch.cuda.is_available():
        multi_gpu_logger.error("[Multi-GPU] CUDA is not available on this system")
        return False
    
    num_available_gpus = torch.cuda.device_count()
    multi_gpu_logger.info(f"[Multi-GPU] System has {num_available_gpus} CUDA devices available")
    
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= num_available_gpus:
            multi_gpu_logger.error(
                f"[Multi-GPU] GPU {gpu_id} is not available. "
                f"Valid range: [0, {num_available_gpus-1}]"
            )
            return False
        
        # Try to access the GPU
        try:
            device = torch.device(f'cuda:{gpu_id}')
            _ = torch.zeros(1, device=device)
            multi_gpu_logger.debug(f"[Multi-GPU] GPU {gpu_id} is accessible")
        except Exception as e:
            multi_gpu_logger.error(
                f"[Multi-GPU] Failed to access GPU {gpu_id}: {e}"
            )
            return False
    
    multi_gpu_logger.info(f"[Multi-GPU] All specified GPUs {gpu_ids} are available and accessible")
    return True


# ============================================================================
# Multi-GPU Result Aggregation Utilities
# ============================================================================

def aggregate_prm_data(
    output_dir: str,
    num_gpus: int,
    batch_name: str
) -> Optional[str]:
    """Aggregate PRM training data collected from multiple GPUs.
    
    This function merges data files from multiple GPU processes into a single
    unified dataset. It combines question records, tree structures, and metadata.
    
    Args:
        output_dir: Directory containing GPU-specific data files
        num_gpus: Number of GPUs that collected data
        batch_name: Base name for the batch (without GPU suffix)
        
    Returns:
        Path to the merged data file, or None if aggregation failed
    """
    multi_gpu_logger.info("=" * 80)
    multi_gpu_logger.info("[Multi-GPU Aggregation] Starting PRM data aggregation")
    multi_gpu_logger.info("=" * 80)
    multi_gpu_logger.info(f"[Multi-GPU Aggregation] Output directory: {output_dir}")
    multi_gpu_logger.info(f"[Multi-GPU Aggregation] Number of GPUs: {num_gpus}")
    multi_gpu_logger.info(f"[Multi-GPU Aggregation] Batch name: {batch_name}")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        multi_gpu_logger.error(f"[Multi-GPU Aggregation] Output directory does not exist: {output_dir}")
        return None
    
    # Collect all GPU data files
    all_question_records = []
    all_tree_structures = []
    gpu_metadata_list = []
    
    for gpu_id in range(num_gpus):
        gpu_batch_name = f"{batch_name}_gpu_{gpu_id}"
        data_file = output_path / f"{gpu_batch_name}.pt"
        metadata_file = output_path / f"{gpu_batch_name}_metadata.json"
        
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Processing GPU {gpu_id} data...")
        multi_gpu_logger.debug(f"[Multi-GPU Aggregation] Looking for data file: {data_file}")
        
        # Load data file
        if not data_file.exists():
            multi_gpu_logger.warning(
                f"[Multi-GPU Aggregation] Data file not found for GPU {gpu_id}: {data_file}"
            )
            continue
        
        try:
            # Load with weights_only=False since we have numpy objects and custom classes
            # This is safe as we trust the source (our own generated data)
            gpu_data = torch.load(data_file, map_location='cpu', weights_only=False)
            question_records = gpu_data.get('question_records', [])
            tree_structures = gpu_data.get('tree_structures', [])
            
            all_question_records.extend(question_records)
            all_tree_structures.extend(tree_structures)
            
            multi_gpu_logger.info(
                f"[Multi-GPU Aggregation] GPU {gpu_id}: Loaded {len(question_records)} questions, "
                f"{len(tree_structures)} trees"
            )
            multi_gpu_logger.debug(
                f"[Multi-GPU Aggregation] GPU {gpu_id} data keys: {list(gpu_data.keys())}"
            )
            
        except Exception as e:
            multi_gpu_logger.error(
                f"[Multi-GPU Aggregation] Failed to load data from GPU {gpu_id}: {e}",
                exc_info=True
            )
            continue
        
        # Load metadata file
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    gpu_metadata = json.load(f)
                    gpu_metadata_list.append(gpu_metadata)
                    multi_gpu_logger.debug(
                        f"[Multi-GPU Aggregation] GPU {gpu_id} metadata: "
                        f"{gpu_metadata.get('num_questions', 0)} questions, "
                        f"accuracy={gpu_metadata.get('accuracy', 0.0):.4f}"
                    )
            except Exception as e:
                multi_gpu_logger.warning(
                    f"[Multi-GPU Aggregation] Failed to load metadata from GPU {gpu_id}: {e}"
                )
    
    # Check if we have any data
    if not all_question_records:
        multi_gpu_logger.error("[Multi-GPU Aggregation] No data collected from any GPU")
        return None
    
    multi_gpu_logger.info(
        f"[Multi-GPU Aggregation] Aggregated {len(all_question_records)} total questions "
        f"from {len(gpu_metadata_list)} GPUs"
    )
    
    # Merge metadata
    merged_metadata = merge_prm_metadata(gpu_metadata_list, batch_name)
    
    # Save merged data
    merged_batch_name = f"{batch_name}_merged"
    merged_data_file = output_path / f"{merged_batch_name}.pt"
    merged_metadata_file = output_path / f"{merged_batch_name}_metadata.json"
    
    try:
        # Save merged data
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Saving merged data to: {merged_data_file}")
        torch.save({
            'question_records': all_question_records,
            'tree_structures': all_tree_structures,
        }, merged_data_file)
        
        # Save merged metadata
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Saving merged metadata to: {merged_metadata_file}")
        with open(merged_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=2, ensure_ascii=False)
        
        multi_gpu_logger.info("=" * 80)
        multi_gpu_logger.info("[Multi-GPU Aggregation] Aggregation completed successfully!")
        multi_gpu_logger.info("=" * 80)
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Merged data file: {merged_data_file}")
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Merged metadata file: {merged_metadata_file}")
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Total questions: {len(all_question_records)}")
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Total trees: {len(all_tree_structures)}")
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Total paths: {merged_metadata.get('num_paths_total', 0)}")
        multi_gpu_logger.info(f"[Multi-GPU Aggregation] Overall accuracy: {merged_metadata.get('accuracy', 0.0):.4f}")
        multi_gpu_logger.info("=" * 80)
        
        return str(merged_data_file)
        
    except Exception as e:
        multi_gpu_logger.error(
            f"[Multi-GPU Aggregation] Failed to save merged data: {e}",
            exc_info=True
        )
        return None


def merge_prm_metadata(
    metadata_list: list,
    batch_name: str
) -> dict:
    """Merge metadata from multiple GPU processes.
    
    Args:
        metadata_list: List of metadata dictionaries from each GPU
        batch_name: Name for the merged batch
        
    Returns:
        Merged metadata dictionary
    """
    if not metadata_list:
        multi_gpu_logger.warning("[Multi-GPU Aggregation] No metadata to merge")
        return {}
    
    multi_gpu_logger.info(f"[Multi-GPU Aggregation] Merging metadata from {len(metadata_list)} GPUs")
    
    # Aggregate statistics
    total_questions = sum(m.get('num_questions', 0) for m in metadata_list)
    total_paths = sum(m.get('num_paths_total', 0) for m in metadata_list)
    correct_questions = sum(
        int(m.get('num_questions', 0) * m.get('accuracy', 0.0))
        for m in metadata_list
    )
    
    overall_accuracy = correct_questions / total_questions if total_questions > 0 else 0.0
    
    # Get common fields from first metadata
    first_metadata = metadata_list[0]
    
    merged_metadata = {
        'task': first_metadata.get('task', 'unknown'),
        'method': first_metadata.get('method', 'unknown'),
        'model': first_metadata.get('model', 'unknown'),
        'num_questions': total_questions,
        'num_paths_total': total_paths,
        'accuracy': overall_accuracy,
        'timestamp': datetime.now().isoformat(),
        'batch_name': batch_name + '_merged',
        'num_gpus': len(metadata_list),
        'gpu_metadata': metadata_list,  # Keep individual GPU metadata for reference
    }
    
    multi_gpu_logger.info(
        f"[Multi-GPU Aggregation] Merged metadata: {total_questions} questions, "
        f"{total_paths} paths, accuracy={overall_accuracy:.4f}"
    )
    multi_gpu_logger.debug(f"[Multi-GPU Aggregation] Merged metadata keys: {list(merged_metadata.keys())}")
    
    return merged_metadata

