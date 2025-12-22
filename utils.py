import os
import random
import re
import time
import functools
import logging
import json
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


def save_question_answer_record(
    output_file: str,
    problem_idx: int,
    question: str,
    prediction: Optional[str],
    gold: Optional[str],
    correct: bool,
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
        correct: Whether the prediction was correct
        additional_info: Optional dictionary with additional information to include
    """
    try:
        record = {
            'problem_idx': problem_idx,
            'timestamp': datetime.now().isoformat(),
            'question': question.strip() if question else "",
            'prediction': str(prediction) if prediction is not None else None,
            'gold': str(gold) if gold is not None else None,
            'correct': correct,
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

