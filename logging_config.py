"""Enhanced logging configuration for LatentMAS project.

This module provides improved logging with:
- Module-specific color-coded formatting
- Reduced console verbosity with detailed file logging
- Support for progress bars at the bottom
- Clear separation between different components
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Component colors
    MAIN = '\033[94m'      # Blue - main execution
    METHOD = '\033[92m'    # Green - method execution
    MODEL = '\033[93m'     # Yellow - model operations
    DATA = '\033[96m'      # Cyan - data loading
    SCORING = '\033[95m'   # Magenta - scoring operations
    PRUNING = '\033[91m'   # Red - pruning operations
    PATH = '\033[97m'      # White - path management
    
    # Log level colors
    DEBUG = '\033[37m'     # Gray
    INFO = '\033[97m'      # White
    WARNING = '\033[93m'   # Yellow
    ERROR = '\033[91m'     # Red


# Module to color mapping
MODULE_COLORS = {
    'run': LogColors.MAIN,
    'run_direct': LogColors.MAIN,
    'methods.latent_mas': LogColors.METHOD,
    'methods.latent_mas_multipath': LogColors.METHOD,
    'methods.baseline': LogColors.METHOD,
    'methods.text_mas': LogColors.METHOD,
    'methods.scoring_metrics': LogColors.SCORING,
    'methods.pruning_strategies': LogColors.PRUNING,
    'methods.path_manager': LogColors.PATH,
    'methods.path_merging': LogColors.PATH,
    'methods.diversity_strategies': LogColors.METHOD,
    'models': LogColors.MODEL,
    'data': LogColors.DATA,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded module names and clean output."""
    
    def __init__(self, use_colors: bool = True, verbose: bool = False):
        """Initialize the formatter.
        
        Args:
            use_colors: Whether to use ANSI colors in output
            verbose: Whether to show full module names and timestamps
        """
        self.use_colors = use_colors
        self.verbose = verbose
        
        if verbose:
            # Detailed format for file logging
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            # Compact format for console
            fmt = '%(levelname)s - %(message)s'
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and module prefixes.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Get module color
        module_name = record.name
        module_color = MODULE_COLORS.get(module_name, LogColors.RESET)
        
        # Extract component name from module
        if '.' in module_name:
            component = module_name.split('.')[-1]
        else:
            component = module_name
        
        # Shorten common component names
        component_map = {
            'latent_mas_multipath': 'MultiPath',
            'latent_mas': 'LatentMAS',
            'scoring_metrics': 'Scoring',
            'pruning_strategies': 'Pruning',
            'path_manager': 'PathMgr',
            'path_merging': 'PathMerge',
            'diversity_strategies': 'Diversity',
        }
        component = component_map.get(component, component.capitalize())
        
        if self.use_colors and not self.verbose:
            # Colored, compact format for console
            level_colors = {
                'DEBUG': LogColors.DEBUG,
                'INFO': LogColors.INFO,
                'WARNING': LogColors.WARNING,
                'ERROR': LogColors.ERROR,
            }
            level_color = level_colors.get(record.levelname, LogColors.RESET)
            
            # Format: [Component] Message
            formatted = f"{module_color}[{component}]{LogColors.RESET} {level_color}{record.getMessage()}{LogColors.RESET}"
            return formatted
        else:
            # Standard format for file logging
            return super().format(record)


class ProgressBarHandler(logging.Handler):
    """Custom handler that works with progress bars.
    
    This handler ensures log messages don't interfere with progress bars
    by writing above them.
    """
    
    def __init__(self, stream=None):
        """Initialize the handler.
        
        Args:
            stream: Output stream (default: sys.stderr)
        """
        super().__init__()
        self.stream = stream or sys.stderr
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record.
        
        Args:
            record: Log record to emit
        """
        try:
            msg = self.format(record)
            # Write to stream
            self.stream.write(msg + '\n')
            self.stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    log_level: str = "INFO",
    console_level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    progress_bar_mode: bool = True
) -> None:
    """Setup logging configuration for the entire project.
    
    Args:
        log_level: Overall logging level (DEBUG, INFO, WARNING, ERROR)
        console_level: Console output level (can be higher than log_level)
        log_file: Optional path to log file for detailed logging
        use_colors: Whether to use colored output in console
        progress_bar_mode: Whether to configure logging for progress bar compatibility
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with compact, colored format
    if progress_bar_mode:
        console_handler = ProgressBarHandler(sys.stderr)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
    
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors, verbose=False))
    root_logger.addHandler(console_handler)
    
    # File handler with detailed format (if specified)
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(ColoredFormatter(use_colors=False, verbose=True))
        root_logger.addHandler(file_handler)
    
    # Set specific module levels to reduce noise
    _configure_module_levels(console_level)


def _configure_module_levels(console_level: str) -> None:
    """Configure logging levels for specific modules to reduce noise.
    
    Args:
        console_level: Base console logging level
    """
    # Reduce verbosity of certain modules
    noisy_modules = [
        'transformers',
        'torch',
        'vllm',
        'urllib3',
        'asyncio',
    ]
    
    for module in noisy_modules:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    # Set method modules to appropriate levels
    if console_level == "INFO":
        # In INFO mode, show less detail from internal operations
        logging.getLogger('methods.scoring_metrics').setLevel(logging.INFO)
        logging.getLogger('methods.pruning_strategies').setLevel(logging.INFO)
        logging.getLogger('methods.path_manager').setLevel(logging.INFO)
        logging.getLogger('methods.path_merging').setLevel(logging.INFO)
        logging.getLogger('methods.diversity_strategies').setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def create_log_file_path(task: str, method: str, model_name: str) -> str:
    """Create a standardized log file path.
    
    Args:
        task: Task name (e.g., 'gsm8k')
        method: Method name (e.g., 'latent_mas_multipath')
        model_name: Model name (e.g., 'Qwen3-4B')
        
    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = model_name.split('/')[-1].replace('-', '_').lower()
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_filename = f"{task}_{method}_{model_short}_{timestamp}.log"
    return str(log_dir / log_filename)

