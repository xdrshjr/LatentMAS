"""Configuration management for LatentMAS Multi-Path reasoning.

This module provides configuration support for multi-path reasoning, including:
- MultiPathConfig dataclass for type-safe configuration
- JSON/YAML configuration file loading
- Preset configurations for common scenarios
- Configuration validation and merging with command-line arguments
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any, List
import argparse

# Logger setup
logger = logging.getLogger(__name__)

# Try to import YAML support (optional dependency)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available, YAML config files will not be supported")


@dataclass
class MultiPathConfig:
    """Configuration for multi-path latent reasoning.
    
    This dataclass encapsulates all configuration parameters for the
    Graph-Structured Multi-Path Latent Reasoning (GMLR) system.
    
    Attributes:
        # Core multi-path parameters
        num_paths: Number of parallel reasoning paths to maintain
        num_parent_paths: Number of top-scoring parent paths to use for next agent
        enable_branching: Whether to use adaptive branching based on uncertainty
        enable_merging: Whether to merge similar paths for efficiency
        pruning_strategy: Strategy for pruning paths ("topk", "adaptive", "diversity", "budget")
        merge_threshold: Cosine similarity threshold for path merging [0.0-1.0]
        branch_threshold: Uncertainty threshold for adaptive branching [0.0-1.0]
        diversity_strategy: Strategy for generating diverse paths ("temperature", "noise", "hybrid")
        latent_consistency_metric: Similarity metric for latent consistency scoring 
            ("cosine", "euclidean", "l2", "kl_divergence")
        
        # Scoring weights for ensemble scorer
        scoring_weights: Weights for different scoring metrics
            - latent_consistency: Weight for latent-based consistency score (default: 0.4)
            - self_consistency: Weight for text-based self-consistency score (slower, optional)
            - perplexity: Weight for perplexity-based score (default: 0.3)
            - verification: Weight for verification score (default: 0.2)
            - hidden_quality: Weight for hidden state quality (default: 0.1)
        
        # Generation parameters
        latent_steps: Number of latent thinking steps per path
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        max_new_tokens: Maximum new tokens for generation
        generate_bs: Batch size for generation
        
        # Pruning parameters
        pruning_keep_ratio: Ratio of paths to keep during pruning [0.0-1.0]
        pruning_min_paths: Minimum number of paths to keep after pruning
        pruning_max_paths: Maximum number of paths to maintain
        
        # Budget parameters (for budget-based pruning)
        max_compute_budget: Maximum computational budget (arbitrary units)
        cost_per_token: Cost per token for budget calculation
        
        # Visualization parameters
        enable_visualization: Whether to generate visualization graphs
    """
    
    # Core multi-path parameters
    num_paths: int = 5
    num_parent_paths: int = 5
    enable_branching: bool = True
    enable_merging: bool = True
    pruning_strategy: str = "adaptive"
    merge_threshold: float = 0.9
    branch_threshold: float = 0.5
    diversity_strategy: str = "hybrid"
    
    # Latent consistency scoring parameters
    latent_consistency_metric: str = "cosine"
    
    # Visualization parameters
    enable_visualization: bool = True
    
    # Scoring weights
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "latent_consistency": 0.4,  # Use latent-based consistency (faster, no decoding)
        "perplexity": 0.3,
        "verification": 0.2,
        "hidden_quality": 0.1,
    })
    
    # Generation parameters
    latent_steps: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 4096
    generate_bs: int = 20
    
    # Pruning parameters
    pruning_keep_ratio: float = 0.5
    pruning_min_paths: int = 2
    pruning_max_paths: int = 20
    
    # Budget parameters
    max_compute_budget: float = 1000000.0
    cost_per_token: float = 1.0
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        logger.debug("[MultiPathConfig] Validating configuration parameters")
        self._validate()
        logger.debug("[MultiPathConfig] Configuration validation passed")
    
    def _validate(self):
        """Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        # Validate num_paths
        if self.num_paths < 1:
            raise ValueError(f"num_paths must be >= 1, got {self.num_paths}")
        if self.num_paths > 100:
            logger.warning(f"[MultiPathConfig] num_paths={self.num_paths} is very large, may cause memory issues")
        
        # Validate num_parent_paths
        if self.num_parent_paths < 1:
            raise ValueError(f"num_parent_paths must be >= 1, got {self.num_parent_paths}")
        if self.num_parent_paths > self.num_paths:
            logger.warning(f"[MultiPathConfig] num_parent_paths ({self.num_parent_paths}) > num_paths ({self.num_paths}), "
                         f"will be capped at num_paths")
        
        # Validate pruning_strategy
        valid_strategies = ["topk", "adaptive", "diversity", "budget"]
        if self.pruning_strategy not in valid_strategies:
            raise ValueError(f"pruning_strategy must be one of {valid_strategies}, got '{self.pruning_strategy}'")
        
        # Validate diversity_strategy
        valid_diversity = ["temperature", "noise", "hybrid"]
        if self.diversity_strategy not in valid_diversity:
            raise ValueError(f"diversity_strategy must be one of {valid_diversity}, got '{self.diversity_strategy}'")
        
        # Validate latent_consistency_metric
        valid_metrics = ["cosine", "euclidean", "l2", "kl_divergence"]
        if self.latent_consistency_metric not in valid_metrics:
            raise ValueError(f"latent_consistency_metric must be one of {valid_metrics}, got '{self.latent_consistency_metric}'")
        logger.debug(f"[MultiPathConfig] Using latent consistency metric: {self.latent_consistency_metric}")
        
        # Validate thresholds (must be in [0, 1])
        if not 0.0 <= self.merge_threshold <= 1.0:
            raise ValueError(f"merge_threshold must be in [0.0, 1.0], got {self.merge_threshold}")
        if not 0.0 <= self.branch_threshold <= 1.0:
            raise ValueError(f"branch_threshold must be in [0.0, 1.0], got {self.branch_threshold}")
        if not 0.0 <= self.pruning_keep_ratio <= 1.0:
            raise ValueError(f"pruning_keep_ratio must be in [0.0, 1.0], got {self.pruning_keep_ratio}")
        
        # Validate scoring weights
        if not isinstance(self.scoring_weights, dict):
            raise ValueError(f"scoring_weights must be a dict, got {type(self.scoring_weights)}")
        
        # Check that at least one scoring metric is present
        valid_weight_keys = ["latent_consistency", "self_consistency", "perplexity", "verification", "hidden_quality"]
        if not any(key in self.scoring_weights for key in valid_weight_keys):
            raise ValueError(f"scoring_weights must contain at least one of: {valid_weight_keys}")
        
        # Validate each weight value is in [0, 1]
        for weight_name, weight_value in self.scoring_weights.items():
            if weight_name not in valid_weight_keys:
                logger.warning(f"[MultiPathConfig] Unknown scoring weight key: '{weight_name}'")
            if not 0.0 <= weight_value <= 1.0:
                raise ValueError(f"scoring_weights['{weight_name}'] must be in [0.0, 1.0], got {weight_value}")
        
        # Validate weights sum to approximately 1.0
        weight_sum = sum(self.scoring_weights.values())
        if not 0.99 <= weight_sum <= 1.01:
            logger.warning(f"[MultiPathConfig] scoring_weights sum to {weight_sum:.3f}, not 1.0. Weights will be normalized.")
        
        # Validate generation parameters
        if self.latent_steps < 0:
            raise ValueError(f"latent_steps must be >= 0, got {self.latent_steps}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.generate_bs < 1:
            raise ValueError(f"generate_bs must be >= 1, got {self.generate_bs}")
        
        # Validate pruning parameters
        if self.pruning_min_paths < 1:
            raise ValueError(f"pruning_min_paths must be >= 1, got {self.pruning_min_paths}")
        if self.pruning_max_paths < self.pruning_min_paths:
            raise ValueError(f"pruning_max_paths ({self.pruning_max_paths}) must be >= pruning_min_paths ({self.pruning_min_paths})")
        
        # Validate budget parameters
        if self.max_compute_budget <= 0:
            raise ValueError(f"max_compute_budget must be > 0, got {self.max_compute_budget}")
        if self.cost_per_token <= 0:
            raise ValueError(f"cost_per_token must be > 0, got {self.cost_per_token}")
        
        # Logical consistency checks
        if self.enable_merging and self.merge_threshold < 0.5:
            logger.warning(f"[MultiPathConfig] merge_threshold={self.merge_threshold} is low, may merge too aggressively")
        
        if self.num_paths < self.pruning_min_paths:
            logger.warning(f"[MultiPathConfig] num_paths ({self.num_paths}) < pruning_min_paths ({self.pruning_min_paths}), pruning may not be effective")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert configuration to JSON string or save to file.
        
        Args:
            filepath: Optional path to save JSON file. If None, returns JSON string.
            
        Returns:
            JSON string representation of the configuration.
        """
        json_str = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"[MultiPathConfig] Configuration saved to {filepath}")
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiPathConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            MultiPathConfig instance.
        """
        logger.debug(f"[MultiPathConfig] Creating config from dict with {len(config_dict)} parameters")
        return cls(**config_dict)


class ConfigLoader:
    """Loader for configuration files and presets."""
    
    @staticmethod
    def load_from_file(filepath: str) -> MultiPathConfig:
        """Load configuration from JSON or YAML file.
        
        Args:
            filepath: Path to configuration file (.json or .yaml/.yml).
            
        Returns:
            MultiPathConfig instance loaded from file.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported or invalid.
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        logger.info(f"[ConfigLoader] Loading configuration from {filepath}")
        
        # Determine file format
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            logger.debug(f"[ConfigLoader] Loaded JSON configuration with {len(config_dict)} parameters")
        
        elif suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ValueError("YAML support not available. Install PyYAML: pip install pyyaml")
            
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            logger.debug(f"[ConfigLoader] Loaded YAML configuration with {len(config_dict)} parameters")
        
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}. Use .json, .yaml, or .yml")
        
        config = MultiPathConfig.from_dict(config_dict)
        logger.info(f"[ConfigLoader] Configuration loaded successfully from {filepath}")
        return config
    
    @staticmethod
    def get_preset(preset_name: str) -> MultiPathConfig:
        """Get a preset configuration by name.
        
        Args:
            preset_name: Name of the preset configuration.
            
        Returns:
            MultiPathConfig instance with preset values.
            
        Raises:
            ValueError: If preset name is not recognized.
        """
        preset_name = preset_name.lower()
        logger.info(f"[ConfigLoader] Loading preset configuration: '{preset_name}'")
        
        if preset_name == "conservative":
            config = PRESET_CONSERVATIVE
        elif preset_name == "balanced":
            config = PRESET_BALANCED
        elif preset_name == "aggressive":
            config = PRESET_AGGRESSIVE
        elif preset_name == "fast":
            config = PRESET_FAST
        elif preset_name == "quality":
            config = PRESET_QUALITY
        else:
            available = ["conservative", "balanced", "aggressive", "fast", "quality"]
            raise ValueError(f"Unknown preset: '{preset_name}'. Available presets: {available}")
        
        logger.info(f"[ConfigLoader] Preset '{preset_name}' loaded: num_paths={config.num_paths}, pruning={config.pruning_strategy}")
        return config
    
    @staticmethod
    def merge_with_args(config: MultiPathConfig, args: argparse.Namespace) -> MultiPathConfig:
        """Merge configuration with command-line arguments.
        
        Command-line arguments override configuration file values.
        
        Args:
            config: Base configuration from file or preset.
            args: Command-line arguments from argparse.
            
        Returns:
            New MultiPathConfig with merged values.
        """
        logger.debug("[ConfigLoader] Merging configuration with command-line arguments")
        
        # Convert config to dict for easier manipulation
        config_dict = config.to_dict()
        
        # List of arguments that can override config
        override_params = [
            'num_paths', 'num_parent_paths', 'enable_branching', 'enable_merging', 'pruning_strategy',
            'merge_threshold', 'branch_threshold', 'diversity_strategy',
            'latent_consistency_metric',
            'latent_steps', 'temperature', 'top_p', 'max_new_tokens', 'generate_bs',
            'enable_visualization'
        ]
        
        overrides_applied = []
        
        for param in override_params:
            if hasattr(args, param):
                arg_value = getattr(args, param)
                # Only override if argument was explicitly provided (not None)
                if arg_value is not None:
                    old_value = config_dict.get(param)
                    # Only apply override if the value is actually different
                    if old_value != arg_value:
                        config_dict[param] = arg_value
                        overrides_applied.append(f"{param}: {old_value} -> {arg_value}")
                        logger.debug(f"[ConfigLoader] Override {param}: {old_value} -> {arg_value}")
        
        if overrides_applied:
            logger.info(f"[ConfigLoader] Applied {len(overrides_applied)} command-line overrides")
        else:
            logger.debug("[ConfigLoader] No command-line overrides applied")
        
        merged_config = MultiPathConfig.from_dict(config_dict)
        return merged_config
    
    @staticmethod
    def validate_config(config: MultiPathConfig) -> bool:
        """Validate a configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            config._validate()
            logger.info("[ConfigLoader] Configuration validation passed")
            return True
        except ValueError as e:
            logger.error(f"[ConfigLoader] Configuration validation failed: {e}")
            return False


# ============================================================================
# Preset Configurations
# ============================================================================

PRESET_CONSERVATIVE = MultiPathConfig(
    num_paths=3,
    enable_branching=False,
    enable_merging=True,
    pruning_strategy="topk",
    merge_threshold=0.95,
    branch_threshold=0.7,
    diversity_strategy="temperature",
    latent_steps=10,
    temperature=0.6,
    top_p=0.9,
    pruning_keep_ratio=0.4,
    pruning_min_paths=2,
    pruning_max_paths=5,
)
"""Conservative configuration: Few paths, aggressive pruning, high thresholds.

Best for: Limited computational resources, simple problems.
"""

PRESET_BALANCED = MultiPathConfig(
    num_paths=5,
    enable_branching=True,
    enable_merging=True,
    pruning_strategy="adaptive",
    merge_threshold=0.9,
    branch_threshold=0.5,
    diversity_strategy="hybrid",
    latent_steps=10,
    temperature=0.7,
    top_p=0.95,
    pruning_keep_ratio=0.5,
    pruning_min_paths=2,
    pruning_max_paths=10,
)
"""Balanced configuration: Default values, good for most cases.

Best for: General use, moderate computational resources.
"""

PRESET_AGGRESSIVE = MultiPathConfig(
    num_paths=10,
    enable_branching=True,
    enable_merging=True,
    pruning_strategy="diversity",
    merge_threshold=0.85,
    branch_threshold=0.4,
    diversity_strategy="hybrid",
    latent_steps=15,
    temperature=0.8,
    top_p=0.95,
    pruning_keep_ratio=0.6,
    pruning_min_paths=3,
    pruning_max_paths=20,
)
"""Aggressive configuration: Many paths, diverse exploration, careful pruning.

Best for: Difficult problems, ample computational resources, maximum accuracy.
"""

PRESET_FAST = MultiPathConfig(
    num_paths=3,
    enable_branching=False,
    enable_merging=True,
    pruning_strategy="topk",
    merge_threshold=0.95,
    branch_threshold=0.6,
    diversity_strategy="temperature",
    latent_steps=5,
    temperature=0.6,
    top_p=0.9,
    pruning_keep_ratio=0.3,
    pruning_min_paths=1,
    pruning_max_paths=5,
)
"""Fast configuration: Minimal paths, quick pruning, optimized for speed.

Best for: Time-sensitive applications, simple problems, quick iterations.
"""

PRESET_QUALITY = MultiPathConfig(
    num_paths=15,
    enable_branching=True,
    enable_merging=False,  # Keep all diverse paths
    pruning_strategy="diversity",
    merge_threshold=0.8,
    branch_threshold=0.3,
    diversity_strategy="hybrid",
    latent_steps=20,
    temperature=0.8,
    top_p=0.95,
    pruning_keep_ratio=0.7,
    pruning_min_paths=5,
    pruning_max_paths=30,
)
"""Quality configuration: Maximum paths, minimal pruning, optimized for accuracy.

Best for: Critical problems, maximum accuracy required, unlimited resources.
"""


# ============================================================================
# Utility Functions
# ============================================================================

def list_presets() -> List[str]:
    """Get list of available preset names.
    
    Returns:
        List of preset configuration names.
    """
    return ["conservative", "balanced", "aggressive", "fast", "quality"]


def get_preset_description(preset_name: str) -> str:
    """Get description of a preset configuration.
    
    Args:
        preset_name: Name of the preset.
        
    Returns:
        Description string.
    """
    descriptions = {
        "conservative": "Few paths, aggressive pruning, high thresholds. Best for limited resources.",
        "balanced": "Default values, good for most cases. Moderate computational resources.",
        "aggressive": "Many paths, diverse exploration, careful pruning. Best for difficult problems.",
        "fast": "Minimal paths, quick pruning, optimized for speed. Best for time-sensitive tasks.",
        "quality": "Maximum paths, minimal pruning, optimized for accuracy. Best for critical problems.",
    }
    return descriptions.get(preset_name.lower(), "Unknown preset")


def create_example_config(filepath: str = "config_example.json"):
    """Create an example configuration file.
    
    Args:
        filepath: Path where to save the example config.
    """
    config = PRESET_BALANCED
    config.to_json(filepath)
    logger.info(f"[Config] Example configuration created at {filepath}")
    print(f"Example configuration file created: {filepath}")
    print("Edit this file and use with: --config {filepath}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== LatentMAS Multi-Path Configuration System ===\n")
    
    # Test 1: Create and validate default config
    print("Test 1: Default configuration")
    config = MultiPathConfig()
    print(f"  num_paths: {config.num_paths}")
    print(f"  pruning_strategy: {config.pruning_strategy}")
    print(f"  Validation: {'PASS' if ConfigLoader.validate_config(config) else 'FAIL'}\n")
    
    # Test 2: Load presets
    print("Test 2: Available presets")
    for preset_name in list_presets():
        preset = ConfigLoader.get_preset(preset_name)
        print(f"  {preset_name}: {preset.num_paths} paths, {preset.pruning_strategy} pruning")
    print()
    
    # Test 3: Create example config file
    print("Test 3: Create example config file")
    create_example_config("config_example.json")
    print()
    
    # Test 4: Load from file
    print("Test 4: Load configuration from file")
    loaded_config = ConfigLoader.load_from_file("config_example.json")
    print(f"  Loaded: num_paths={loaded_config.num_paths}")
    print()
    
    print("=== All tests completed ===")

