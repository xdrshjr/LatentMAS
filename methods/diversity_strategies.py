"""Diversity generation strategies for multi-path latent reasoning.

This module implements various strategies to generate diverse reasoning paths
during multi-path exploration, ensuring paths explore different reasoning trajectories.
"""

import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

# Logger setup
logger = logging.getLogger(__name__)


class DiversityStrategy(ABC):
    """Abstract base class for diversity generation strategies.
    
    All diversity strategies should inherit from this class and implement
    the apply method to generate diverse latent paths.
    """
    
    @abstractmethod
    def apply(
        self,
        hidden_states: torch.Tensor,
        path_index: int,
        total_paths: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply diversity strategy to hidden states.
        
        Args:
            hidden_states: Input hidden states tensor [B, D]
            path_index: Index of current path (0 to total_paths-1)
            total_paths: Total number of paths being generated
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Modified hidden states with diversity applied
        """
        pass
    
    def get_temperature(self, path_index: int, total_paths: int) -> float:
        """Get temperature for a specific path.
        
        Args:
            path_index: Index of current path
            total_paths: Total number of paths
            
        Returns:
            Temperature value for this path
        """
        # Default: linearly interpolate between base temperatures
        return 0.7 + 0.6 * (path_index / max(1, total_paths - 1))


class TemperatureDiversityStrategy(DiversityStrategy):
    """Generate diversity using different temperature settings.
    
    Uses different temperatures for each path to encourage exploration.
    Since latent generation is deterministic, this strategy applies
    temperature-scaled noise to hidden states.
    
    Attributes:
        base_temp: Base temperature value
        temp_range: Range of temperature variation
        noise_scale: Base noise scale for temperature-based perturbation
    """
    
    def __init__(
        self,
        base_temp: float = 0.7,
        temp_range: float = 0.6,
        noise_scale: float = 0.2
    ):
        """Initialize temperature diversity strategy.
        
        Args:
            base_temp: Base temperature (for path 0)
            temp_range: Temperature range (max_temp = base_temp + temp_range)
            noise_scale: Base noise scale for perturbation
        """
        self.base_temp = base_temp
        self.temp_range = temp_range
        self.noise_scale = noise_scale
        logger.debug(f"[TemperatureDiversityStrategy] Initialized with base_temp={base_temp}, "
                    f"temp_range={temp_range}, noise_scale={noise_scale}")
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        path_index: int,
        total_paths: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply temperature-based diversity through noise perturbation.
        
        Since latent generation is deterministic, we use temperature to scale
        noise that's added to hidden states, simulating the effect of sampling
        at different temperatures.
        
        Args:
            hidden_states: Input hidden states [B, D]
            path_index: Current path index
            total_paths: Total number of paths
            **kwargs: Additional parameters (temperature, step, total_steps)
            
        Returns:
            Hidden states with temperature-scaled perturbation
        """
        temp = kwargs.get('temperature', self.get_temperature(path_index, total_paths))
        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}/{total_paths}: temperature={temp:.3f}")
        
        # Path 0 is deterministic
        if path_index == 0:
            logger.debug(f"[TemperatureDiversityStrategy] Path 0: no perturbation (deterministic)")
            return hidden_states
        
        # Get step info for progressive noise reduction
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 1)
        
        # Scale noise by temperature (higher temp = more noise)
        # Normalize temperature to [0, 1] range for scaling
        temp_normalized = (temp - self.base_temp) / self.temp_range if self.temp_range > 0 else 0.0
        temp_normalized = max(0.0, min(1.0, temp_normalized))
        
        # Reduce noise as we progress through steps
        step_factor = 1.0 - (0.2 * step / total_steps) if total_steps > 0 else 1.0
        
        # Calculate final noise scale
        noise_scale = self.noise_scale * (0.5 + temp_normalized) * step_factor
        
        # Add temperature-scaled Gaussian noise
        noise = torch.randn_like(hidden_states) * noise_scale
        perturbed = hidden_states + noise
        
        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}: noise_scale={noise_scale:.4f}, "
                    f"noise_norm={noise.norm().item():.4f}")
        
        return perturbed
    
    def get_temperature(self, path_index: int, total_paths: int) -> float:
        """Get temperature for a specific path.
        
        Args:
            path_index: Index of current path
            total_paths: Total number of paths
            
        Returns:
            Temperature value for this path
        """
        if total_paths == 1:
            return self.base_temp
        
        # Linearly interpolate from base_temp to base_temp + temp_range
        ratio = path_index / (total_paths - 1)
        temp = self.base_temp + self.temp_range * ratio
        return temp


class NoiseDiversityStrategy(DiversityStrategy):
    """Generate diversity by adding Gaussian noise to hidden states.
    
    Adds controlled noise to hidden states before realignment to explore
    different regions of the latent space.
    
    Attributes:
        noise_scale: Standard deviation of Gaussian noise
        adaptive: Whether to adapt noise based on path index
        continuous: Whether to apply noise at every step
    """
    
    def __init__(
        self,
        noise_scale: float = 0.3,
        adaptive: bool = True,
        continuous: bool = True
    ):
        """Initialize noise diversity strategy.
        
        Args:
            noise_scale: Standard deviation of Gaussian noise (increased default from 0.1 to 0.3)
            adaptive: If True, scale noise based on path index
            continuous: If True, apply noise at every latent step
        """
        self.noise_scale = noise_scale
        self.adaptive = adaptive
        self.continuous = continuous
        logger.debug(f"[NoiseDiversityStrategy] Initialized with noise_scale={noise_scale}, "
                    f"adaptive={adaptive}, continuous={continuous}")
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        path_index: int,
        total_paths: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply noise-based diversity to hidden states.
        
        Args:
            hidden_states: Input hidden states [B, D]
            path_index: Current path index
            total_paths: Total number of paths
            **kwargs: Additional parameters (step, total_steps, temperature)
            
        Returns:
            Hidden states with added noise
        """
        # Path 0 is kept deterministic (no noise)
        if path_index == 0:
            logger.debug(f"[NoiseDiversityStrategy] Path 0: no noise applied (deterministic)")
            return hidden_states
        
        # Get current step info for continuous noise application
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 1)
        temperature = kwargs.get('temperature', 1.0)
        
        # Calculate noise scale for this path
        if self.adaptive:
            # More noise for higher path indices
            path_factor = (path_index / max(1, total_paths - 1))
            
            # If continuous, reduce noise as we progress through steps
            if self.continuous and total_steps > 0:
                step_factor = 1.0 - (0.3 * step / total_steps)  # Reduce noise by up to 30% as we progress
            else:
                step_factor = 1.0
            
            # Use temperature to scale noise
            temp_factor = min(temperature / 0.7, 2.0)  # Scale by temperature, cap at 2x
            
            scale = self.noise_scale * path_factor * step_factor * temp_factor
        else:
            scale = self.noise_scale
        
        # Add Gaussian noise
        noise = torch.randn_like(hidden_states) * scale
        noisy_hidden = hidden_states + noise
        
        logger.debug(f"[NoiseDiversityStrategy] Path {path_index}/{total_paths}, step {step}/{total_steps}: "
                    f"noise_scale={scale:.4f}, noise_norm={noise.norm().item():.4f}")
        
        return noisy_hidden


class InitializationDiversityStrategy(DiversityStrategy):
    """Generate diversity through different starting points.
    
    Uses different initialization strategies for latent thinking,
    such as perturbing the initial hidden state or using different
    layers' hidden states as starting points.
    
    Attributes:
        perturbation_scale: Scale of initial perturbation
        use_layer_variation: Whether to use different layers for initialization
    """
    
    def __init__(
        self,
        perturbation_scale: float = 0.05,
        use_layer_variation: bool = False
    ):
        """Initialize initialization diversity strategy.
        
        Args:
            perturbation_scale: Scale of initial perturbation
            use_layer_variation: Use different layers for different paths
        """
        self.perturbation_scale = perturbation_scale
        self.use_layer_variation = use_layer_variation
        logger.debug(f"[InitializationDiversityStrategy] Initialized with "
                    f"perturbation_scale={perturbation_scale}, "
                    f"use_layer_variation={use_layer_variation}")
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        path_index: int,
        total_paths: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply initialization-based diversity.
        
        Args:
            hidden_states: Input hidden states [B, D]
            path_index: Current path index
            total_paths: Total number of paths
            **kwargs: May contain 'all_layer_hiddens' for layer variation
            
        Returns:
            Modified hidden states
        """
        # Path 0 uses original hidden states
        if path_index == 0:
            logger.debug(f"[InitializationDiversityStrategy] Path 0: using original hidden states")
            return hidden_states
        
        # Apply perturbation
        perturbation = torch.randn_like(hidden_states) * self.perturbation_scale
        perturbed = hidden_states + perturbation
        
        logger.debug(f"[InitializationDiversityStrategy] Path {path_index}/{total_paths}: "
                    f"perturbation_norm={perturbation.norm().item():.4f}")
        
        return perturbed


class HybridDiversityStrategy(DiversityStrategy):
    """Combine multiple diversity strategies.
    
    Applies multiple diversity strategies in sequence or with weighted combination.
    
    Attributes:
        strategies: List of (strategy, weight) tuples
        combination_mode: How to combine strategies ('sequential' or 'weighted')
    """
    
    def __init__(
        self,
        strategies: Optional[List[tuple]] = None,
        combination_mode: str = 'sequential'
    ):
        """Initialize hybrid diversity strategy.
        
        Args:
            strategies: List of (strategy, weight) tuples
            combination_mode: 'sequential' applies strategies in order,
                            'weighted' combines their effects
        """
        self.strategies = strategies or []
        self.combination_mode = combination_mode
        
        if not self.strategies:
            # Default: combine temperature and noise strategies with increased diversity
            logger.info("[HybridDiversityStrategy] No strategies provided, using default combination")
            self.strategies = [
                (TemperatureDiversityStrategy(noise_scale=0.2), 0.5),
                (NoiseDiversityStrategy(noise_scale=0.3, continuous=True), 0.5)
            ]
        
        logger.info(f"[HybridDiversityStrategy] Initialized with {len(self.strategies)} strategies, "
                   f"mode={combination_mode}")
        for i, (strategy, weight) in enumerate(self.strategies):
            logger.debug(f"[HybridDiversityStrategy] Strategy {i}: {strategy.__class__.__name__} (weight={weight:.3f})")
    
    def apply(
        self,
        hidden_states: torch.Tensor,
        path_index: int,
        total_paths: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply hybrid diversity strategy.
        
        Args:
            hidden_states: Input hidden states [B, D]
            path_index: Current path index
            total_paths: Total number of paths
            **kwargs: Additional parameters passed to sub-strategies
            
        Returns:
            Modified hidden states
        """
        logger.debug(f"[HybridDiversityStrategy] Applying {len(self.strategies)} strategies "
                    f"for path {path_index}/{total_paths}")
        
        if self.combination_mode == 'sequential':
            # Apply strategies sequentially
            result = hidden_states
            for strategy, _ in self.strategies:
                result = strategy.apply(result, path_index, total_paths, **kwargs)
            return result
        
        elif self.combination_mode == 'weighted':
            # Apply strategies and combine with weights
            results = []
            weights = []
            
            for strategy, weight in self.strategies:
                modified = strategy.apply(hidden_states, path_index, total_paths, **kwargs)
                results.append(modified)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                logger.warning("[HybridDiversityStrategy] Total weight is zero, returning original hidden states")
                return hidden_states
            
            # Weighted combination
            combined = torch.zeros_like(hidden_states)
            for result, weight in zip(results, weights):
                combined += result * (weight / total_weight)
            
            return combined
        
        else:
            logger.error(f"[HybridDiversityStrategy] Unknown combination mode: {self.combination_mode}")
            return hidden_states
    
    def add_strategy(self, strategy: DiversityStrategy, weight: float = 1.0) -> None:
        """Add a strategy to the hybrid.
        
        Args:
            strategy: Diversity strategy to add
            weight: Weight for this strategy
        """
        self.strategies.append((strategy, weight))
        logger.info(f"[HybridDiversityStrategy] Added strategy {strategy.__class__.__name__} "
                   f"with weight {weight:.3f}")
    
    def get_temperature(self, path_index: int, total_paths: int) -> float:
        """Get temperature from temperature-based sub-strategies.
        
        Args:
            path_index: Index of current path
            total_paths: Total number of paths
            
        Returns:
            Temperature value (from first temperature strategy found)
        """
        for strategy, _ in self.strategies:
            if isinstance(strategy, TemperatureDiversityStrategy):
                return strategy.get_temperature(path_index, total_paths)
        
        # Default temperature if no temperature strategy found
        return 0.7 + 0.6 * (path_index / max(1, total_paths - 1))


def create_diversity_strategy(
    strategy_type: str = 'hybrid',
    **kwargs
) -> DiversityStrategy:
    """Factory function to create diversity strategies.
    
    Args:
        strategy_type: Type of strategy ('temperature', 'noise', 'initialization', 'hybrid')
        **kwargs: Strategy-specific parameters
        
    Returns:
        Diversity strategy instance
    """
    logger.info(f"[DiversityStrategyFactory] Creating strategy: {strategy_type}")
    
    if strategy_type == 'temperature':
        return TemperatureDiversityStrategy(**kwargs)
    elif strategy_type == 'noise':
        return NoiseDiversityStrategy(**kwargs)
    elif strategy_type == 'initialization':
        return InitializationDiversityStrategy(**kwargs)
    elif strategy_type == 'hybrid':
        return HybridDiversityStrategy(**kwargs)
    else:
        logger.warning(f"[DiversityStrategyFactory] Unknown strategy type: {strategy_type}, "
                      f"defaulting to hybrid")
        return HybridDiversityStrategy(**kwargs)

