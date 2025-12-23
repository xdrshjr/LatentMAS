"""Diversity generation strategies for multi-path latent reasoning.

This module implements various strategies to generate diverse reasoning paths
during multi-path exploration, ensuring paths explore different reasoning trajectories.
"""

import logging
import random
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
    
    Attributes:
        base_temperature: Baseline temperature for generating temperature series
    """
    
    def __init__(self, base_temperature: float = 0.7):
        """Initialize diversity strategy with baseline temperature.
        
        Args:
            base_temperature: Baseline temperature for diversity generation (default: 0.7)
        """
        self.base_temperature = base_temperature
        logger.debug(f"[DiversityStrategy] Initialized with base_temperature={base_temperature}")
    
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
        """Get temperature for a specific path based on baseline temperature.
        
        Generates randomized temperatures centered around the baseline temperature.
        The temperature range is: [base_temperature - 0.3, base_temperature + 0.3]
        
        Args:
            path_index: Index of current path (used for logging only, not for calculation)
            total_paths: Total number of paths
            
        Returns:
            Temperature value for this path
        """
        if total_paths == 1:
            logger.debug(f"[DiversityStrategy] Single path, using base_temperature={self.base_temperature:.3f}")
            return self.base_temperature
        
        # Generate random temperature within range: base_temp ± 0.3
        # This breaks the monotonic pattern and ensures diversity is not index-dependent
        temp_range = 0.6  # Total range: ±0.3
        random_offset = random.uniform(-0.3, 0.3)
        temperature = self.base_temperature + random_offset
        
        logger.debug(f"[DiversityStrategy] Path {path_index}/{total_paths}: "
                    f"base_temp={self.base_temperature:.3f}, random_offset={random_offset:.3f}, "
                    f"generated_temp={temperature:.3f}")
        
        return temperature


class TemperatureDiversityStrategy(DiversityStrategy):
    """Generate diversity using different temperature settings.
    
    Uses different temperatures for each path to encourage exploration.
    Since latent generation is deterministic, this strategy applies
    temperature-scaled noise to hidden states.
    
    Attributes:
        base_temperature: Baseline temperature value (inherited from DiversityStrategy)
        temp_range: Range of temperature variation (±0.3 from baseline)
        noise_scale: Base noise scale for temperature-based perturbation
    """
    
    def __init__(
        self,
        base_temperature: float = 0.7,
        noise_scale: float = 0.2
    ):
        """Initialize temperature diversity strategy.
        
        Args:
            base_temperature: Baseline temperature for generating temperature series
            noise_scale: Base noise scale for perturbation
        """
        super().__init__(base_temperature=base_temperature)
        self.temp_range = 0.6  # Fixed range: ±0.3 from baseline
        self.noise_scale = noise_scale
        logger.info(f"[TemperatureDiversityStrategy] Initialized with base_temperature={base_temperature}, "
                   f"temp_range=±0.3, noise_scale={noise_scale}")

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
        # Note: We no longer use a deterministic path to avoid bias in consistency scoring
        # All paths receive random perturbations, but with varying scales (including very small ones)
        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}/{total_paths}: applying random perturbation")

        # Get temperature (now randomized, not index-based)
        temp = kwargs.get('temperature', self.get_temperature(path_index, total_paths))
        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}/{total_paths}: temperature={temp:.3f}")

        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 1)

        # Random noise scale with very small minimum to include near-deterministic paths
        # This breaks the monotonic pattern while maintaining some stability
        noise_scale_min = 0.01 * self.noise_scale  # Very small perturbation (near-deterministic)
        noise_scale_max = 2.0 * self.noise_scale   # Large perturbation
        random_noise_scale = random.uniform(noise_scale_min, noise_scale_max)
        
        # Step factor: reduce noise as we progress through steps
        step_factor = 1.0 - (0.2 * step / total_steps) if total_steps > 0 else 1.0

        # Use relative noise based on input standard deviation for adaptive scaling
        hidden_std = hidden_states.std(dim=-1, keepdim=True).clamp(min=1e-6)
        final_noise_scale = random_noise_scale * step_factor

        # Generate noise relative to input magnitude
        noise = torch.randn_like(hidden_states) * final_noise_scale * hidden_std
        perturbed = hidden_states + noise

        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}: "
                     f"random_noise_scale={random_noise_scale:.4f}, step_factor={step_factor:.4f}, "
                     f"final_noise_scale={final_noise_scale:.4f}, hidden_std={hidden_std.mean():.4f}, "
                     f"relative_noise={noise.std() / hidden_std.mean():.4f}")

        return perturbed
    
    def get_temperature(self, path_index: int, total_paths: int) -> float:
        """Get temperature for a specific path based on baseline temperature.
        
        Generates randomized temperatures in range: [base_temperature - 0.3, base_temperature + 0.3]
        
        Args:
            path_index: Index of current path (used for logging only)
            total_paths: Total number of paths
            
        Returns:
            Temperature value for this path
        """
        if total_paths == 1:
            logger.debug(f"[TemperatureDiversityStrategy] Single path, using base_temperature={self.base_temperature:.3f}")
            return self.base_temperature
        
        # Generate random temperature within range: base_temperature ± 0.3
        # This breaks the monotonic index-based pattern
        random_offset = random.uniform(-0.3, 0.3)
        temp = self.base_temperature + random_offset
        
        logger.debug(f"[TemperatureDiversityStrategy] Path {path_index}/{total_paths}: "
                    f"base_temp={self.base_temperature:.3f}, random_offset={random_offset:.3f}, "
                    f"generated_temp={temp:.3f}")
        
        return temp


class NoiseDiversityStrategy(DiversityStrategy):
    """Generate diversity by adding Gaussian noise to hidden states.
    
    Adds controlled noise to hidden states before realignment to explore
    different regions of the latent space.
    
    Attributes:
        base_temperature: Baseline temperature value (inherited from DiversityStrategy)
        noise_scale: Standard deviation of Gaussian noise
        adaptive: Whether to adapt noise based on path index
        continuous: Whether to apply noise at every step
    """
    
    def __init__(
        self,
        base_temperature: float = 0.7,
        noise_scale: float = 0.3,
        adaptive: bool = True,
        continuous: bool = True
    ):
        """Initialize noise diversity strategy.
        
        Args:
            base_temperature: Baseline temperature for generating temperature series
            noise_scale: Standard deviation of Gaussian noise (increased default from 0.1 to 0.3)
            adaptive: If True, scale noise based on path index
            continuous: If True, apply noise at every latent step
        """
        super().__init__(base_temperature=base_temperature)
        self.noise_scale = noise_scale
        self.adaptive = adaptive
        self.continuous = continuous
        logger.info(f"[NoiseDiversityStrategy] Initialized with base_temperature={base_temperature}, "
                   f"noise_scale={noise_scale}, adaptive={adaptive}, continuous={continuous}")
    
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
            **kwargs: Additional parameters (step, total_steps, temperature, deterministic_path)
            
        Returns:
            Hidden states with added noise
        """
        # Note: We no longer use a deterministic path to avoid bias in consistency scoring
        # All paths receive random noise, but with varying scales (including very small ones)
        logger.debug(f"[NoiseDiversityStrategy] Path {path_index}/{total_paths}: applying random noise")
        
        # Get current step info for continuous noise application
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 1)
        temperature = kwargs.get('temperature', 1.0)
        
        # Calculate noise scale for this path
        if self.adaptive:
            # Random noise factor instead of index-based monotonic scaling
            # Include very small factors to create near-deterministic paths
            path_factor = random.uniform(0.01, 1.5)  # 0.01 = near-deterministic, 1.5 = high diversity
            
            # If continuous, reduce noise as we progress through steps
            if self.continuous and total_steps > 0:
                step_factor = 1.0 - (0.3 * step / total_steps)  # Reduce noise by up to 30% as we progress
            else:
                step_factor = 1.0
            
            # Use temperature to scale noise (normalized by baseline)
            temp_factor = min(temperature / self.base_temperature, 2.0)  # Scale by temperature, cap at 2x
            
            scale = self.noise_scale * path_factor * step_factor * temp_factor
            
            logger.debug(f"[NoiseDiversityStrategy] Path {path_index}/{total_paths}, step {step}/{total_steps}: "
                        f"random_path_factor={path_factor:.4f}, step_factor={step_factor:.4f}, "
                        f"temp_factor={temp_factor:.4f}")
        else:
            scale = self.noise_scale
            logger.debug(f"[NoiseDiversityStrategy] Path {path_index}/{total_paths}: "
                        f"non-adaptive, using base noise_scale={scale:.4f}")
        
        # Add Gaussian noise
        noise = torch.randn_like(hidden_states) * scale
        noisy_hidden = hidden_states + noise
        
        logger.debug(f"[NoiseDiversityStrategy] Path {path_index}/{total_paths}, step {step}/{total_steps}: "
                    f"final_noise_scale={scale:.4f}, noise_norm={noise.norm().item():.4f}")
        
        return noisy_hidden


class InitializationDiversityStrategy(DiversityStrategy):
    """Generate diversity through different starting points.
    
    Uses different initialization strategies for latent thinking,
    such as perturbing the initial hidden state or using different
    layers' hidden states as starting points.
    
    Attributes:
        base_temperature: Baseline temperature value (inherited from DiversityStrategy)
        perturbation_scale: Scale of initial perturbation
        use_layer_variation: Whether to use different layers for initialization
    """
    
    def __init__(
        self,
        base_temperature: float = 0.7,
        perturbation_scale: float = 0.05,
        use_layer_variation: bool = False
    ):
        """Initialize initialization diversity strategy.
        
        Args:
            base_temperature: Baseline temperature for generating temperature series
            perturbation_scale: Scale of initial perturbation
            use_layer_variation: Use different layers for different paths
        """
        super().__init__(base_temperature=base_temperature)
        self.perturbation_scale = perturbation_scale
        self.use_layer_variation = use_layer_variation
        logger.info(f"[InitializationDiversityStrategy] Initialized with base_temperature={base_temperature}, "
                   f"perturbation_scale={perturbation_scale}, use_layer_variation={use_layer_variation}")
    
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
            **kwargs: May contain 'all_layer_hiddens', 'deterministic_path' for layer variation
            
        Returns:
            Modified hidden states
        """
        # Note: We no longer use a deterministic path to avoid bias in consistency scoring
        # All paths receive random perturbation, but with varying scales (including very small ones)
        logger.debug(f"[InitializationDiversityStrategy] Path {path_index}/{total_paths}: applying random perturbation")
        
        # Apply random perturbation with random scale
        # Include very small scales to create near-deterministic paths
        random_scale_factor = random.uniform(0.01, 2.0)  # 0.01 = near-deterministic, 2.0 = high diversity
        actual_perturbation_scale = self.perturbation_scale * random_scale_factor
        
        perturbation = torch.randn_like(hidden_states) * actual_perturbation_scale
        perturbed = hidden_states + perturbation
        
        logger.debug(f"[InitializationDiversityStrategy] Path {path_index}/{total_paths}: "
                    f"random_scale_factor={random_scale_factor:.4f}, "
                    f"actual_perturbation_scale={actual_perturbation_scale:.4f}, "
                    f"perturbation_norm={perturbation.norm().item():.4f}")
        
        return perturbed


class HybridDiversityStrategy(DiversityStrategy):
    """Combine multiple diversity strategies.
    
    Applies multiple diversity strategies in sequence or with weighted combination.
    
    Attributes:
        base_temperature: Baseline temperature value (inherited from DiversityStrategy)
        strategies: List of (strategy, weight) tuples
        combination_mode: How to combine strategies ('sequential' or 'weighted')
    """
    
    def __init__(
        self,
        base_temperature: float = 0.7,
        strategies: Optional[List[tuple]] = None,
        combination_mode: str = 'sequential'
    ):
        """Initialize hybrid diversity strategy.
        
        Args:
            base_temperature: Baseline temperature for generating temperature series
            strategies: List of (strategy, weight) tuples
            combination_mode: 'sequential' applies strategies in order,
                            'weighted' combines their effects
        """
        super().__init__(base_temperature=base_temperature)
        self.strategies = strategies or []
        self.combination_mode = combination_mode
        
        if not self.strategies:
            # Default: combine temperature and noise strategies with increased diversity
            logger.info(f"[HybridDiversityStrategy] No strategies provided, using default combination "
                       f"with base_temperature={base_temperature}")
            self.strategies = [
                (TemperatureDiversityStrategy(base_temperature=base_temperature, noise_scale=0.2), 0.5),
                (NoiseDiversityStrategy(base_temperature=base_temperature, noise_scale=0.3, continuous=True), 0.5)
            ]
        
        logger.info(f"[HybridDiversityStrategy] Initialized with base_temperature={base_temperature}, "
                   f"{len(self.strategies)} strategies, mode={combination_mode}")
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
        
        Uses baseline temperature to generate temperature series.
        
        Args:
            path_index: Index of current path
            total_paths: Total number of paths
            
        Returns:
            Temperature value (from first temperature strategy found, or base class method)
        """
        for strategy, _ in self.strategies:
            if isinstance(strategy, TemperatureDiversityStrategy):
                return strategy.get_temperature(path_index, total_paths)
        
        # Use base class method which uses baseline temperature
        return super().get_temperature(path_index, total_paths)


def create_diversity_strategy(
    strategy_type: str = 'hybrid',
    base_temperature: float = 0.7,
    **kwargs
) -> DiversityStrategy:
    """Factory function to create diversity strategies.
    
    Args:
        strategy_type: Type of strategy ('temperature', 'noise', 'initialization', 'hybrid')
        base_temperature: Baseline temperature for generating temperature series (default: 0.7)
        **kwargs: Strategy-specific parameters
        
    Returns:
        Diversity strategy instance
    """
    logger.info(f"[DiversityStrategyFactory] Creating strategy: {strategy_type} "
               f"with base_temperature={base_temperature}")
    
    if strategy_type == 'temperature':
        return TemperatureDiversityStrategy(base_temperature=base_temperature, **kwargs)
    elif strategy_type == 'noise':
        return NoiseDiversityStrategy(base_temperature=base_temperature, **kwargs)
    elif strategy_type == 'initialization':
        return InitializationDiversityStrategy(base_temperature=base_temperature, **kwargs)
    elif strategy_type == 'hybrid':
        return HybridDiversityStrategy(base_temperature=base_temperature, **kwargs)
    else:
        logger.warning(f"[DiversityStrategyFactory] Unknown strategy type: {strategy_type}, "
                      f"defaulting to hybrid")
        return HybridDiversityStrategy(base_temperature=base_temperature, **kwargs)

