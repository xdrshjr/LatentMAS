"""Pruning strategies module for multi-path reasoning.

This module implements various strategies to prune low-quality reasoning paths
while maintaining diversity and staying within computational budgets.
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class PruningStatistics:
    """Statistics about a pruning operation.
    
    Attributes:
        num_input_paths: Number of paths before pruning
        num_output_paths: Number of paths after pruning
        num_pruned: Number of paths removed
        pruning_ratio: Ratio of paths removed
        score_threshold: Score threshold used (if applicable)
        avg_score_before: Average score before pruning
        avg_score_after: Average score after pruning
        min_score_kept: Minimum score among kept paths
        max_score_pruned: Maximum score among pruned paths
        metadata: Additional strategy-specific information
    """
    num_input_paths: int
    num_output_paths: int
    num_pruned: int
    pruning_ratio: float
    score_threshold: Optional[float] = None
    avg_score_before: float = 0.0
    avg_score_after: float = 0.0
    min_score_kept: Optional[float] = None
    max_score_pruned: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary.
        
        Returns:
            Dictionary representation of statistics
        """
        return {
            'num_input_paths': self.num_input_paths,
            'num_output_paths': self.num_output_paths,
            'num_pruned': self.num_pruned,
            'pruning_ratio': self.pruning_ratio,
            'score_threshold': self.score_threshold,
            'avg_score_before': self.avg_score_before,
            'avg_score_after': self.avg_score_after,
            'min_score_kept': self.min_score_kept,
            'max_score_pruned': self.max_score_pruned,
            'metadata': self.metadata
        }


class PruningStrategy(ABC):
    """Abstract base class for pruning strategies.
    
    All pruning strategies should inherit from this class and implement
    the prune method.
    """
    
    def __init__(self):
        """Initialize the pruning strategy."""
        self.statistics_history: List[PruningStatistics] = []
        logger.debug(f"[{self.__class__.__name__}] Initialized")
    
    @abstractmethod
    def prune(
        self,
        paths: List[Any],
        **kwargs
    ) -> List[Any]:
        """Prune paths based on strategy-specific criteria.
        
        Args:
            paths: List of PathState objects to prune
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of PathState objects after pruning
        """
        pass
    
    def _compute_statistics(
        self,
        original_paths: List[Any],
        pruned_paths: List[Any],
        **kwargs
    ) -> PruningStatistics:
        """Compute statistics about the pruning operation.
        
        Args:
            original_paths: Paths before pruning
            pruned_paths: Paths after pruning
            **kwargs: Additional metadata
            
        Returns:
            PruningStatistics object
        """
        num_input = len(original_paths)
        num_output = len(pruned_paths)
        num_pruned = num_input - num_output
        
        # Compute score statistics
        if num_input > 0:
            scores_before = [p.score for p in original_paths]
            avg_score_before = np.mean(scores_before)
        else:
            avg_score_before = 0.0
        
        if num_output > 0:
            scores_after = [p.score for p in pruned_paths]
            avg_score_after = np.mean(scores_after)
            min_score_kept = min(scores_after)
        else:
            avg_score_after = 0.0
            min_score_kept = None
        
        # Find max score among pruned paths
        # Compare by path_id to avoid tensor comparison issues
        pruned_path_ids = {p.path_id for p in pruned_paths}
        removed_paths = [p for p in original_paths if p.path_id not in pruned_path_ids]
        if removed_paths:
            max_score_pruned = max(p.score for p in removed_paths)
        else:
            max_score_pruned = None
        
        stats = PruningStatistics(
            num_input_paths=num_input,
            num_output_paths=num_output,
            num_pruned=num_pruned,
            pruning_ratio=num_pruned / num_input if num_input > 0 else 0.0,
            avg_score_before=avg_score_before,
            avg_score_after=avg_score_after,
            min_score_kept=min_score_kept,
            max_score_pruned=max_score_pruned,
            metadata=kwargs
        )
        
        self.statistics_history.append(stats)
        return stats
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of all pruning operations.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.statistics_history:
            return {}
        
        total_pruned = sum(s.num_pruned for s in self.statistics_history)
        total_input = sum(s.num_input_paths for s in self.statistics_history)
        avg_pruning_ratio = np.mean([s.pruning_ratio for s in self.statistics_history])
        
        return {
            'num_operations': len(self.statistics_history),
            'total_paths_pruned': total_pruned,
            'total_paths_processed': total_input,
            'avg_pruning_ratio': avg_pruning_ratio,
            'operations': [s.to_dict() for s in self.statistics_history]
        }


class TopKPruning(PruningStrategy):
    """Prunes paths by keeping only the top-k highest scoring paths.
    
    This is the simplest pruning strategy that keeps a fixed number of
    best-performing paths based on their scores.
    
    Attributes:
        k: Number of paths to keep
        min_paths: Minimum number of paths to keep (overrides k if needed)
    """
    
    def __init__(self, k: int = 5, min_paths: int = 1):
        """Initialize the top-k pruning strategy.
        
        Args:
            k: Number of paths to keep
            min_paths: Minimum number of paths to keep
        """
        super().__init__()
        self.k = k
        self.min_paths = min_paths
        logger.info(f"[TopKPruning] Initialized with k={k}, min_paths={min_paths}")
    
    def prune(
        self,
        paths: List[Any],
        k: Optional[int] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Keep only the top-k paths by score.
        
        Args:
            paths: List of PathState objects to prune
            k: Number of paths to keep (overrides default if provided)
            current_step: Current step (ignored)
            total_steps: Total steps (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of top-k PathState objects
        """
        if not paths:
            logger.warning("[TopKPruning] No paths to prune, returning empty list")
            return []
        
        # Use provided k or default
        keep_count = k if k is not None else self.k
        keep_count = max(keep_count, self.min_paths)
        keep_count = min(keep_count, len(paths))
        
        logger.debug(f"[TopKPruning] Pruning {len(paths)} paths, keeping top {keep_count}")
        
        # Log individual path scores
        for path in paths:
            logger.debug(f"[TopKPruning] Path {path.path_id}: score={path.score:.4f}")
        
        # Sort paths by score (descending) and keep top-k
        sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
        pruned_paths = sorted_paths[:keep_count]
        
        # Compute and log statistics
        stats = self._compute_statistics(
            paths,
            pruned_paths,
            strategy='top_k',
            k=keep_count
        )
        
        logger.info(
            f"[TopKPruning] Pruned {stats.num_pruned} paths "
            f"({stats.num_input_paths} -> {stats.num_output_paths}), "
            f"avg_score: {stats.avg_score_before:.4f} -> {stats.avg_score_after:.4f}"
        )
        
        if stats.max_score_pruned is not None and stats.min_score_kept is not None:
            logger.debug(
                f"[TopKPruning] Score range - kept: [{stats.min_score_kept:.4f}, ...], "
                f"pruned: [..., {stats.max_score_pruned:.4f}]"
            )
        
        # Clean up temporary data structures
        del sorted_paths
        
        return pruned_paths


class ThresholdPruning(PruningStrategy):
    """Prunes paths that fall below a score threshold.
    
    This strategy removes all paths with scores below a specified threshold,
    keeping only paths that meet the quality bar.
    
    Attributes:
        threshold: Score threshold below which paths are pruned
        min_paths: Minimum number of paths to keep (overrides threshold if needed)
    """
    
    def __init__(self, threshold: float = 0.5, min_paths: int = 1):
        """Initialize the threshold pruning strategy.
        
        Args:
            threshold: Score threshold for pruning
            min_paths: Minimum number of paths to keep
        """
        super().__init__()
        self.threshold = threshold
        self.min_paths = min_paths
        logger.info(f"[ThresholdPruning] Initialized with threshold={threshold}, min_paths={min_paths}")
    
    def prune(
        self,
        paths: List[Any],
        threshold: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Remove paths below the score threshold.
        
        Args:
            paths: List of PathState objects to prune
            threshold: Score threshold (overrides default if provided)
            current_step: Current step (ignored)
            total_steps: Total steps (ignored)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of PathState objects above threshold
        """
        if not paths:
            logger.warning("[ThresholdPruning] No paths to prune, returning empty list")
            return []
        
        # Use provided threshold or default
        score_threshold = threshold if threshold is not None else self.threshold
        
        logger.debug(f"[ThresholdPruning] Pruning {len(paths)} paths with threshold={score_threshold:.4f}")
        
        # Log individual path scores
        for path in paths:
            logger.debug(f"[ThresholdPruning] Path {path.path_id}: score={path.score:.4f}")
        
        # Keep paths above threshold
        pruned_paths = [p for p in paths if p.score >= score_threshold]
        
        # Ensure minimum number of paths
        if len(pruned_paths) < self.min_paths and len(paths) > 0:
            logger.debug(
                f"[ThresholdPruning] Only {len(pruned_paths)} paths above threshold, "
                f"keeping top {self.min_paths} to meet minimum"
            )
            sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
            pruned_paths = sorted_paths[:self.min_paths]
        
        # Compute and log statistics
        stats = self._compute_statistics(
            paths,
            pruned_paths,
            strategy='threshold',
            threshold=score_threshold
        )
        stats.score_threshold = score_threshold
        
        logger.info(
            f"[ThresholdPruning] Pruned {stats.num_pruned} paths "
            f"({stats.num_input_paths} -> {stats.num_output_paths}), "
            f"threshold={score_threshold:.4f}, "
            f"avg_score: {stats.avg_score_before:.4f} -> {stats.avg_score_after:.4f}"
        )
        
        # Clean up temporary data structures if sorted_paths was created
        if 'sorted_paths' in locals():
            del sorted_paths
        
        return pruned_paths


class AdaptivePruning(PruningStrategy):
    """Adaptive pruning that adjusts pruning rate based on progress.
    
    This strategy is more aggressive in early steps (to explore broadly)
    and less aggressive in later steps (to exploit best paths).
    
    Attributes:
        min_keep_ratio: Minimum ratio of paths to keep (early steps)
        max_keep_ratio: Maximum ratio of paths to keep (later steps)
        min_paths: Minimum absolute number of paths to keep
        consistency_threshold: Threshold below which paths are considered low-consistency
        prioritize_consistency: Whether to prioritize high-consistency paths
    """
    
    def __init__(
        self,
        min_keep_ratio: float = 0.3,
        max_keep_ratio: float = 0.8,
        min_paths: int = 2,
        consistency_threshold: float = 0.3,
        prioritize_consistency: bool = True
    ):
        """Initialize the adaptive pruning strategy.
        
        Args:
            min_keep_ratio: Minimum ratio of paths to keep (early)
            max_keep_ratio: Maximum ratio of paths to keep (later)
            min_paths: Minimum absolute number of paths to keep
            consistency_threshold: Minimum consistency score to avoid immediate pruning
            prioritize_consistency: Whether to filter low-consistency paths first
        """
        super().__init__()
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.min_paths = min_paths
        self.consistency_threshold = consistency_threshold
        self.prioritize_consistency = prioritize_consistency
        logger.info(
            f"[AdaptivePruning] Initialized with "
            f"keep_ratio=[{min_keep_ratio:.2f}, {max_keep_ratio:.2f}], "
            f"min_paths={min_paths}, "
            f"consistency_threshold={consistency_threshold:.2f}, "
            f"prioritize_consistency={prioritize_consistency}"
        )
    
    def prune(
        self,
        paths: List[Any],
        current_step: int,
        total_steps: int,
        path_diversity: Optional[float] = None,
        force_keep_count: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Prune paths with adaptive rate based on progress.
        
        Implements consistency-aware pruning: filters out low-consistency paths first,
        then sorts by final score among high-consistency paths.
        
        Args:
            paths: List of PathState objects to prune
            current_step: Current step in the reasoning process
            total_steps: Total number of steps
            path_diversity: Optional diversity metric (0-1, higher = more diverse)
            force_keep_count: Force keeping exactly this many paths (overrides adaptive calculation)
            **kwargs: Additional parameters
            
        Returns:
            List of PathState objects after adaptive pruning
        """
        if not paths:
            logger.warning("[AdaptivePruning] No paths to prune, returning empty list")
            return []
        
        # Check if force_keep_count is specified (e.g., for Refiner stage)
        if force_keep_count is not None:
            keep_count = max(1, min(force_keep_count, len(paths)))
            logger.info(f"[AdaptivePruning] Using forced keep_count={keep_count} (overriding adaptive calculation)")
        else:
            # Calculate adaptive keep ratio based on progress
            # Formula: keep_ratio = min_ratio + (max_ratio - min_ratio) * (step / total_steps)
            progress = current_step / max(total_steps, 1)
            keep_ratio = self.min_keep_ratio + (self.max_keep_ratio - self.min_keep_ratio) * progress
            
            # Adjust based on path diversity if provided
            if path_diversity is not None:
                # If diversity is low, prune more aggressively to force exploration
                # If diversity is high, keep more paths
                diversity_adjustment = (path_diversity - 0.5) * 0.2  # ±0.1 adjustment
                keep_ratio = np.clip(keep_ratio + diversity_adjustment, self.min_keep_ratio, self.max_keep_ratio)
                logger.debug(
                    f"[AdaptivePruning] Adjusted keep_ratio by {diversity_adjustment:.3f} "
                    f"based on diversity={path_diversity:.3f}"
                )
            
            # Calculate number of paths to keep
            keep_count = max(self.min_paths, int(len(paths) * keep_ratio))
            keep_count = min(keep_count, len(paths))
            
            logger.debug(
                f"[AdaptivePruning] Step {current_step}/{total_steps} (progress={progress:.2f}), "
                f"keep_ratio={keep_ratio:.3f}, keeping {keep_count}/{len(paths)} paths"
            )
        
        # Log individual path scores and consistency
        for path in paths:
            consistency = path.metadata.get('latent_consistency', None)
            if consistency is not None:
                logger.debug(f"[AdaptivePruning] Path {path.path_id}: score={path.score:.4f}, consistency={consistency:.4f}")
            else:
                logger.debug(f"[AdaptivePruning] Path {path.path_id}: score={path.score:.4f}")
        
        # CONSISTENCY-AWARE PRUNING LOGIC
        # Step 1: Filter out low-consistency paths if prioritize_consistency is enabled
        if self.prioritize_consistency:
            # Separate paths into high and low consistency groups
            high_consistency_paths = []
            low_consistency_paths = []
            
            for path in paths:
                consistency = path.metadata.get('latent_consistency', None)
                if consistency is not None:
                    if consistency >= self.consistency_threshold:
                        high_consistency_paths.append(path)
                        logger.debug(f"[AdaptivePruning] Path {path.path_id}: HIGH consistency={consistency:.4f}, score={path.score:.4f}")
                    else:
                        low_consistency_paths.append(path)
                        logger.debug(f"[AdaptivePruning] Path {path.path_id}: LOW consistency={consistency:.4f}, score={path.score:.4f}")
                else:
                    # If no consistency score, treat as medium consistency
                    high_consistency_paths.append(path)
                    logger.debug(f"[AdaptivePruning] Path {path.path_id}: NO consistency score, treating as medium, score={path.score:.4f}")
            
            logger.info(f"[AdaptivePruning] Consistency filtering: {len(high_consistency_paths)} high-consistency "
                       f"(>={self.consistency_threshold:.2f}), {len(low_consistency_paths)} low-consistency paths")
            
            # Log statistics for each group
            if high_consistency_paths:
                high_cons_scores = [p.metadata.get('latent_consistency', 0) for p in high_consistency_paths]
                high_path_scores = [p.score for p in high_consistency_paths]
                logger.info(f"[AdaptivePruning] High-consistency group: "
                           f"consistency=[{min(high_cons_scores):.4f}, {max(high_cons_scores):.4f}], "
                           f"scores=[{min(high_path_scores):.4f}, {max(high_path_scores):.4f}]")
            
            if low_consistency_paths:
                low_cons_scores = [p.metadata.get('latent_consistency', 0) for p in low_consistency_paths]
                low_path_scores = [p.score for p in low_consistency_paths]
                logger.info(f"[AdaptivePruning] Low-consistency group: "
                           f"consistency=[{min(low_cons_scores):.4f}, {max(low_cons_scores):.4f}], "
                           f"scores=[{min(low_path_scores):.4f}, {max(low_path_scores):.4f}]")
            
            # Step 2: Prioritize high-consistency paths
            # If we have enough high-consistency paths, only consider those
            if len(high_consistency_paths) >= keep_count:
                logger.info(f"[AdaptivePruning] Sufficient high-consistency paths ({len(high_consistency_paths)} >= {keep_count}), "
                           f"pruning ALL {len(low_consistency_paths)} low-consistency paths")
                if low_consistency_paths:
                    pruned_ids = [p.path_id for p in low_consistency_paths]
                    logger.info(f"[AdaptivePruning] Pruned low-consistency path IDs: {pruned_ids}")
                candidate_paths = high_consistency_paths
            else:
                # Not enough high-consistency paths, need to include some low-consistency ones
                logger.warning(f"[AdaptivePruning] Insufficient high-consistency paths ({len(high_consistency_paths)} < {keep_count}), "
                              f"will include some low-consistency paths to meet keep_count requirement")
                candidate_paths = paths
        else:
            # No consistency filtering
            candidate_paths = paths
            logger.debug("[AdaptivePruning] Consistency prioritization disabled, using all paths")
        
        # Step 3: Sort by final score and keep top-k
        sorted_paths = sorted(candidate_paths, key=lambda p: p.score, reverse=True)
        pruned_paths = sorted_paths[:keep_count]
        
        # Log details of kept paths
        logger.info(f"[AdaptivePruning] Kept paths:")
        for path in pruned_paths:
            consistency = path.metadata.get('latent_consistency', None)
            if consistency is not None:
                logger.info(f"  - Path {path.path_id}: score={path.score:.4f}, consistency={consistency:.4f}")
            else:
                logger.info(f"  - Path {path.path_id}: score={path.score:.4f}")
        
        # Compute and log statistics
        stats = self._compute_statistics(
            paths,
            pruned_paths,
            strategy='adaptive_consistency_aware' if self.prioritize_consistency else 'adaptive',
            current_step=current_step,
            total_steps=total_steps,
            progress=progress if force_keep_count is None else None,
            keep_ratio=keep_ratio if force_keep_count is None else None,
            path_diversity=path_diversity,
            forced_count=force_keep_count
        )
        
        logger.info(
            f"[AdaptivePruning] Pruned {stats.num_pruned} paths ({stats.num_input_paths} -> {stats.num_output_paths}), "
            f"avg_score: {stats.avg_score_before:.4f} -> {stats.avg_score_after:.4f}"
        )
        
        # Clean up temporary data structures
        del sorted_paths
        
        return pruned_paths


class DiversityAwarePruning(PruningStrategy):
    """Pruning that balances score and diversity.
    
    This strategy keeps the highest-scoring path and then selects remaining
    paths to maximize both score and diversity from already-selected paths.
    
    Attributes:
        target_count: Target number of paths to keep
        min_diversity_threshold: Minimum cosine distance between kept paths
        score_weight: Weight for score vs diversity (0-1, higher = prioritize score)
        min_paths: Minimum number of paths to keep
    """
    
    def __init__(
        self,
        target_count: int = 5,
        min_diversity_threshold: float = 0.1,
        score_weight: float = 0.6,
        min_paths: int = 2
    ):
        """Initialize the diversity-aware pruning strategy.
        
        Args:
            target_count: Target number of paths to keep
            min_diversity_threshold: Minimum diversity (1 - cosine_sim) required
            score_weight: Weight for score (1 - score_weight = diversity weight)
            min_paths: Minimum number of paths to keep
        """
        super().__init__()
        self.target_count = target_count
        self.min_diversity_threshold = min_diversity_threshold
        self.score_weight = score_weight
        self.diversity_weight = 1.0 - score_weight
        self.min_paths = min_paths
        logger.info(
            f"[DiversityAwarePruning] Initialized with "
            f"target_count={target_count}, "
            f"min_diversity={min_diversity_threshold:.3f}, "
            f"score_weight={score_weight:.2f}"
        )
    
    def prune(
        self,
        paths: List[Any],
        target_count: Optional[int] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Prune paths while preserving diversity.
        
        Args:
            paths: List of PathState objects to prune
            target_count: Target number of paths (overrides default if provided)
            current_step: Current step (ignored)
            total_steps: Total steps (ignored)
            **kwargs: Additional parameters
            
        Returns:
            List of diverse, high-scoring PathState objects
        """
        if not paths:
            logger.warning("[DiversityAwarePruning] No paths to prune, returning empty list")
            return []
        
        # Use provided target or default
        keep_count = target_count if target_count is not None else self.target_count
        keep_count = max(keep_count, self.min_paths)
        keep_count = min(keep_count, len(paths))
        
        logger.debug(
            f"[DiversityAwarePruning] Pruning {len(paths)} paths, "
            f"target={keep_count}, score_weight={self.score_weight:.2f}"
        )
        
        # Sort paths by score
        sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
        
        # Always keep the highest-scoring path
        selected_paths = [sorted_paths[0]]
        remaining_paths = sorted_paths[1:]
        
        logger.debug(
            f"[DiversityAwarePruning] Selected best path {sorted_paths[0].path_id} "
            f"with score={sorted_paths[0].score:.4f}"
        )
        
        # Select remaining paths balancing score and diversity
        while len(selected_paths) < keep_count and remaining_paths:
            best_candidate = None
            best_combined_score = -float('inf')
            
            for candidate in remaining_paths:
                # Calculate diversity from selected paths
                min_diversity = self._compute_min_diversity(candidate, selected_paths)
                
                # Normalize score to [0, 1] range for fair combination
                normalized_score = candidate.score
                
                # Combined score: weighted sum of score and diversity
                combined_score = (
                    self.score_weight * normalized_score +
                    self.diversity_weight * min_diversity
                )
                
                logger.debug(
                    f"[DiversityAwarePruning] Path {candidate.path_id}: "
                    f"score={normalized_score:.4f}, "
                    f"min_diversity={min_diversity:.4f}, "
                    f"combined={combined_score:.4f}"
                )
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_paths.append(best_candidate)
                remaining_paths.remove(best_candidate)
                
                # Get diversity of selected path
                min_div = self._compute_min_diversity(best_candidate, selected_paths[:-1])
                logger.debug(
                    f"[DiversityAwarePruning] Selected path {best_candidate.path_id} "
                    f"with score={best_candidate.score:.4f}, "
                    f"min_diversity={min_div:.4f}"
                )
            else:
                break
        
        # Compute pairwise diversity statistics
        avg_diversity = self._compute_average_pairwise_diversity(selected_paths)
        
        # Compute and log statistics
        stats = self._compute_statistics(
            paths,
            selected_paths,
            strategy='diversity_aware',
            target_count=keep_count,
            avg_pairwise_diversity=avg_diversity,
            score_weight=self.score_weight
        )
        
        logger.info(
            f"[DiversityAwarePruning] Pruned {stats.num_pruned} paths "
            f"({stats.num_input_paths} -> {stats.num_output_paths}), "
            f"avg_diversity={avg_diversity:.4f}, "
            f"avg_score: {stats.avg_score_before:.4f} -> {stats.avg_score_after:.4f}"
        )
        
        # Clean up temporary data structures
        del sorted_paths, remaining_paths
        
        # Clean up any GPU tensors created during diversity calculations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[DiversityAwarePruning] GPU cache cleaned after diversity-aware pruning")
        
        return selected_paths
    
    def _compute_min_diversity(
        self,
        candidate: Any,
        selected_paths: List[Any]
    ) -> float:
        """Compute minimum diversity between candidate and selected paths.
        
        Args:
            candidate: Candidate PathState
            selected_paths: List of already selected PathState objects
            
        Returns:
            Minimum diversity (1 - cosine_similarity) in range [0, 1]
        """
        if not selected_paths:
            return 1.0
        
        # Get hidden states for comparison
        if candidate.hidden_states is None:
            logger.debug(f"[DiversityAwarePruning] Path {candidate.path_id} has no hidden states")
            return 0.5  # Default diversity
        
        min_diversity = 1.0
        
        for selected in selected_paths:
            if selected.hidden_states is None:
                continue
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(
                candidate.hidden_states,
                selected.hidden_states
            )
            
            # Diversity = 1 - similarity
            diversity = 1.0 - similarity
            min_diversity = min(min_diversity, diversity)
            
            logger.debug(
                f"[DiversityAwarePruning] Diversity between path {candidate.path_id} "
                f"and {selected.path_id}: {diversity:.4f}"
            )
        
        # Note: We don't delete tensors here as they're references from PathState objects
        # They will be cleaned up when paths are removed from PathManager
        
        return max(0.0, min_diversity)
    
    def _compute_average_pairwise_diversity(self, paths: List[Any]) -> float:
        """Compute average pairwise diversity among paths.
        
        Args:
            paths: List of PathState objects
            
        Returns:
            Average pairwise diversity
        """
        if len(paths) < 2:
            return 1.0
        
        diversities = []
        for i, path_i in enumerate(paths):
            for path_j in paths[i+1:]:
                if path_i.hidden_states is not None and path_j.hidden_states is not None:
                    similarity = self._cosine_similarity(
                        path_i.hidden_states,
                        path_j.hidden_states
                    )
                    diversity = 1.0 - similarity
                    diversities.append(diversity)
        
        if not diversities:
            return 0.5
        
        return float(np.mean(diversities))
    
    def _cosine_similarity(
        self,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two hidden state tensors.
        
        Args:
            hidden_states_1: First hidden states tensor
            hidden_states_2: Second hidden states tensor
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Flatten tensors if needed
        vec1 = hidden_states_1.flatten()
        vec2 = hidden_states_2.flatten()
        
        # Handle dimension mismatch
        if vec1.shape[0] != vec2.shape[0]:
            logger.debug(
                f"[DiversityAwarePruning] Dimension mismatch: "
                f"{vec1.shape[0]} vs {vec2.shape[0]}, using default similarity"
            )
            return 0.5
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            vec1.unsqueeze(0),
            vec2.unsqueeze(0),
            dim=1
        )
        
        result = float(similarity.item())
        
        # Clean up temporary tensors
        del vec1, vec2, similarity
        
        return result


class BudgetBasedPruning(PruningStrategy):
    """Pruning based on computational budget constraints.
    
    This strategy tracks computational cost and prunes paths to stay within
    budget, considering cost-benefit ratio (score per compute).
    
    Attributes:
        max_budget: Maximum computational budget (in tokens or FLOPs)
        cost_metric: Type of cost to track ('tokens', 'flops', 'memory')
        min_paths: Minimum number of paths to keep
        prioritize_efficiency: Whether to prioritize cost-efficiency over raw score
    """
    
    def __init__(
        self,
        max_budget: float = 100000,
        cost_metric: str = 'tokens',
        min_paths: int = 1,
        prioritize_efficiency: bool = True
    ):
        """Initialize the budget-based pruning strategy.
        
        Args:
            max_budget: Maximum computational budget
            cost_metric: Type of cost to track
            min_paths: Minimum number of paths to keep
            prioritize_efficiency: Whether to prioritize efficiency
        """
        super().__init__()
        self.max_budget = max_budget
        self.cost_metric = cost_metric
        self.min_paths = min_paths
        self.prioritize_efficiency = prioritize_efficiency
        self.current_budget_used = 0.0
        logger.info(
            f"[BudgetBasedPruning] Initialized with "
            f"max_budget={max_budget}, "
            f"cost_metric={cost_metric}, "
            f"prioritize_efficiency={prioritize_efficiency}"
        )
    
    def prune(
        self,
        paths: List[Any],
        current_budget_used: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Prune paths to stay within computational budget.
        
        Args:
            paths: List of PathState objects to prune
            current_budget_used: Current budget used (updates internal tracker)
            current_step: Current step (ignored)
            total_steps: Total steps (ignored)
            **kwargs: Additional parameters
            
        Returns:
            List of PathState objects within budget
        """
        if not paths:
            logger.warning("[BudgetBasedPruning] No paths to prune, returning empty list")
            return []
        
        # Update budget tracker if provided
        if current_budget_used is not None:
            self.current_budget_used = current_budget_used
        
        # Calculate remaining budget
        remaining_budget = self.max_budget - self.current_budget_used
        budget_usage_ratio = self.current_budget_used / self.max_budget
        
        logger.debug(
            f"[BudgetBasedPruning] Budget status: "
            f"{self.current_budget_used:.0f}/{self.max_budget:.0f} "
            f"({budget_usage_ratio:.1%} used), "
            f"remaining={remaining_budget:.0f}"
        )
        
        # Estimate cost for each path
        path_costs = []
        for path in paths:
            cost = self._estimate_path_cost(path)
            efficiency = path.score / max(cost, 1.0)  # Score per unit cost
            path_costs.append({
                'path': path,
                'cost': cost,
                'efficiency': efficiency
            })
            logger.debug(
                f"[BudgetBasedPruning] Path {path.path_id}: "
                f"score={path.score:.4f}, "
                f"cost={cost:.0f}, "
                f"efficiency={efficiency:.6f}"
            )
        
        # Sort by efficiency or score
        if self.prioritize_efficiency:
            path_costs.sort(key=lambda x: x['efficiency'], reverse=True)
            logger.debug("[BudgetBasedPruning] Sorting by cost-efficiency")
        else:
            path_costs.sort(key=lambda x: x['path'].score, reverse=True)
            logger.debug("[BudgetBasedPruning] Sorting by raw score")
        
        # Select paths within budget
        selected_paths = []
        total_cost = 0.0
        
        for item in path_costs:
            path = item['path']
            cost = item['cost']
            
            # Always keep minimum number of paths
            if len(selected_paths) < self.min_paths:
                selected_paths.append(path)
                total_cost += cost
                logger.debug(
                    f"[BudgetBasedPruning] Keeping path {path.path_id} "
                    f"(min_paths requirement), cost={cost:.0f}"
                )
                continue
            
            # Check if we can afford this path
            if total_cost + cost <= remaining_budget:
                selected_paths.append(path)
                total_cost += cost
                logger.debug(
                    f"[BudgetBasedPruning] Selected path {path.path_id}, "
                    f"cost={cost:.0f}, total_cost={total_cost:.0f}"
                )
            else:
                logger.debug(
                    f"[BudgetBasedPruning] Skipped path {path.path_id}, "
                    f"would exceed budget (cost={cost:.0f})"
                )
        
        # Update budget tracker
        self.current_budget_used += total_cost
        
        # Compute and log statistics
        stats = self._compute_statistics(
            paths,
            selected_paths,
            strategy='budget_based',
            budget_used=self.current_budget_used,
            max_budget=self.max_budget,
            remaining_budget=self.max_budget - self.current_budget_used,
            total_cost=total_cost,
            cost_metric=self.cost_metric
        )
        
        logger.info(
            f"[BudgetBasedPruning] Pruned {stats.num_pruned} paths "
            f"({stats.num_input_paths} -> {stats.num_output_paths}), "
            f"cost={total_cost:.0f}, "
            f"budget={self.current_budget_used:.0f}/{self.max_budget:.0f} "
            f"({self.current_budget_used/self.max_budget:.1%})"
        )
        
        # Clean up temporary data structures
        del path_costs
        
        return selected_paths
    
    def _estimate_path_cost(self, path: Any) -> float:
        """Estimate computational cost for a path.
        
        Args:
            path: PathState object
            
        Returns:
            Estimated cost
        """
        # Check if cost is stored in metadata
        if 'cost' in path.metadata:
            return path.metadata['cost']
        
        # Estimate based on cost metric
        if self.cost_metric == 'tokens':
            # Estimate tokens from latent history length
            # Assume each latent step processes ~100 tokens
            return len(path.latent_history) * 100
        
        elif self.cost_metric == 'flops':
            # Rough FLOP estimate based on hidden state size and steps
            if path.hidden_states is not None:
                hidden_size = path.hidden_states.numel()
                # Approximate FLOPs per step
                return len(path.latent_history) * hidden_size * 2
            return len(path.latent_history) * 1000000  # Default estimate
        
        elif self.cost_metric == 'memory':
            # Estimate memory usage
            memory = 0
            if path.hidden_states is not None:
                memory += path.hidden_states.numel() * 4  # 4 bytes per float32
            for latent in path.latent_history:
                memory += latent.numel() * 4
            return memory
        
        else:
            # Default: use path length as proxy
            return len(path.latent_history)
    
    def reset_budget(self):
        """Reset the budget tracker."""
        self.current_budget_used = 0.0
        logger.info("[BudgetBasedPruning] Budget tracker reset")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status.
        
        Returns:
            Dictionary with budget information
        """
        return {
            'current_budget_used': self.current_budget_used,
            'max_budget': self.max_budget,
            'remaining_budget': self.max_budget - self.current_budget_used,
            'usage_ratio': self.current_budget_used / self.max_budget,
            'cost_metric': self.cost_metric
        }

