"""Path merging module for multi-path reasoning.

This module implements strategies to detect and merge similar reasoning paths
to reduce redundancy and computational costs while preserving reasoning quality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class MergeCandidate:
    """Represents a candidate for path merging.
    
    Attributes:
        path_ids: List of path IDs that are similar
        similarity_scores: Pairwise similarity scores
        avg_similarity: Average similarity across all pairs
        merge_priority: Priority score for merging (higher = merge sooner)
        metadata: Additional information about this merge candidate
    """
    path_ids: List[int]
    similarity_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    avg_similarity: float = 0.0
    merge_priority: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the merge candidate
        """
        return {
            'path_ids': self.path_ids,
            'avg_similarity': self.avg_similarity,
            'merge_priority': self.merge_priority,
            'num_paths': len(self.path_ids),
            'metadata': self.metadata
        }


@dataclass
class MergeStatistics:
    """Statistics about a merge operation.
    
    Attributes:
        num_input_paths: Number of paths before merging
        num_output_paths: Number of paths after merging
        num_merged: Number of paths that were merged
        merge_groups: Number of merge groups created
        avg_similarity: Average similarity of merged paths
        score_before_merge: Average score before merging
        score_after_merge: Average score after merging
        metadata: Additional merge-specific information
    """
    num_input_paths: int
    num_output_paths: int
    num_merged: int
    merge_groups: int
    avg_similarity: float = 0.0
    score_before_merge: float = 0.0
    score_after_merge: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary.
        
        Returns:
            Dictionary representation of statistics
        """
        return {
            'num_input_paths': self.num_input_paths,
            'num_output_paths': self.num_output_paths,
            'num_merged': self.num_merged,
            'merge_groups': self.merge_groups,
            'avg_similarity': self.avg_similarity,
            'score_before_merge': self.score_before_merge,
            'score_after_merge': self.score_after_merge,
            'metadata': self.metadata
        }


class PathSimilarityDetector:
    """Detects similar paths for potential merging.
    
    This class implements various similarity metrics to identify paths
    that are similar enough to be merged without losing information.
    
    Attributes:
        cosine_threshold: Threshold for cosine similarity (default: 0.9)
        kl_threshold: Threshold for KL divergence (default: 0.1)
        convergence_window: Number of recent steps to check for convergence
    """
    
    def __init__(
        self,
        cosine_threshold: float = 0.9,
        kl_threshold: float = 0.1,
        convergence_window: int = 3
    ):
        """Initialize the similarity detector.
        
        Args:
            cosine_threshold: Minimum cosine similarity for merge candidates
            kl_threshold: Maximum KL divergence for merge candidates
            convergence_window: Number of recent steps to check for convergence
        """
        self.cosine_threshold = cosine_threshold
        self.kl_threshold = kl_threshold
        self.convergence_window = convergence_window
        logger.info(f"[PathSimilarityDetector] Initialized with cosine_threshold={cosine_threshold}, "
                   f"kl_threshold={kl_threshold}, convergence_window={convergence_window}")
    
    def compute_cosine_similarity(
        self,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two hidden state tensors.
        
        Args:
            hidden_states1: First hidden states tensor
            hidden_states2: Second hidden states tensor
            
        Returns:
            Cosine similarity score in [-1, 1]
        """
        try:
            # Flatten tensors for comparison
            h1_flat = hidden_states1.flatten()
            h2_flat = hidden_states2.flatten()
            
            # Ensure same shape
            if h1_flat.shape != h2_flat.shape:
                logger.debug(f"[PathSimilarityDetector] Shape mismatch in cosine similarity: "
                           f"{h1_flat.shape} vs {h2_flat.shape}")
                return 0.0
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                h1_flat.unsqueeze(0),
                h2_flat.unsqueeze(0),
                dim=1
            ).item()
            
            logger.debug(f"[PathSimilarityDetector] Cosine similarity: {similarity:.4f}")
            
            # Clean up temporary tensors
            del h1_flat, h2_flat
            
            return similarity
            
        except Exception as e:
            logger.warning(f"[PathSimilarityDetector] Error computing cosine similarity: {e}")
            return 0.0
    
    def compute_kl_divergence(
        self,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        model_lm_head: Optional[torch.nn.Module] = None
    ) -> float:
        """Compute KL divergence between distributions of two hidden states.
        
        Args:
            hidden_states1: First hidden states tensor
            hidden_states2: Second hidden states tensor
            model_lm_head: Language model head to convert hidden states to logits
            
        Returns:
            KL divergence score (lower = more similar)
        """
        try:
            if model_lm_head is None:
                # If no LM head provided, use distribution of hidden state values
                h1_flat = hidden_states1.flatten()
                h2_flat = hidden_states2.flatten()
                
                # Convert to probability distributions (softmax)
                p = F.softmax(h1_flat, dim=0)
                q = F.softmax(h2_flat, dim=0)
            else:
                # Use LM head to get token distributions
                logits1 = model_lm_head(hidden_states1)
                logits2 = model_lm_head(hidden_states2)
                
                # Take last token position
                p = F.softmax(logits1[:, -1, :].flatten(), dim=0)
                q = F.softmax(logits2[:, -1, :].flatten(), dim=0)
            
            # Compute KL divergence: KL(P || Q)
            kl_div = F.kl_div(
                q.log(),
                p,
                reduction='batchmean',
                log_target=False
            ).item()
            
            logger.debug(f"[PathSimilarityDetector] KL divergence: {kl_div:.4f}")
            
            # Clean up temporary tensors
            if model_lm_head is None:
                del h1_flat, h2_flat, p, q
            else:
                del logits1, logits2, p, q
            
            return kl_div
            
        except Exception as e:
            logger.warning(f"[PathSimilarityDetector] Error computing KL divergence: {e}")
            return float('inf')
    
    def check_convergence(
        self,
        path1_history: List[torch.Tensor],
        path2_history: List[torch.Tensor]
    ) -> Tuple[bool, float]:
        """Check if two paths are converging (becoming more similar over time).
        
        Args:
            path1_history: List of hidden states for path 1
            path2_history: List of hidden states for path 2
            
        Returns:
            Tuple of (is_converging, convergence_score)
        """
        try:
            # Need at least convergence_window steps
            if len(path1_history) < self.convergence_window or len(path2_history) < self.convergence_window:
                logger.debug("[PathSimilarityDetector] Not enough history for convergence check")
                return False, 0.0
            
            # Get recent steps
            recent_steps = min(self.convergence_window, len(path1_history), len(path2_history))
            path1_recent = path1_history[-recent_steps:]
            path2_recent = path2_history[-recent_steps:]
            
            # Compute similarity for each step
            similarities = []
            for h1, h2 in zip(path1_recent, path2_recent):
                sim = self.compute_cosine_similarity(h1, h2)
                similarities.append(sim)
            
            # Check if similarity is increasing (converging)
            if len(similarities) >= 2:
                # Compare first half to second half
                mid = len(similarities) // 2
                first_half_avg = np.mean(similarities[:mid])
                second_half_avg = np.mean(similarities[mid:])
                
                is_converging = second_half_avg > first_half_avg
                convergence_score = second_half_avg - first_half_avg
                
                logger.debug(f"[PathSimilarityDetector] Convergence check: is_converging={is_converging}, "
                           f"score={convergence_score:.4f}")
                return is_converging, convergence_score
            
            return False, 0.0
            
        except Exception as e:
            logger.warning(f"[PathSimilarityDetector] Error checking convergence: {e}")
            return False, 0.0
    
    def are_paths_similar(
        self,
        path1: Any,
        path2: Any,
        model_lm_head: Optional[torch.nn.Module] = None,
        use_kl: bool = False
    ) -> Tuple[bool, float]:
        """Check if two paths are similar enough to be merged.
        
        Args:
            path1: First PathState object
            path2: Second PathState object
            model_lm_head: Optional LM head for KL divergence
            use_kl: Whether to use KL divergence in addition to cosine similarity
            
        Returns:
            Tuple of (are_similar, similarity_score)
        """
        # Check if both paths have hidden states
        if path1.hidden_states is None or path2.hidden_states is None:
            logger.debug(f"[PathSimilarityDetector] Cannot compare paths {path1.path_id} and {path2.path_id}: "
                        "missing hidden states")
            return False, 0.0
        
        # Compute cosine similarity
        cosine_sim = self.compute_cosine_similarity(path1.hidden_states, path2.hidden_states)
        
        # Check cosine threshold
        if cosine_sim < self.cosine_threshold:
            logger.debug(f"[PathSimilarityDetector] Paths {path1.path_id} and {path2.path_id} not similar: "
                        f"cosine={cosine_sim:.4f} < threshold={self.cosine_threshold}")
            return False, cosine_sim
        
        # Optionally check KL divergence
        if use_kl and model_lm_head is not None:
            kl_div = self.compute_kl_divergence(path1.hidden_states, path2.hidden_states, model_lm_head)
            if kl_div > self.kl_threshold:
                logger.debug(f"[PathSimilarityDetector] Paths {path1.path_id} and {path2.path_id} not similar: "
                            f"kl_div={kl_div:.4f} > threshold={self.kl_threshold}")
                return False, cosine_sim
        
        logger.debug(f"[PathSimilarityDetector] Paths {path1.path_id} and {path2.path_id} are similar: "
                    f"cosine={cosine_sim:.4f}")
        return True, cosine_sim
    
    def find_merge_candidates(
        self,
        paths: List[Any],
        model_lm_head: Optional[torch.nn.Module] = None,
        use_kl: bool = False,
        min_group_size: int = 2
    ) -> List[MergeCandidate]:
        """Find groups of similar paths that can be merged.
        
        Args:
            paths: List of PathState objects
            model_lm_head: Optional LM head for KL divergence
            use_kl: Whether to use KL divergence
            min_group_size: Minimum number of paths in a merge group
            
        Returns:
            List of MergeCandidate objects
        """
        logger.info(f"[PathSimilarityDetector] Finding merge candidates among {len(paths)} paths")
        
        if len(paths) < min_group_size:
            logger.debug("[PathSimilarityDetector] Not enough paths for merging")
            return []
        
        # Build similarity matrix
        n = len(paths)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                are_similar, sim_score = self.are_paths_similar(
                    paths[i], paths[j], model_lm_head, use_kl
                )
                similarity_matrix[i, j] = sim_score
                similarity_matrix[j, i] = sim_score
        
        # Find connected components (groups of similar paths)
        visited = set()
        merge_candidates = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find all paths similar to path i
            similar_group = [i]
            for j in range(n):
                if i != j and j not in visited and similarity_matrix[i, j] >= self.cosine_threshold:
                    similar_group.append(j)
            
            # If group is large enough, create merge candidate
            if len(similar_group) >= min_group_size:
                # Mark all as visited
                visited.update(similar_group)
                
                # Compute pairwise similarities
                pairwise_sims = {}
                for idx1 in similar_group:
                    for idx2 in similar_group:
                        if idx1 < idx2:
                            pairwise_sims[(paths[idx1].path_id, paths[idx2].path_id)] = \
                                similarity_matrix[idx1, idx2]
                
                # Compute average similarity
                avg_sim = np.mean([similarity_matrix[idx1, idx2] 
                                  for idx1 in similar_group 
                                  for idx2 in similar_group 
                                  if idx1 < idx2])
                
                # Compute merge priority (higher similarity + more paths = higher priority)
                merge_priority = avg_sim * len(similar_group)
                
                candidate = MergeCandidate(
                    path_ids=[paths[idx].path_id for idx in similar_group],
                    similarity_scores=pairwise_sims,
                    avg_similarity=float(avg_sim),
                    merge_priority=float(merge_priority),
                    metadata={
                        'num_paths': len(similar_group),
                        'avg_score': np.mean([paths[idx].score for idx in similar_group])
                    }
                )
                merge_candidates.append(candidate)
                
                logger.info(f"[PathSimilarityDetector] Found merge candidate: "
                          f"{len(similar_group)} paths with avg_similarity={avg_sim:.4f}")
        
        # Sort by merge priority (descending)
        merge_candidates.sort(key=lambda c: c.merge_priority, reverse=True)
        
        logger.info(f"[PathSimilarityDetector] Found {len(merge_candidates)} merge candidates")
        
        if merge_candidates:
            for idx, candidate in enumerate(merge_candidates, 1):
                logger.debug(f"[PathSimilarityDetector] Candidate {idx}: "
                           f"{len(candidate.path_ids)} paths {candidate.path_ids}, "
                           f"avg_similarity={candidate.avg_similarity:.4f}, "
                           f"priority={candidate.merge_priority:.4f}")
        else:
            logger.debug(f"[PathSimilarityDetector] No similar path groups found "
                        f"(similarity threshold={self.cosine_threshold})")
        
        # Explicitly delete similarity matrix to free memory
        del similarity_matrix, visited
        logger.debug(f"[PathSimilarityDetector] Cleaned up similarity matrix and visited set")
        
        # Force GPU memory cleanup (in case GPU tensors were used in comparisons)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[PathSimilarityDetector] GPU cache cleaned after similarity detection")
        
        return merge_candidates


class MergeStrategy(ABC):
    """Abstract base class for path merging strategies.
    
    All merge strategies should inherit from this class and implement
    the merge method.
    """
    
    def __init__(self):
        """Initialize the merge strategy."""
        self.statistics_history: List[MergeStatistics] = []
        logger.debug(f"[{self.__class__.__name__}] Initialized")
    
    @abstractmethod
    def merge(
        self,
        paths: List[Any],
        **kwargs
    ) -> Any:
        """Merge multiple paths into a single path.
        
        Args:
            paths: List of PathState objects to merge
            **kwargs: Strategy-specific parameters
            
        Returns:
            Merged PathState object
        """
        pass
    
    def _compute_statistics(
        self,
        original_paths: List[Any],
        merged_paths: List[Any],
        merge_groups: int,
        avg_similarity: float = 0.0
    ) -> MergeStatistics:
        """Compute statistics about the merge operation.
        
        Args:
            original_paths: Paths before merging
            merged_paths: Paths after merging
            merge_groups: Number of merge groups
            avg_similarity: Average similarity of merged paths
            
        Returns:
            MergeStatistics object
        """
        num_input = len(original_paths)
        num_output = len(merged_paths)
        num_merged = num_input - num_output
        
        score_before = np.mean([p.score for p in original_paths]) if original_paths else 0.0
        score_after = np.mean([p.score for p in merged_paths]) if merged_paths else 0.0
        
        stats = MergeStatistics(
            num_input_paths=num_input,
            num_output_paths=num_output,
            num_merged=num_merged,
            merge_groups=merge_groups,
            avg_similarity=avg_similarity,
            score_before_merge=float(score_before),
            score_after_merge=float(score_after)
        )
        
        self.statistics_history.append(stats)
        return stats
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of all merge operations.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.statistics_history:
            return {'num_operations': 0}
        
        return {
            'num_operations': len(self.statistics_history),
            'total_merged': sum(s.num_merged for s in self.statistics_history),
            'avg_merge_groups': np.mean([s.merge_groups for s in self.statistics_history]),
            'avg_similarity': np.mean([s.avg_similarity for s in self.statistics_history]),
            'avg_score_change': np.mean([s.score_after_merge - s.score_before_merge 
                                        for s in self.statistics_history])
        }


class AverageMergeStrategy(MergeStrategy):
    """Simple averaging merge strategy.
    
    Merges paths by averaging their hidden states and other properties.
    """
    
    def merge(
        self,
        paths: List[Any],
        **kwargs
    ) -> Any:
        """Merge paths using simple averaging.
        
        Args:
            paths: List of PathState objects to merge
            **kwargs: Additional parameters (unused)
            
        Returns:
            Merged PathState object
        """
        if not paths:
            logger.warning("[AverageMergeStrategy] Cannot merge empty path list")
            return None
        
        if len(paths) == 1:
            logger.debug("[AverageMergeStrategy] Only one path, returning as-is")
            return paths[0]
        
        logger.info(f"[AverageMergeStrategy] Merging {len(paths)} paths using simple averaging")
        
        # Import PathState here to avoid circular dependency
        from .path_manager import PathState
        
        # Create new merged path
        merged_path = PathState(
            path_id=paths[0].path_id,  # Will be reassigned by path manager
            latent_history=paths[0].latent_history.copy(),
            score=np.mean([p.score for p in paths]),
            metadata={
                'merged_from': [p.path_id for p in paths],
                'merge_strategy': 'average',
                'num_merged': len(paths)
            }
        )
        
        # Average hidden states if available
        if all(p.hidden_states is not None for p in paths):
            hidden_states_list = [p.hidden_states for p in paths]
            merged_path.hidden_states = torch.stack(hidden_states_list).mean(dim=0)
            logger.debug("[AverageMergeStrategy] Averaged hidden states")
        else:
            # Use hidden states from best-scoring path
            best_path = max(paths, key=lambda p: p.score)
            merged_path.hidden_states = best_path.hidden_states
            logger.debug("[AverageMergeStrategy] Using hidden states from best path")
        
        # Use KV cache from best-scoring path (cannot average KV caches easily)
        best_path = max(paths, key=lambda p: p.score)
        merged_path.kv_cache = best_path.kv_cache
        
        logger.info(f"[AverageMergeStrategy] Merged {len(paths)} paths into one with score={merged_path.score:.4f}")
        return merged_path


class WeightedMergeStrategy(MergeStrategy):
    """Score-weighted merge strategy.
    
    Merges paths by computing weighted averages based on their quality scores.
    Higher-scoring paths have more influence on the merged result.
    """
    
    def merge(
        self,
        paths: List[Any],
        **kwargs
    ) -> Any:
        """Merge paths using score-weighted averaging.
        
        Args:
            paths: List of PathState objects to merge
            **kwargs: Additional parameters (unused)
            
        Returns:
            Merged PathState object
        """
        if not paths:
            logger.warning("[WeightedMergeStrategy] Cannot merge empty path list")
            return None
        
        if len(paths) == 1:
            logger.debug("[WeightedMergeStrategy] Only one path, returning as-is")
            return paths[0]
        
        logger.info(f"[WeightedMergeStrategy] Merging {len(paths)} paths using weighted averaging")
        
        # Import PathState here to avoid circular dependency
        from .path_manager import PathState
        
        # Compute weights based on scores
        scores = np.array([p.score for p in paths])
        total_score = scores.sum()
        
        if total_score > 0:
            weights = scores / total_score
        else:
            # If all scores are 0, use uniform weights
            weights = np.ones(len(paths)) / len(paths)
        
        logger.debug(f"[WeightedMergeStrategy] Computed weights: {weights}")
        
        # Create new merged path
        merged_score = float(np.sum(weights * scores))
        
        merged_path = PathState(
            path_id=paths[0].path_id,  # Will be reassigned by path manager
            latent_history=paths[0].latent_history.copy(),
            score=merged_score,
            metadata={
                'merged_from': [p.path_id for p in paths],
                'merge_strategy': 'weighted',
                'num_merged': len(paths),
                'weights': weights.tolist()
            }
        )
        
        # Weighted average of hidden states if available
        if all(p.hidden_states is not None for p in paths):
            weighted_hidden = None
            for i, path in enumerate(paths):
                if weighted_hidden is None:
                    weighted_hidden = weights[i] * path.hidden_states
                else:
                    weighted_hidden += weights[i] * path.hidden_states
            merged_path.hidden_states = weighted_hidden
            logger.debug("[WeightedMergeStrategy] Computed weighted average of hidden states")
        else:
            # Use hidden states from best-scoring path
            best_path = max(paths, key=lambda p: p.score)
            merged_path.hidden_states = best_path.hidden_states
            logger.debug("[WeightedMergeStrategy] Using hidden states from best path")
        
        # Use KV cache from best-scoring path
        best_path = max(paths, key=lambda p: p.score)
        merged_path.kv_cache = best_path.kv_cache
        
        logger.info(f"[WeightedMergeStrategy] Merged {len(paths)} paths into one with score={merged_path.score:.4f}")
        return merged_path


class SelectiveMergeStrategy(MergeStrategy):
    """Selective merge strategy.
    
    Merges paths by selectively choosing the best components from each path
    rather than averaging everything.
    """
    
    def __init__(self, selection_criterion: str = 'best_score'):
        """Initialize the selective merge strategy.
        
        Args:
            selection_criterion: Criterion for selection ('best_score', 'median', 'consensus')
        """
        super().__init__()
        self.selection_criterion = selection_criterion
        logger.info(f"[SelectiveMergeStrategy] Initialized with criterion={selection_criterion}")
    
    def merge(
        self,
        paths: List[Any],
        **kwargs
    ) -> Any:
        """Merge paths using selective component selection.
        
        Args:
            paths: List of PathState objects to merge
            **kwargs: Additional parameters (unused)
            
        Returns:
            Merged PathState object
        """
        if not paths:
            logger.warning("[SelectiveMergeStrategy] Cannot merge empty path list")
            return None
        
        if len(paths) == 1:
            logger.debug("[SelectiveMergeStrategy] Only one path, returning as-is")
            return paths[0]
        
        logger.info(f"[SelectiveMergeStrategy] Merging {len(paths)} paths using selective strategy "
                   f"(criterion={self.selection_criterion})")
        
        # Import PathState here to avoid circular dependency
        from .path_manager import PathState
        
        # Select based on criterion
        if self.selection_criterion == 'best_score':
            # Use components from best-scoring path
            best_path = max(paths, key=lambda p: p.score)
            merged_path = PathState(
                path_id=best_path.path_id,
                latent_history=best_path.latent_history.copy(),
                hidden_states=best_path.hidden_states,
                kv_cache=best_path.kv_cache,
                score=best_path.score,
                metadata={
                    'merged_from': [p.path_id for p in paths],
                    'merge_strategy': 'selective_best',
                    'num_merged': len(paths),
                    'selected_path': best_path.path_id
                }
            )
            logger.debug(f"[SelectiveMergeStrategy] Selected best path {best_path.path_id}")
        
        elif self.selection_criterion == 'median':
            # Use median-scoring path
            sorted_paths = sorted(paths, key=lambda p: p.score)
            median_path = sorted_paths[len(sorted_paths) // 2]
            merged_path = PathState(
                path_id=median_path.path_id,
                latent_history=median_path.latent_history.copy(),
                hidden_states=median_path.hidden_states,
                kv_cache=median_path.kv_cache,
                score=median_path.score,
                metadata={
                    'merged_from': [p.path_id for p in paths],
                    'merge_strategy': 'selective_median',
                    'num_merged': len(paths),
                    'selected_path': median_path.path_id
                }
            )
            logger.debug(f"[SelectiveMergeStrategy] Selected median path {median_path.path_id}")
        
        else:  # consensus - average of top-k paths
            k = max(1, len(paths) // 2)
            top_k_paths = sorted(paths, key=lambda p: p.score, reverse=True)[:k]
            
            # Average hidden states of top-k
            if all(p.hidden_states is not None for p in top_k_paths):
                hidden_states_list = [p.hidden_states for p in top_k_paths]
                merged_hidden = torch.stack(hidden_states_list).mean(dim=0)
            else:
                merged_hidden = top_k_paths[0].hidden_states
            
            merged_path = PathState(
                path_id=top_k_paths[0].path_id,
                latent_history=top_k_paths[0].latent_history.copy(),
                hidden_states=merged_hidden,
                kv_cache=top_k_paths[0].kv_cache,
                score=np.mean([p.score for p in top_k_paths]),
                metadata={
                    'merged_from': [p.path_id for p in paths],
                    'merge_strategy': 'selective_consensus',
                    'num_merged': len(paths),
                    'consensus_size': k
                }
            )
            logger.debug(f"[SelectiveMergeStrategy] Computed consensus from top-{k} paths")
        
        logger.info(f"[SelectiveMergeStrategy] Merged {len(paths)} paths into one with score={merged_path.score:.4f}")
        return merged_path


class PathMerger:
    """High-level path merging orchestrator.
    
    This class coordinates path similarity detection and merging using
    configurable strategies.
    
    Attributes:
        similarity_detector: PathSimilarityDetector instance
        merge_strategy: MergeStrategy instance
        auto_select_strategy: Whether to automatically select merge strategy
        use_metadata_for_merge: Whether to use pre-computed consistency scores for merge decisions
        metadata_consistency_threshold: Minimum consistency required for merging
        metadata_score_diff_threshold: Maximum score difference allowed for merging
    """
    
    def __init__(
        self,
        similarity_detector: Optional[PathSimilarityDetector] = None,
        merge_strategy: Optional[MergeStrategy] = None,
        auto_select_strategy: bool = True,
        use_metadata_for_merge: bool = True,
        metadata_consistency_threshold: float = 0.85,
        metadata_score_diff_threshold: float = 0.05
    ):
        """Initialize the path merger.
        
        Args:
            similarity_detector: Custom similarity detector (creates default if None)
            merge_strategy: Custom merge strategy (creates default if None)
            auto_select_strategy: Whether to auto-select strategy based on paths
            use_metadata_for_merge: Whether to use pre-computed consistency from metadata
            metadata_consistency_threshold: Min consistency for merge candidates
            metadata_score_diff_threshold: Max score difference for merge candidates
        """
        self.similarity_detector = similarity_detector or PathSimilarityDetector()
        self.merge_strategy = merge_strategy or WeightedMergeStrategy()
        self.auto_select_strategy = auto_select_strategy
        self.use_metadata_for_merge = use_metadata_for_merge
        self.metadata_consistency_threshold = metadata_consistency_threshold
        self.metadata_score_diff_threshold = metadata_score_diff_threshold
        self.statistics_history: List[MergeStatistics] = []
        
        logger.info(f"[PathMerger] Initialized with auto_select={auto_select_strategy}, "
                   f"use_metadata={use_metadata_for_merge}, "
                   f"consistency_threshold={metadata_consistency_threshold:.2f}, "
                   f"score_diff_threshold={metadata_score_diff_threshold:.2f}")
    
    def should_merge_paths_by_metadata(
        self,
        path1: Any,
        path2: Any
    ) -> Tuple[bool, str]:
        """Check if two paths should be merged based on pre-computed metadata.
        
        This method uses already-computed consistency scores to avoid re-computing
        similarity, following the principle: "一致性高的路径是高质量路径".
        
        Args:
            path1: First PathState object
            path2: Second PathState object
            
        Returns:
            Tuple of (should_merge, reason)
        """
        # Extract consistency scores from metadata
        consistency1 = path1.metadata.get('latent_consistency', None)
        consistency2 = path2.metadata.get('latent_consistency', None)
        
        # Check if both paths have consistency scores
        if consistency1 is None or consistency2 is None:
            logger.debug(f"[PathMerger] Cannot use metadata-based merge: missing consistency scores "
                        f"(path {path1.path_id}: {consistency1}, path {path2.path_id}: {consistency2})")
            return False, "missing_consistency_scores"
        
        # Check 1: Both paths must have high consistency
        if consistency1 < self.metadata_consistency_threshold:
            logger.debug(f"[PathMerger] ✗ Check 1 failed: Path {path1.path_id} has low consistency "
                        f"({consistency1:.4f} < threshold {self.metadata_consistency_threshold:.2f})")
            return False, f"path1_low_consistency_{consistency1:.4f}"
        
        if consistency2 < self.metadata_consistency_threshold:
            logger.debug(f"[PathMerger] ✗ Check 1 failed: Path {path2.path_id} has low consistency "
                        f"({consistency2:.4f} < threshold {self.metadata_consistency_threshold:.2f})")
            return False, f"path2_low_consistency_{consistency2:.4f}"
        
        logger.debug(f"[PathMerger] ✓ Check 1 passed: Both paths have high consistency "
                    f"(path {path1.path_id}: {consistency1:.4f}, path {path2.path_id}: {consistency2:.4f})")
        
        # Check 2: Scores must be close
        score_diff = abs(path1.score - path2.score)
        if score_diff > self.metadata_score_diff_threshold:
            logger.debug(f"[PathMerger] ✗ Check 2 failed: Paths {path1.path_id} and {path2.path_id} have large score difference "
                        f"({score_diff:.4f} > threshold {self.metadata_score_diff_threshold:.2f})")
            return False, f"score_diff_too_large_{score_diff:.4f}"
        
        logger.debug(f"[PathMerger] ✓ Check 2 passed: Scores are close "
                    f"(path {path1.path_id}: {path1.score:.4f}, path {path2.path_id}: {path2.score:.4f}, diff={score_diff:.4f})")
        
        # Check 3 (optional): Both should be reasonably high quality
        min_quality_threshold = 0.5  # Paths with very low scores probably shouldn't merge
        if path1.score < min_quality_threshold or path2.score < min_quality_threshold:
            logger.debug(f"[PathMerger] ✗ Check 3 failed: At least one path has low quality score "
                        f"(path {path1.path_id}: {path1.score:.4f}, path {path2.path_id}: {path2.score:.4f}, "
                        f"threshold: {min_quality_threshold:.2f})")
            return False, f"low_quality_scores_{path1.score:.4f}_{path2.score:.4f}"
        
        logger.debug(f"[PathMerger] ✓ Check 3 passed: Both paths have reasonable quality "
                    f"(path {path1.path_id}: {path1.score:.4f}, path {path2.path_id}: {path2.score:.4f})")
        
        # All checks passed
        logger.debug(f"[PathMerger] ✓✓✓ All checks passed for paths {path1.path_id} and {path2.path_id}")
        logger.debug(f"[PathMerger]   Consistency: ({consistency1:.4f}, {consistency2:.4f})")
        logger.debug(f"[PathMerger]   Scores: ({path1.score:.4f}, {path2.score:.4f}), diff={score_diff:.4f}")
        return True, "all_checks_passed"
    
    def select_merge_strategy(
        self,
        paths: List[Any],
        merge_candidate: MergeCandidate
    ) -> MergeStrategy:
        """Automatically select the best merge strategy for given paths.
        
        Args:
            paths: List of PathState objects
            merge_candidate: MergeCandidate with similarity information
            
        Returns:
            Selected MergeStrategy instance
        """
        logger.debug("[PathMerger] Auto-selecting merge strategy")
        
        # Get paths to merge
        paths_to_merge = [p for p in paths if p.path_id in merge_candidate.path_ids]
        
        if not paths_to_merge:
            return self.merge_strategy
        
        # Compute score variance
        scores = [p.score for p in paths_to_merge]
        score_variance = np.var(scores)
        
        # Decision logic
        if score_variance < 0.01:
            # Low variance: paths have similar scores, use simple average
            strategy = AverageMergeStrategy()
            logger.info("[PathMerger] Selected AverageMergeStrategy (low score variance)")
        
        elif score_variance > 0.1:
            # High variance: paths have very different scores, use selective
            strategy = SelectiveMergeStrategy(selection_criterion='best_score')
            logger.info("[PathMerger] Selected SelectiveMergeStrategy (high score variance)")
        
        else:
            # Medium variance: use weighted average
            strategy = WeightedMergeStrategy()
            logger.info("[PathMerger] Selected WeightedMergeStrategy (medium score variance)")
        
        return strategy
    
    def merge_similar_paths(
        self,
        paths: List[Any],
        path_manager: Any,
        model_lm_head: Optional[torch.nn.Module] = None,
        use_kl: bool = False,
        min_group_size: int = 2
    ) -> List[Any]:
        """Find and merge similar paths.
        
        Uses metadata-based merge decisions when available to avoid re-computing similarity.
        
        Args:
            paths: List of PathState objects
            path_manager: PathManager instance to handle merging
            model_lm_head: Optional LM head for KL divergence
            use_kl: Whether to use KL divergence
            min_group_size: Minimum paths per merge group
            
        Returns:
            List of PathState objects after merging
        """
        logger.info(f"[PathMerger] Starting merge operation on {len(paths)} paths")
        
        if len(paths) < min_group_size:
            logger.debug("[PathMerger] Not enough paths for merging")
            return paths
        
        # STRATEGY: Use metadata-based merge decisions if enabled
        if self.use_metadata_for_merge:
            logger.info(f"[PathMerger] Using metadata-based merge decisions (avoiding re-computation)")
            
            # Find merge candidates using pre-computed consistency scores
            merge_candidates = []
            checked_pairs = set()
            
            for i, path1 in enumerate(paths):
                for j, path2 in enumerate(paths[i+1:], start=i+1):
                    # Avoid checking the same pair twice
                    pair_key = (min(path1.path_id, path2.path_id), max(path1.path_id, path2.path_id))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    
                    # Check if should merge using metadata
                    should_merge, reason = self.should_merge_paths_by_metadata(path1, path2)
                    
                    if should_merge:
                        # Create a merge candidate for this pair
                        consistency1 = path1.metadata.get('latent_consistency', 0.0)
                        consistency2 = path2.metadata.get('latent_consistency', 0.0)
                        avg_consistency = (consistency1 + consistency2) / 2.0
                        score_diff = abs(path1.score - path2.score)
                        
                        candidate = MergeCandidate(
                            path_ids=[path1.path_id, path2.path_id],
                            similarity_scores={(path1.path_id, path2.path_id): avg_consistency},
                            avg_similarity=avg_consistency,
                            merge_priority=avg_consistency * 2,  # 2 paths
                            metadata={
                                'merge_method': 'metadata_based',
                                'consistency1': consistency1,
                                'consistency2': consistency2,
                                'score1': path1.score,
                                'score2': path2.score,
                                'score_diff': score_diff,
                                'reason': reason
                            }
                        )
                        merge_candidates.append(candidate)
                        logger.debug(f"[PathMerger] ✓ Merge candidate approved: paths [{path1.path_id}, {path2.path_id}]")
                        logger.debug(f"[PathMerger]   - Consistency: [{consistency1:.4f}, {consistency2:.4f}] (avg={avg_consistency:.4f})")
                        logger.debug(f"[PathMerger]   - Scores: [{path1.score:.4f}, {path2.score:.4f}] (diff={score_diff:.4f})")
                        logger.debug(f"[PathMerger]   - Reason: {reason}")
                    else:
                        # Log why merge was rejected
                        logger.debug(f"[PathMerger] ✗ Merge rejected: paths [{path1.path_id}, {path2.path_id}] - {reason}")
            
            logger.info(f"[PathMerger] Found {len(merge_candidates)} metadata-based merge candidates")
            
            # If no metadata-based candidates found, fall back to similarity detection
            if not merge_candidates:
                logger.info(f"[PathMerger] No metadata-based merge candidates, falling back to similarity detection")
                merge_candidates = self.similarity_detector.find_merge_candidates(
                    paths, model_lm_head, use_kl, min_group_size
                )
        else:
            # Use traditional similarity detection
            logger.info(f"[PathMerger] Using traditional similarity detection")
            merge_candidates = self.similarity_detector.find_merge_candidates(
                paths, model_lm_head, use_kl, min_group_size
            )
        
        if not merge_candidates:
            logger.info("[PathMerger] No merge candidates found")
            return paths
        
        # Track which paths have been merged
        merged_path_ids = set()
        result_paths = []
        merge_groups = 0
        total_similarity = 0.0
        
        # Process each merge candidate
        for idx, candidate in enumerate(merge_candidates, 1):
            logger.debug(f"[PathMerger] Processing merge candidate {idx}/{len(merge_candidates)}: "
                        f"{len(candidate.path_ids)} paths with avg_similarity={candidate.avg_similarity:.4f}")
            
            # Skip if any path in this group has already been merged
            if any(pid in merged_path_ids for pid in candidate.path_ids):
                logger.debug(f"[PathMerger] Skipping candidate (paths already merged): {candidate.path_ids}")
                continue
            
            # Get paths to merge
            paths_to_merge = [p for p in paths if p.path_id in candidate.path_ids]
            
            if len(paths_to_merge) < min_group_size:
                logger.debug(f"[PathMerger] Skipping candidate (insufficient paths): "
                           f"found {len(paths_to_merge)}, need {min_group_size}")
                continue
            
            # Log paths to merge details
            paths_info = [f"Path{p.path_id}(score={p.score:.4f})" for p in paths_to_merge]
            logger.info(f"[PathMerger] Attempting to merge {len(paths_to_merge)} similar paths: {paths_info}")
            
            # Select merge strategy
            if self.auto_select_strategy:
                strategy = self.select_merge_strategy(paths, candidate)
            else:
                strategy = self.merge_strategy
                logger.debug(f"[PathMerger] Using pre-configured strategy: {strategy.__class__.__name__}")
            
            # Merge paths at data level
            merged_path = strategy.merge(paths_to_merge)
            
            if merged_path is not None:
                # Use path manager to officially merge and register
                merged_path_id = path_manager.merge_paths(
                    candidate.path_ids,
                    merge_strategy='custom'  # We already merged, just need to register
                )
                
                if merged_path_id is not None:
                    # Get the officially merged path from path manager
                    official_merged_path = path_manager.get_path(merged_path_id)
                    if official_merged_path is not None:
                        # Update with our merged data
                        official_merged_path.hidden_states = merged_path.hidden_states
                        official_merged_path.score = merged_path.score
                        official_merged_path.metadata.update(merged_path.metadata)
                        
                        result_paths.append(official_merged_path)
                        merged_path_ids.update(candidate.path_ids)
                        merge_groups += 1
                        total_similarity += candidate.avg_similarity
                        
                        logger.info(f"[PathMerger] Successfully merged {len(candidate.path_ids)} paths "
                                  f"{candidate.path_ids} into new path {merged_path_id} "
                                  f"(score={official_merged_path.score:.4f})")
                    else:
                        logger.warning(f"[PathMerger] Failed to retrieve merged path {merged_path_id} from path manager")
                else:
                    logger.warning(f"[PathMerger] PathManager.merge_paths() returned None for paths {candidate.path_ids}")
            else:
                logger.warning(f"[PathMerger] Strategy.merge() returned None for paths {candidate.path_ids}")
        
        # Add paths that were not merged
        unmerged_paths = []
        for path in paths:
            if path.path_id not in merged_path_ids:
                result_paths.append(path)
                unmerged_paths.append(path.path_id)
        
        logger.info(f"[PathMerger] Kept {len(unmerged_paths)} unmerged paths: {unmerged_paths}")
        
        # Compute statistics
        avg_similarity = total_similarity / merge_groups if merge_groups > 0 else 0.0
        stats = MergeStatistics(
            num_input_paths=len(paths),
            num_output_paths=len(result_paths),
            num_merged=len(merged_path_ids),
            merge_groups=merge_groups,
            avg_similarity=avg_similarity,
            score_before_merge=np.mean([p.score for p in paths]),
            score_after_merge=np.mean([p.score for p in result_paths])
        )
        self.statistics_history.append(stats)
        
        # Log final statistics
        logger.info(f"[PathMerger] Merge complete: {len(paths)} -> {len(result_paths)} paths "
                   f"({merge_groups} merge groups)")
        logger.info(f"[PathMerger] Merged {len(merged_path_ids)} paths into {merge_groups} new merged path(s)")
        logger.info(f"[PathMerger] Score change: {stats.score_before_merge:.4f} -> {stats.score_after_merge:.4f}")
        
        if merge_groups == 0:
            logger.debug(f"[PathMerger] No paths were merged (no suitable candidates found or all skipped)")
        
        # Clean up temporary data structures
        del merged_path_ids, merge_candidates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[PathMerger] GPU cache cleaned after merge operation")
        
        return result_paths
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of all merge operations.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.statistics_history:
            return {'num_operations': 0}
        
        return {
            'num_operations': len(self.statistics_history),
            'total_input_paths': sum(s.num_input_paths for s in self.statistics_history),
            'total_output_paths': sum(s.num_output_paths for s in self.statistics_history),
            'total_merged': sum(s.num_merged for s in self.statistics_history),
            'total_merge_groups': sum(s.merge_groups for s in self.statistics_history),
            'avg_similarity': np.mean([s.avg_similarity for s in self.statistics_history if s.avg_similarity > 0]),
            'avg_score_change': np.mean([s.score_after_merge - s.score_before_merge 
                                        for s in self.statistics_history])
        }

