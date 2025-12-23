"""Path management module for tracking and managing reasoning paths.

This module provides utilities for managing the lifecycle of reasoning paths,
including creation, branching, merging, and comparison.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class PathState:
    """Encapsulates information about a single reasoning path.
    
    Attributes:
        path_id: Unique identifier for this path
        latent_history: List of latent vectors generated along this path
        hidden_states: Current hidden states tensor
        kv_cache: Current KV cache from the model
        score: Quality score for this path
        metadata: Additional information about this path
        node_ids: List of node IDs in the graph that make up this path
    """
    path_id: int
    latent_history: List[torch.Tensor] = field(default_factory=list)
    hidden_states: Optional[torch.Tensor] = None
    kv_cache: Optional[Tuple] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_ids: List[int] = field(default_factory=list)
    
    def add_latent_step(self, latent_vec: torch.Tensor) -> None:
        """Add a latent vector to the history.
        
        Args:
            latent_vec: Latent vector to add
        """
        self.latent_history.append(latent_vec.detach().clone())
        logger.debug(f"[PathState] Added latent step to path {self.path_id}, total steps: {len(self.latent_history)}")
    
    def update_state(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        score: Optional[float] = None
    ) -> None:
        """Update the current state of this path.
        
        Args:
            hidden_states: New hidden states (if provided)
            kv_cache: New KV cache (if provided)
            score: New score (if provided)
        """
        if hidden_states is not None:
            self.hidden_states = hidden_states
        if kv_cache is not None:
            self.kv_cache = kv_cache
        if score is not None:
            old_score = self.score
            self.score = score
            logger.debug(f"[PathState] Updated score for path {self.path_id}: {old_score:.4f} -> {score:.4f}")
    
    def get_length(self) -> int:
        """Get the length of this path (number of latent steps).
        
        Returns:
            Number of latent steps in this path
        """
        return len(self.latent_history)
    
    def clone(self, new_path_id: int) -> 'PathState':
        """Create a deep copy of this path with a new ID.
        
        Args:
            new_path_id: ID for the cloned path
            
        Returns:
            New PathState object with copied data
        """
        cloned = PathState(
            path_id=new_path_id,
            latent_history=[vec.clone() for vec in self.latent_history],
            hidden_states=self.hidden_states.clone() if self.hidden_states is not None else None,
            kv_cache=self.kv_cache,  # KV cache is typically not cloned
            score=self.score,
            metadata=self.metadata.copy(),
            node_ids=self.node_ids.copy()
        )
        logger.debug(f"[PathState] Cloned path {self.path_id} to new path {new_path_id}")
        return cloned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary (excluding tensors).
        
        Returns:
            Dictionary representation of the path
        """
        return {
            'path_id': self.path_id,
            'num_latent_steps': len(self.latent_history),
            'score': self.score,
            'metadata': self.metadata,
            'node_ids': self.node_ids,
            'has_hidden_states': self.hidden_states is not None,
            'has_kv_cache': self.kv_cache is not None,
        }


class PathManager:
    """Manages the lifecycle of reasoning paths.
    
    This class provides methods to create, branch, merge, and track multiple
    reasoning paths during multi-path latent reasoning.
    
    Attributes:
        paths: Dictionary mapping path IDs to PathState objects
        active_paths: Set of currently active path IDs
        next_path_id: Counter for generating unique path IDs
    """
    
    def __init__(self):
        """Initialize an empty path manager."""
        self.paths: Dict[int, PathState] = {}
        self.active_paths: set = set()
        self.next_path_id: int = 0
        logger.info("[PathManager] Initialized path manager")
    
    def create_path(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create a new reasoning path.
        
        Args:
            hidden_states: Initial hidden states
            kv_cache: Initial KV cache
            score: Initial score
            metadata: Additional metadata
            
        Returns:
            ID of the newly created path
        """
        path_id = self.next_path_id
        self.next_path_id += 1
        
        path = PathState(
            path_id=path_id,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            score=score,
            metadata=metadata or {}
        )
        
        self.paths[path_id] = path
        self.active_paths.add(path_id)
        
        logger.info(f"[PathManager] Created new path: path_id={path_id}, score={score:.4f}")
        return path_id
    
    def get_path(self, path_id: int) -> Optional[PathState]:
        """Get a path by its ID.
        
        Args:
            path_id: ID of the path to retrieve
            
        Returns:
            PathState object or None if not found
        """
        return self.paths.get(path_id)
    
    def branch_path(self, source_path_id: int, num_branches: int = 2) -> List[int]:
        """Create multiple branches from an existing path.
        
        Args:
            source_path_id: ID of the path to branch from
            num_branches: Number of branches to create
            
        Returns:
            List of new path IDs
        """
        source_path = self.get_path(source_path_id)
        if source_path is None:
            logger.warning(f"[PathManager] Cannot branch from non-existent path {source_path_id}")
            return []
        
        new_path_ids = []
        for i in range(num_branches):
            new_path_id = self.next_path_id
            self.next_path_id += 1
            
            # Clone the source path
            new_path = source_path.clone(new_path_id)
            new_path.metadata['branched_from'] = source_path_id
            new_path.metadata['branch_index'] = i
            
            self.paths[new_path_id] = new_path
            self.active_paths.add(new_path_id)
            new_path_ids.append(new_path_id)
        
        logger.info(f"[PathManager] Branched path {source_path_id} into {num_branches} new paths: {new_path_ids}")
        return new_path_ids
    
    def merge_paths(
        self,
        path_ids: List[int],
        merge_strategy: str = 'weighted_average'
    ) -> Optional[int]:
        """Merge multiple paths into a single path.
        
        Args:
            path_ids: List of path IDs to merge
            merge_strategy: Strategy for merging ('weighted_average', 'best', 'average')
            
        Returns:
            ID of the merged path, or None if merge failed
        """
        if len(path_ids) < 2:
            logger.warning("[PathManager] Cannot merge less than 2 paths")
            return None
        
        # Validate all paths exist
        paths_to_merge = []
        for path_id in path_ids:
            path = self.get_path(path_id)
            if path is None:
                logger.warning(f"[PathManager] Cannot merge: path {path_id} not found")
                return None
            paths_to_merge.append(path)
        
        # Create merged path
        merged_path_id = self.next_path_id
        self.next_path_id += 1
        
        # Merge based on strategy
        if merge_strategy == 'best':
            # Use the best-scoring path
            best_path = max(paths_to_merge, key=lambda p: p.score)
            merged_path = best_path.clone(merged_path_id)
            merged_score = best_path.score
        
        elif merge_strategy == 'average':
            # Simple average
            merged_path = paths_to_merge[0].clone(merged_path_id)
            merged_score = sum(p.score for p in paths_to_merge) / len(paths_to_merge)
            
            # Average hidden states if available
            if all(p.hidden_states is not None for p in paths_to_merge):
                hidden_states_list = [p.hidden_states for p in paths_to_merge]
                merged_path.hidden_states = torch.stack(hidden_states_list).mean(dim=0)
        
        elif merge_strategy == 'weighted_average':
            # Weighted average by scores
            total_score = sum(p.score for p in paths_to_merge)
            if total_score > 0:
                weights = [p.score / total_score for p in paths_to_merge]
            else:
                weights = [1.0 / len(paths_to_merge)] * len(paths_to_merge)
            
            merged_path = paths_to_merge[0].clone(merged_path_id)
            merged_score = sum(w * p.score for w, p in zip(weights, paths_to_merge))
            
            # Weighted average of hidden states
            if all(p.hidden_states is not None for p in paths_to_merge):
                hidden_states_list = [p.hidden_states for p in paths_to_merge]
                weighted_hidden = sum(
                    w * hs for w, hs in zip(weights, hidden_states_list)
                )
                merged_path.hidden_states = weighted_hidden
        
        elif merge_strategy == 'custom':
            # Custom merge: caller has already performed the merge
            # Create a placeholder path that will be updated by caller
            merged_path = paths_to_merge[0].clone(merged_path_id)
            merged_score = merged_path.score  # Will be updated by caller
            logger.debug(f"[PathManager] Using custom merge strategy (path will be updated by caller)")
        
        else:
            logger.warning(f"[PathManager] Unknown merge strategy: {merge_strategy}")
            return None
        
        # Update merged path
        merged_path.score = merged_score
        merged_path.metadata['merged_from'] = path_ids
        merged_path.metadata['merge_strategy'] = merge_strategy
        
        # Register merged path
        self.paths[merged_path_id] = merged_path
        self.active_paths.add(merged_path_id)
        logger.debug(f"[PathManager] Registered merged path {merged_path_id} in path manager")
        
        # Deactivate original paths
        deactivated_count = 0
        for path_id in path_ids:
            if path_id in self.active_paths:
                self.active_paths.discard(path_id)
                deactivated_count += 1
        
        logger.info(f"[PathManager] Merged {len(path_ids)} paths {path_ids} into path {merged_path_id} "
                   f"using '{merge_strategy}' strategy, score={merged_score:.4f}")
        logger.info(f"[PathManager] Deactivated {deactivated_count} original paths from active set")
        logger.debug(f"[PathManager] Active paths count: {len(self.active_paths)}")
        
        return merged_path_id
    
    def get_active_paths(self) -> List[PathState]:
        """Get all currently active paths.
        
        Returns:
            List of active PathState objects
        """
        active = [self.paths[path_id] for path_id in self.active_paths if path_id in self.paths]
        logger.debug(f"[PathManager] Retrieved {len(active)} active paths")
        return active
    
    def prune_paths(self, path_ids_to_remove: List[int]) -> int:
        """Remove paths from the active set.
        
        Args:
            path_ids_to_remove: List of path IDs to prune
            
        Returns:
            Number of paths successfully pruned
        """
        pruned_count = 0
        for path_id in path_ids_to_remove:
            if path_id in self.active_paths:
                self.active_paths.discard(path_id)
                pruned_count += 1
        
        logger.info(f"[PathManager] Pruned {pruned_count}/{len(path_ids_to_remove)} paths from active set")
        return pruned_count
    
    def deactivate_path(self, path_id: int) -> bool:
        """Deactivate a path (keep in memory but mark as inactive).
        
        Args:
            path_id: ID of the path to deactivate
            
        Returns:
            True if path was deactivated, False if not found
        """
        if path_id in self.active_paths:
            self.active_paths.discard(path_id)
            logger.info(f"[PathManager] Deactivated path {path_id}")
            return True
        return False
    
    def reactivate_path(self, path_id: int) -> bool:
        """Reactivate a previously deactivated path.
        
        Args:
            path_id: ID of the path to reactivate
            
        Returns:
            True if path was reactivated, False if not found
        """
        if path_id in self.paths and path_id not in self.active_paths:
            self.active_paths.add(path_id)
            logger.info(f"[PathManager] Reactivated path {path_id}")
            return True
        return False
    
    def compute_path_similarity(
        self,
        path_id1: int,
        path_id2: int,
        method: str = 'cosine'
    ) -> Optional[float]:
        """Compute similarity between two paths.
        
        Args:
            path_id1: ID of first path
            path_id2: ID of second path
            method: Similarity method ('cosine', 'euclidean')
            
        Returns:
            Similarity score, or None if paths not found or incompatible
        """
        path1 = self.get_path(path_id1)
        path2 = self.get_path(path_id2)
        
        if path1 is None or path2 is None:
            logger.warning(f"[PathManager] Cannot compute similarity: path not found")
            return None
        
        if path1.hidden_states is None or path2.hidden_states is None:
            logger.warning(f"[PathManager] Cannot compute similarity: missing hidden states")
            return None
        
        h1 = path1.hidden_states
        h2 = path2.hidden_states
        
        # Ensure same shape
        if h1.shape != h2.shape:
            logger.warning(f"[PathManager] Cannot compute similarity: shape mismatch {h1.shape} vs {h2.shape}")
            return None
        
        if method == 'cosine':
            # Flatten if needed
            h1_flat = h1.flatten()
            h2_flat = h2.flatten()
            similarity = F.cosine_similarity(h1_flat.unsqueeze(0), h2_flat.unsqueeze(0)).item()
        
        elif method == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = torch.norm(h1 - h2).item()
            similarity = 1.0 / (1.0 + distance)
        
        else:
            logger.warning(f"[PathManager] Unknown similarity method: {method}")
            return None
        
        logger.debug(f"[PathManager] Similarity between paths {path_id1} and {path_id2} ({method}): {similarity:.4f}")
        return similarity
    
    def find_similar_paths(
        self,
        threshold: float = 0.9,
        method: str = 'cosine'
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of similar paths in the active set.
        
        Args:
            threshold: Minimum similarity threshold
            method: Similarity method to use
            
        Returns:
            List of (path_id1, path_id2, similarity) tuples
        """
        similar_pairs = []
        active_path_ids = list(self.active_paths)
        
        for i, path_id1 in enumerate(active_path_ids):
            for path_id2 in active_path_ids[i+1:]:
                similarity = self.compute_path_similarity(path_id1, path_id2, method)
                if similarity is not None and similarity >= threshold:
                    similar_pairs.append((path_id1, path_id2, similarity))
        
        logger.debug(f"[PathManager] Found {len(similar_pairs)} similar path pairs (threshold={threshold})")
        return similar_pairs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed paths.
        
        Returns:
            Dictionary containing path statistics
        """
        active_paths = self.get_active_paths()
        
        stats = {
            'total_paths': len(self.paths),
            'active_paths': len(self.active_paths),
            'inactive_paths': len(self.paths) - len(self.active_paths),
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'avg_length': 0.0,
        }
        
        if active_paths:
            scores = [p.score for p in active_paths]
            stats['avg_score'] = sum(scores) / len(scores)
            stats['max_score'] = max(scores)
            stats['min_score'] = min(scores)
            stats['avg_length'] = sum(p.get_length() for p in active_paths) / len(active_paths)
        
        return stats
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save path manager state to a checkpoint file.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        try:
            checkpoint_dir = Path(checkpoint_path).parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data (excluding large tensors)
            checkpoint_data = {
                'next_path_id': self.next_path_id,
                'active_paths': list(self.active_paths),
                'paths_metadata': {
                    path_id: path.to_dict()
                    for path_id, path in self.paths.items()
                }
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"[PathManager] Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"[PathManager] Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load path manager state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.next_path_id = checkpoint_data['next_path_id']
            self.active_paths = set(checkpoint_data['active_paths'])
            
            logger.info(f"[PathManager] Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"[PathManager] Failed to load checkpoint: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all paths and reset the manager."""
        num_paths = len(self.paths)
        self.paths.clear()
        self.active_paths.clear()
        logger.info(f"[PathManager] Cleared {num_paths} paths")
    
    def __len__(self) -> int:
        """Return the number of active paths."""
        return len(self.active_paths)
    
    def __repr__(self) -> str:
        """String representation of the path manager."""
        stats = self.get_statistics()
        return (f"PathManager(total={stats['total_paths']}, "
                f"active={stats['active_paths']}, "
                f"avg_score={stats['avg_score']:.3f})")

