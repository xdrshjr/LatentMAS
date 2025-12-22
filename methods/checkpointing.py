"""Checkpointing module for saving and resuming multi-path reasoning experiments.

This module provides utilities to save and restore the state of multi-path reasoning
experiments, including graph states, path states, scores, and metadata.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import pickle
import json
import gzip
from datetime import datetime
from dataclasses import asdict

from .graph_structure import ReasoningGraph, ReasoningNode
from .path_manager import PathState, PathManager

# Logger setup
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpointing for multi-path reasoning experiments.
    
    This class provides methods to save and restore experiment state,
    including reasoning graphs, path states, and metadata.
    
    Attributes:
        checkpoint_dir: Directory to store checkpoints
        compress: Whether to compress checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        checkpoint_interval: Steps between checkpoints
        last_checkpoint_step: Step number of last checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        compress: bool = True,
        max_checkpoints: int = 5,
        checkpoint_interval: int = 100
    ):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            compress: Whether to compress checkpoint files
            max_checkpoints: Maximum number of checkpoints to keep (0 for unlimited)
            checkpoint_interval: Steps between automatic checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_step = -1
        
        logger.info(
            f"[CheckpointManager] Initialized with dir={checkpoint_dir}, "
            f"compress={compress}, max_checkpoints={max_checkpoints}"
        )
    
    def save_checkpoint(
        self,
        step: int,
        graph: Optional[ReasoningGraph] = None,
        path_manager: Optional[PathManager] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """Save a checkpoint of the current state.
        
        Args:
            step: Current step number
            graph: Reasoning graph to save
            path_manager: Path manager to save
            metadata: Additional metadata to save
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            Path to the saved checkpoint file
        """
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step{step}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        # Save graph state
        if graph is not None:
            checkpoint_data['graph'] = self._serialize_graph(graph)
            logger.debug(
                f"[CheckpointManager] Serialized graph with {len(graph.nodes)} nodes"
            )
        
        # Save path manager state
        if path_manager is not None:
            checkpoint_data['path_manager'] = self._serialize_path_manager(path_manager)
            logger.debug(
                f"[CheckpointManager] Serialized path manager with "
                f"{len(path_manager.paths)} paths"
            )
        
        # Save to file
        if self.compress:
            checkpoint_file = checkpoint_path.with_suffix('.pkl.gz')
            with gzip.open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            checkpoint_file = checkpoint_path.with_suffix('.pkl')
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata as JSON for easy inspection
        metadata_file = checkpoint_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'step': step,
                'timestamp': checkpoint_data['timestamp'],
                'num_nodes': len(graph.nodes) if graph else 0,
                'num_paths': len(path_manager.paths) if path_manager else 0,
                'metadata': metadata or {},
            }, f, indent=2)
        
        self.last_checkpoint_step = step
        
        logger.info(
            f"[CheckpointManager] Saved checkpoint at step {step} to {checkpoint_file}"
        )
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_file)
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """Load a checkpoint from file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        if checkpoint_file.suffix == '.gz':
            with gzip.open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
        else:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        logger.info(
            f"[CheckpointManager] Loaded checkpoint from step {checkpoint_data['step']}"
        )
        logger.debug(
            f"[CheckpointManager] Checkpoint timestamp: {checkpoint_data['timestamp']}"
        )
        
        return checkpoint_data
    
    def restore_graph(
        self,
        checkpoint_data: Dict[str, Any]
    ) -> Optional[ReasoningGraph]:
        """Restore a reasoning graph from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            Restored ReasoningGraph or None if not present
        """
        if 'graph' not in checkpoint_data:
            logger.warning("[CheckpointManager] No graph data in checkpoint")
            return None
        
        graph = self._deserialize_graph(checkpoint_data['graph'])
        
        logger.info(
            f"[CheckpointManager] Restored graph with {len(graph.nodes)} nodes"
        )
        
        return graph
    
    def restore_path_manager(
        self,
        checkpoint_data: Dict[str, Any]
    ) -> Optional[PathManager]:
        """Restore a path manager from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            Restored PathManager or None if not present
        """
        if 'path_manager' not in checkpoint_data:
            logger.warning("[CheckpointManager] No path manager data in checkpoint")
            return None
        
        path_manager = self._deserialize_path_manager(checkpoint_data['path_manager'])
        
        logger.info(
            f"[CheckpointManager] Restored path manager with "
            f"{len(path_manager.paths)} paths"
        )
        
        return path_manager
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step.
        
        Args:
            step: Current step number
            
        Returns:
            True if checkpoint should be saved
        """
        if self.checkpoint_interval <= 0:
            return False
        
        return (step - self.last_checkpoint_step) >= self.checkpoint_interval
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for json_file in sorted(self.checkpoint_dir.glob("*.json")):
            try:
                with open(json_file, 'r') as f:
                    info = json.load(f)
                    info['path'] = str(json_file.with_suffix('.pkl.gz' if self.compress else '.pkl'))
                    checkpoints.append(info)
            except Exception as e:
                logger.warning(
                    f"[CheckpointManager] Error reading checkpoint info from {json_file}: {e}"
                )
        
        logger.info(f"[CheckpointManager] Found {len(checkpoints)} checkpoints")
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            logger.info("[CheckpointManager] No checkpoints found")
            return None
        
        # Sort by step number
        latest = max(checkpoints, key=lambda x: x['step'])
        
        logger.info(
            f"[CheckpointManager] Latest checkpoint: step {latest['step']}, "
            f"path={latest['path']}"
        )
        
        return latest['path']
    
    def _serialize_graph(self, graph: ReasoningGraph) -> Dict[str, Any]:
        """Serialize a reasoning graph to a dictionary.
        
        Args:
            graph: Reasoning graph to serialize
            
        Returns:
            Serialized graph data
        """
        # Serialize nodes (excluding tensors)
        nodes_data = {}
        for node_id, node in graph.nodes.items():
            nodes_data[node_id] = {
                'node_id': node.node_id,
                'parent_id': node.parent_id,
                'children_ids': node.children_ids,
                'score': node.score,
                'metadata': node.metadata,
                # Note: hidden_states and kv_cache are not serialized
                # They will need to be regenerated
            }
        
        return {
            'nodes': nodes_data,
            'edges': graph.edges,
            'root_id': graph.root_id,
            'leaf_ids': list(graph.leaf_ids),
            'next_node_id': graph.next_node_id,
        }
    
    def _deserialize_graph(self, graph_data: Dict[str, Any]) -> ReasoningGraph:
        """Deserialize a reasoning graph from a dictionary.
        
        Args:
            graph_data: Serialized graph data
            
        Returns:
            Restored ReasoningGraph
        """
        graph = ReasoningGraph()
        
        # Restore graph metadata
        graph.edges = graph_data['edges']
        graph.root_id = graph_data['root_id']
        graph.leaf_ids = set(graph_data['leaf_ids'])
        graph.next_node_id = graph_data['next_node_id']
        
        # Restore nodes
        for node_id, node_data in graph_data['nodes'].items():
            node = ReasoningNode(
                node_id=node_data['node_id'],
                hidden_states=None,  # Will need to be regenerated
                kv_cache=None,  # Will need to be regenerated
                parent_id=node_data['parent_id'],
                children_ids=node_data['children_ids'],
                score=node_data['score'],
                metadata=node_data['metadata']
            )
            graph.nodes[int(node_id)] = node
        
        return graph
    
    def _serialize_path_manager(self, path_manager: PathManager) -> Dict[str, Any]:
        """Serialize a path manager to a dictionary.
        
        Args:
            path_manager: Path manager to serialize
            
        Returns:
            Serialized path manager data
        """
        # Serialize paths (excluding tensors)
        paths_data = {}
        for path_id, path in path_manager.paths.items():
            paths_data[path_id] = {
                'path_id': path.path_id,
                'num_latent_steps': len(path.latent_history),
                'score': path.score,
                'metadata': path.metadata,
                'node_ids': path.node_ids,
                # Note: latent_history, hidden_states, and kv_cache are not serialized
            }
        
        return {
            'paths': paths_data,
            'active_paths': list(path_manager.active_paths),
            'next_path_id': path_manager.next_path_id,
        }
    
    def _deserialize_path_manager(
        self,
        path_manager_data: Dict[str, Any]
    ) -> PathManager:
        """Deserialize a path manager from a dictionary.
        
        Args:
            path_manager_data: Serialized path manager data
            
        Returns:
            Restored PathManager
        """
        path_manager = PathManager()
        
        # Restore path manager metadata
        path_manager.active_paths = set(path_manager_data['active_paths'])
        path_manager.next_path_id = path_manager_data['next_path_id']
        
        # Restore paths (with empty tensors)
        for path_id, path_data in path_manager_data['paths'].items():
            path = PathState(
                path_id=path_data['path_id'],
                latent_history=[],  # Will need to be regenerated
                hidden_states=None,  # Will need to be regenerated
                kv_cache=None,  # Will need to be regenerated
                score=path_data['score'],
                metadata=path_data['metadata'],
                node_ids=path_data['node_ids']
            )
            path_manager.paths[int(path_id)] = path
        
        return path_manager
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x['step'])
        
        # Remove oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:num_to_remove]:
            try:
                # Remove both .pkl/.pkl.gz and .json files
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                json_path = checkpoint_path.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                
                logger.debug(
                    f"[CheckpointManager] Removed old checkpoint: step {checkpoint['step']}"
                )
            except Exception as e:
                logger.warning(
                    f"[CheckpointManager] Error removing checkpoint: {e}"
                )
        
        logger.info(
            f"[CheckpointManager] Cleaned up {num_to_remove} old checkpoints"
        )


class IncrementalCheckpointer:
    """Implements incremental checkpointing to save only changes.
    
    This class tracks changes since the last checkpoint and saves only
    the differences, reducing checkpoint size and save time.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """Initialize the incremental checkpointer.
        
        Args:
            checkpoint_manager: Checkpoint manager to use
        """
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_data = None
        
        logger.info("[IncrementalCheckpointer] Initialized")
    
    def save_incremental(
        self,
        step: int,
        graph: Optional[ReasoningGraph] = None,
        path_manager: Optional[PathManager] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save an incremental checkpoint with only changes.
        
        Args:
            step: Current step number
            graph: Current reasoning graph
            path_manager: Current path manager
            metadata: Additional metadata
            
        Returns:
            Path to the saved checkpoint file
        """
        # Detect changes
        changes = self._detect_changes(graph, path_manager)
        
        if not changes['has_changes']:
            logger.info(
                f"[IncrementalCheckpointer] No changes detected at step {step}, "
                f"skipping checkpoint"
            )
            return None
        
        # Save full checkpoint (for now, incremental diff not implemented)
        # In a full implementation, we would save only the changes
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            step=step,
            graph=graph,
            path_manager=path_manager,
            metadata={
                **(metadata or {}),
                'incremental': True,
                'changes': changes,
            }
        )
        
        # Update last checkpoint data
        if graph is not None:
            self.last_checkpoint_data = {
                'num_nodes': len(graph.nodes),
                'num_paths': len(path_manager.paths) if path_manager else 0,
            }
        
        logger.info(
            f"[IncrementalCheckpointer] Saved incremental checkpoint: "
            f"{changes['num_new_nodes']} new nodes, "
            f"{changes['num_new_paths']} new paths"
        )
        
        return checkpoint_path
    
    def _detect_changes(
        self,
        graph: Optional[ReasoningGraph],
        path_manager: Optional[PathManager]
    ) -> Dict[str, Any]:
        """Detect changes since last checkpoint.
        
        Args:
            graph: Current reasoning graph
            path_manager: Current path manager
            
        Returns:
            Dictionary describing changes
        """
        changes = {
            'has_changes': False,
            'num_new_nodes': 0,
            'num_new_paths': 0,
            'num_updated_scores': 0,
        }
        
        if self.last_checkpoint_data is None:
            changes['has_changes'] = True
            if graph is not None:
                changes['num_new_nodes'] = len(graph.nodes)
            if path_manager is not None:
                changes['num_new_paths'] = len(path_manager.paths)
        else:
            if graph is not None:
                new_nodes = len(graph.nodes) - self.last_checkpoint_data.get('num_nodes', 0)
                if new_nodes > 0:
                    changes['has_changes'] = True
                    changes['num_new_nodes'] = new_nodes
            
            if path_manager is not None:
                new_paths = len(path_manager.paths) - self.last_checkpoint_data.get('num_paths', 0)
                if new_paths > 0:
                    changes['has_changes'] = True
                    changes['num_new_paths'] = new_paths
        
        logger.debug(
            f"[IncrementalCheckpointer] Change detection: "
            f"has_changes={changes['has_changes']}, "
            f"new_nodes={changes['num_new_nodes']}, "
            f"new_paths={changes['num_new_paths']}"
        )
        
        return changes


def create_checkpoint_from_state(
    step: int,
    graph: Optional[ReasoningGraph] = None,
    path_manager: Optional[PathManager] = None,
    metadata: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "./checkpoints",
    compress: bool = True
) -> str:
    """Convenience function to create a checkpoint.
    
    Args:
        step: Current step number
        graph: Reasoning graph to save
        path_manager: Path manager to save
        metadata: Additional metadata
        checkpoint_dir: Directory to store checkpoint
        compress: Whether to compress checkpoint
        
    Returns:
        Path to the saved checkpoint file
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, compress=compress)
    return manager.save_checkpoint(
        step=step,
        graph=graph,
        path_manager=path_manager,
        metadata=metadata
    )


def restore_from_checkpoint(
    checkpoint_path: str
) -> Tuple[Optional[ReasoningGraph], Optional[PathManager], Dict[str, Any]]:
    """Convenience function to restore from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (graph, path_manager, metadata)
    """
    manager = CheckpointManager()
    checkpoint_data = manager.load_checkpoint(checkpoint_path)
    
    graph = manager.restore_graph(checkpoint_data)
    path_manager = manager.restore_path_manager(checkpoint_data)
    metadata = checkpoint_data.get('metadata', {})
    
    return graph, path_manager, metadata

