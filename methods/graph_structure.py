"""Graph data structure module for managing reasoning paths in multi-path latent reasoning.

This module provides the core graph structure to represent and manage reasoning states,
including nodes, edges, and graph operations like traversal, pruning, and merging.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import torch
from collections import deque
import json

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class ReasoningNode:
    """Represents a single reasoning state in the graph.
    
    Attributes:
        node_id: Unique identifier for this node
        hidden_states: Hidden states tensor at this reasoning step
        kv_cache: Key-value cache from the model
        parent_id: ID of the parent node (None for root)
        children_ids: List of child node IDs
        score: Quality score for this reasoning path
        metadata: Additional information about this node
    """
    node_id: int
    hidden_states: Optional[torch.Tensor] = None
    kv_cache: Optional[Tuple] = None
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_id: int) -> None:
        """Add a child node ID to this node.
        
        Args:
            child_id: ID of the child node to add
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            logger.debug(f"[ReasoningNode] Added child {child_id} to node {self.node_id}")
    
    def update_score(self, new_score: float) -> None:
        """Update the quality score of this node.
        
        Args:
            new_score: New score value
        """
        old_score = self.score
        self.score = new_score
        logger.debug(f"[ReasoningNode] Updated score for node {self.node_id}: {old_score:.4f} -> {new_score:.4f}")
    
    def get_depth(self, graph: 'ReasoningGraph') -> int:
        """Calculate the depth of this node in the graph.
        
        Args:
            graph: The reasoning graph containing this node
            
        Returns:
            Depth of the node (root has depth 0)
        """
        depth = 0
        current_id = self.node_id
        while True:
            node = graph.get_node(current_id)
            if node is None or node.parent_id is None:
                break
            current_id = node.parent_id
            depth += 1
        return depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary (excluding tensors for serialization).
        
        Returns:
            Dictionary representation of the node
        """
        return {
            'node_id': self.node_id,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'score': self.score,
            'metadata': self.metadata,
            'has_hidden_states': self.hidden_states is not None,
            'has_kv_cache': self.kv_cache is not None,
        }


class ReasoningGraph:
    """Manages the entire reasoning graph structure.
    
    This class provides methods to build, traverse, prune, and merge reasoning paths
    represented as a directed acyclic graph (DAG).
    
    Attributes:
        nodes: Dictionary mapping node IDs to ReasoningNode objects
        edges: List of (parent_id, child_id) tuples
        root_id: ID of the root node
        leaf_ids: Set of leaf node IDs (nodes with no children)
        next_node_id: Counter for generating unique node IDs
    """
    
    def __init__(self):
        """Initialize an empty reasoning graph."""
        self.nodes: Dict[int, ReasoningNode] = {}
        self.edges: List[Tuple[int, int]] = []
        self.root_id: Optional[int] = None
        self.leaf_ids: Set[int] = set()
        self.next_node_id: int = 0
        logger.info("[ReasoningGraph] Initialized empty reasoning graph")
    
    def add_node(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        parent_id: Optional[int] = None,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a new node to the graph.
        
        Args:
            hidden_states: Hidden states tensor for this node
            kv_cache: KV cache for this node
            parent_id: ID of parent node (None for root)
            score: Initial score for this node
            metadata: Additional metadata
            
        Returns:
            ID of the newly created node
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = ReasoningNode(
            node_id=node_id,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            parent_id=parent_id,
            score=score,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Update root if this is the first node
        if self.root_id is None:
            self.root_id = node_id
            logger.info(f"[ReasoningGraph] Set root node: node_id={node_id}")
        
        # Update parent-child relationships
        if parent_id is not None:
            if parent_id in self.nodes:
                self.nodes[parent_id].add_child(node_id)
                self.edges.append((parent_id, node_id))
                # Parent is no longer a leaf
                self.leaf_ids.discard(parent_id)
            else:
                logger.warning(f"[ReasoningGraph] Parent node {parent_id} not found for node {node_id}")
        
        # New node is a leaf
        self.leaf_ids.add(node_id)
        
        logger.info(f"[ReasoningGraph] Added node: node_id={node_id}, parent_id={parent_id}, score={score:.4f}")
        return node_id
    
    def get_node(self, node_id: int) -> Optional[ReasoningNode]:
        """Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            ReasoningNode object or None if not found
        """
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: int) -> bool:
        """Remove a node from the graph.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        if node_id not in self.nodes:
            logger.warning(f"[ReasoningGraph] Cannot remove node {node_id}: not found")
            return False
        
        node = self.nodes[node_id]
        
        # Cannot remove root node
        if node_id == self.root_id:
            logger.warning(f"[ReasoningGraph] Cannot remove root node {node_id}")
            return False
        
        # Remove edges involving this node
        self.edges = [(p, c) for p, c in self.edges if p != node_id and c != node_id]
        
        # Update parent's children list
        if node.parent_id is not None and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
            # If parent has no more children, it becomes a leaf
            if len(parent.children_ids) == 0:
                self.leaf_ids.add(node.parent_id)
        
        # Remove from leaf set
        self.leaf_ids.discard(node_id)
        
        # Remove children recursively
        children_to_remove = list(node.children_ids)
        for child_id in children_to_remove:
            self.remove_node(child_id)
        
        # Remove the node itself
        del self.nodes[node_id]
        logger.info(f"[ReasoningGraph] Removed node {node_id}")
        return True
    
    def get_path(self, node_id: int) -> List[int]:
        """Get the path from root to a specific node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of node IDs from root to target node
        """
        path = []
        current_id = node_id
        
        while current_id is not None:
            path.append(current_id)
            node = self.get_node(current_id)
            if node is None:
                break
            current_id = node.parent_id
        
        path.reverse()
        logger.debug(f"[ReasoningGraph] Path to node {node_id}: {path}")
        return path
    
    def get_all_paths(self) -> List[List[int]]:
        """Get all paths from root to leaf nodes.
        
        Returns:
            List of paths, where each path is a list of node IDs
        """
        if self.root_id is None:
            logger.debug("[ReasoningGraph] No root node, returning empty paths")
            return []
        
        paths = []
        for leaf_id in self.leaf_ids:
            path = self.get_path(leaf_id)
            paths.append(path)
        
        logger.debug(f"[ReasoningGraph] Found {len(paths)} paths from root to leaves")
        return paths
    
    def bfs_traversal(self, start_node_id: Optional[int] = None) -> List[int]:
        """Perform breadth-first search traversal.
        
        Args:
            start_node_id: Starting node ID (defaults to root)
            
        Returns:
            List of node IDs in BFS order
        """
        if start_node_id is None:
            start_node_id = self.root_id
        
        if start_node_id is None or start_node_id not in self.nodes:
            logger.debug("[ReasoningGraph] Invalid start node for BFS")
            return []
        
        visited = []
        queue = deque([start_node_id])
        seen = {start_node_id}
        
        while queue:
            node_id = queue.popleft()
            visited.append(node_id)
            
            node = self.nodes[node_id]
            for child_id in node.children_ids:
                if child_id not in seen:
                    seen.add(child_id)
                    queue.append(child_id)
        
        logger.debug(f"[ReasoningGraph] BFS traversal from {start_node_id}: {len(visited)} nodes")
        return visited
    
    def dfs_traversal(self, start_node_id: Optional[int] = None) -> List[int]:
        """Perform depth-first search traversal.
        
        Args:
            start_node_id: Starting node ID (defaults to root)
            
        Returns:
            List of node IDs in DFS order
        """
        if start_node_id is None:
            start_node_id = self.root_id
        
        if start_node_id is None or start_node_id not in self.nodes:
            logger.debug("[ReasoningGraph] Invalid start node for DFS")
            return []
        
        visited = []
        stack = [start_node_id]
        seen = {start_node_id}
        
        while stack:
            node_id = stack.pop()
            visited.append(node_id)
            
            node = self.nodes[node_id]
            for child_id in reversed(node.children_ids):
                if child_id not in seen:
                    seen.add(child_id)
                    stack.append(child_id)
        
        logger.debug(f"[ReasoningGraph] DFS traversal from {start_node_id}: {len(visited)} nodes")
        return visited
    
    def prune_nodes(self, node_ids_to_remove: List[int]) -> int:
        """Remove multiple nodes from the graph.
        
        Args:
            node_ids_to_remove: List of node IDs to remove
            
        Returns:
            Number of nodes successfully removed
        """
        removed_count = 0
        for node_id in node_ids_to_remove:
            if self.remove_node(node_id):
                removed_count += 1
        
        logger.info(f"[ReasoningGraph] Pruned {removed_count}/{len(node_ids_to_remove)} nodes")
        return removed_count
    
    def merge_nodes(self, node_ids: List[int], merged_score: Optional[float] = None) -> Optional[int]:
        """Merge multiple nodes into a single node.
        
        This creates a new node that combines information from the specified nodes.
        The merged node will have the same parent as the first node in the list.
        
        Args:
            node_ids: List of node IDs to merge
            merged_score: Score for merged node (defaults to average of input nodes)
            
        Returns:
            ID of the newly created merged node, or None if merge failed
        """
        if len(node_ids) < 2:
            logger.warning("[ReasoningGraph] Cannot merge less than 2 nodes")
            return None
        
        # Validate all nodes exist
        nodes_to_merge = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node is None:
                logger.warning(f"[ReasoningGraph] Cannot merge: node {node_id} not found")
                return None
            nodes_to_merge.append(node)
        
        # Check all nodes have the same parent
        parent_ids = set(node.parent_id for node in nodes_to_merge)
        if len(parent_ids) > 1:
            logger.warning(f"[ReasoningGraph] Cannot merge nodes with different parents: {parent_ids}")
            return None
        
        parent_id = nodes_to_merge[0].parent_id
        
        # Calculate merged score
        if merged_score is None:
            merged_score = sum(node.score for node in nodes_to_merge) / len(nodes_to_merge)
        
        # Merge hidden states (average)
        merged_hidden_states = None
        if all(node.hidden_states is not None for node in nodes_to_merge):
            hidden_states_list = [node.hidden_states for node in nodes_to_merge]
            merged_hidden_states = torch.stack(hidden_states_list).mean(dim=0)
        
        # Use KV cache from highest-scoring node
        best_node = max(nodes_to_merge, key=lambda n: n.score)
        merged_kv_cache = best_node.kv_cache
        
        # Merge metadata
        merged_metadata = {
            'merged_from': node_ids,
            'merge_type': 'average',
        }
        for node in nodes_to_merge:
            for key, value in node.metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value
        
        # Create merged node
        merged_node_id = self.add_node(
            hidden_states=merged_hidden_states,
            kv_cache=merged_kv_cache,
            parent_id=parent_id,
            score=merged_score,
            metadata=merged_metadata
        )
        
        # Collect all children from nodes being merged
        all_children = set()
        for node in nodes_to_merge:
            all_children.update(node.children_ids)
        
        # Update children to point to merged node
        for child_id in all_children:
            child = self.get_node(child_id)
            if child is not None:
                child.parent_id = merged_node_id
                self.nodes[merged_node_id].add_child(child_id)
        
        # Remove original nodes (but not their children, already reassigned)
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Clear children to prevent recursive deletion
                node.children_ids = []
                self.remove_node(node_id)
        
        logger.info(f"[ReasoningGraph] Merged nodes {node_ids} into node {merged_node_id} with score {merged_score:.4f}")
        return merged_node_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        stats = {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_leaves': len(self.leaf_ids),
            'num_paths': len(self.get_all_paths()),
            'max_depth': 0,
            'avg_score': 0.0,
        }
        
        if self.nodes:
            if self.root_id is not None:
                root = self.get_node(self.root_id)
                if root is not None:
                    stats['max_depth'] = max(
                        node.get_depth(self) for node in self.nodes.values()
                    )
            stats['avg_score'] = sum(node.score for node in self.nodes.values()) / len(self.nodes)
        
        return stats
    
    def export_to_dot(self, output_path: str) -> None:
        """Export graph to DOT format for visualization with Graphviz.
        
        Args:
            output_path: Path to save the DOT file
        """
        try:
            with open(output_path, 'w') as f:
                f.write("digraph ReasoningGraph {\n")
                f.write("  rankdir=TB;\n")
                f.write("  node [shape=box, style=rounded];\n\n")
                
                # Write nodes
                for node_id, node in self.nodes.items():
                    color = self._get_node_color(node.score)
                    label = f"Node {node_id}\\nscore: {node.score:.3f}"
                    if node.metadata:
                        label += f"\\n{list(node.metadata.keys())}"
                    f.write(f'  {node_id} [label="{label}", fillcolor="{color}", style=filled];\n')
                
                # Write edges
                f.write("\n")
                for parent_id, child_id in self.edges:
                    f.write(f"  {parent_id} -> {child_id};\n")
                
                f.write("}\n")
            
            logger.info(f"[ReasoningGraph] Exported graph to DOT format: {output_path}")
        except Exception as e:
            logger.error(f"[ReasoningGraph] Failed to export graph to DOT: {e}")
    
    def _get_node_color(self, score: float) -> str:
        """Get color for node based on score.
        
        Args:
            score: Node score
            
        Returns:
            Color string for visualization
        """
        if score >= 0.8:
            return "#90EE90"  # Light green
        elif score >= 0.6:
            return "#FFFFE0"  # Light yellow
        elif score >= 0.4:
            return "#FFD700"  # Gold
        elif score >= 0.2:
            return "#FFA500"  # Orange
        else:
            return "#FFB6C1"  # Light red
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': self.edges,
            'root_id': self.root_id,
            'leaf_ids': list(self.leaf_ids),
            'statistics': self.get_statistics(),
        }
    
    def save_to_json(self, output_path: str) -> None:
        """Save graph structure to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        try:
            graph_dict = self.to_dict()
            with open(output_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)
            logger.info(f"[ReasoningGraph] Saved graph to JSON: {output_path}")
        except Exception as e:
            logger.error(f"[ReasoningGraph] Failed to save graph to JSON: {e}")
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)
    
    def clear(self) -> None:
        """Clear all nodes and free GPU memory from tensors.
        
        This method recursively clears all nodes in the graph and explicitly
        deletes tensors to free GPU memory.
        """
        num_nodes = len(self.nodes)
        tensor_count = 0
        
        if num_nodes == 0:
            logger.debug("[ReasoningGraph.clear] No nodes to clear")
            return
        
        logger.info(f"[ReasoningGraph.clear] Clearing {num_nodes} nodes from graph")
        
        # Explicitly delete tensors from all nodes
        for node_id, node in self.nodes.items():
            # Delete hidden states
            if node.hidden_states is not None:
                tensor_count += 1
                del node.hidden_states
                node.hidden_states = None
            
            # Delete KV cache (recursively)
            if node.kv_cache is not None:
                tensor_count += 1
                self._deep_clean_kv_cache(node.kv_cache)
                node.kv_cache = None
        
        # Clear all data structures
        self.nodes.clear()
        self.edges.clear()
        self.leaf_ids.clear()
        self.root_id = None
        
        logger.info(f"[ReasoningGraph] Cleared {num_nodes} nodes and freed {tensor_count} tensor references")
        logger.debug(f"[ReasoningGraph] Graph state after clear: nodes={len(self.nodes)}, edges={len(self.edges)}")
    
    def _deep_clean_kv_cache(self, kv_cache: Any) -> None:
        """Recursively clean KV cache structure.
        
        KV cache is a nested structure of tuples/lists containing tensors.
        This method recursively traverses and deletes all tensors.
        
        Args:
            kv_cache: KV cache structure to clean
        """
        if kv_cache is None:
            return
        
        if isinstance(kv_cache, torch.Tensor):
            del kv_cache
        elif isinstance(kv_cache, (tuple, list)):
            for item in kv_cache:
                self._deep_clean_kv_cache(item)
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        stats = self.get_statistics()
        return (f"ReasoningGraph(nodes={stats['num_nodes']}, "
                f"edges={stats['num_edges']}, "
                f"paths={stats['num_paths']}, "
                f"max_depth={stats['max_depth']})")

