"""Path tree builder for organizing reasoning paths into tree structures.

This module builds tree structures from collected paths and computes
PRM scores based on descendant success rates.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Node in the reasoning path tree.
    
    Attributes:
        path_id: ID of the path this node represents
        agent_name: Name of the agent
        agent_idx: Index of the agent
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        score: Original path score
        prm_score: PRM score based on descendant success
        is_leaf: Whether this is a leaf node
        depth: Depth in the tree (0 for root)
    """
    path_id: int
    agent_name: str
    agent_idx: int
    parent_id: Optional[int]
    children_ids: List[int]
    score: float
    prm_score: Optional[float] = None
    is_leaf: bool = False
    depth: int = 0


class PathTreeBuilder:
    """Builder for constructing reasoning path trees.
    
    This class takes collected path records and builds a tree structure,
    computing PRM scores based on descendant success rates.
    """
    
    def __init__(self):
        """Initialize the path tree builder."""
        logger.debug("[PathTreeBuilder] Initialized")
    
    def build_tree(
        self,
        path_records: List[Any],
        is_correct: bool
    ) -> Dict[str, Any]:
        """Build a tree structure from path records.
        
        Args:
            path_records: List of PathRecord objects
            is_correct: Whether the final answer is correct
            
        Returns:
            Dictionary representing the tree structure with PRM scores
        """
        logger.info(f"[PathTreeBuilder] Building tree from {len(path_records)} paths")
        logger.info(f"[PathTreeBuilder] Final answer correct: {is_correct}")
        
        if not path_records:
            logger.warning("[PathTreeBuilder] No path records provided")
            return {"nodes": [], "edges": [], "root_ids": []}
        
        # Create tree nodes
        nodes = {}
        for path_record in path_records:
            node = TreeNode(
                path_id=path_record.path_id,
                agent_name=path_record.agent_name,
                agent_idx=path_record.agent_idx,
                parent_id=path_record.parent_path_id,
                children_ids=path_record.child_path_ids.copy(),
                score=path_record.score,
                is_leaf=(len(path_record.child_path_ids) == 0),
                depth=0  # Will be computed later
            )
            nodes[path_record.path_id] = node
        
        logger.debug(f"[PathTreeBuilder] Created {len(nodes)} tree nodes")
        
        # Find root nodes and compute depths
        root_ids = [
            path_id for path_id, node in nodes.items()
            if node.parent_id is None
        ]
        logger.info(f"[PathTreeBuilder] Found {len(root_ids)} root nodes: {root_ids}")
        
        # Compute depths using BFS
        self._compute_depths(nodes, root_ids)
        
        # Compute PRM scores based on descendant success
        self._compute_prm_scores(nodes, is_correct)
        
        # Build edges list
        edges = []
        for path_id, node in nodes.items():
            for child_id in node.children_ids:
                edges.append((path_id, child_id))
        
        logger.info(f"[PathTreeBuilder] Built tree with {len(nodes)} nodes, "
                   f"{len(edges)} edges, {len(root_ids)} roots")
        
        # Create tree structure dictionary
        tree_structure = {
            "nodes": [
                {
                    "path_id": node.path_id,
                    "agent_name": node.agent_name,
                    "agent_idx": node.agent_idx,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "score": float(node.score),
                    "prm_score": float(node.prm_score) if node.prm_score is not None else None,
                    "is_leaf": node.is_leaf,
                    "depth": node.depth,
                }
                for node in nodes.values()
            ],
            "edges": edges,
            "root_ids": root_ids,
            "is_correct": is_correct,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "max_depth": max(node.depth for node in nodes.values()) if nodes else 0,
        }
        
        return tree_structure
    
    def _compute_depths(
        self,
        nodes: Dict[int, TreeNode],
        root_ids: List[int]
    ) -> None:
        """Compute depth for each node using BFS.
        
        Args:
            nodes: Dictionary of tree nodes
            root_ids: List of root node IDs
        """
        logger.debug("[PathTreeBuilder] Computing node depths")
        
        # BFS to compute depths
        queue = [(root_id, 0) for root_id in root_ids]
        visited = set()
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            if node_id in nodes:
                nodes[node_id].depth = depth
                
                # Add children to queue
                for child_id in nodes[node_id].children_ids:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
        
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        logger.debug(f"[PathTreeBuilder] Computed depths, max_depth={max_depth}")
    
    def _compute_prm_scores(
        self,
        nodes: Dict[int, TreeNode],
        is_correct: bool
    ) -> None:
        """Compute PRM scores based on descendant success rates.
        
        For leaf nodes: score = 1.0 if final answer is correct, else 0.0
        For internal nodes: score = average of children's PRM scores
        
        Args:
            nodes: Dictionary of tree nodes
            is_correct: Whether the final answer is correct
        """
        logger.info("[PathTreeBuilder] Computing PRM scores")
        logger.debug(f"[PathTreeBuilder] Final answer correct: {is_correct}")
        
        # Find all leaf nodes
        leaf_nodes = [node for node in nodes.values() if node.is_leaf]
        logger.info(f"[PathTreeBuilder] Found {len(leaf_nodes)} leaf nodes")
        
        # Assign scores to leaf nodes based on final correctness
        leaf_score = 1.0 if is_correct else 0.0
        for leaf_node in leaf_nodes:
            leaf_node.prm_score = leaf_score
            logger.debug(f"[PathTreeBuilder] Leaf node {leaf_node.path_id}: "
                        f"prm_score={leaf_score}")
        
        # Propagate scores backward from leaves to roots
        # Use topological sort (reverse BFS by depth)
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        
        for depth in range(max_depth, -1, -1):
            nodes_at_depth = [
                node for node in nodes.values()
                if node.depth == depth and not node.is_leaf
            ]
            
            logger.debug(f"[PathTreeBuilder] Processing depth {depth}: "
                        f"{len(nodes_at_depth)} internal nodes")
            
            for node in nodes_at_depth:
                # Compute average PRM score of children
                child_scores = []
                for child_id in node.children_ids:
                    if child_id in nodes and nodes[child_id].prm_score is not None:
                        child_scores.append(nodes[child_id].prm_score)
                
                if child_scores:
                    node.prm_score = np.mean(child_scores)
                    logger.debug(f"[PathTreeBuilder] Node {node.path_id} at depth {depth}: "
                                f"prm_score={node.prm_score:.4f} "
                                f"(avg of {len(child_scores)} children)")
                else:
                    # No children with scores, assign neutral score
                    node.prm_score = 0.5
                    logger.warning(f"[PathTreeBuilder] Node {node.path_id} has no scored children, "
                                 f"assigning neutral score 0.5")
        
        # Log PRM score statistics
        prm_scores = [node.prm_score for node in nodes.values() if node.prm_score is not None]
        if prm_scores:
            logger.info(f"[PathTreeBuilder] PRM score statistics:")
            logger.info(f"  - Min: {min(prm_scores):.4f}")
            logger.info(f"  - Max: {max(prm_scores):.4f}")
            logger.info(f"  - Mean: {np.mean(prm_scores):.4f}")
            logger.info(f"  - Std: {np.std(prm_scores):.4f}")
    
    def get_path_to_root(
        self,
        nodes: Dict[int, TreeNode],
        leaf_id: int
    ) -> List[int]:
        """Get the path from a leaf node to root.
        
        Args:
            nodes: Dictionary of tree nodes
            leaf_id: ID of the leaf node
            
        Returns:
            List of node IDs from leaf to root
        """
        path = []
        current_id = leaf_id
        
        while current_id is not None:
            path.append(current_id)
            if current_id in nodes:
                current_id = nodes[current_id].parent_id
            else:
                break
        
        logger.debug(f"[PathTreeBuilder] Path from leaf {leaf_id} to root: {path}")
        return path
    
    def get_subtree_nodes(
        self,
        nodes: Dict[int, TreeNode],
        root_id: int
    ) -> Set[int]:
        """Get all nodes in the subtree rooted at a given node.
        
        Args:
            nodes: Dictionary of tree nodes
            root_id: ID of the root node of the subtree
            
        Returns:
            Set of node IDs in the subtree
        """
        subtree = set()
        queue = [root_id]
        
        while queue:
            node_id = queue.pop(0)
            if node_id in subtree:
                continue
            
            subtree.add(node_id)
            
            if node_id in nodes:
                queue.extend(nodes[node_id].children_ids)
        
        logger.debug(f"[PathTreeBuilder] Subtree rooted at {root_id}: "
                    f"{len(subtree)} nodes")
        return subtree

