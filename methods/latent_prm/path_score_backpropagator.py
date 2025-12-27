"""Path score backpropagator for computing PRM scores via backward propagation.

This module implements backward propagation of scores from judger nodes to all
ancestor paths in the reasoning tree. The PRM score of each path reflects the
success rate of its descendant judger nodes.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Node in the reasoning path tree for backpropagation.
    
    Attributes:
        path_id: ID of the path this node represents
        agent_name: Name of the agent that generated this path
        agent_idx: Index of the agent in the reasoning chain
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        original_score: Original path score (e.g., consistency score)
        prm_score: PRM score based on descendant success rate
        is_leaf: Whether this is a leaf (judger) node
        depth: Depth in the tree (0 for root)
        is_correct: For leaf nodes, whether the answer is correct
    """
    path_id: int
    agent_name: str
    agent_idx: int
    parent_id: Optional[int]
    children_ids: List[int]
    original_score: float
    prm_score: Optional[float] = None
    is_leaf: bool = False
    depth: int = 0
    is_correct: Optional[bool] = None


class PathScoreBackpropagator:
    """Backpropagates PRM scores from judger nodes to ancestor paths.
    
    This class builds a tree structure from collected paths and computes
    PRM scores by propagating correctness information backward from leaf
    (judger) nodes to root nodes.
    
    The PRM score of a path represents the success rate of its descendants:
    - Leaf nodes: 1.0 if correct, 0.0 if incorrect
    - Internal nodes: average of children's PRM scores
    """
    
    def __init__(self):
        """Initialize the path score backpropagator."""
        logger.debug("[PathScoreBackpropagator] Initialized")
    
    def build_and_score_tree(
        self,
        path_records: List[Any],
        question_id: str,
        question_text: str
    ) -> Dict[str, Any]:
        """Build tree structure and compute PRM scores for all paths.
        
        This method:
        1. Builds a tree from path records
        2. Identifies leaf (judger) nodes and their correctness
        3. Backpropagates PRM scores from leaves to roots
        4. Updates path_records with computed PRM scores
        5. Returns tree structure for visualization
        
        Args:
            path_records: List of PathRecord objects
            question_id: ID of the question being processed
            question_text: Text of the question (for logging)
            
        Returns:
            Dictionary representing the tree structure with PRM scores
        """
        logger.info("=" * 80)
        logger.info(f"[PathScoreBackpropagator] Building tree for question {question_id}")
        logger.info("=" * 80)
        logger.debug(f"[PathScoreBackpropagator] Question: {question_text[:100]}...")
        logger.info(f"[PathScoreBackpropagator] Processing {len(path_records)} paths")
        
        if not path_records:
            logger.warning("[PathScoreBackpropagator] No path records provided")
            return {
                "nodes": [],
                "edges": [],
                "root_ids": [],
                "num_nodes": 0,
                "num_edges": 0,
                "max_depth": 0
            }
        
        # Step 1: Build tree nodes
        logger.info("[PathScoreBackpropagator] Step 1: Building tree nodes")
        nodes = self._build_tree_nodes(path_records)
        logger.info(f"[PathScoreBackpropagator] Created {len(nodes)} tree nodes")
        
        # Step 2: Identify root nodes and compute depths
        logger.info("[PathScoreBackpropagator] Step 2: Computing tree structure")
        root_ids = self._find_root_nodes(nodes)
        logger.info(f"[PathScoreBackpropagator] Found {len(root_ids)} root nodes: {root_ids}")
        
        self._compute_depths(nodes, root_ids)
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        logger.info(f"[PathScoreBackpropagator] Tree max depth: {max_depth}")
        
        # Step 3: Identify leaf nodes and classify them (Judger vs Pruned)
        logger.info("[PathScoreBackpropagator] Step 3: Identifying and classifying leaf nodes")
        leaf_nodes = [node for node in nodes.values() if node.is_leaf]
        logger.info(f"[PathScoreBackpropagator] Found {len(leaf_nodes)} total leaf nodes")
        
        # Classify leaf nodes into Judger leaves (complete paths) and pruned leaves (incomplete paths)
        judger_leaves = [node for node in leaf_nodes if node.agent_name.lower() == "judger"]
        pruned_leaves = [node for node in leaf_nodes if node.agent_name.lower() != "judger"]
        
        logger.info(f"[PathScoreBackpropagator] Leaf classification:")
        logger.info(f"  - Judger leaves (complete paths): {len(judger_leaves)}")
        logger.info(f"  - Pruned leaves (incomplete paths): {len(pruned_leaves)}")
        
        if pruned_leaves:
            pruned_agents = {}
            for node in pruned_leaves:
                agent = node.agent_name
                if agent not in pruned_agents:
                    pruned_agents[agent] = []
                pruned_agents[agent].append(node.path_id)
            
            logger.warning(f"[PathScoreBackpropagator] Found {len(pruned_leaves)} pruned intermediate paths:")
            for agent, path_ids in pruned_agents.items():
                logger.warning(f"  - {agent}: {len(path_ids)} paths pruned (IDs: {path_ids})")
            logger.warning("[PathScoreBackpropagator] These paths will be assigned penalty score 0.0 "
                         "as they were abandoned before reaching Judger")
        
        # Check if we have individual path correctness data for Judger leaves
        has_individual_correctness, num_correct, num_incorrect = self._check_correctness_data(
            judger_leaves, path_records
        )
        
        if has_individual_correctness:
            logger.info(f"[PathScoreBackpropagator] Individual path verification available:")
            logger.info(f"  - Correct Judger paths: {num_correct}")
            logger.info(f"  - Incorrect Judger paths: {num_incorrect}")
            logger.info(f"  - Judger path accuracy: {num_correct / len(judger_leaves) * 100:.1f}%")
        else:
            logger.warning("[PathScoreBackpropagator] No individual path correctness data found for Judger leaves")
            logger.warning("[PathScoreBackpropagator] Cannot compute accurate PRM scores")
        
        # Step 4: Backpropagate PRM scores from leaves to roots
        logger.info("[PathScoreBackpropagator] Step 4: Backpropagating PRM scores")
        self._backpropagate_scores(nodes, path_records, max_depth)
        logger.info("[PathScoreBackpropagator] PRM score backpropagation complete")
        
        # Log score statistics
        self._log_score_statistics(nodes)
        
        # Step 5: Transfer PRM scores back to PathRecords
        logger.info("[PathScoreBackpropagator] Step 5: Transferring scores to PathRecords")
        num_transferred = self._transfer_scores_to_records(nodes, path_records)
        logger.info(f"[PathScoreBackpropagator] Transferred {num_transferred} PRM scores")
        
        # Step 6: Print tree structure to console
        logger.info("[PathScoreBackpropagator] Step 6: Generating tree visualization")
        self.print_tree_structure(nodes, root_ids, question_id, question_text)
        
        # Step 7: Extract and print successful paths
        logger.info("[PathScoreBackpropagator] Step 7: Extracting and printing successful paths")
        successful_paths = self._extract_successful_paths(nodes, root_ids)
        self._print_successful_paths(nodes, successful_paths, question_id)
        
        # Step 8: Build tree structure dictionary for storage
        logger.info("[PathScoreBackpropagator] Step 8: Creating tree structure data")
        tree_structure = self._build_tree_structure_dict(nodes, root_ids, max_depth)
        logger.info(f"[PathScoreBackpropagator] Tree structure created: "
                   f"{tree_structure['num_nodes']} nodes, {tree_structure['num_edges']} edges")
        
        logger.info("=" * 80)
        logger.info(f"[PathScoreBackpropagator] Tree building complete for question {question_id}")
        logger.info("=" * 80)
        
        return tree_structure
    
    def _build_tree_nodes(self, path_records: List[Any]) -> Dict[int, TreeNode]:
        """Build tree nodes from path records.
        
        Args:
            path_records: List of PathRecord objects
            
        Returns:
            Dictionary mapping path_id to TreeNode
        """
        nodes = {}
        for path_record in path_records:
            is_leaf = (len(path_record.child_path_ids) == 0)
            node = TreeNode(
                path_id=path_record.path_id,
                agent_name=path_record.agent_name,
                agent_idx=path_record.agent_idx,
                parent_id=path_record.parent_path_id,
                children_ids=path_record.child_path_ids.copy(),
                original_score=path_record.score,
                is_leaf=is_leaf,
                depth=0  # Will be computed later
            )
            nodes[path_record.path_id] = node
            
            # Log leaf node type for debugging
            if is_leaf:
                if path_record.agent_name.lower() == "judger":
                    logger.debug(f"[PathScoreBackpropagator] Created JUDGER LEAF node for path {path_record.path_id}")
                else:
                    logger.debug(f"[PathScoreBackpropagator] Created PRUNED LEAF node for path {path_record.path_id}: "
                               f"agent={path_record.agent_name} (not Judger)")
            else:
                logger.debug(f"[PathScoreBackpropagator] Created INTERNAL node for path {path_record.path_id}: "
                            f"agent={path_record.agent_name}, {len(path_record.child_path_ids)} children")
        
        return nodes
    
    def _find_root_nodes(self, nodes: Dict[int, TreeNode]) -> List[int]:
        """Find root nodes (nodes with no parent).
        
        Args:
            nodes: Dictionary of tree nodes
            
        Returns:
            List of root node IDs
        """
        root_ids = [
            path_id for path_id, node in nodes.items()
            if node.parent_id is None
        ]
        return root_ids
    
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
        logger.debug("[PathScoreBackpropagator] Computing node depths via BFS")
        
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
                logger.debug(f"[PathScoreBackpropagator] Node {node_id} depth: {depth}")
                
                # Add children to queue
                for child_id in nodes[node_id].children_ids:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
        
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        logger.debug(f"[PathScoreBackpropagator] Depth computation complete, max_depth={max_depth}")
    
    def _check_correctness_data(
        self,
        leaf_nodes: List[TreeNode],
        path_records: List[Any]
    ) -> Tuple[bool, int, int]:
        """Check if individual path correctness data is available.
        
        Args:
            leaf_nodes: List of leaf nodes
            path_records: List of PathRecord objects
            
        Returns:
            Tuple of (has_correctness_data, num_correct, num_incorrect)
        """
        has_correctness = False
        num_correct = 0
        num_incorrect = 0
        
        # Build path_id to record mapping
        path_dict = {pr.path_id: pr for pr in path_records}
        
        for leaf_node in leaf_nodes:
            path_record = path_dict.get(leaf_node.path_id)
            if path_record and 'is_correct' in path_record.metadata:
                has_correctness = True
                is_correct = path_record.metadata['is_correct']
                leaf_node.is_correct = is_correct
                
                if is_correct:
                    num_correct += 1
                else:
                    num_incorrect += 1
                
                logger.debug(f"[PathScoreBackpropagator] Leaf node {leaf_node.path_id}: "
                           f"is_correct={is_correct}")
        
        return has_correctness, num_correct, num_incorrect
    
    def _backpropagate_scores(
        self,
        nodes: Dict[int, TreeNode],
        path_records: List[Any],
        max_depth: int
    ) -> None:
        """Backpropagate PRM scores from leaf nodes to root nodes.
        
        CRITICAL FIX: Only Judger leaf nodes get correctness-based scores.
        Pruned intermediate leaf nodes get penalty scores (0.0) as they were abandoned.
        
        For Judger leaf nodes: PRM score = 1.0 if correct, 0.0 if incorrect
        For pruned intermediate leaf nodes: PRM score = 0.0 (penalty for incomplete path)
        For internal nodes: PRM score = average of children's PRM scores
        
        Args:
            nodes: Dictionary of tree nodes
            path_records: List of PathRecord objects (for metadata lookup)
            max_depth: Maximum depth of the tree
        """
        logger.info("[PathScoreBackpropagator] Starting backward propagation")
        
        # Step 1: Assign PRM scores to leaf nodes
        logger.info("[PathScoreBackpropagator] Step 1.1: Assigning scores to Judger leaf nodes")
        judger_leaf_count = 0
        judger_with_correctness = 0
        judger_without_correctness = 0
        
        for node in nodes.values():
            if node.is_leaf and node.agent_name.lower() == "judger":
                if node.is_correct is not None:
                    # Judger node with correctness data
                    node.prm_score = 1.0 if node.is_correct else 0.0
                    judger_with_correctness += 1
                    logger.debug(f"[PathScoreBackpropagator] Judger leaf {node.path_id}: "
                               f"is_correct={node.is_correct}, prm_score={node.prm_score}")
                else:
                    # Judger node without correctness data (shouldn't happen in PRM mode)
                    node.prm_score = 0.5
                    judger_without_correctness += 1
                    logger.warning(f"[PathScoreBackpropagator] Judger leaf {node.path_id}: "
                                 f"No correctness data (unexpected!), assigned neutral score 0.5")
                judger_leaf_count += 1
        
        logger.info(f"[PathScoreBackpropagator] Assigned PRM scores to {judger_leaf_count} Judger leaves:")
        logger.info(f"  - With correctness data: {judger_with_correctness}")
        logger.info(f"  - Without correctness data: {judger_without_correctness}")
        
        # Step 1.2: Assign penalty scores to pruned intermediate leaf nodes
        logger.info("[PathScoreBackpropagator] Step 1.2: Assigning penalty scores to pruned leaf nodes")
        pruned_leaf_count = 0
        
        for node in nodes.values():
            if node.is_leaf and node.agent_name.lower() != "judger":
                # This is a pruned intermediate path that never reached Judger
                # Assign penalty score of 0.0 to indicate it was abandoned
                node.prm_score = 0.0
                pruned_leaf_count += 1
                logger.info(f"[PathScoreBackpropagator] Pruned leaf {node.path_id} "
                          f"(agent={node.agent_name}, depth={node.depth}): "
                          f"assigned penalty score 0.0 (path abandoned before Judger)")
        
        logger.info(f"[PathScoreBackpropagator] Assigned penalty scores to {pruned_leaf_count} pruned leaves")
        
        total_leaves = judger_leaf_count + pruned_leaf_count
        logger.info(f"[PathScoreBackpropagator] Total leaf nodes processed: {total_leaves} "
                   f"({judger_leaf_count} Judger + {pruned_leaf_count} pruned)")
        
        # Step 2: Propagate scores backward from leaves to roots
        # Process nodes level by level, from deepest to shallowest
        logger.debug("[PathScoreBackpropagator] Propagating scores backward")
        
        for depth in range(max_depth - 1, -1, -1):
            nodes_at_depth = [
                node for node in nodes.values()
                if node.depth == depth and not node.is_leaf
            ]
            
            if not nodes_at_depth:
                logger.debug(f"[PathScoreBackpropagator] Depth {depth}: No internal nodes")
                continue
            
            logger.debug(f"[PathScoreBackpropagator] Processing depth {depth}: "
                        f"{len(nodes_at_depth)} internal nodes")
            
            for node in nodes_at_depth:
                # Compute average PRM score of children
                child_scores = []
                for child_id in node.children_ids:
                    if child_id in nodes and nodes[child_id].prm_score is not None:
                        child_scores.append(nodes[child_id].prm_score)
                
                if child_scores:
                    node.prm_score = np.mean(child_scores)
                    logger.debug(f"[PathScoreBackpropagator] Node {node.path_id} at depth {depth}: "
                               f"prm_score={node.prm_score:.4f} "
                               f"(avg of {len(child_scores)} children: {[f'{s:.4f}' for s in child_scores]})")
                else:
                    # No children with scores, assign neutral score
                    node.prm_score = 0.5
                    logger.warning(f"[PathScoreBackpropagator] Node {node.path_id} at depth {depth}: "
                                 f"No scored children, assigned neutral score 0.5")
        
        logger.info("[PathScoreBackpropagator] Backward propagation complete")
    
    def _log_score_statistics(self, nodes: Dict[int, TreeNode]) -> None:
        """Log statistics about computed PRM scores.
        
        Args:
            nodes: Dictionary of tree nodes
        """
        prm_scores = [node.prm_score for node in nodes.values() if node.prm_score is not None]
        
        if not prm_scores:
            logger.warning("[PathScoreBackpropagator] No PRM scores computed")
            return
        
        logger.info("[PathScoreBackpropagator] PRM Score Statistics:")
        logger.info(f"  - Total scored nodes: {len(prm_scores)}")
        logger.info(f"  - Min score: {min(prm_scores):.4f}")
        logger.info(f"  - Max score: {max(prm_scores):.4f}")
        logger.info(f"  - Mean score: {np.mean(prm_scores):.4f}")
        logger.info(f"  - Std deviation: {np.std(prm_scores):.4f}")
        logger.info(f"  - Median score: {np.median(prm_scores):.4f}")
    
    def _transfer_scores_to_records(
        self,
        nodes: Dict[int, TreeNode],
        path_records: List[Any]
    ) -> int:
        """Transfer computed PRM scores back to PathRecord objects.
        
        Args:
            nodes: Dictionary of tree nodes
            path_records: List of PathRecord objects
            
        Returns:
            Number of scores transferred
        """
        num_transferred = 0
        path_dict = {pr.path_id: pr for pr in path_records}
        
        for node in nodes.values():
            if node.prm_score is not None:
                path_record = path_dict.get(node.path_id)
                if path_record:
                    path_record.prm_score = node.prm_score
                    num_transferred += 1
                    logger.debug(f"[PathScoreBackpropagator] Transferred PRM score {node.prm_score:.4f} "
                               f"to PathRecord {path_record.path_id}")
        
        return num_transferred
    
    def _build_tree_structure_dict(
        self,
        nodes: Dict[int, TreeNode],
        root_ids: List[int],
        max_depth: int
    ) -> Dict[str, Any]:
        """Build tree structure dictionary for storage.
        
        Args:
            nodes: Dictionary of tree nodes
            root_ids: List of root node IDs
            max_depth: Maximum depth of the tree
            
        Returns:
            Dictionary representing the tree structure
        """
        # Build edges list
        edges = []
        for path_id, node in nodes.items():
            for child_id in node.children_ids:
                edges.append((path_id, child_id))
        
        # Create nodes list
        nodes_list = [
            {
                "path_id": node.path_id,
                "agent_name": node.agent_name,
                "agent_idx": node.agent_idx,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "original_score": float(node.original_score),
                "prm_score": float(node.prm_score) if node.prm_score is not None else None,
                "is_leaf": node.is_leaf,
                "depth": node.depth,
                "is_correct": node.is_correct,
            }
            for node in nodes.values()
        ]
        
        return {
            "nodes": nodes_list,
            "edges": edges,
            "root_ids": root_ids,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "max_depth": max_depth,
        }
    
    def print_tree_structure(
        self,
        nodes: Dict[int, TreeNode],
        root_ids: List[int],
        question_id: str,
        question_text: str
    ) -> None:
        """Print tree structure as a multi-way tree with clear parent-child relationships.
        
        Args:
            nodes: Dictionary of tree nodes
            root_ids: List of root node IDs
            question_id: ID of the question
            question_text: Text of the question
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[PATH TREE STRUCTURE] Question {question_id}")
        logger.info("=" * 80)
        logger.info(f"Question: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
        logger.info("")
        
        if not nodes:
            logger.info("  (Empty tree)")
            logger.info("=" * 80)
            return
        
        # Print tree statistics
        max_depth = max(node.depth for node in nodes.values()) if nodes else 0
        num_leaves = sum(1 for node in nodes.values() if node.is_leaf)
        
        # Classify leaf nodes
        judger_leaves = [node for node in nodes.values() if node.is_leaf and node.agent_name.lower() == "judger"]
        pruned_leaves = [node for node in nodes.values() if node.is_leaf and node.agent_name.lower() != "judger"]
        num_correct_judger = sum(1 for node in judger_leaves if node.is_correct is True)
        num_incorrect_judger = sum(1 for node in judger_leaves if node.is_correct is False)
        
        logger.info(f"Tree Statistics:")
        logger.info(f"  - Total nodes: {len(nodes)}")
        logger.info(f"  - Root nodes: {len(root_ids)}")
        logger.info(f"  - Max depth: {max_depth}")
        logger.info(f"  - Total leaf nodes: {num_leaves}")
        logger.info(f"    * Judger leaves (complete paths): {len(judger_leaves)}")
        logger.info(f"    * Pruned leaves (incomplete paths): {len(pruned_leaves)}")
        
        if len(judger_leaves) > 0:
            logger.info(f"  - Judger path results:")
            logger.info(f"    * Correct: {num_correct_judger}/{len(judger_leaves)} ({num_correct_judger/len(judger_leaves)*100:.1f}%)")
            logger.info(f"    * Incorrect: {num_incorrect_judger}/{len(judger_leaves)} ({num_incorrect_judger/len(judger_leaves)*100:.1f}%)")
        
        if len(pruned_leaves) > 0:
            pruned_by_agent = {}
            for node in pruned_leaves:
                agent = node.agent_name
                if agent not in pruned_by_agent:
                    pruned_by_agent[agent] = 0
                pruned_by_agent[agent] += 1
            logger.info(f"  - Pruned paths by agent:")
            for agent, count in sorted(pruned_by_agent.items()):
                logger.info(f"    * {agent}: {count} paths")
        
        logger.info("")
        logger.info("Multi-Way Tree Structure:")
        logger.info("")
        
        # Print tree starting from each root using recursive tree traversal
        for root_idx, root_id in enumerate(sorted(root_ids)):
            if root_idx > 0:
                logger.info("")  # Separate different root trees
            
            if root_id in nodes:
                self._print_tree_node_recursive(
                    node=nodes[root_id],
                    nodes=nodes,
                    prefix="",
                    is_last=True,
                    is_root=True
                )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("[PATH TREE STRUCTURE] End")
        logger.info("=" * 80)
        logger.info("")
    
    def _print_tree_node_recursive(
        self,
        node: TreeNode,
        nodes: Dict[int, TreeNode],
        prefix: str,
        is_last: bool,
        is_root: bool = False
    ) -> None:
        """Recursively print a node and its descendants in tree format.
        
        Args:
            node: Current node to print
            nodes: Dictionary of all nodes
            prefix: Prefix string for indentation
            is_last: Whether this is the last child of its parent
            is_root: Whether this is a root node
        """
        # Build the connection symbol
        if is_root:
            connector = ""
        else:
            connector = "└── " if is_last else "├── "
        
        # Build node info string
        node_info = f"Path {node.path_id:3d} [{node.agent_name}]"
        
        # Add scores
        scores_info = f" orig={node.original_score:.4f}"
        if node.prm_score is not None:
            scores_info += f", prm={node.prm_score:.4f}"
        else:
            scores_info += ", prm=N/A"
        
        # Add status/correctness indicator
        status_info = ""
        if node.is_leaf:
            if node.agent_name.lower() == "judger":
                # Judger leaf: show correctness
                if node.is_correct is not None:
                    status_info = " ✓ CORRECT" if node.is_correct else " ✗ INCORRECT"
                else:
                    status_info = " ? UNKNOWN"
            else:
                # Pruned intermediate leaf
                status_info = " ✂ PRUNED"
        
        # Add parent info
        parent_info = f" parent={node.parent_id}" if node.parent_id is not None else " (ROOT)"
        
        # Add children info
        if node.children_ids:
            children_info = f" children=[{','.join(str(c) for c in node.children_ids)}]"
        else:
            children_info = " (LEAF)"
        
        # Combine all info
        full_info = f"{node_info}{scores_info}{status_info}{parent_info}{children_info}"
        
        # Print the node
        logger.info(f"{prefix}{connector}{full_info}")
        
        # Update prefix for children
        if is_root:
            new_prefix = ""
        else:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
        
        # Recursively print children
        sorted_children_ids = sorted(node.children_ids)
        for child_idx, child_id in enumerate(sorted_children_ids):
            if child_id in nodes:
                is_last_child = (child_idx == len(sorted_children_ids) - 1)
                self._print_tree_node_recursive(
                    node=nodes[child_id],
                    nodes=nodes,
                    prefix=new_prefix,
                    is_last=is_last_child,
                    is_root=False
                )
    
    def _extract_successful_paths(
        self,
        nodes: Dict[int, TreeNode],
        root_ids: List[int]
    ) -> List[List[int]]:
        """Extract all paths that lead to correct answers.
        
        This method identifies all Judger leaf nodes with correct answers
        and traces back to their root nodes to construct complete reasoning paths.
        
        Args:
            nodes: Dictionary of tree nodes
            root_ids: List of root node IDs
            
        Returns:
            List of successful paths, where each path is a list of node IDs
            from root to correct Judger leaf (ordered from root to leaf)
        """
        logger.debug("[PathScoreBackpropagator] Extracting successful paths")
        
        successful_paths = []
        
        # Find all Judger leaf nodes that are correct
        correct_judger_leaves = [
            node for node in nodes.values()
            if node.is_leaf 
            and node.agent_name.lower() == "judger"
            and node.is_correct is True
        ]
        
        logger.info(f"[PathScoreBackpropagator] Found {len(correct_judger_leaves)} correct Judger leaves")
        
        # For each correct leaf, trace back to root
        for leaf_node in correct_judger_leaves:
            path = []
            current_id = leaf_node.path_id
            
            # Trace backward from leaf to root
            visited = set()  # Prevent infinite loops
            while current_id is not None and current_id not in visited:
                visited.add(current_id)
                path.insert(0, current_id)  # Insert at beginning to maintain root-to-leaf order
                
                # Get parent
                if current_id in nodes:
                    current_id = nodes[current_id].parent_id
                else:
                    logger.warning(f"[PathScoreBackpropagator] Node {current_id} not found in nodes dict")
                    break
            
            if path:
                successful_paths.append(path)
                logger.debug(f"[PathScoreBackpropagator] Extracted successful path: {path}")
        
        logger.info(f"[PathScoreBackpropagator] Extracted {len(successful_paths)} successful paths")
        return successful_paths
    
    def _print_successful_paths(
        self,
        nodes: Dict[int, TreeNode],
        successful_paths: List[List[int]],
        question_id: str
    ) -> None:
        """Print all successful reasoning paths in arrow notation format.
        
        This method displays each successful path with clear arrow notation,
        showing the complete reasoning chain from root to correct answer.
        
        Args:
            nodes: Dictionary of tree nodes
            successful_paths: List of successful paths (each path is a list of node IDs)
            question_id: ID of the question
        """
        if not successful_paths:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[SUCCESSFUL REASONING PATHS] Question {question_id}")
            logger.info("=" * 80)
            logger.info("  No successful paths found (no correct Judger nodes)")
            logger.info("=" * 80)
            logger.info("")
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[SUCCESSFUL REASONING PATHS] Question {question_id}")
        logger.info("=" * 80)
        logger.info(f"")
        logger.info(f"Found {len(successful_paths)} successful path(s) leading to correct answers:")
        logger.info("")
        
        # Statistics for summary
        total_length = 0
        total_final_prm_score = 0.0
        
        # Print each successful path
        for path_idx, path in enumerate(successful_paths, 1):
            # Build arrow notation string
            arrow_parts = []
            for node_id in path:
                if node_id in nodes:
                    node = nodes[node_id]
                    arrow_parts.append(f"path_{node_id}[{node.agent_name}]")
                else:
                    arrow_parts.append(f"path_{node_id}[Unknown]")
            
            arrow_string = " → ".join(arrow_parts)
            
            # Get final node info
            final_node_id = path[-1]
            final_node = nodes.get(final_node_id)
            final_prm_score = final_node.prm_score if final_node and final_node.prm_score is not None else 0.0
            
            # Update statistics
            total_length += len(path)
            total_final_prm_score += final_prm_score
            
            # Print path header
            logger.info(f"Path {path_idx} (Length: {len(path)}, Final PRM Score: {final_prm_score:.4f}):")
            logger.info(f"  {arrow_string} ✓")
            logger.info("")
            
            # Print detailed node information
            logger.info(f"  Detailed Node Information:")
            for step_idx, node_id in enumerate(path, 1):
                if node_id in nodes:
                    node = nodes[node_id]
                    
                    # Node header
                    logger.info(f"    [{step_idx}] path_{node.path_id} ({node.agent_name}, depth={node.depth})")
                    
                    # Scores
                    prm_score_str = f"{node.prm_score:.4f}" if node.prm_score is not None else "N/A"
                    logger.info(f"        - Original Score: {node.original_score:.4f}, PRM Score: {prm_score_str}")
                    
                    # Parent info
                    if node.parent_id is not None:
                        logger.info(f"        - Parent: {node.parent_id}, Children: {node.children_ids}")
                    else:
                        logger.info(f"        - Parent: None (ROOT), Children: {node.children_ids}")
                    
                    # Special indicators
                    if node.is_leaf:
                        if node.agent_name.lower() == "judger":
                            if node.is_correct:
                                logger.info(f"        - Status: ✓ CORRECT (LEAF)")
                            else:
                                logger.info(f"        - Status: ✗ INCORRECT (LEAF)")
                        else:
                            logger.info(f"        - Status: ✂ PRUNED (LEAF)")
                    else:
                        logger.info(f"        - Status: Internal node")
                    
                    logger.info("")
                else:
                    logger.warning(f"    [{step_idx}] path_{node_id} (Node not found in tree)")
                    logger.info("")
            
            # Separator between paths
            if path_idx < len(successful_paths):
                logger.info("  " + "-" * 76)
                logger.info("")
        
        # Print summary statistics
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info(f"  - Total successful paths: {len(successful_paths)}")
        
        if len(successful_paths) > 0:
            avg_length = total_length / len(successful_paths)
            avg_prm_score = total_final_prm_score / len(successful_paths)
            logger.info(f"  - Average path length: {avg_length:.1f}")
            logger.info(f"  - Average final PRM score: {avg_prm_score:.4f}")
            
            # Calculate success rate (correct Judger leaves / total Judger leaves)
            total_judger_leaves = sum(
                1 for node in nodes.values()
                if node.is_leaf and node.agent_name.lower() == "judger"
            )
            if total_judger_leaves > 0:
                success_rate = len(successful_paths) / total_judger_leaves * 100
                logger.info(f"  - Success rate: {len(successful_paths)}/{total_judger_leaves} ({success_rate:.1f}%)")
        
        logger.info("=" * 80)
        logger.info("")