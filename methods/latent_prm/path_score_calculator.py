"""Path score calculator for computing PRM scores immediately after batch completion.

This module provides a lightweight calculator that computes PRM scores for all paths
in a question immediately after the Judger agent verifies the answers. This ensures
that PRM scores are computed while the path data is still fresh in memory.

The calculator uses the PathScoreBackpropagator to perform the actual score computation,
but is designed to be called per-batch rather than after all batches are complete.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .path_score_backpropagator import PathScoreBackpropagator
from .data_collector import PathRecord

logger = logging.getLogger(__name__)


class PathScoreCalculator:
    """Calculator for computing PRM scores immediately after batch completion.
    
    This class is designed to be called after each batch is processed by the Judger agent.
    It takes the path records with verified answer correctness and computes PRM scores
    by backpropagating the correctness information through the path tree.
    
    Key differences from PathTreeBuilder:
    - Lightweight: Only computes scores, doesn't build full tree structure
    - Per-batch: Called after each batch, not after all batches
    - In-place: Updates PathRecord objects directly with PRM scores
    
    Attributes:
        backpropagator: PathScoreBackpropagator instance for score computation
        correct_score: Score assigned to correct leaf paths
        incorrect_score: Score assigned to incorrect leaf paths
        aggregation_method: Method for aggregating child scores
    """
    
    def __init__(
        self,
        correct_score: float = 1.0,
        incorrect_score: float = 0.0,
        aggregation_method: str = "proportion"
    ):
        """Initialize the path score calculator.
        
        Args:
            correct_score: Score assigned to correct leaf paths (default: 1.0)
            incorrect_score: Score assigned to incorrect leaf paths (default: 0.0)
            aggregation_method: Method for aggregating child scores
                - "proportion": Direct proportion of correct leaf descendants (RECOMMENDED)
                - "mean": Simple average of child scores (not recommended for unbalanced trees)
                - "max": Maximum child score (optimistic)
                - "weighted_mean": Weighted average by descendant count (equivalent to proportion)
        """
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        self.aggregation_method = aggregation_method
        
        # Create backpropagator for score computation
        self.backpropagator = PathScoreBackpropagator(
            correct_score=correct_score,
            incorrect_score=incorrect_score,
            aggregation_method=aggregation_method
        )
        
        logger.info("[PathScoreCalculator] Initialized with:")
        logger.info(f"  - correct_score: {correct_score}")
        logger.info(f"  - incorrect_score: {incorrect_score}")
        logger.info(f"  - aggregation_method: {aggregation_method}")
        
        if aggregation_method == "proportion":
            logger.info("[PathScoreCalculator] Using 'proportion' method: score = correct_leaves / total_leaves")
            logger.info("[PathScoreCalculator] This correctly handles unbalanced tree structures")
    
    def calculate_and_update_scores(
        self,
        path_records: List[PathRecord],
        question_id: str
    ) -> Dict[int, float]:
        """Calculate PRM scores and update PathRecord objects in-place.
        
        This is the main entry point called after each batch is processed.
        It computes PRM scores based on the verified answer correctness
        and updates the PathRecord objects directly.
        
        Args:
            path_records: List of PathRecord objects with verified correctness
            question_id: ID of the question being processed
        
        Returns:
            Dictionary mapping path_id to computed PRM score
        """
        logger.info("=" * 80)
        logger.info(f"[PathScoreCalculator] Computing PRM scores for question {question_id}")
        logger.info("=" * 80)
        logger.info(f"[PathScoreCalculator] Total paths: {len(path_records)}")
        
        if not path_records:
            logger.warning(f"[PathScoreCalculator] No path records for question {question_id}")
            return {}
        
        # Step 1: Verify that leaf paths have individual correctness
        leaf_paths = [p for p in path_records if len(p.child_path_ids) == 0]
        has_correctness = any('is_correct' in p.metadata for p in leaf_paths)
        
        logger.info(f"[PathScoreCalculator] Leaf paths: {len(leaf_paths)}/{len(path_records)}")
        logger.info(f"[PathScoreCalculator] Has individual correctness: {has_correctness}")
        
        if not has_correctness:
            logger.warning(f"[PathScoreCalculator] No individual correctness found for question {question_id}")
            logger.warning("[PathScoreCalculator] Cannot compute meaningful PRM scores without answer verification")
            return {}
        
        # Step 2: Use backpropagator to compute PRM scores
        logger.info("[PathScoreCalculator] Starting score backpropagation")
        prm_scores = self.backpropagator.backpropagate_scores(
            path_records=path_records,
            use_individual_correctness=True
        )
        
        logger.info(f"[PathScoreCalculator] Computed {len(prm_scores)} PRM scores")
        
        # Step 3: Update PathRecord objects in-place
        logger.info("[PathScoreCalculator] Updating PathRecord objects with PRM scores")
        num_updated = 0
        
        # Build a lookup for descendant counts from backpropagator
        path_lookup = self.backpropagator._build_path_lookup(path_records)
        
        for path_record in path_records:
            if path_record.path_id in prm_scores:
                old_score = path_record.prm_score
                new_score = prm_scores[path_record.path_id]
                path_record.prm_score = new_score
                num_updated += 1
                
                # Get descendant counts for detailed logging
                path_info = path_lookup.get(path_record.path_id)
                if path_info:
                    num_correct = path_info.num_correct_descendants
                    num_total = path_info.num_total_descendants
                    success_rate = (num_correct / num_total * 100) if num_total > 0 else 0.0
                    
                    if old_score is not None and abs(old_score - new_score) > 1e-6:
                        logger.debug(f"[PathScoreCalculator] Path {path_record.path_id} ({path_record.agent_name}): "
                                   f"updated prm_score from {old_score:.4f} to {new_score:.4f}, "
                                   f"descendants={num_correct}/{num_total} ({success_rate:.1f}%)")
                    else:
                        logger.debug(f"[PathScoreCalculator] Path {path_record.path_id} ({path_record.agent_name}): "
                                   f"set prm_score to {new_score:.4f}, "
                                   f"descendants={num_correct}/{num_total} ({success_rate:.1f}%)")
                else:
                    logger.debug(f"[PathScoreCalculator] Path {path_record.path_id}: "
                               f"set prm_score to {new_score:.4f}")
        
        logger.info(f"[PathScoreCalculator] Updated {num_updated}/{len(path_records)} PathRecords")
        
        # Step 4: Log statistics
        self._log_score_statistics(prm_scores, path_records, question_id)
        
        logger.info("=" * 80)
        logger.info(f"[PathScoreCalculator] PRM score calculation complete for question {question_id}")
        logger.info("=" * 80)
        
        return prm_scores
    
    def _log_score_statistics(
        self,
        prm_scores: Dict[int, float],
        path_records: List[PathRecord],
        question_id: str
    ) -> None:
        """Log detailed statistics about computed scores.
        
        Args:
            prm_scores: Dictionary of computed PRM scores
            path_records: List of PathRecord objects
            question_id: ID of the question
        """
        if not prm_scores:
            logger.warning(f"[PathScoreCalculator] No scores to log for question {question_id}")
            return
        
        logger.info("=" * 80)
        logger.info(f"[PathScoreCalculator] SCORE STATISTICS - Question {question_id}")
        logger.info("=" * 80)
        
        scores_array = np.array(list(prm_scores.values()))
        
        logger.info(f"[PathScoreCalculator] Overall statistics:")
        logger.info(f"  - Total paths scored: {len(prm_scores)}")
        logger.info(f"  - Score range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
        logger.info(f"  - Mean score: {scores_array.mean():.4f}")
        logger.info(f"  - Median score: {np.median(scores_array):.4f}")
        logger.info(f"  - Std deviation: {scores_array.std():.4f}")
        
        # Count correct vs incorrect leaf paths
        leaf_paths = [p for p in path_records if len(p.child_path_ids) == 0]
        num_correct_leaves = sum(
            1 for p in leaf_paths 
            if p.metadata.get('is_correct', False)
        )
        num_incorrect_leaves = len(leaf_paths) - num_correct_leaves
        
        logger.info(f"[PathScoreCalculator] Leaf path statistics:")
        logger.info(f"  - Total leaf paths: {len(leaf_paths)}")
        logger.info(f"  - Correct leaf paths: {num_correct_leaves}")
        logger.info(f"  - Incorrect leaf paths: {num_incorrect_leaves}")
        if len(leaf_paths) > 0:
            accuracy = num_correct_leaves / len(leaf_paths) * 100
            logger.info(f"  - Leaf accuracy: {accuracy:.1f}%")
        
        # Score distribution by bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(scores_array, bins=bins)
        logger.info(f"[PathScoreCalculator] Score distribution:")
        for i in range(len(bins) - 1):
            count = hist[i]
            percentage = count / len(prm_scores) * 100 if len(prm_scores) > 0 else 0
            logger.info(f"  - [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} paths ({percentage:.1f}%)")
        
        # Statistics by agent with descendant information
        agent_scores = {}
        agent_descendants = {}
        path_lookup = self.backpropagator._build_path_lookup(path_records)
        
        for path_record in path_records:
            if path_record.path_id in prm_scores:
                agent_name = path_record.agent_name
                if agent_name not in agent_scores:
                    agent_scores[agent_name] = []
                    agent_descendants[agent_name] = []
                
                agent_scores[agent_name].append(prm_scores[path_record.path_id])
                
                # Get descendant info
                path_info = path_lookup.get(path_record.path_id)
                if path_info:
                    agent_descendants[agent_name].append(
                        (path_info.num_correct_descendants, path_info.num_total_descendants)
                    )
        
        logger.info(f"[PathScoreCalculator] Statistics by agent:")
        for agent_name in sorted(agent_scores.keys()):
            scores = np.array(agent_scores[agent_name])
            descendants = agent_descendants.get(agent_name, [])
            
            # Calculate average success rate for this agent
            if descendants:
                total_correct = sum(d[0] for d in descendants)
                total_leaves = sum(d[1] for d in descendants)
                avg_success_rate = (total_correct / total_leaves * 100) if total_leaves > 0 else 0.0
                
                logger.info(f"  - {agent_name}: {len(scores)} paths, "
                           f"mean_score={scores.mean():.4f}, std={scores.std():.4f}, "
                           f"range=[{scores.min():.4f}, {scores.max():.4f}], "
                           f"avg_success_rate={avg_success_rate:.1f}% "
                           f"({total_correct}/{total_leaves} correct descendants)")
            else:
                logger.info(f"  - {agent_name}: {len(scores)} paths, "
                           f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
                           f"range=[{scores.min():.4f}, {scores.max():.4f}]")
        
        # Root path statistics
        root_paths = [p for p in path_records if p.parent_path_id is None]
        root_scores = [
            prm_scores[p.path_id] for p in root_paths 
            if p.path_id in prm_scores
        ]
        if root_scores:
            logger.info(f"[PathScoreCalculator] Root path statistics:")
            logger.info(f"  - Number of roots: {len(root_scores)}")
            logger.info(f"  - Mean root score: {np.mean(root_scores):.4f}")
            logger.info(f"  - Root scores: {[f'{s:.4f}' for s in root_scores]}")
        
        logger.info("=" * 80)
    
    def validate_path_structure(
        self,
        path_records: List[PathRecord]
    ) -> bool:
        """Validate that the path structure is consistent.
        
        This method checks for common issues like:
        - Missing parent references
        - Circular dependencies
        - Orphaned paths
        
        Args:
            path_records: List of PathRecord objects to validate
        
        Returns:
            True if structure is valid, False otherwise
        """
        logger.debug("[PathScoreCalculator] Validating path structure")
        
        if not path_records:
            logger.warning("[PathScoreCalculator] No paths to validate")
            return True
        
        # Build path lookup
        path_lookup = {p.path_id: p for p in path_records}
        
        # Check for missing parents
        missing_parents = []
        for path in path_records:
            if path.parent_path_id is not None:
                if path.parent_path_id not in path_lookup:
                    missing_parents.append((path.path_id, path.parent_path_id))
        
        if missing_parents:
            logger.warning(f"[PathScoreCalculator] Found {len(missing_parents)} paths with missing parents:")
            for path_id, parent_id in missing_parents[:5]:  # Log first 5
                logger.warning(f"  - Path {path_id} references missing parent {parent_id}")
            return False
        
        # Check for circular dependencies (simple check: no path should be its own ancestor)
        def has_cycle(path_id: int, visited: set) -> bool:
            if path_id in visited:
                return True
            if path_id not in path_lookup:
                return False
            
            visited.add(path_id)
            parent_id = path_lookup[path_id].parent_path_id
            if parent_id is None:
                return False
            
            return has_cycle(parent_id, visited)
        
        cycles = []
        for path in path_records:
            if has_cycle(path.path_id, set()):
                cycles.append(path.path_id)
        
        if cycles:
            logger.error(f"[PathScoreCalculator] Found {len(cycles)} paths with circular dependencies:")
            for path_id in cycles[:5]:  # Log first 5
                logger.error(f"  - Path {path_id} has circular dependency")
            return False
        
        # Check for orphaned paths (paths with parent but parent doesn't list them as child)
        orphaned = []
        for path in path_records:
            if path.parent_path_id is not None:
                parent = path_lookup.get(path.parent_path_id)
                if parent and path.path_id not in parent.child_path_ids:
                    orphaned.append(path.path_id)
        
        if orphaned:
            logger.warning(f"[PathScoreCalculator] Found {len(orphaned)} orphaned paths (not in parent's child list):")
            for path_id in orphaned[:5]:  # Log first 5
                logger.warning(f"  - Path {path_id} is orphaned")
            # This is a warning, not an error - we can still compute scores
        
        logger.debug("[PathScoreCalculator] Path structure validation complete")
        return True

