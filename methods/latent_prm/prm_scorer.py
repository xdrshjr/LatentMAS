"""PRM scorer for computing process reward model scores.

This module provides scoring functionality specifically designed for
training Process Reward Models (PRMs) on latent reasoning paths.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class PRMScorer:
    """Scorer for computing PRM training labels.
    
    This scorer computes scores for reasoning paths based on whether
    the final answer is correct, propagating scores backward through
    the reasoning tree.
    
    Attributes:
        correct_score: Score assigned to paths leading to correct answers
        incorrect_score: Score assigned to paths leading to incorrect answers
    """
    
    def __init__(
        self,
        correct_score: float = 1.0,
        incorrect_score: float = 0.0
    ):
        """Initialize the PRM scorer.
        
        Args:
            correct_score: Score for correct paths (default: 1.0)
            incorrect_score: Score for incorrect paths (default: 0.0)
        """
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        
        logger.info(f"[PRMScorer] Initialized with correct_score={correct_score}, "
                   f"incorrect_score={incorrect_score}")
    
    def score_paths(
        self,
        path_records: List[Any],
        is_correct: bool
    ) -> None:
        """Score all paths based on final answer correctness.
        
        This method updates the prm_score field of each PathRecord in place.
        
        Scoring strategy:
        - Leaf nodes: correct_score if answer is correct, else incorrect_score
        - Internal nodes: average of children's PRM scores
        
        Args:
            path_records: List of PathRecord objects
            is_correct: Whether the final answer is correct
        """
        logger.info(f"[PRMScorer] Scoring {len(path_records)} paths")
        logger.info(f"[PRMScorer] Final answer correct: {is_correct}")
        
        if not path_records:
            logger.warning("[PRMScorer] No path records to score")
            return
        
        # Build path lookup dictionary
        paths_dict = {path.path_id: path for path in path_records}
        
        # Find leaf nodes (nodes with no children)
        leaf_paths = [path for path in path_records if not path.child_path_ids]
        internal_paths = [path for path in path_records if path.child_path_ids]
        
        logger.info(f"[PRMScorer] Found {len(leaf_paths)} leaf paths, "
                   f"{len(internal_paths)} internal paths")
        
        # Score leaf nodes based on final correctness
        leaf_score = self.correct_score if is_correct else self.incorrect_score
        for leaf_path in leaf_paths:
            leaf_path.prm_score = leaf_score
            logger.debug(f"[PRMScorer] Leaf path {leaf_path.path_id}: "
                        f"prm_score={leaf_score}")
        
        # Score internal nodes by propagating from children
        # Use topological sort based on agent_idx (reverse order)
        max_agent_idx = max(path.agent_idx for path in path_records)
        
        for agent_idx in range(max_agent_idx, -1, -1):
            paths_at_agent = [
                path for path in internal_paths
                if path.agent_idx == agent_idx
            ]
            
            logger.debug(f"[PRMScorer] Processing agent_idx {agent_idx}: "
                        f"{len(paths_at_agent)} internal paths")
            
            for path in paths_at_agent:
                # Compute average score of children
                child_scores = []
                for child_id in path.child_path_ids:
                    if child_id in paths_dict:
                        child_path = paths_dict[child_id]
                        if child_path.prm_score is not None:
                            child_scores.append(child_path.prm_score)
                
                if child_scores:
                    sum_scores = sum(child_scores)
                    num_children = len(child_scores)
                    path.prm_score = np.mean(child_scores)
                    logger.debug(f"[PRMScorer] Internal path {path.path_id} "
                               f"(agent {agent_idx}): prm_score={path.prm_score:.4f} "
                               f"(calculated from {num_children} children: "
                               f"sum={sum_scores:.4f} / count={num_children})")
                    logger.debug(f"[PRMScorer] Path {path.path_id} child scores: "
                                f"{[f'{s:.4f}' for s in child_scores]}")
                else:
                    # No children with scores, assign neutral score
                    path.prm_score = 0.5
                    logger.warning(f"[PRMScorer] Path {path.path_id} has no scored children, "
                                 f"assigning neutral score 0.5")
        
        # Log scoring statistics
        scored_paths = [path for path in path_records if path.prm_score is not None]
        if scored_paths:
            prm_scores = [path.prm_score for path in scored_paths]
            logger.info(f"[PRMScorer] Scoring complete:")
            logger.info(f"  - Scored paths: {len(scored_paths)}/{len(path_records)}")
            logger.info(f"  - Min score: {min(prm_scores):.4f}")
            logger.info(f"  - Max score: {max(prm_scores):.4f}")
            logger.info(f"  - Mean score: {np.mean(prm_scores):.4f}")
            logger.info(f"  - Std score: {np.std(prm_scores):.4f}")
        else:
            logger.warning("[PRMScorer] No paths were scored")
    
    def compute_path_quality_score(
        self,
        path_record: Any,
        include_original_score: bool = True,
        weight_original: float = 0.3
    ) -> float:
        """Compute a combined quality score for a path.
        
        Combines the PRM score with the original path score.
        
        Args:
            path_record: PathRecord object
            include_original_score: Whether to include original score
            weight_original: Weight for original score (0-1)
            
        Returns:
            Combined quality score
        """
        if path_record.prm_score is None:
            logger.warning(f"[PRMScorer] Path {path_record.path_id} has no PRM score")
            return path_record.score if include_original_score else 0.5
        
        if not include_original_score:
            return path_record.prm_score
        
        # Weighted combination
        combined_score = (
            weight_original * path_record.score +
            (1 - weight_original) * path_record.prm_score
        )
        
        logger.debug(f"[PRMScorer] Path {path_record.path_id} combined score: "
                    f"{combined_score:.4f} (original={path_record.score:.4f}, "
                    f"prm={path_record.prm_score:.4f})")
        
        return combined_score
    
    def get_score_distribution(
        self,
        path_records: List[Any]
    ) -> Dict[str, Any]:
        """Get distribution statistics of PRM scores.
        
        Args:
            path_records: List of PathRecord objects
            
        Returns:
            Dictionary with distribution statistics
        """
        logger.debug(f"[PRMScorer] Computing score distribution for {len(path_records)} paths")
        
        prm_scores = [
            path.prm_score for path in path_records
            if path.prm_score is not None
        ]
        
        if not prm_scores:
            logger.warning("[PRMScorer] No PRM scores available")
            return {
                "num_scored": 0,
                "num_total": len(path_records),
            }
        
        distribution = {
            "num_scored": len(prm_scores),
            "num_total": len(path_records),
            "min": float(np.min(prm_scores)),
            "max": float(np.max(prm_scores)),
            "mean": float(np.mean(prm_scores)),
            "median": float(np.median(prm_scores)),
            "std": float(np.std(prm_scores)),
            "quartiles": {
                "q25": float(np.percentile(prm_scores, 25)),
                "q50": float(np.percentile(prm_scores, 50)),
                "q75": float(np.percentile(prm_scores, 75)),
            }
        }
        
        logger.debug(f"[PRMScorer] Score distribution: mean={distribution['mean']:.4f}, "
                    f"std={distribution['std']:.4f}")
        
        return distribution
    
    def identify_critical_paths(
        self,
        path_records: List[Any],
        threshold: float = 0.7
    ) -> List[int]:
        """Identify high-quality paths above a threshold.
        
        Args:
            path_records: List of PathRecord objects
            threshold: Score threshold for critical paths
            
        Returns:
            List of path IDs for critical paths
        """
        logger.info(f"[PRMScorer] Identifying critical paths with threshold={threshold}")
        
        critical_path_ids = [
            path.path_id for path in path_records
            if path.prm_score is not None and path.prm_score >= threshold
        ]
        
        logger.info(f"[PRMScorer] Found {len(critical_path_ids)} critical paths "
                   f"out of {len(path_records)} total paths")
        logger.debug(f"[PRMScorer] Critical path IDs: {critical_path_ids}")
        
        return critical_path_ids
    
    def identify_failure_paths(
        self,
        path_records: List[Any],
        threshold: float = 0.3
    ) -> List[int]:
        """Identify low-quality paths below a threshold.
        
        Args:
            path_records: List of PathRecord objects
            threshold: Score threshold for failure paths
            
        Returns:
            List of path IDs for failure paths
        """
        logger.info(f"[PRMScorer] Identifying failure paths with threshold={threshold}")
        
        failure_path_ids = [
            path.path_id for path in path_records
            if path.prm_score is not None and path.prm_score <= threshold
        ]
        
        logger.info(f"[PRMScorer] Found {len(failure_path_ids)} failure paths "
                   f"out of {len(path_records)} total paths")
        logger.debug(f"[PRMScorer] Failure path IDs: {failure_path_ids}")
        
        return failure_path_ids

