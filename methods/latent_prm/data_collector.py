"""Data collector for latent PRM training data.

This module collects multi-path latent reasoning data during inference,
including latent vectors, path relationships, and final answer correctness.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Forward declaration for type hint
PathScoreBackpropagator = Any


@dataclass
class PathRecord:
    """Record for a single reasoning path.
    
    Attributes:
        path_id: Unique identifier for this path
        agent_name: Name of the agent that generated this path
        agent_idx: Index of the agent in the reasoning chain
        parent_path_id: ID of the parent path (None for root paths)
        latent_history: List of latent vectors from reasoning steps
        hidden_states: Final hidden state of this path
        score: Quality score of this path
        metadata: Additional metadata about this path
        child_path_ids: List of child path IDs
    """
    path_id: int
    agent_name: str
    agent_idx: int
    parent_path_id: Optional[int]
    latent_history: List[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    child_path_ids: List[int] = field(default_factory=list)
    prm_score: Optional[float] = None  # Score based on descendant success rate


@dataclass
class QuestionRecord:
    """Record for a complete question with all its reasoning paths.
    
    Attributes:
        question_id: Unique identifier for this question
        question: Question text
        gold_answer: Ground truth answer
        paths: List of all reasoning paths for this question
        final_answer: Final predicted answer
        is_correct: Whether the final answer is correct
        path_tree: Tree structure representation
        timestamp: When this record was created
    """
    question_id: str
    question: str
    gold_answer: str
    paths: List[PathRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    path_tree: Optional[Dict] = None
    timestamp: Optional[str] = None


class LatentPRMDataCollector:
    """Collector for latent PRM training data.
    
    This class collects reasoning paths during multi-path latent reasoning,
    maintaining the tree structure and relationships between paths.
    
    Attributes:
        enabled: Whether data collection is enabled
        current_question: Current question being processed
        path_records: Dictionary mapping path_id to PathRecord
        question_records: List of completed question records
        backpropagator: PathScoreBackpropagator for computing PRM scores
    """
    
    def __init__(
        self,
        enabled: bool = False,
        backpropagator: Optional[PathScoreBackpropagator] = None
    ):
        """Initialize the data collector.
        
        Args:
            enabled: Whether to enable data collection
            backpropagator: Optional PathScoreBackpropagator for computing PRM scores
        """
        self.enabled = enabled
        self.current_question: Optional[QuestionRecord] = None
        self.path_records: Dict[int, PathRecord] = {}
        self.question_records: List[QuestionRecord] = []
        self.backpropagator = backpropagator
        
        if self.enabled:
            logger.info("[LatentPRMDataCollector] Data collection ENABLED")
            if self.backpropagator:
                logger.info("[LatentPRMDataCollector] PathScoreBackpropagator ENABLED")
            else:
                logger.warning("[LatentPRMDataCollector] PathScoreBackpropagator NOT provided - "
                             "PRM scores will not be computed per batch")
        else:
            logger.debug("[LatentPRMDataCollector] Data collection DISABLED")
    
    def start_question(
        self,
        question_id: str,
        question: str,
        gold_answer: str
    ) -> None:
        """Start collecting data for a new question.
        
        Args:
            question_id: Unique identifier for the question
            question: Question text
            gold_answer: Ground truth answer
        """
        if not self.enabled:
            return
        
        logger.info(f"[DataCollector] Starting question {question_id}")
        logger.debug(f"[DataCollector] Question: {question[:100]}...")
        
        self.current_question = QuestionRecord(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer
        )
        self.path_records.clear()
    
    def record_path(
        self,
        path_id: int,
        agent_name: str,
        agent_idx: int,
        parent_path_id: Optional[int],
        latent_history: List[torch.Tensor],
        hidden_states: Optional[torch.Tensor],
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a reasoning path.
        
        Args:
            path_id: Unique identifier for this path
            agent_name: Name of the agent
            agent_idx: Index of the agent
            parent_path_id: ID of parent path (None for root)
            latent_history: List of latent vectors
            hidden_states: Final hidden state
            score: Path quality score
            metadata: Additional metadata
        """
        if not self.enabled or self.current_question is None:
            return
        
        logger.debug(f"[DataCollector] Recording path {path_id} "
                    f"(agent: {agent_name}, parent: {parent_path_id})")
        
        # Deep copy tensors to CPU to avoid GPU memory issues
        latent_history_cpu = [
            tensor.detach().cpu().clone() if tensor is not None else None
            for tensor in latent_history
        ]
        hidden_states_cpu = (
            hidden_states.detach().cpu().clone()
            if hidden_states is not None else None
        )
        
        # Clean metadata to remove tensor references
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if not isinstance(v, torch.Tensor):
                    clean_metadata[k] = v
        
        path_record = PathRecord(
            path_id=path_id,
            agent_name=agent_name,
            agent_idx=agent_idx,
            parent_path_id=parent_path_id,
            latent_history=latent_history_cpu,
            hidden_states=hidden_states_cpu,
            score=score,
            metadata=clean_metadata,
            child_path_ids=[]
        )
        
        self.path_records[path_id] = path_record
        
        # Update parent's child list
        if parent_path_id is not None and parent_path_id in self.path_records:
            self.path_records[parent_path_id].child_path_ids.append(path_id)
            logger.debug(f"[DataCollector] Linked path {path_id} to parent {parent_path_id}")
        
        logger.info(f"[DataCollector] Recorded path {path_id}: "
                   f"agent={agent_name}, latent_steps={len(latent_history_cpu)}, "
                   f"score={score:.4f}, parent={parent_path_id}")
    
    def finish_question(
        self,
        final_answer: str,
        is_correct: bool
    ) -> None:
        """Finish collecting data for the current question.
        
        This method now:
        1. Collects all path data
        2. IMMEDIATELY builds tree structure and computes PRM scores
        3. Prints tree to console
        4. Stores tree structure with question record
        
        Args:
            final_answer: Final predicted answer (for reporting, may be aggregated)
            is_correct: Whether the final answer is correct (for reporting, may be aggregated)
        """
        if not self.enabled or self.current_question is None:
            return
        
        logger.info(f"[DataCollector] Finishing question {self.current_question.question_id}")
        logger.info(f"[DataCollector] Final answer: {final_answer}, "
                   f"Correct: {is_correct}")
        
        # Transfer individual path correctness from metadata to path records
        # This is set by the judger in PRM data collection mode
        for path_id, path_record in self.path_records.items():
            if 'is_correct' in path_record.metadata:
                # Individual path correctness was verified by judger
                path_is_correct = path_record.metadata['is_correct']
                logger.debug(f"[DataCollector] Path {path_id} individual correctness: {path_is_correct}")
                # Store in path record for later use (not just metadata)
                # This will be used by PathScoreBackpropagator for PRM scoring
            else:
                logger.debug(f"[DataCollector] Path {path_id} has no individual correctness (normal inference mode)")
        
        # Update question record
        self.current_question.final_answer = final_answer
        self.current_question.is_correct = is_correct
        self.current_question.paths = list(self.path_records.values())
        
        # Log statistics
        num_paths = len(self.path_records)
        num_agents = len(set(p.agent_idx for p in self.path_records.values()))
        total_latent_steps = sum(
            len(p.latent_history) for p in self.path_records.values()
        )
        
        logger.info(f"[DataCollector] Question {self.current_question.question_id} stats:")
        logger.info(f"  - Total paths: {num_paths}")
        logger.info(f"  - Num agents: {num_agents}")
        logger.info(f"  - Total latent steps: {total_latent_steps}")
        logger.info(f"  - Answer correct: {is_correct}")
        
        # Log detailed path information with traversal visualization
        self._log_detailed_path_info()
        
        # ===================================================================
        # CRITICAL: Build tree and compute PRM scores IMMEDIATELY after batch
        # This is the fix for the issue described in the requirements
        # ===================================================================
        if self.backpropagator and num_paths > 0:
            logger.info("=" * 80)
            logger.info("[DataCollector] Computing PRM scores via backward propagation")
            logger.info("[DataCollector] This happens IMMEDIATELY after batch completion")
            logger.info("=" * 80)
            
            try:
                # Build tree and compute PRM scores
                tree_structure = self.backpropagator.build_and_score_tree(
                    path_records=self.current_question.paths,
                    question_id=self.current_question.question_id,
                    question_text=self.current_question.question
                )
                
                # Store tree structure in question record
                self.current_question.path_tree = tree_structure
                
                logger.info("[DataCollector] PRM scores computed and stored successfully")
                logger.debug(f"[DataCollector] Tree structure: {tree_structure['num_nodes']} nodes, "
                           f"{tree_structure['num_edges']} edges, max_depth={tree_structure['max_depth']}")
                
            except Exception as e:
                logger.error(f"[DataCollector] Failed to compute PRM scores: {e}", exc_info=True)
                logger.warning("[DataCollector] Continuing without PRM scores for this question")
                self.current_question.path_tree = None
        else:
            if not self.backpropagator:
                logger.warning("[DataCollector] No backpropagator configured - skipping PRM score computation")
            if num_paths == 0:
                logger.warning("[DataCollector] No paths collected - skipping PRM score computation")
            self.current_question.path_tree = None
        
        # Add to question records
        self.question_records.append(self.current_question)
        
        # Clear current question
        self.current_question = None
        self.path_records.clear()
    
    def _build_path_traversal(self, path_id: int) -> List[int]:
        """Build the full traversal path from root to the given path.
        
        Args:
            path_id: ID of the target path
            
        Returns:
            List of path IDs from root to target (inclusive)
        """
        traversal = []
        current_id = path_id
        
        # Traverse backwards from target to root
        visited = set()  # Prevent infinite loops
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            traversal.insert(0, current_id)
            
            # Get parent
            if current_id in self.path_records:
                current_id = self.path_records[current_id].parent_path_id
            else:
                break
        
        return traversal
    
    def _format_path_traversal(self, path_id: int) -> str:
        """Format a path traversal as a visual string.
        
        Args:
            path_id: ID of the target path
            
        Returns:
            Formatted string showing path traversal with agent info
        """
        traversal = self._build_path_traversal(path_id)
        
        if not traversal:
            return f"path_{path_id} (isolated)"
        
        # Build formatted string with agent information
        parts = []
        for pid in traversal:
            if pid in self.path_records:
                path = self.path_records[pid]
                parts.append(f"path_{pid}({path.agent_name})")
            else:
                parts.append(f"path_{pid}(unknown)")
        
        return " -> ".join(parts)
    
    def _log_detailed_path_info(self) -> None:
        """Log detailed information about all collected paths with traversal visualization."""
        if not self.path_records:
            logger.info("[DataCollector] No paths collected for this question")
            return
        
        logger.info("=" * 80)
        logger.info("[DataCollector] DETAILED PATH COLLECTION REPORT")
        logger.info("=" * 80)
        
        # Group paths by agent for organized display
        paths_by_agent = {}
        for path_id, path in self.path_records.items():
            agent_key = f"{path.agent_name} (idx={path.agent_idx})"
            if agent_key not in paths_by_agent:
                paths_by_agent[agent_key] = []
            paths_by_agent[agent_key].append(path)
        
        # Log paths organized by agent
        for agent_key in sorted(paths_by_agent.keys()):
            paths = paths_by_agent[agent_key]
            logger.info(f"\n[Agent: {agent_key}]")
            logger.info(f"  Paths generated: {len(paths)}")
            
            # Sort paths by score (descending)
            sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
            
            for i, path in enumerate(sorted_paths, 1):
                # Build traversal visualization
                traversal_str = self._format_path_traversal(path.path_id)
                
                # Log path details
                logger.info(f"  [{i}] Path ID: {path.path_id}")
                logger.info(f"      Traversal: {traversal_str}")
                logger.info(f"      Score: {path.score:.4f}")
                logger.info(f"      Latent steps: {len(path.latent_history)}")
                logger.info(f"      Parent: {path.parent_path_id if path.parent_path_id is not None else 'None (root)'}")
                logger.info(f"      Children: {len(path.child_path_ids)} paths")
                
                # Log PRM score with calculation details if available
                if path.prm_score is not None:
                    # Get child scores for calculation breakdown
                    if path.child_path_ids:
                        child_scores = []
                        for child_id in path.child_path_ids:
                            if child_id in self.path_records:
                                child_path = self.path_records[child_id]
                                if child_path.prm_score is not None:
                                    child_scores.append(child_path.prm_score)
                        
                        if child_scores:
                            sum_scores = sum(child_scores)
                            num_children = len(child_scores)
                            logger.info(f"      PRM Score: {path.prm_score:.4f} "
                                      f"(calculated from {num_children} children: "
                                      f"sum={sum_scores:.4f} / count={num_children})")
                            logger.debug(f"      Child scores: {[f'{s:.4f}' for s in child_scores]}")
                        else:
                            logger.info(f"      PRM Score: {path.prm_score:.4f} (no scored children)")
                    else:
                        # Leaf node
                        logger.info(f"      PRM Score: {path.prm_score:.4f} (leaf node)")
        
        # Log overall statistics
        logger.info("\n" + "=" * 80)
        logger.info("[DataCollector] COLLECTION SUMMARY")
        logger.info("=" * 80)
        
        # Count root paths (no parent)
        root_paths = [p for p in self.path_records.values() if p.parent_path_id is None]
        leaf_paths = [p for p in self.path_records.values() if len(p.child_path_ids) == 0]
        
        # Calculate training data samples
        # Each path with latent history contributes training samples
        total_training_samples = sum(
            len(p.latent_history) for p in self.path_records.values()
        )
        
        logger.info(f"  Total paths collected: {len(self.path_records)}")
        logger.info(f"  Root paths (no parent): {len(root_paths)}")
        logger.info(f"  Leaf paths (no children): {len(leaf_paths)}")
        logger.info(f"  Training data samples: {total_training_samples}")
        logger.info(f"  Average score: {np.mean([p.score for p in self.path_records.values()]):.4f}")
        logger.info(f"  Score range: [{min(p.score for p in self.path_records.values()):.4f}, "
                   f"{max(p.score for p in self.path_records.values()):.4f}]")
        
        # Log score distribution by quartiles
        scores = sorted([p.score for p in self.path_records.values()])
        if len(scores) >= 4:
            q1_idx = len(scores) // 4
            q2_idx = len(scores) // 2
            q3_idx = 3 * len(scores) // 4
            logger.info(f"  Score quartiles: Q1={scores[q1_idx]:.4f}, "
                       f"Q2={scores[q2_idx]:.4f}, Q3={scores[q3_idx]:.4f}")
        
        logger.info("=" * 80)
    
    def get_collected_data(self) -> List[QuestionRecord]:
        """Get all collected question records.
        
        Returns:
            List of QuestionRecord objects
        """
        if not self.enabled:
            logger.warning("[DataCollector] Data collection is disabled, returning empty list")
            return []
        
        logger.info(f"[DataCollector] Retrieved {len(self.question_records)} question records")
        return self.question_records
    
    def clear(self) -> None:
        """Clear all collected data."""
        if not self.enabled:
            return
        
        num_questions = len(self.question_records)
        logger.info(f"[DataCollector] Clearing {num_questions} question records")
        
        self.current_question = None
        self.path_records.clear()
        self.question_records.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.enabled:
            return {"enabled": False}
        
        total_questions = len(self.question_records)
        total_paths = sum(len(q.paths) for q in self.question_records)
        correct_questions = sum(1 for q in self.question_records if q.is_correct)
        
        stats = {
            "enabled": True,
            "total_questions": total_questions,
            "total_paths": total_paths,
            "correct_questions": correct_questions,
            "accuracy": correct_questions / total_questions if total_questions > 0 else 0.0,
            "avg_paths_per_question": total_paths / total_questions if total_questions > 0 else 0.0,
        }
        
        logger.debug(f"[DataCollector] Statistics: {stats}")
        return stats

