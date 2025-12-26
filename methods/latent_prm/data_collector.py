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
    """
    
    def __init__(self, enabled: bool = False):
        """Initialize the data collector.
        
        Args:
            enabled: Whether to enable data collection
        """
        self.enabled = enabled
        self.current_question: Optional[QuestionRecord] = None
        self.path_records: Dict[int, PathRecord] = {}
        self.question_records: List[QuestionRecord] = []
        
        if self.enabled:
            logger.info("[LatentPRMDataCollector] Data collection ENABLED")
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
        
        Args:
            final_answer: Final predicted answer
            is_correct: Whether the answer is correct
        """
        if not self.enabled or self.current_question is None:
            return
        
        logger.info(f"[DataCollector] Finishing question {self.current_question.question_id}")
        logger.info(f"[DataCollector] Final answer: {final_answer}, "
                   f"Correct: {is_correct}")
        
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
        
        # Add to question records
        self.question_records.append(self.current_question)
        
        # Clear current question
        self.current_question = None
        self.path_records.clear()
    
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

