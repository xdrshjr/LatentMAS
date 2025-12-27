"""Latent Process Reward Model (PRM) training data collection and fine-tuning module.

This module provides functionality for collecting multi-path latent reasoning data
and fine-tuning a Process Reward Model (PRM) for the Qwen-0.6B model.

Key components:
- DataCollector: Collects path information during multi-path reasoning
- PathTreeBuilder: Builds tree structures from collected paths (legacy, for backward compatibility)
- PathScoreBackpropagator: Computes PRM scores via backward propagation after each batch
- DataStorage: Saves collected data in appropriate formats
- PRMScorer: Scores paths based on descendant success rates (legacy)
- Dataset: Loads collected data for training
- Model: QwenLatentPRM model for fine-tuning
- Trainer: Training loop for full-parameter fine-tuning
"""

from .data_collector import LatentPRMDataCollector
from .path_tree_builder import PathTreeBuilder
from .path_score_backpropagator import PathScoreBackpropagator
from .data_storage import PRMDataStorage
from .prm_scorer import PRMScorer
from .dataset import LatentPRMDataset, create_dataloader, collate_fn
from .model import QwenLatentPRM, create_model
from .trainer import LatentPRMTrainer

__all__ = [
    # Data collection
    'LatentPRMDataCollector',
    'PathTreeBuilder',  # Legacy
    'PathScoreBackpropagator',  # New: per-batch PRM scoring
    'PRMDataStorage',
    'PRMScorer',  # Legacy
    # Training
    'LatentPRMDataset',
    'create_dataloader',
    'collate_fn',
    'QwenLatentPRM',
    'create_model',
    'LatentPRMTrainer',
]

