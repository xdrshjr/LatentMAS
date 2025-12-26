"""Latent Process Reward Model (PRM) training data collection module.

This module provides functionality for collecting multi-path latent reasoning data
to train a Process Reward Model (PRM) for the Qwen-0.6B model.

Key components:
- DataCollector: Collects path information during multi-path reasoning
- PathTreeBuilder: Builds tree structures from collected paths
- DataStorage: Saves collected data in appropriate formats
- PRMScorer: Scores paths based on descendant success rates
"""

from .data_collector import LatentPRMDataCollector
from .path_tree_builder import PathTreeBuilder
from .data_storage import PRMDataStorage
from .prm_scorer import PRMScorer

__all__ = [
    'LatentPRMDataCollector',
    'PathTreeBuilder',
    'PRMDataStorage',
    'PRMScorer',
]

