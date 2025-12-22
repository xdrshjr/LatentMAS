from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


# Import path merging components
from .path_merging import (
    PathSimilarityDetector,
    MergeStrategy,
    AverageMergeStrategy,
    WeightedMergeStrategy,
    SelectiveMergeStrategy,
    PathMerger,
    MergeCandidate,
    MergeStatistics,
)

# Import multi-path method
from .latent_mas_multipath import LatentMASMultiPathMethod


__all__ = [
    "Agent",
    "default_agents",
    "PathSimilarityDetector",
    "MergeStrategy",
    "AverageMergeStrategy",
    "WeightedMergeStrategy",
    "SelectiveMergeStrategy",
    "PathMerger",
    "MergeCandidate",
    "MergeStatistics",
    "LatentMASMultiPathMethod",
]
