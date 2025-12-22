# LatentMAS Multi-Path API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Graph Structure](#graph-structure)
3. [Path Management](#path-management)
4. [Scoring Metrics](#scoring-metrics)
5. [Pruning Strategies](#pruning-strategies)
6. [Path Merging](#path-merging)
7. [Diversity Strategies](#diversity-strategies)
8. [Optimization Utilities](#optimization-utilities)
9. [Visualization](#visualization)

---

## Core Classes

### LatentMASMultiPathMethod

Main class implementing multi-path latent reasoning.

**Location:** `methods/latent_mas_multipath.py`

#### Constructor

```python
LatentMASMultiPathMethod(
    model: ModelWrapper,
    *,
    latent_steps: int = 10,
    judger_max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    generate_bs: int = 1,
    args: argparse.Namespace = None,
    num_paths: int = 5,
    enable_branching: bool = True,
    enable_merging: bool = True,
    pruning_strategy: str = "adaptive",
    scoring_weights: Optional[Dict[str, float]] = None,
    merge_threshold: float = 0.9,
    branch_threshold: float = 0.5,
    diversity_strategy: str = "hybrid"
) -> None
```

**Parameters:**
- `model` (ModelWrapper): Model wrapper for inference
- `latent_steps` (int): Number of latent thinking steps per path (default: 10)
- `judger_max_new_tokens` (int): Max tokens for judger generation (default: 256)
- `temperature` (float): Sampling temperature (default: 0.7)
- `top_p` (float): Nucleus sampling parameter (default: 0.95)
- `generate_bs` (int): Batch size for generation (default: 1)
- `args` (argparse.Namespace): Additional arguments namespace
- `num_paths` (int): Number of parallel paths (default: 5)
- `enable_branching` (bool): Enable adaptive branching (default: True)
- `enable_merging` (bool): Enable path merging (default: True)
- `pruning_strategy` (str): Pruning strategy name (default: "adaptive")
  - Choices: "topk", "adaptive", "diversity", "budget"
- `scoring_weights` (Dict[str, float], optional): Custom weights for ensemble scorer
  - Keys: "self_consistency", "perplexity", "verification", "hidden_quality"
  - Default: {"self_consistency": 0.4, "perplexity": 0.3, "verification": 0.2, "hidden_quality": 0.1}
- `merge_threshold` (float): Similarity threshold for merging (default: 0.9)
- `branch_threshold` (float): Uncertainty threshold for branching (default: 0.5)
- `diversity_strategy` (str): Diversity strategy name (default: "hybrid")
  - Choices: "temperature", "noise", "hybrid"

**Example:**
```python
from models import ModelWrapper
from methods.latent_mas_multipath import LatentMASMultiPathMethod

model = ModelWrapper(model_name="Qwen/Qwen3-14B")
method = LatentMASMultiPathMethod(
    model=model,
    num_paths=5,
    pruning_strategy="adaptive",
    enable_branching=True,
    enable_merging=True
)
```

#### Methods

##### run_batch

```python
def run_batch(
    self,
    questions: List[str],
    agent_messages: List[str],
    judger_message: str
) -> List[str]
```

Run multi-path reasoning on a batch of questions.

**Parameters:**
- `questions` (List[str]): List of input questions
- `agent_messages` (List[str]): Messages for non-judger agents
- `judger_message` (str): Message for judger agent

**Returns:**
- `List[str]`: Generated answers for each question

**Example:**
```python
questions = ["What is 2+2?", "Calculate 5*6"]
agent_messages = ["Think step by step", "Analyze carefully"]
judger_message = "Provide the final answer"

answers = method.run_batch(questions, agent_messages, judger_message)
```

##### run_batch_vllm

```python
def run_batch_vllm(
    self,
    questions: List[str],
    agent_messages: List[str],
    judger_message: str
) -> List[str]
```

Run multi-path reasoning with vLLM backend.

**Parameters:**
- Same as `run_batch`

**Returns:**
- `List[str]`: Generated answers

**Note:** Requires vLLM installation and `--use_vllm` flag.

---

## Graph Structure

### ReasoningNode

Represents a single node in the reasoning graph.

**Location:** `methods/graph_structure.py`

#### Constructor

```python
ReasoningNode(
    node_id: str,
    hidden_states: torch.Tensor,
    kv_cache: Optional[Tuple] = None,
    parent_id: Optional[str] = None,
    score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `node_id` (str): Unique identifier for the node
- `hidden_states` (torch.Tensor): Hidden state tensor at this node
- `kv_cache` (Tuple, optional): KV cache at this node
- `parent_id` (str, optional): ID of parent node
- `score` (float): Quality score for this node (default: 0.0)
- `metadata` (Dict, optional): Additional metadata

**Attributes:**
- `children_ids` (List[str]): List of child node IDs
- `depth` (int): Depth in the graph (0 for root)

#### Methods

##### add_child

```python
def add_child(self, child_id: str) -> None
```

Add a child node ID to this node.

##### update_score

```python
def update_score(self, score: float) -> None
```

Update the quality score for this node.

##### get_depth

```python
def get_depth(self) -> int
```

Get the depth of this node in the graph.

**Returns:**
- `int`: Depth (0 for root)

---

### ReasoningGraph

Manages the entire reasoning graph structure.

**Location:** `methods/graph_structure.py`

#### Constructor

```python
ReasoningGraph(root_hidden_states: torch.Tensor)
```

**Parameters:**
- `root_hidden_states` (torch.Tensor): Hidden states for root node

#### Methods

##### add_node

```python
def add_node(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Optional[Tuple] = None,
    parent_id: Optional[str] = None,
    score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

Add a new node to the graph.

**Parameters:**
- Same as ReasoningNode constructor

**Returns:**
- `str`: ID of the newly created node

##### remove_node

```python
def remove_node(self, node_id: str) -> None
```

Remove a node from the graph.

**Parameters:**
- `node_id` (str): ID of node to remove

##### get_path

```python
def get_path(self, node_id: str) -> List[ReasoningNode]
```

Get the path from root to specified node.

**Parameters:**
- `node_id` (str): Target node ID

**Returns:**
- `List[ReasoningNode]`: List of nodes from root to target

##### get_all_paths

```python
def get_all_paths(self) -> List[List[ReasoningNode]]
```

Get all paths from root to leaf nodes.

**Returns:**
- `List[List[ReasoningNode]]`: List of paths (each path is a list of nodes)

##### prune_nodes

```python
def prune_nodes(self, node_ids_to_keep: List[str]) -> None
```

Prune all nodes except those specified.

**Parameters:**
- `node_ids_to_keep` (List[str]): IDs of nodes to keep

##### merge_nodes

```python
def merge_nodes(
    self,
    node_ids: List[str],
    merge_strategy: str = "weighted"
) -> str
```

Merge multiple nodes into one.

**Parameters:**
- `node_ids` (List[str]): IDs of nodes to merge
- `merge_strategy` (str): Strategy for merging ("weighted", "average", "selective")

**Returns:**
- `str`: ID of the merged node

##### export_dot

```python
def export_dot(self, output_file: str) -> None
```

Export graph to DOT format for visualization.

**Parameters:**
- `output_file` (str): Path to output .dot file

---

## Path Management

### PathState

Dataclass representing the state of a reasoning path.

**Location:** `methods/path_manager.py`

#### Attributes

```python
@dataclass
class PathState:
    path_id: str
    latent_history: List[torch.Tensor]
    hidden_states: torch.Tensor
    kv_cache: Optional[Tuple]
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Fields:**
- `path_id` (str): Unique identifier for the path
- `latent_history` (List[torch.Tensor]): History of latent states
- `hidden_states` (torch.Tensor): Current hidden states
- `kv_cache` (Tuple, optional): Current KV cache
- `score` (float): Quality score for this path
- `metadata` (Dict): Additional metadata (parent_id, branch_point, etc.)

---

### PathManager

Manages path lifecycle and operations.

**Location:** `methods/path_manager.py`

#### Constructor

```python
PathManager()
```

#### Methods

##### create_path

```python
def create_path(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Optional[Tuple] = None,
    parent_id: Optional[str] = None
) -> PathState
```

Create a new path.

**Parameters:**
- `hidden_states` (torch.Tensor): Initial hidden states
- `kv_cache` (Tuple, optional): Initial KV cache
- `parent_id` (str, optional): Parent path ID (for branching)

**Returns:**
- `PathState`: Newly created path

##### branch_path

```python
def branch_path(
    self,
    parent_path: PathState,
    num_branches: int = 2
) -> List[PathState]
```

Create multiple branches from a parent path.

**Parameters:**
- `parent_path` (PathState): Path to branch from
- `num_branches` (int): Number of branches to create

**Returns:**
- `List[PathState]`: List of branched paths

##### merge_paths

```python
def merge_paths(
    self,
    paths: List[PathState],
    strategy: str = "weighted"
) -> PathState
```

Merge multiple paths into one.

**Parameters:**
- `paths` (List[PathState]): Paths to merge
- `strategy` (str): Merge strategy ("weighted", "average", "selective")

**Returns:**
- `PathState`: Merged path

##### get_active_paths

```python
def get_active_paths(self) -> List[PathState]
```

Get all currently active paths.

**Returns:**
- `List[PathState]`: List of active paths

##### prune_paths

```python
def prune_paths(
    self,
    paths: List[PathState],
    keep_count: int
) -> List[PathState]
```

Prune paths to keep only top-k.

**Parameters:**
- `paths` (List[PathState]): Paths to prune
- `keep_count` (int): Number of paths to keep

**Returns:**
- `List[PathState]`: Pruned list of paths

---

## Scoring Metrics

### EnsembleScorer

Combines multiple scoring metrics with configurable weights.

**Location:** `methods/scoring_metrics.py`

#### Constructor

```python
EnsembleScorer(
    model: ModelWrapper,
    weights: Optional[Dict[str, float]] = None
)
```

**Parameters:**
- `model` (ModelWrapper): Model for scoring
- `weights` (Dict[str, float], optional): Weights for each metric
  - Default: {"self_consistency": 0.4, "perplexity": 0.3, "verification": 0.2, "hidden_quality": 0.1}

#### Methods

##### score

```python
def score(
    self,
    path: PathState,
    question: str = "",
    answer: str = ""
) -> float
```

Compute ensemble score for a path.

**Parameters:**
- `path` (PathState): Path to score
- `question` (str): Original question (for verification)
- `answer` (str): Generated answer (for verification)

**Returns:**
- `float`: Score in range [0, 1]

##### get_score_breakdown

```python
def get_score_breakdown(
    self,
    path: PathState,
    question: str = "",
    answer: str = ""
) -> Dict[str, float]
```

Get detailed breakdown of scores from each metric.

**Returns:**
- `Dict[str, float]`: Dictionary with scores from each metric

---

### SelfConsistencyScorer

Scores paths based on self-consistency of generated answers.

**Location:** `methods/scoring_metrics.py`

#### Constructor

```python
SelfConsistencyScorer(
    model: ModelWrapper,
    num_samples: int = 5
)
```

**Parameters:**
- `model` (ModelWrapper): Model for generation
- `num_samples` (int): Number of samples to generate (default: 5)

#### Methods

##### score

```python
def score(self, path: PathState) -> float
```

Compute self-consistency score.

**Returns:**
- `float`: Frequency of most common answer (range: [0, 1])

---

### PerplexityScorer

Scores paths based on perplexity of latent steps.

**Location:** `methods/scoring_metrics.py`

#### Constructor

```python
PerplexityScorer(model: ModelWrapper)
```

#### Methods

##### score

```python
def score(self, path: PathState) -> float
```

Compute perplexity-based score.

**Returns:**
- `float`: Normalized confidence score (lower perplexity = higher score)

---

### VerificationScorer

Scores paths using model-based verification.

**Location:** `methods/scoring_metrics.py`

#### Constructor

```python
VerificationScorer(model: ModelWrapper)
```

#### Methods

##### score

```python
def score(
    self,
    path: PathState,
    question: str,
    answer: str
) -> float
```

Compute verification score.

**Parameters:**
- `path` (PathState): Path to verify
- `question` (str): Original question
- `answer` (str): Generated answer

**Returns:**
- `float`: Verification confidence (range: [0, 1])

---

### HiddenStateQualityScorer

Scores paths based on hidden state properties.

**Location:** `methods/scoring_metrics.py`

#### Constructor

```python
HiddenStateQualityScorer()
```

#### Methods

##### score

```python
def score(self, path: PathState) -> float
```

Compute hidden state quality score.

**Returns:**
- `float`: Quality score based on norm stability and smoothness

---

## Pruning Strategies

### PruningStrategy (Base Class)

Abstract base class for pruning strategies.

**Location:** `methods/pruning_strategies.py`

#### Methods

##### prune

```python
def prune(
    self,
    paths: List[PathState],
    target_count: int,
    current_step: int = 0,
    total_steps: int = 1
) -> List[PathState]
```

Prune paths to target count.

**Parameters:**
- `paths` (List[PathState]): Paths to prune
- `target_count` (int): Number of paths to keep
- `current_step` (int): Current reasoning step
- `total_steps` (int): Total reasoning steps

**Returns:**
- `List[PathState]`: Pruned paths

---

### TopKPruning

Keep top-k paths by score.

**Location:** `methods/pruning_strategies.py`

#### Constructor

```python
TopKPruning()
```

**Example:**
```python
from methods.pruning_strategies import TopKPruning

pruner = TopKPruning()
pruned_paths = pruner.prune(paths, target_count=5)
```

---

### AdaptivePruning

Adjust pruning rate based on progress.

**Location:** `methods/pruning_strategies.py`

#### Constructor

```python
AdaptivePruning(
    min_keep_ratio: float = 0.3,
    max_keep_ratio: float = 0.8
)
```

**Parameters:**
- `min_keep_ratio` (float): Minimum ratio of paths to keep (early steps)
- `max_keep_ratio` (float): Maximum ratio of paths to keep (late steps)

**Formula:**
```
keep_ratio = min_keep_ratio + (max_keep_ratio - min_keep_ratio) * (current_step / total_steps)
```

---

### DiversityAwarePruning

Balance score and diversity when pruning.

**Location:** `methods/pruning_strategies.py`

#### Constructor

```python
DiversityAwarePruning(min_distance: float = 0.3)
```

**Parameters:**
- `min_distance` (float): Minimum cosine distance between kept paths

---

### BudgetBasedPruning

Prune based on computational budget.

**Location:** `methods/pruning_strategies.py`

#### Constructor

```python
BudgetBasedPruning(
    max_budget: float,
    budget_type: str = "tokens"
)
```

**Parameters:**
- `max_budget` (float): Maximum computational budget
- `budget_type` (str): Type of budget ("tokens", "flops", "memory")

---

## Path Merging

### PathMerger

Handles path similarity detection and merging.

**Location:** `methods/path_merging.py`

#### Constructor

```python
PathMerger(similarity_threshold: float = 0.9)
```

**Parameters:**
- `similarity_threshold` (float): Cosine similarity threshold for merging

#### Methods

##### find_similar_paths

```python
def find_similar_paths(
    self,
    paths: List[PathState]
) -> List[Tuple[int, int, float]]
```

Find pairs of similar paths.

**Parameters:**
- `paths` (List[PathState]): Paths to analyze

**Returns:**
- `List[Tuple[int, int, float]]`: List of (idx1, idx2, similarity) tuples

##### merge_paths

```python
def merge_paths(
    self,
    paths: List[PathState],
    strategy: str = "weighted"
) -> PathState
```

Merge multiple paths.

**Parameters:**
- `paths` (List[PathState]): Paths to merge
- `strategy` (str): Merge strategy

**Returns:**
- `PathState`: Merged path

##### merge_all_similar

```python
def merge_all_similar(
    self,
    paths: List[PathState]
) -> List[PathState]
```

Automatically merge all similar paths.

**Returns:**
- `List[PathState]`: Paths after merging

---

### Merge Strategies

#### AverageMergeStrategy

Simple averaging of hidden states and KV caches.

**Location:** `methods/path_merging.py`

```python
strategy = AverageMergeStrategy()
merged_path = strategy.merge(paths)
```

#### WeightedMergeStrategy

Score-weighted merging.

**Location:** `methods/path_merging.py`

```python
strategy = WeightedMergeStrategy()
merged_path = strategy.merge(paths)
```

#### SelectiveMergeStrategy

Keep best components from each path.

**Location:** `methods/path_merging.py`

```python
strategy = SelectiveMergeStrategy()
merged_path = strategy.merge(paths)
```

---

## Diversity Strategies

### DiversityStrategy (Base Class)

Abstract base class for diversity strategies.

**Location:** `methods/diversity_strategies.py`

#### Methods

##### apply

```python
def apply(
    self,
    hidden_states: torch.Tensor,
    path_index: int,
    total_paths: int
) -> torch.Tensor
```

Apply diversity transformation to hidden states.

**Parameters:**
- `hidden_states` (torch.Tensor): Input hidden states
- `path_index` (int): Index of current path
- `total_paths` (int): Total number of paths

**Returns:**
- `torch.Tensor`: Transformed hidden states

---

### TemperatureDiversityStrategy

Use different temperatures for diversity.

**Location:** `methods/diversity_strategies.py`

#### Constructor

```python
TemperatureDiversityStrategy(
    min_temp: float = 0.7,
    max_temp: float = 1.3
)
```

---

### NoiseDiversityStrategy

Add Gaussian noise to hidden states.

**Location:** `methods/diversity_strategies.py`

#### Constructor

```python
NoiseDiversityStrategy(noise_scale: float = 0.1)
```

---

### HybridDiversityStrategy

Combine temperature and noise strategies.

**Location:** `methods/diversity_strategies.py`

#### Constructor

```python
HybridDiversityStrategy(
    min_temp: float = 0.7,
    max_temp: float = 1.3,
    noise_scale: float = 0.1
)
```

---

## Optimization Utilities

### KVCacheOptimizer

Optimize KV-cache usage for multi-path.

**Location:** `methods/cache_optimization.py`

#### Methods

##### detect_shared_prefix

```python
def detect_shared_prefix(
    self,
    kv_caches: List[Tuple]
) -> Tuple
```

Detect shared prefix across KV caches.

##### reuse_cache

```python
def reuse_cache(
    self,
    base_cache: Tuple,
    new_tokens: torch.Tensor
) -> Tuple
```

Reuse KV cache for new tokens.

---

### BatchOptimizer

Optimize batch processing for multiple paths.

**Location:** `methods/batch_optimization.py`

#### Methods

##### group_similar_paths

```python
def group_similar_paths(
    self,
    paths: List[PathState]
) -> List[List[PathState]]
```

Group similar paths for batch processing.

##### optimize_batch

```python
def optimize_batch(
    self,
    paths: List[PathState]
) -> torch.Tensor
```

Optimize tensor operations for batch.

---

### Checkpointer

Save and restore experiment state.

**Location:** `methods/checkpointing.py`

#### Methods

##### save_checkpoint

```python
def save_checkpoint(
    self,
    graph: ReasoningGraph,
    paths: List[PathState],
    metadata: Dict,
    checkpoint_path: str
) -> None
```

Save checkpoint to disk.

##### load_checkpoint

```python
def load_checkpoint(
    self,
    checkpoint_path: str
) -> Tuple[ReasoningGraph, List[PathState], Dict]
```

Load checkpoint from disk.

---

## Visualization

### visualize_reasoning_graph

Visualize reasoning graph structure.

**Location:** `visualization/graph_viz.py`

```python
def visualize_reasoning_graph(
    graph: ReasoningGraph,
    output_file: str,
    format: str = "html",
    color_by: str = "score"
) -> None
```

**Parameters:**
- `graph` (ReasoningGraph): Graph to visualize
- `output_file` (str): Output file path
- `format` (str): Output format ("html", "png", "svg")
- `color_by` (str): Node coloring ("score", "depth", "path")

---

### analyze_path_behavior

Analyze path statistics and behavior.

**Location:** `visualization/path_analysis.py`

```python
def analyze_path_behavior(
    paths: List[PathState],
    output_file: str
) -> Dict[str, Any]
```

**Parameters:**
- `paths` (List[PathState]): Paths to analyze
- `output_file` (str): Output report file

**Returns:**
- `Dict[str, Any]`: Analysis statistics

---

### create_dashboard

Create real-time monitoring dashboard.

**Location:** `visualization/dashboard.py`

```python
def create_dashboard(
    log_file: str,
    output_file: str,
    refresh_interval: int = 5
) -> None
```

**Parameters:**
- `log_file` (str): Path to experiment log file
- `output_file` (str): Output HTML dashboard file
- `refresh_interval` (int): Refresh interval in seconds

---

## Configuration

### Load Configuration

```python
from config import load_config, MultiPathConfig

# Load from file
config = load_config("config.json")

# Create programmatically
config = MultiPathConfig(
    num_paths=5,
    pruning_strategy="adaptive",
    enable_branching=True,
    enable_merging=True
)
```

### Configuration Presets

```python
from config import get_preset_config, list_presets

# List available presets
presets = list_presets()

# Load preset
config = get_preset_config("balanced")
```

---

## Error Handling

All methods may raise the following exceptions:

- `ValueError`: Invalid parameter values
- `RuntimeError`: Execution errors (OOM, model errors)
- `FileNotFoundError`: Missing configuration or checkpoint files
- `KeyError`: Missing required configuration keys

**Example:**
```python
try:
    method = LatentMASMultiPathMethod(
        model=model,
        num_paths=5,
        pruning_strategy="invalid"  # Will raise ValueError
    )
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
```

---

## Logging

All modules use Python's standard logging framework:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)

# For detailed debugging
logging.basicConfig(level=logging.DEBUG)
```

**Log Levels:**
- `DEBUG`: Detailed information (tensor shapes, intermediate values)
- `INFO`: General information (method calls, results)
- `WARNING`: Unexpected situations (fallbacks, deprecated features)
- `ERROR`: Errors preventing specific operations

---

## Type Hints

All functions include complete type hints:

```python
from typing import List, Dict, Optional, Tuple, Any
import torch

def example_function(
    paths: List[PathState],
    scores: Dict[str, float],
    threshold: Optional[float] = None
) -> Tuple[List[PathState], Dict[str, Any]]:
    ...
```

---

## Version Compatibility

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **vLLM**: 0.2.0+ (optional)

---

## See Also

- [Multi-Path User Guide](multipath-guide.md) - Comprehensive usage guide
- [Experiment Guide](experiment-guide.md) - Guide for running experiments
- [GitHub Repository](https://github.com/Gen-Verse/LatentMAS) - Source code and issues

