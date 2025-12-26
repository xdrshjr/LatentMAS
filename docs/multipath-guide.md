# LatentMAS Multi-Path User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Configuration Guide](#configuration-guide)
5. [Components Deep Dive](#components-deep-dive)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Introduction

### What is Multi-Path Reasoning?

The **LatentMAS Multi-Path** method extends the original LatentMAS framework with **Graph-Structured Multi-Path Latent Reasoning (GMLR)**. Instead of following a single reasoning trajectory, the method explores multiple parallel paths through the latent space, evaluates them using training-free metrics, and intelligently prunes low-quality paths while merging similar ones.

### Key Benefits

- **Improved Accuracy**: Exploring multiple reasoning paths increases the likelihood of finding the correct solution
- **Robustness**: Less sensitive to suboptimal initial reasoning directions
- **Training-Free**: All evaluation metrics and strategies require no additional training
- **Flexible**: Configurable trade-offs between accuracy and computational cost

### When to Use Multi-Path

✅ **Use multi-path when:**
- Working on complex reasoning tasks (math, code, multi-step problems)
- Accuracy is more important than speed
- You have sufficient GPU memory (typically 2-3× single-path requirements)
- Problems have multiple valid reasoning approaches

❌ **Consider single-path when:**
- Working on simple classification or QA tasks
- Speed is critical
- GPU memory is limited
- Problems have straightforward, linear reasoning

---

## Core Concepts

### 1. Reasoning Graph

The multi-path method constructs a **reasoning graph** where:
- **Nodes** represent reasoning states (hidden states + KV cache)
- **Edges** represent latent thinking transitions
- **Paths** are sequences of nodes from root to leaf

```
                    Root (Initial Prompt)
                    /      |      \
                Path-1   Path-2   Path-3
                  |        |        |
              [Latent]  [Latent] [Latent]
                  |        |        |
              Agent-1   Agent-1  Agent-1
                   \      |      /
                    \     |     /
                   [Merge Similar]
                         |
                      Judger
                         |
                   Final Answer
```

### 2. Path Diversity

To explore different reasoning strategies, paths are diversified using:

- **Temperature Diversity**: Different sampling temperatures (0.7, 1.0, 1.3)
- **Noise Injection**: Adding Gaussian noise to hidden states
- **Initialization Diversity**: Different starting points for latent thinking

### 3. Training-Free Evaluation

Paths are scored using an ensemble of metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Self-Consistency** | 40% | Frequency of most common answer across samples |
| **Perplexity** | 30% | Model confidence in the reasoning path |
| **Verification** | 20% | Model's self-verification of reasoning correctness |
| **Hidden State Quality** | 10% | Stability and smoothness of latent representations |

### 4. Intelligent Pruning

Low-quality paths are removed using various strategies:

- **Top-K**: Keep only the top-k highest-scoring paths
- **Adaptive**: Adjust pruning rate based on reasoning progress
- **Diversity-Aware**: Balance score and diversity to avoid redundancy
- **Budget-Based**: Prune to stay within computational constraints

### 5. Path Merging

Similar paths are merged to reduce redundancy:
- Detect similarity using cosine similarity of hidden states
- Merge KV caches and hidden states (weighted by scores)
- Update graph structure to reflect merged paths

---

## Getting Started

### Basic Usage

```bash
# Simple multi-path with default settings
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --prompt sequential \
  --max_samples -1 \
  --max_new_tokens 2048
```

### Using Configuration Presets

```bash
# List available presets
python run.py --list_presets

# Use balanced preset (recommended for most tasks)
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --config_preset balanced \
  --max_samples -1 \
  --max_new_tokens 2048
```

### Available Presets

| Preset | Paths | Pruning | Use Case |
|--------|-------|---------|----------|
| **conservative** | 3 | Aggressive | Fast, memory-efficient |
| **balanced** | 5 | Adaptive | Good accuracy/speed trade-off |
| **aggressive** | 10 | Diversity | Maximum exploration |
| **fast** | 3 | Top-K | Fastest multi-path option |
| **quality** | 10 | Adaptive | Highest accuracy |

---

## Configuration Guide

### Command-Line Arguments

#### Core Multi-Path Arguments

```bash
--num_paths N              # Number of parallel paths (default: 5)
--pruning_strategy STRATEGY # topk, adaptive, diversity, budget
--diversity_strategy STRATEGY # temperature, noise, hybrid
--enable_branching         # Enable adaptive branching
--enable_merging           # Enable path merging
--merge_threshold FLOAT    # Similarity threshold for merging (default: 0.9)
--branch_threshold FLOAT   # Uncertainty threshold for branching (default: 0.5)
```

#### Configuration Files

Create a JSON or YAML configuration file:

**config_multipath.json:**
```json
{
  "num_paths": 5,
  "enable_branching": true,
  "enable_merging": true,
  "pruning_strategy": "adaptive",
  "diversity_strategy": "hybrid",
  "merge_threshold": 0.9,
  "branch_threshold": 0.5,
  "scoring_weights": {
    "self_consistency": 0.4,
    "perplexity": 0.3,
    "verification": 0.2,
    "hidden_quality": 0.1
  }
}
```

**Usage:**
```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --config config_multipath.json \
  --max_samples -1 \
  --max_new_tokens 2048
```

### Recommended Configurations by Task Type

#### Math Reasoning (GSM8K, MATH, AIME)
```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --num_paths 7 \
  --pruning_strategy diversity \
  --diversity_strategy hybrid \
  --enable_branching \
  --enable_merging \
  --merge_threshold 0.85 \
  --latent_steps 15 \
  --max_new_tokens 2048
```

#### Code Generation (HumanEval+, MBPP+)
```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task humaneval_plus \
  --num_paths 5 \
  --pruning_strategy adaptive \
  --diversity_strategy temperature \
  --enable_merging \
  --merge_threshold 0.9 \
  --latent_steps 10 \
  --max_new_tokens 2048
```

#### Complex Reasoning (GPQA, ARC-Challenge)
```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gpqa \
  --num_paths 10 \
  --pruning_strategy diversity \
  --diversity_strategy hybrid \
  --enable_branching \
  --enable_merging \
  --branch_threshold 0.4 \
  --latent_steps 20 \
  --max_new_tokens 2048
```

---

## Components Deep Dive

### Diversity Strategies

#### Temperature Diversity
Uses different sampling temperatures for each path:
- Path 0: Temperature 0.7 (conservative)
- Path 1: Temperature 1.0 (balanced)
- Path 2: Temperature 1.3 (exploratory)

**When to use:** Code generation, math problems with multiple solution approaches

#### Noise Diversity
Adds Gaussian noise to hidden states before realignment:
- Noise scale: 0.1 × hidden state norm
- Applied before each latent thinking step

**When to use:** Problems requiring creative exploration, avoiding local optima

#### Hybrid Diversity (Recommended)
Combines temperature and noise strategies:
- Uses temperature for initial diversity
- Adds noise for continued exploration
- Balances exploitation and exploration

**When to use:** Most tasks, especially complex multi-step reasoning

### Pruning Strategies

#### Top-K Pruning
Keeps the top-k paths by score at each step.

**Parameters:**
- `k`: Number of paths to keep (set via `--num_paths`)

**Pros:** Simple, predictable
**Cons:** May prune too aggressively early on

**When to use:** When you need consistent computational cost

#### Adaptive Pruning (Recommended)
Adjusts pruning rate based on reasoning progress:
- Early steps: Keep 30% of paths (aggressive)
- Middle steps: Keep 50% of paths
- Late steps: Keep 80% of paths (conservative)

**Formula:** `keep_ratio = 0.3 + 0.5 × (current_step / total_steps)`

**Pros:** Balances exploration and exploitation
**Cons:** Less predictable computational cost

**When to use:** Most tasks, especially when unsure about difficulty

#### Diversity-Aware Pruning
Balances score and diversity:
1. Always keep highest-scoring path
2. For remaining slots, select paths that are both high-scoring and diverse
3. Ensure minimum cosine distance between kept paths

**Parameters:**
- `min_distance`: Minimum cosine distance (default: 0.3)

**Pros:** Maintains diverse reasoning approaches
**Cons:** May keep lower-scoring but diverse paths

**When to use:** Complex problems with multiple valid approaches

#### Budget-Based Pruning
Prunes to stay within computational budget:
- Tracks tokens, FLOPs, or memory usage
- Prunes paths with lowest score-per-cost ratio

**Parameters:**
- `max_budget`: Maximum computational budget

**Pros:** Guarantees computational constraints
**Cons:** Requires careful budget estimation

**When to use:** Production systems with strict resource limits

### Scoring Metrics

#### Self-Consistency Score (40% weight)
Generates multiple answers from the same path and measures agreement.

**How it works:**
1. Generate 5 answers from the path with temperature 0.7
2. Normalize and compare answers
3. Score = frequency of most common answer / total samples

**Interpretation:**
- Score 1.0: All samples agree (high confidence)
- Score 0.6: Majority agree (moderate confidence)
- Score 0.2: High disagreement (low confidence)

#### Perplexity Score (30% weight)
Measures model confidence in the reasoning path.

**How it works:**
1. Track log-probabilities during latent generation
2. Calculate perplexity for each step
3. Normalize by path length
4. Return negative perplexity as score (lower perplexity = higher score)

**Interpretation:**
- High score: Model is confident in this reasoning path
- Low score: Model is uncertain about this path

#### Verification Score (20% weight)
Uses the model to verify its own reasoning.

**How it works:**
1. Generate verification prompt: "Is this reasoning correct?"
2. Parse model's confidence level (Low/Medium/High)
3. Map to numerical score (0.3/0.6/1.0)

**Interpretation:**
- High score: Model verifies reasoning is correct
- Low score: Model finds issues with reasoning

#### Hidden State Quality Score (10% weight)
Measures stability and smoothness of latent representations.

**How it works:**
1. Calculate norm stability across latent steps
2. Measure cosine similarity progression (smoothness)
3. Combine sub-metrics into overall quality score

**Interpretation:**
- High score: Stable, smooth reasoning trajectory
- Low score: Erratic or unstable reasoning

### Path Merging

#### When Paths are Merged
Paths are merged when:
1. Cosine similarity of hidden states > `merge_threshold` (default: 0.9)
2. Both paths are in the same reasoning stage
3. Merging would reduce computational cost

#### How Merging Works
1. **Detect similar paths** using cosine similarity
2. **Merge KV caches** (concatenation or weighted average)
3. **Merge hidden states** (weighted by path scores)
4. **Update graph structure** (remove merged nodes, create new node)
5. **Preserve metadata** from both paths

#### Merge Strategies

**Average Merge:**
- Simple averaging of hidden states and KV caches
- Fast, but may lose information

**Weighted Merge:**
- Weight by path scores
- Preserves information from higher-quality paths

**Selective Merge:**
- Keep best components from each path
- Most sophisticated, but slower

---

## Performance Tuning

### Memory Optimization

#### Reduce Number of Paths
```bash
--num_paths 3  # Instead of default 5
```
**Impact:** 40% memory reduction, ~10-15% accuracy decrease

#### Aggressive Pruning
```bash
--pruning_strategy topk --num_paths 3
```
**Impact:** Consistent low memory usage, may prune good paths early

#### Enable Path Merging
```bash
--enable_merging --merge_threshold 0.85
```
**Impact:** Reduces redundant paths, 20-30% memory savings

### Speed Optimization

#### Use Fast Preset
```bash
--config_preset fast
```
**Impact:** 2-3× faster than default, ~5-10% accuracy decrease

#### Reduce Latent Steps
```bash
--latent_steps 5  # Instead of default 10
```
**Impact:** Proportional speedup, may reduce reasoning quality

#### Disable Branching
```bash
# Don't use --enable_branching
```
**Impact:** Simpler graph structure, faster execution

### Accuracy Optimization

#### Use Quality Preset
```bash
--config_preset quality
```
**Impact:** Best accuracy, 3-4× slower and more memory

#### Increase Number of Paths
```bash
--num_paths 10
```
**Impact:** Better exploration, 2× memory and time

#### Use Diversity-Aware Pruning
```bash
--pruning_strategy diversity
```
**Impact:** Maintains diverse reasoning approaches, better accuracy

#### Enable Branching
```bash
--enable_branching --branch_threshold 0.4
```
**Impact:** Adaptive exploration at uncertain points, better accuracy

### Balanced Configuration (Recommended)

```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --num_paths 5 \
  --pruning_strategy adaptive \
  --diversity_strategy hybrid \
  --enable_merging \
  --merge_threshold 0.9 \
  --latent_steps 10 \
  --max_new_tokens 2048
```

**Expected Performance:**
- Accuracy: +10-15% over single-path
- Memory: 2-2.5× single-path
- Time: 2-3× single-path

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:** CUDA out of memory error during multi-path execution

**Solutions:**
```bash
# Reduce number of paths
--num_paths 3

# Use aggressive pruning
--pruning_strategy topk

# Enable merging
--enable_merging --merge_threshold 0.85

# Reduce batch size
--generate_bs 1

# Use gradient checkpointing (if available)
--gradient_checkpointing
```

#### 2. Very Slow Execution

**Symptoms:** Multi-path takes 5-10× longer than expected

**Solutions:**
```bash
# Use fast preset
--config_preset fast

# Reduce latent steps
--latent_steps 5

# Disable branching
# (don't use --enable_branching)

# Use top-k pruning
--pruning_strategy topk --num_paths 3

# Check if vLLM is properly configured
--use_vllm --enable_prefix_caching
```

#### 3. No Accuracy Improvement

**Symptoms:** Multi-path accuracy similar to single-path

**Solutions:**
```bash
# Increase number of paths
--num_paths 7

# Use diversity-aware pruning
--pruning_strategy diversity

# Enable branching
--enable_branching --branch_threshold 0.5

# Increase latent steps
--latent_steps 15

# Try different diversity strategy
--diversity_strategy hybrid
```

#### 4. Paths Converge Too Quickly

**Symptoms:** All paths become very similar after first agent

**Solutions:**
```bash
# Lower merge threshold
--merge_threshold 0.8

# Use noise diversity
--diversity_strategy noise

# Disable merging temporarily
# (don't use --enable_merging)

# Increase branch threshold
--branch_threshold 0.6
```

#### 5. Inconsistent Results Across Runs

**Symptoms:** Large variance in accuracy across multiple runs

**Solutions:**
```bash
# Set random seed
--seed 42

# Increase number of paths for stability
--num_paths 7

# Use lower temperature
--temperature 0.6

# Use adaptive pruning
--pruning_strategy adaptive
```

### Debugging Tips

#### Enable Verbose Logging
```bash
# Set logging level to DEBUG
export LOGLEVEL=DEBUG
python run.py ...
```

#### Visualize Reasoning Graphs
```python
from visualization.graph_viz import visualize_reasoning_graph

# After running experiment
visualize_reasoning_graph(
    graph_data="path/to/graph.pkl",
    output_file="reasoning_graph.html"
)
```

#### Analyze Path Statistics
```python
from visualization.path_analysis import analyze_path_behavior

# Generate path analysis report
analyze_path_behavior(
    log_file="experiment_log.txt",
    output_file="path_analysis.html"
)
```

#### Monitor Real-Time
```python
from visualization.dashboard import create_dashboard

# Create monitoring dashboard
create_dashboard(
    log_file="experiment_log.txt",
    output_file="dashboard.html",
    refresh_interval=5
)
```

---

## Best Practices

### 1. Start Simple, Then Optimize

```bash
# Step 1: Baseline
python run.py --method latent_mas --model_name MODEL --task TASK

# Step 2: Simple multi-path
python run.py --method latent_mas_multipath --model_name MODEL --task TASK

# Step 3: Optimize configuration
python run.py --method latent_mas_multipath --model_name MODEL --task TASK \
  --config_preset balanced

# Step 4: Fine-tune parameters
python run.py --method latent_mas_multipath --model_name MODEL --task TASK \
  --num_paths 7 --pruning_strategy diversity --enable_branching
```

### 2. Use Presets for Quick Experiments

```bash
# Quick test
--config_preset fast

# Production baseline
--config_preset balanced

# Maximum quality
--config_preset quality
```

### 3. Monitor Resource Usage

```bash
# Use nvidia-smi to monitor GPU memory
watch -n 1 nvidia-smi

# Profile execution time
time python run.py --method latent_mas_multipath ...

# Enable profiling in code
--enable_profiling
```

### 4. Validate on Small Sample First

```bash
# Test on 10 samples first
--max_samples 10

# If results look good, run full evaluation
--max_samples -1
```

### 5. Save Intermediate Results

```bash
# Enable checkpointing
--enable_checkpointing --checkpoint_interval 100

# Save graph visualizations
--save_graphs --graph_output_dir graphs/
```

### 6. Compare Against Baselines

```bash
# Always compare against single-path
python run.py --method latent_mas ...
python run.py --method latent_mas_multipath ...

# Compare against text-based MAS
python run.py --method text_mas ...
```

### 7. Document Your Configuration

Create a configuration file for reproducibility:

```json
{
  "experiment_name": "gsm8k_multipath_v1",
  "method": "latent_mas_multipath",
  "model_name": "Qwen/Qwen3-14B",
  "task": "gsm8k",
  "num_paths": 5,
  "pruning_strategy": "adaptive",
  "diversity_strategy": "hybrid",
  "enable_branching": true,
  "enable_merging": true,
  "merge_threshold": 0.9,
  "latent_steps": 10,
  "notes": "Balanced configuration for GSM8K"
}
```

### 8. Analyze Failures

When accuracy doesn't improve:
1. Check if paths are too similar (enable visualization)
2. Verify scoring metrics are working (check logs)
3. Ensure pruning isn't too aggressive (try diversity-aware)
4. Consider if task benefits from multi-path (some tasks may not)

### 9. Optimize for Your Hardware

```bash
# Single GPU with limited memory
--num_paths 3 --pruning_strategy topk --enable_merging

# Multiple GPUs with vLLM
--use_vllm --use_second_HF_model --device2 cuda:1 --num_paths 7

# High-memory GPU
--num_paths 10 --enable_branching --pruning_strategy diversity
```

### 10. Keep Logs for Analysis

```bash
# Redirect output to log file
python run.py --method latent_mas_multipath ... 2>&1 | tee experiment.log

# Analyze logs later
grep "Path score" experiment.log
grep "Pruning" experiment.log
grep "Merging" experiment.log
```

---

## Advanced Topics

### Custom Scoring Weights

Override default scoring weights:

```bash
# Emphasize self-consistency
--scoring_weights '{"self_consistency": 0.6, "perplexity": 0.2, "verification": 0.1, "hidden_quality": 0.1}'

# Emphasize perplexity for code generation
--scoring_weights '{"self_consistency": 0.2, "perplexity": 0.5, "verification": 0.2, "hidden_quality": 0.1}'
```

### Dynamic Path Count

Adjust number of paths based on problem difficulty:

```python
# In your custom script
if problem_difficulty == "easy":
    num_paths = 3
elif problem_difficulty == "medium":
    num_paths = 5
else:  # hard
    num_paths = 10
```

### Hybrid Pruning Strategies

Combine multiple pruning strategies:

```python
# Use adaptive early, diversity-aware later
if current_step < total_steps // 2:
    pruning_strategy = "adaptive"
else:
    pruning_strategy = "diversity"
```

---

## Conclusion

The LatentMAS Multi-Path method provides a powerful framework for improving reasoning accuracy through parallel path exploration. By understanding the core concepts, properly configuring the system, and following best practices, you can achieve significant accuracy improvements while managing computational costs.

For more information:
- [API Reference](api-reference.md) - Detailed API documentation
- [Experiment Guide](experiment-guide.md) - Guide for running experiments
- [GitHub Issues](https://github.com/Gen-Verse/LatentMAS/issues) - Report bugs or request features

Happy reasoning! 🚀


