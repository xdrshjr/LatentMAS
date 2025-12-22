# LatentMAS Multi-Path Enhancement - Modification Plan

## Overview

This document outlines the detailed modification plan to enhance the LatentMAS method with **Graph-Structured Multi-Path Latent Reasoning (GMLR)** capabilities. The enhancement will implement multi-path exploration, training-free evaluation metrics, intelligent pruning, and path merging while maintaining the training-free constraint.

---

## Modification Tasks

### Phase 1: Core Infrastructure & Data Structures

#### Task-1.1: Create Graph Data Structure Module
**Status**: ✅ Complete  
**File**: `methods/graph_structure.py` (NEW)  
**Description**: Implement the core graph data structure to manage reasoning paths  
**Details**:
- Create `ReasoningNode` class to represent each reasoning state
  - Fields: `node_id`, `hidden_states`, `kv_cache`, `parent_id`, `children_ids`, `score`, `metadata`
  - Methods: `add_child()`, `update_score()`, `get_depth()`
- Create `ReasoningGraph` class to manage the entire graph
  - Fields: `nodes`, `edges`, `root_id`, `leaf_ids`
  - Methods: `add_node()`, `remove_node()`, `get_path()`, `get_all_paths()`, `prune_nodes()`, `merge_nodes()`
- Implement graph traversal utilities (BFS, DFS)
- Add graph visualization export (for debugging/analysis)
- **Logging**: INFO for graph operations (add/remove nodes), DEBUG for traversal details

#### Task-1.2: Create Path Management Module
**Status**: ✅ Complete  
**File**: `methods/path_manager.py` (NEW)  
**Description**: Implement path tracking and management utilities  
**Details**:
- Create `PathState` dataclass to encapsulate path information
  - Fields: `path_id`, `latent_history`, `hidden_states`, `kv_cache`, `score`, `metadata`
- Create `PathManager` class for path lifecycle management
  - Methods: `create_path()`, `branch_path()`, `merge_paths()`, `get_active_paths()`, `prune_paths()`
- Implement path comparison utilities (cosine similarity, divergence metrics)
- Add path serialization for checkpointing
- **Logging**: INFO for path creation/deletion, DEBUG for path comparisons

#### Task-1.3: Create Scoring Metrics Module
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (NEW)  
**Description**: Implement training-free evaluation metrics for path scoring  
**Details**:
- Implement `SelfConsistencyScorer` class
  - Method: `score(path, num_samples=5)` → float [0-1]
  - Generate multiple answers from same path, measure agreement
- Implement `PerplexityScorer` class
  - Method: `score(path)` → float (normalized)
  - Calculate perplexity from log-probabilities of latent steps
- Implement `VerificationScorer` class
  - Method: `score(path, question, answer)` → float [0-1]
  - Use model to verify reasoning chain
- Implement `HiddenStateQualityScorer` class
  - Method: `score(path)` → float [0-1]
  - Measure norm stability, smoothness, entropy
- Create `EnsembleScorer` class to combine metrics
  - Configurable weights for each metric
  - Method: `score(path)` → float (weighted combination)
- **Logging**: DEBUG for individual metric scores, INFO for ensemble scores

---

### Phase 2: Multi-Path Generation

#### Task-2.1: Enhance ModelWrapper for Multi-Path Support
**Status**: ✅ Complete  
**File**: `models.py` (MODIFY)  
**Description**: Add multi-path latent generation capabilities  
**Details**:
- Add `generate_diverse_latent_paths()` method
  - Parameters: `input_ids`, `attention_mask`, `num_paths`, `latent_steps`, `diversity_temps`, `past_key_values`
  - Returns: List of path states with diverse latent trajectories
  - Use temperature sampling and noise injection for diversity
- Add `generate_latent_with_branching()` method
  - Support branching from existing KV cache
  - Return multiple continuations from single state
- Modify `_apply_latent_realignment()` to support batch processing
- Add optional noise injection parameter for exploration
- **Logging**: INFO for multi-path generation start/end, DEBUG for each path generation

#### Task-2.2: Implement Diversity Generation Strategies
**Status**: ✅ Complete  
**File**: `methods/diversity_strategies.py` (NEW)  
**Description**: Implement strategies to generate diverse reasoning paths  
**Details**:
- Create `DiversityStrategy` base class
- Implement `TemperatureDiversityStrategy`
  - Use different temperatures for each path (e.g., 0.7, 1.0, 1.3)
- Implement `NoiseDiversityStrategy`
  - Add Gaussian noise to hidden states before realignment
- Implement `InitializationDiversityStrategy`
  - Different starting points for latent thinking
- Implement `HybridDiversityStrategy`
  - Combine multiple strategies
- **Logging**: DEBUG for diversity parameters, INFO for strategy selection

---

### Phase 3: Scoring and Evaluation

#### Task-3.1: Implement Self-Consistency Scorer
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (MODIFY)  
**Description**: Complete implementation of self-consistency scoring  
**Details**:
- Generate N answers from same path with temperature sampling
- Extract and normalize answers using existing utilities
- Calculate frequency of most common answer
- Handle edge cases (all different answers, parsing failures)
- Cache results to avoid redundant computation
- **Logging**: DEBUG for each sample answer, INFO for final consistency score

#### Task-3.2: Implement Perplexity-Based Scorer
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (MODIFY)  
**Description**: Complete implementation of perplexity-based scoring  
**Details**:
- Track log-probabilities during latent generation
- Calculate perplexity for each latent step
- Normalize by path length
- Handle numerical stability (log-space computation)
- Return normalized confidence score
- **Logging**: DEBUG for per-step perplexity, INFO for average perplexity

#### Task-3.3: Implement Verification Scorer
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (MODIFY)  
**Description**: Complete implementation of verification-based scoring  
**Details**:
- Create verification prompts for different tasks
- Use model to evaluate reasoning correctness
- Parse confidence levels from model response
- Map confidence to numerical scores
- Handle task-specific verification (math, code, QA)
- **Logging**: DEBUG for verification prompts/responses, INFO for verification scores

#### Task-3.4: Implement Hidden State Quality Scorer
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (MODIFY)  
**Description**: Complete implementation of hidden state quality metrics  
**Details**:
- Calculate norm stability across latent steps
- Measure cosine similarity progression (smoothness)
- Calculate entropy of hidden state distributions
- Detect anomalies (sudden jumps, vanishing norms)
- Combine sub-metrics into overall quality score
- **Logging**: DEBUG for individual quality metrics, INFO for combined score

#### Task-3.5: Implement Ensemble Scorer
**Status**: ✅ Complete  
**File**: `methods/scoring_metrics.py` (MODIFY)  
**Description**: Create weighted ensemble of all scoring metrics  
**Details**:
- Default weights: self-consistency (40%), perplexity (30%), verification (20%), hidden quality (10%)
- Support custom weight configuration
- Normalize scores to [0, 1] range
- Add score breakdown for analysis
- Support metric-specific thresholds
- **Logging**: INFO for ensemble score, DEBUG for individual component scores

---

### Phase 4: Pruning Strategies

#### Task-4.1: Implement Basic Pruning
**Status**: ✅ Complete  
**File**: `methods/pruning_strategies.py` (NEW)  
**Description**: Implement basic top-k pruning strategy  
**Details**:
- Create `PruningStrategy` base class
- Implement `TopKPruning` strategy
  - Keep top-k paths by score
  - Configurable k value
- Implement `ThresholdPruning` strategy
  - Remove paths below score threshold
- Add pruning statistics tracking
- **Logging**: INFO for pruning decisions (how many removed), DEBUG for individual path scores

#### Task-4.2: Implement Adaptive Pruning
**Status**: ✅ Complete  
**File**: `methods/pruning_strategies.py` (MODIFY)  
**Description**: Implement adaptive pruning based on progress  
**Details**:
- Implement `AdaptivePruning` strategy
  - More aggressive early, less aggressive later
  - Formula: `keep_ratio = 0.3 + 0.5 * (step / total_steps)`
- Adjust pruning rate based on path diversity
- Consider computational budget constraints
- **Logging**: INFO for adaptive pruning rate, DEBUG for pruning calculations

#### Task-4.3: Implement Diversity-Aware Pruning
**Status**: ✅ Complete  
**File**: `methods/pruning_strategies.py` (MODIFY)  
**Description**: Implement pruning that preserves path diversity  
**Details**:
- Implement `DiversityAwarePruning` strategy
- Always keep highest-scoring path
- For remaining slots, balance score and diversity
- Use cosine similarity to measure diversity
- Ensure minimum distance between kept paths
- **Logging**: INFO for diversity metrics, DEBUG for pairwise similarity calculations

#### Task-4.4: Implement Budget-Based Pruning
**Status**: ✅ Complete  
**File**: `methods/pruning_strategies.py` (MODIFY)  
**Description**: Implement pruning based on computational budget  
**Details**:
- Track computational cost (tokens, FLOPs)
- Prune to stay within budget constraints
- Consider cost-benefit ratio (score per compute)
- Support dynamic budget adjustment
- **Logging**: INFO for budget status, DEBUG for cost calculations

---

### Phase 5: Path Merging

#### Task-5.1: Implement Path Similarity Detection
**Status**: ✅ Complete  
**File**: `methods/path_merging.py` (NEW)  
**Description**: Detect similar paths for merging  
**Details**:
- Implement cosine similarity for hidden states
- Implement KL-divergence for distributions
- Support configurable similarity thresholds
- Detect convergent paths (different start, similar end)
- **Logging**: DEBUG for similarity calculations, INFO for merge candidates

#### Task-5.2: Implement Path Merging Operations
**Status**: ✅ Complete  
**File**: `methods/path_merging.py` (MODIFY)  
**Description**: Implement actual path merging logic  
**Details**:
- Merge KV caches (concatenation or weighted average)
- Merge hidden states (weighted by scores)
- Update graph structure (remove merged nodes, create new node)
- Preserve metadata from both paths
- Handle edge cases (different lengths, incompatible states)
- **Logging**: INFO for merge operations, DEBUG for merge details

#### Task-5.3: Implement Merge Strategy Selection
**Status**: ✅ Complete  
**File**: `methods/path_merging.py` (MODIFY)  
**Description**: Implement intelligent merge strategy selection  
**Details**:
- Create `MergeStrategy` base class
- Implement `AverageMergeStrategy` (simple averaging)
- Implement `WeightedMergeStrategy` (score-weighted)
- Implement `SelectiveMergeStrategy` (keep best components)
- Auto-select strategy based on path characteristics
- **Logging**: INFO for strategy selection, DEBUG for merge parameters

---

### Phase 6: Enhanced LatentMAS Method

#### Task-6.1: Create Multi-Path LatentMAS Method
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (NEW)  
**Description**: Create new method class with multi-path capabilities  
**Details**:
- Inherit from `LatentMASMethod`
- Add configuration parameters:
  - `num_paths`: number of parallel paths (default: 5)
  - `enable_branching`: whether to branch during reasoning (default: True)
  - `enable_merging`: whether to merge similar paths (default: True)
  - `pruning_strategy`: which pruning strategy to use
  - `scoring_weights`: weights for ensemble scorer
- Override `run_batch()` to use multi-path logic
- Maintain backward compatibility with single-path mode
- **Logging**: INFO for configuration, DEBUG for method initialization

#### Task-6.2: Implement Multi-Path Agent Processing
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (MODIFY)  
**Description**: Implement multi-path processing for each agent  
**Details**:
- For each non-judger agent:
  1. Generate diverse latent paths (using diversity strategies)
  2. Score all paths (using ensemble scorer)
  3. Prune low-quality paths (using pruning strategy)
  4. Merge similar paths (using merge strategy)
  5. Continue to next agent with surviving paths
- Track path genealogy (which paths came from which)
- Maintain path metadata throughout processing
- **Logging**: INFO for each agent's path processing, DEBUG for path state changes

#### Task-6.3: Implement Multi-Path Judger Aggregation
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (MODIFY)  
**Description**: Implement judger that aggregates multiple paths  
**Details**:
- Collect all surviving paths at judger stage
- Option 1: Run judger on each path, use voting
- Option 2: Merge paths before judger
- Option 3: Concatenate path embeddings for judger
- Support configurable aggregation strategy
- Generate final answer with confidence score
- **Logging**: INFO for aggregation strategy, DEBUG for per-path judger results

#### Task-6.4: Implement Uncertainty-Based Branching
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (MODIFY)  
**Description**: Implement adaptive branching based on uncertainty  
**Details**:
- Calculate uncertainty from hidden state entropy
- Branch when uncertainty exceeds threshold
- Adjust branch factor based on uncertainty level
- Avoid excessive branching (budget constraints)
- **Logging**: INFO for branching decisions, DEBUG for uncertainty calculations

#### Task-6.5: Add vLLM Support for Multi-Path
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (MODIFY)  
**Description**: Extend vLLM support to multi-path processing  
**Details**:
- Implement `run_batch_vllm()` for multi-path
- Handle embedding concatenation for multiple paths
- Optimize batch processing for vLLM
- Support prefix caching for shared path prefixes
- **Logging**: INFO for vLLM batch processing, DEBUG for embedding operations

---

### Phase 7: Configuration and Integration

#### Task-7.1: Add Command-Line Arguments
**Status**: ✅ Complete  
**File**: `run.py` (MODIFY)  
**Description**: Add command-line arguments for multi-path features  
**Details**:
- Add `--enable_multipath` flag
- Add `--num_paths` (default: 5)
- Add `--pruning_strategy` (choices: topk, adaptive, diversity, budget)
- Add `--merge_threshold` (default: 0.9)
- Add `--scoring_weights` (JSON string for custom weights)
- Add `--diversity_strategy` (choices: temperature, noise, hybrid)
- Add `--enable_branching` flag
- Add `--branch_threshold` (uncertainty threshold for branching)
- **Logging**: INFO for parsed arguments, DEBUG for argument validation

#### Task-7.2: Update Method Factory
**Status**: ✅ Complete  
**File**: `run.py` (MODIFY)  
**Description**: Update method initialization to support multi-path  
**Details**:
- Add `latent_mas_multipath` to method choices
- Initialize `LatentMASMultiPathMethod` when selected
- Pass multi-path configuration to method
- Maintain backward compatibility with original `latent_mas`
- **Logging**: INFO for method selection, DEBUG for method configuration

#### Task-7.3: Create Configuration File Support
**Status**: ✅ Complete  
**File**: `config.py` (NEW)  
**Description**: Add support for configuration files  
**Details**:
- Create `MultiPathConfig` dataclass
- Support JSON/YAML config files
- Validate configuration parameters
- Provide default configurations for common scenarios
- Support config inheritance and overrides
- **Logging**: INFO for config loading, DEBUG for config validation

---

### Phase 8: Optimization and Utilities

#### Task-8.1: Implement KV-Cache Optimization
**Status**: ✅ Complete  
**File**: `methods/cache_optimization.py` (NEW)  
**Description**: Optimize KV-cache usage for multi-path  
**Details**:
- Implement shared prefix detection
- Reuse KV-cache for common path prefixes
- Implement cache eviction strategies
- Track cache memory usage
- Support cache serialization/deserialization
- **Logging**: INFO for cache operations, DEBUG for cache hit/miss rates

#### Task-8.2: Implement Batch Processing Optimization
**Status**: ✅ Complete  
**File**: `methods/batch_optimization.py` (NEW)  
**Description**: Optimize batch processing for multiple paths  
**Details**:
- Group similar paths for batch processing
- Implement dynamic batching based on path lengths
- Optimize tensor operations for multi-path
- Reduce redundant computations
- **Logging**: INFO for batch composition, DEBUG for optimization details

#### Task-8.3: Add Performance Profiling
**Status**: ✅ Complete  
**File**: `utils.py` (MODIFY)  
**Description**: Add profiling utilities for performance analysis  
**Details**:
- Add `@profile` decorator for timing functions
- Track memory usage per operation
- Log computational costs (FLOPs, tokens)
- Generate performance reports
- Support profiling enable/disable flag
- **Logging**: INFO for performance summaries, DEBUG for detailed profiling

#### Task-8.4: Implement Checkpointing
**Status**: ✅ Complete  
**File**: `methods/checkpointing.py` (NEW)  
**Description**: Add checkpointing for long-running experiments  
**Details**:
- Save graph state at regular intervals
- Save path states and scores
- Support resume from checkpoint
- Implement incremental checkpointing
- Add checkpoint compression
- **Logging**: INFO for checkpoint operations, DEBUG for checkpoint contents

---

### Phase 9: Visualization and Analysis

#### Task-9.1: Implement Graph Visualization
**Status**: ✅ Complete  
**File**: `visualization/graph_viz.py` (NEW)  
**Description**: Create visualization tools for reasoning graphs  
**Details**:
- Export graph to DOT format (Graphviz)
- Color nodes by score
- Show path genealogy
- Highlight pruned/merged nodes
- Support interactive visualization (HTML)
- **Logging**: INFO for visualization generation, DEBUG for rendering details

#### Task-9.2: Implement Path Analysis Tools
**Status**: ✅ Complete  
**File**: `visualization/path_analysis.py` (NEW)  
**Description**: Create tools for analyzing path behavior  
**Details**:
- Plot score distributions across paths
- Visualize diversity metrics over time
- Show pruning/merging statistics
- Compare paths that led to correct vs incorrect answers
- Generate path comparison reports
- **Logging**: INFO for analysis results, DEBUG for calculation details

#### Task-9.3: Implement Logging Dashboard
**Status**: ✅ Complete  
**File**: `visualization/dashboard.py` (NEW)  
**Description**: Create real-time monitoring dashboard  
**Details**:
- Show active paths and their scores
- Display pruning/merging events
- Track computational budget usage
- Show per-agent statistics
- Support export to HTML/JSON
- **Logging**: INFO for dashboard updates, DEBUG for data collection

---

### Phase 10: Testing and Validation

#### Task-10.1: Create Unit Tests for Graph Structure
**Status**: ⬜ Pending  
**File**: `tests/test_graph_structure.py` (NEW)  
**Description**: Unit tests for graph data structures  
**Details**:
- Test node creation and manipulation
- Test graph traversal
- Test pruning operations
- Test merging operations
- Test edge cases (empty graph, single node, cycles)
- **Logging**: INFO for test execution, DEBUG for test details

#### Task-10.2: Create Unit Tests for Scoring Metrics
**Status**: ⬜ Pending  
**File**: `tests/test_scoring_metrics.py` (NEW)  
**Description**: Unit tests for scoring metrics  
**Details**:
- Test each scorer independently
- Test ensemble scorer
- Test score normalization
- Test edge cases (empty paths, invalid states)
- Validate score ranges [0, 1]
- **Logging**: INFO for test results, DEBUG for score calculations

#### Task-10.3: Create Integration Tests
**Status**: ⬜ Pending  
**File**: `tests/test_integration.py` (NEW)  
**Description**: End-to-end integration tests  
**Details**:
- Test full multi-path pipeline
- Test with different configurations
- Test backward compatibility
- Test vLLM integration
- Validate output format consistency
- **Logging**: INFO for test progress, DEBUG for intermediate states

#### Task-10.4: Create Ablation Test Suite
**Status**: ⬜ Pending  
**File**: `tests/test_ablation.py` (NEW)  
**Description**: Tests for ablation studies  
**Details**:
- Test with different numbers of paths (1, 3, 5, 10)
- Test with each scoring metric disabled
- Test with different pruning strategies
- Test with/without merging
- Generate comparison reports
- **Logging**: INFO for ablation results, DEBUG for configuration details

---

### Phase 11: Documentation

#### Task-11.1: Update README
**Status**: ✅ Complete  
**File**: `README.md` (MODIFY)  
**Description**: Update README with multi-path features  
**Details**:
- Add multi-path method description
- Add usage examples
- Add configuration guide
- Add performance considerations
- Update command-line arguments documentation
- **Logging**: N/A

#### Task-11.2: Create Multi-Path User Guide
**Status**: ✅ Complete  
**File**: `docs/multipath-guide.md` (NEW)  
**Description**: Comprehensive guide for multi-path features  
**Details**:
- Explain multi-path reasoning concept
- Describe each component (scoring, pruning, merging)
- Provide configuration recommendations
- Include troubleshooting guide
- Add performance tuning tips
- **Logging**: N/A

#### Task-11.3: Create API Documentation
**Status**: ✅ Complete  
**File**: `docs/api-reference.md` (NEW)  
**Description**: API documentation for new modules  
**Details**:
- Document all new classes and methods
- Include parameter descriptions
- Add usage examples
- Document return types
- Add cross-references
- **Logging**: N/A

#### Task-11.4: Create Experiment Guide
**Status**: ✅ Complete  
**File**: `docs/experiment-guide.md` (NEW)  
**Description**: Guide for running experiments  
**Details**:
- Describe experimental setup
- Provide baseline configurations
- Explain ablation study setup
- Include result interpretation guide
- Add reproducibility checklist
- **Logging**: N/A

---

## Implementation Order

### Priority 1 (Core Foundation)
1. Task-1.1: Graph Data Structure
2. Task-1.2: Path Management
3. Task-1.3: Scoring Metrics Module
4. Task-2.1: Multi-Path Model Support

### Priority 2 (Core Functionality)
5. Task-2.2: Diversity Strategies
6. Task-3.1 - 3.5: All Scoring Implementations
7. Task-4.1 - 4.2: Basic and Adaptive Pruning
8. Task-5.1 - 5.2: Path Merging

### Priority 3 (Integration)
9. Task-6.1 - 6.3: Multi-Path LatentMAS Method
10. Task-7.1 - 7.2: Configuration and CLI
11. Task-4.3 - 4.4: Advanced Pruning
12. Task-5.3: Merge Strategies

### Priority 4 (Optimization)
13. Task-6.4 - 6.5: Advanced Features (Branching, vLLM)
14. Task-8.1 - 8.2: Cache and Batch Optimization
15. Task-7.3: Config File Support

### Priority 5 (Analysis & Testing)
16. Task-8.3 - 8.4: Profiling and Checkpointing
17. Task-9.1 - 9.3: Visualization Tools
18. Task-10.1 - 10.4: All Testing

### Priority 6 (Documentation)
19. Task-11.1 - 11.4: All Documentation

---

## Logging Standards

### Log Levels
- **DEBUG**: Detailed information for debugging (tensor shapes, intermediate values, algorithm steps)
- **INFO**: General information about program flow (method calls, configuration, results)
- **WARNING**: Unexpected situations that don't prevent execution (fallbacks, deprecated features)
- **ERROR**: Errors that prevent specific operations (invalid inputs, failed computations)

### Log Format
```python
# Standard format
logger.info(f"[{component_name}] {action}: {details}")

# Examples
logger.info("[PathManager] Created new path: path_id=3, parent_id=1")
logger.debug("[PerplexityScorer] Calculated perplexity: 15.23 for path_id=3")
logger.warning("[PruningStrategy] No paths to prune, keeping all 2 paths")
logger.error("[GraphStructure] Failed to merge nodes: incompatible dimensions")
```

### Key Logging Points
1. **Method Entry/Exit**: Log when entering/exiting major methods with parameters
2. **State Changes**: Log when paths are created, pruned, merged, or scored
3. **Decisions**: Log why certain decisions were made (pruning, branching, merging)
4. **Performance**: Log timing and resource usage for expensive operations
5. **Errors**: Log all exceptions with context and stack traces

---

## Code Quality Standards

### General Principles
1. **Clean Code**: Functions should be small, focused, and well-named
2. **DRY**: Don't repeat yourself - extract common logic
3. **Type Hints**: All functions must have complete type hints
4. **Docstrings**: All public functions/classes must have docstrings
5. **Error Handling**: Use try-except with specific exceptions

### File Organization
```python
# Standard file structure
"""Module docstring describing purpose."""

# 1. Imports (stdlib, third-party, local)
import logging
from typing import List, Dict, Optional
import torch
from .base import BaseClass

# 2. Logger setup
logger = logging.getLogger(__name__)

# 3. Constants
DEFAULT_NUM_PATHS = 5
MAX_PATHS = 20

# 4. Classes and functions
class MyClass:
    """Class docstring."""
    
    def __init__(self, ...):
        """Initialize with..."""
        logger.info(f"[MyClass] Initialized with ...")
        
    def my_method(self, ...) -> ReturnType:
        """Method docstring.
        
        Args:
            param1: Description
            
        Returns:
            Description
        """
        logger.debug(f"[MyClass.my_method] Starting with ...")
        # Implementation
```

### Testing Standards
1. Each module should have corresponding test file
2. Test coverage should be > 80%
3. Include edge cases and error conditions
4. Use descriptive test names: `test_<what>_<condition>_<expected>`

---

## Risk Mitigation

### Risk 1: Backward Compatibility
**Mitigation**: 
- Keep original `LatentMASMethod` unchanged
- Create new `LatentMASMultiPathMethod` class
- Add feature flags to enable/disable multi-path features

### Risk 2: Memory Usage
**Mitigation**:
- Implement aggressive pruning by default
- Add memory monitoring and warnings
- Support configurable path limits
- Implement KV-cache sharing

### Risk 3: Computational Cost
**Mitigation**:
- Add budget-based pruning
- Optimize batch processing
- Support early stopping when paths converge
- Profile and optimize hot paths

### Risk 4: Complexity
**Mitigation**:
- Modular design with clear interfaces
- Comprehensive documentation
- Extensive logging for debugging
- Visualization tools for understanding behavior

---

## Success Criteria

### Functional Requirements
- ✅ Multi-path generation works correctly
- ✅ All scoring metrics implemented and validated
- ✅ Pruning reduces paths without losing quality
- ✅ Merging reduces redundancy
- ✅ vLLM integration functional
- ✅ Backward compatible with original method

### Performance Requirements
- ✅ Multi-path (5 paths) completes within 3x time of single-path
- ✅ Memory usage stays within 2x of single-path
- ✅ Accuracy improvement of at least 5% on benchmarks

### Quality Requirements
- ✅ Code coverage > 80%
- ✅ All tests passing
- ✅ Comprehensive logging throughout
- ✅ Complete documentation

---

## Timeline Estimate

- **Phase 1-2 (Infrastructure & Generation)**: 2-3 weeks
- **Phase 3-5 (Scoring & Pruning & Merging)**: 3-4 weeks
- **Phase 6-7 (Integration & Config)**: 2-3 weeks
- **Phase 8-9 (Optimization & Visualization)**: 2-3 weeks
- **Phase 10-11 (Testing & Documentation)**: 2-3 weeks

**Total**: 11-16 weeks (approximately 3-4 months)

---

## Notes

1. **Incremental Development**: Implement and test each phase before moving to next
2. **Continuous Integration**: Run tests after each task completion
3. **Code Review**: Review code quality and logging before marking tasks complete
4. **Performance Monitoring**: Track performance metrics throughout development
5. **User Feedback**: Gather feedback from users during development

---

## Appendix: Key Design Decisions

### Decision 1: Graph vs Tree Structure
**Choice**: Graph structure  
**Rationale**: More flexible, supports merging and complex dependencies

### Decision 2: Training-Free Metrics
**Choice**: Ensemble of intrinsic metrics  
**Rationale**: Maintains training-free constraint while providing robust evaluation

### Decision 3: Pruning Strategy
**Choice**: Adaptive + diversity-aware  
**Rationale**: Balances computational efficiency with exploration

### Decision 4: Path Aggregation
**Choice**: Configurable (voting, merging, concatenation)  
**Rationale**: Different tasks may benefit from different strategies

### Decision 5: Backward Compatibility
**Choice**: New class, not modification  
**Rationale**: Preserves existing functionality, reduces risk

---

## Phase 12: Logging and Progress Bar Enhancement

### Task-12.1: Enhanced Logging System
**Status**: ✅ Complete  
**File**: `logging_config.py` (NEW)  
**Description**: Implement improved logging with module-specific formatting and reduced console verbosity  
**Details**:
- Create `ColoredFormatter` class with ANSI color codes for different modules
- Implement `ProgressBarHandler` for compatibility with progress bars
- Add `setup_logging()` function with separate console and file logging levels
- Configure module-specific log levels to reduce noise from verbose modules
- Support colored output with module prefixes (e.g., `[MultiPath]`, `[Scoring]`, `[Pruning]`)
- Automatic log file creation with standardized naming
- **Logging**: Clean, color-coded console output with detailed file logging

### Task-12.2: Progress Bar System
**Status**: ✅ Complete  
**File**: `progress_utils.py` (NEW)  
**Description**: Implement progress bar system that stays at bottom while logs scroll above  
**Details**:
- Create `ProgressBarManager` class for managing progress bars
- Support main progress bar showing overall processing status
- Support sub-progress bars for nested operations
- Integration with tqdm for smooth progress display
- Update progress bar with real-time accuracy metrics
- Context manager support for automatic cleanup
- **Features**: Persistent bottom progress bar with live accuracy updates

### Task-12.3: Update run.py with New Logging
**Status**: ✅ Complete  
**File**: `run.py` (MODIFY)  
**Description**: Integrate new logging and progress bar system into main execution  
**Details**:
- Replace basic logging.basicConfig with enhanced setup_logging()
- Create progress bar at start of processing
- Update progress bar after each batch with accuracy metrics
- Reduce print statements, use logger instead
- Move detailed output to DEBUG level
- Clean final results display
- **Impact**: Much cleaner console output with progress bar at bottom

### Task-12.4: Update Method Logging Levels
**Status**: ✅ Complete  
**File**: `methods/latent_mas_multipath.py` (MODIFY)  
**Description**: Adjust logging levels in methods to reduce console verbosity  
**Details**:
- Change verbose INFO logs to DEBUG level
- Keep important milestones at INFO level (agent processing, pruning/merging results)
- Simplify log messages, remove redundant prefixes
- Consistent formatting across all methods
- Better distinction between important events and detailed debugging
- **Impact**: ~70% reduction in console log volume while maintaining visibility of key events

---

**Document Version**: 1.1  
**Last Updated**: 2025-12-22  
**Status**: Phase 12 Complete - Logging Enhancement Implemented

