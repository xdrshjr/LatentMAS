# Fact-Checking Dataset Inference Guide

## Overview

This guide explains how to use the newly implemented fact-checking dataset inference functionality for the CoT (Chain-of-Thought) fact-checking dataset from Wikipedia.

## Dataset Information

- **Dataset Name**: CoT Fact-Checking Wiki Dataset
- **Location**: `data/cot-fact-wiki/test-00000-of-00001.json`
- **Format**: JSON array with objects containing:
  - `problem`: The fact-checking question
  - `solution`: Chain-of-thought reasoning with final answer in `\boxed{...}` format
  - `messages`: Conversation format (user/assistant messages)

### Dataset Format Example

```json
[
  {
    "problem": "What were the key details surrounding the 2013 Pittsburgh mayoral election and its outcome?",
    "solution": "To analyze the 2013 Pittsburgh mayoral election... \\[\\boxed{The 2013 Pittsburgh mayoral election resulted in Democrat Bill Peduto being elected...}\\]",
    "messages": [
      {
        "content": "What were the key details...",
        "role": "user"
      },
      {
        "content": "To analyze the 2013 Pittsburgh mayoral election...",
        "role": "assistant"
      }
    ]
  }
]
```

## Implementation Details

### 1. Data Loader (`data.py`)

A new data loader function `load_cot_fact_wiki()` has been added to handle the fact-checking dataset:

### 2. Prompt Templates (`prompts.py`)

All prompt building functions have been updated to support the `cot_fact_wiki` task:

- **`build_agent_message_sequential_latent_mas()`**: Sequential latent MAS prompts
- **`build_agent_message_hierarchical_latent_mas()`**: Hierarchical latent MAS prompts  
- **`build_agent_messages_sequential_text_mas()`**: Sequential text MAS prompts
- **`build_agent_messages_hierarchical_text_mas()`**: Hierarchical text MAS prompts
- **`build_agent_messages_single_agent()`**: Single agent (baseline) prompts

The fact-checking task uses the same prompt format as `gsm8k`, `aime2024`, and `aime2025` tasks, which require:
- Step-by-step reasoning
- Final answer in `\boxed{...}` format
- No additional constraints on answer format (unlike multiple choice tasks)

### 3. Data Loader Implementation (`data.py`)

The data loader function `load_cot_fact_wiki()` handles the fact-checking dataset:

- **Function**: `load_cot_fact_wiki(split: str = "test", cache_dir: Optional[str] = None)`
- **Purpose**: Loads and processes the CoT fact-checking dataset
- **Features**:
  - Reads JSON file from local path
  - Extracts questions from `problem` field
  - Extracts gold answers from `\boxed{...}` format in `solution` field
  - Normalizes answers for comparison
  - Provides detailed logging for debugging

### 4. Task Registration (`run.py`)

The new task has been registered in the main inference pipeline:

- **Task Name**: `cot_fact_wiki`
- **Added to**: Command-line argument choices
- **Integration**: Seamlessly integrated with existing task loading logic

### 5. Answer Extraction Logic

All inference methods have been updated to handle fact-checking answers:

#### Methods Updated:
- `methods/baseline.py` - Baseline single-agent method
- `methods/text_mas.py` - Text-based multi-agent system
- `methods/latent_mas.py` - Latent multi-agent system
- `methods/latent_mas_multipath.py` - Multi-path latent reasoning (both standard and vLLM versions)

#### Answer Extraction Strategy:
1. **Primary**: Extract text from `\boxed{...}` format using regex
2. **Fallback**: Use existing `extract_gsm8k_answer()` function
3. **Normalization**: Convert to lowercase and strip whitespace
4. **Comparison**: Exact string match after normalization

### 6. Inference Script (`fact-inference.sh`)

A new bash script has been created for easy inference execution:

**Key Features**:
- Configurable number of samples to process
- Configurable output directory
- Support for all inference methods (baseline, text_mas, latent_mas, latent_mas_multipath)
- Adjustable generation parameters
- Visualization control
- Environment configuration for both cloud and local compute

## Usage

### Basic Usage

```bash
# Make the script executable (Linux/Mac)
chmod +x fact-inference.sh

# Run inference with default settings
./fact-inference.sh
```

### Configuration Options

Edit the script to customize parameters:

```bash
# Number of samples to process (-1 for all)
MAX_SAMPLES=50

# Output directory
OUTPUT_DIR="output/fact_checking_results"

# Model configuration
MODEL_NAME="/path/to/your/model"

# Inference method
METHOD="latent_mas_multipath"  # Options: baseline, text_mas, latent_mas, latent_mas_multipath

# Generation parameters
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.95
GENERATE_BS=1
SEED=42

# Latent reasoning parameters
LATENT_STEPS=3

# Multi-path parameters (for latent_mas_multipath)
NUM_PATHS=30
NUM_PARENT_PATHS=10
DIVERSITY_STRATEGY="temperature"
PRUNING_STRATEGY="adaptive"
TOPK_K=3
MERGE_THRESHOLD=0.9
BRANCH_THRESHOLD=0.5
LATENT_CONSISTENCY_METRIC="kl_divergence"

# Visualization
DISABLE_VISUALIZATION="--disable_visualization"  # Comment out to enable
```

### Direct Python Usage

You can also run inference directly using Python:

```bash
python run.py \
  --method latent_mas_multipath \
  --model_name "Qwen/Qwen3-0.6B" \
  --task cot_fact_wiki \
  --max_samples 50 \
  --max_new_tokens 2048 \
  --temperature 0.7 \
  --latent_steps 3 \
  --num_paths 30 \
  --disable_visualization
```

## Output

### Output Files

The inference will generate several output files:

1. **Question-Answer Records**: `output/res/cot_fact_wiki_<method>_<model>_<timestamp>.txt`
   - Contains detailed question-answer pairs
   - Includes predictions, gold answers, and correctness

2. **Log Files**: `output/logs/cot_fact_wiki_<method>_<model>_<timestamp>.log`
   - Detailed execution logs
   - Debug information
   - Performance metrics

3. **Result Summary**: `output/simple_res/cot_fact_wiki_<method>_<timestamp>.log`
   - Run parameters
   - Final accuracy and statistics
   - JSON format results

4. **CSV Results**: `output/csv_res/results.csv`
   - Aggregated results across runs
   - Easy to analyze in spreadsheet software

### Output Format

Each question-answer record includes:
- Problem index
- Question text
- Model prediction
- Gold answer
- Correctness indicator
- Method and model information

Example:
```
================================================================================
PROBLEM #1
================================================================================
Question: What were the key details surrounding the 2013 Pittsburgh mayoral election and its outcome?
Prediction: the 2013 pittsburgh mayoral election resulted in democrat bill peduto being elected as the 60th mayor of pittsburgh, following incumbent mayor luke ravenstahl's decision not to seek reelection.
Gold Answer: the 2013 pittsburgh mayoral election resulted in democrat bill peduto being elected as the 60th mayor of pittsburgh, following incumbent mayor luke ravenstahl's decision not to seek reelection.
Result: ✓ CORRECT
================================================================================
```

## Logging

The implementation includes comprehensive logging:

### Log Levels:
- **INFO**: High-level progress and results
- **DEBUG**: Detailed processing information, answer extraction details
- **WARNING**: Non-critical issues (e.g., missing fields)
- **ERROR**: Critical errors that prevent processing

### Key Log Messages:
- Dataset loading progress
- Sample processing details
- Answer extraction results
- Evaluation outcomes
- Performance metrics

### Example Log Output:
```
INFO - Loading CoT fact-checking dataset from: data/cot-fact-wiki/test-00000-of-00001.json
INFO - Loaded 1794 samples from CoT fact-checking dataset
DEBUG - Sample 0: question length=89, solution length=1234, gold length=156
INFO - Item 1: Evaluation result: CORRECT (pred=the 2013 pittsburgh..., gold=the 2013 pittsburgh...)
```

## Evaluation Metrics

The system evaluates predictions using:

1. **Exact Match**: Normalized string comparison
   - Both prediction and gold answer are converted to lowercase
   - Whitespace is stripped
   - Exact string match is required

2. **Accuracy**: Percentage of correct predictions
   - Calculated as: `correct_count / total_count`
   - Reported in final results

3. **Per-Sample Results**: Individual correctness for each question
   - Stored in output files
   - Used for detailed analysis

## Advanced Features

### Multi-Path Reasoning

For the `latent_mas_multipath` method, you can configure:

- **Number of Paths**: How many reasoning paths to explore
- **Diversity Strategy**: How to generate diverse paths (temperature, noise, hybrid)
- **Pruning Strategy**: How to select best paths (topk, adaptive, diversity, budget)
- **Branching/Merging**: Enable adaptive path exploration

### Visualization

When enabled, the system generates:
- Reasoning graph visualizations (DOT format)
- Interactive HTML visualizations
- Path genealogy JSON files
- Graph statistics

To enable visualization, remove or comment out `DISABLE_VISUALIZATION` in the script.

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Error: `Dataset file not found: data/cot-fact-wiki/test-00000-of-00001.json`
   - Solution: Ensure the dataset file exists at the specified path

2. **No Answer Extracted**
   - Issue: Prediction is empty or None
   - Check: Model output format, ensure `\boxed{...}` pattern is present
   - Debug: Enable DEBUG logging to see extraction details

3. **Low Accuracy**
   - Possible causes:
     - Model not suitable for fact-checking tasks
     - Temperature too high (try 0.5-0.7)
     - Insufficient reasoning steps (increase `LATENT_STEPS`)
   - Solution: Experiment with different parameters

4. **Memory Issues**
   - Reduce `GENERATE_BS` (batch size)
   - Reduce `NUM_PATHS` for multi-path methods
   - Use smaller model
   - Enable gradient checkpointing

## Performance Considerations

### Recommended Settings

**For Fast Inference**:
```bash
METHOD="baseline"
MAX_SAMPLES=10
GENERATE_BS=4
DISABLE_VISUALIZATION="--disable_visualization"
```

**For High Quality**:
```bash
METHOD="latent_mas_multipath"
MAX_SAMPLES=-1  # All samples
LATENT_STEPS=5
NUM_PATHS=50
ENABLE_BRANCHING="--enable_branching"
ENABLE_MERGING="--enable_merging"
```

**For Balanced Performance**:
```bash
METHOD="latent_mas_multipath"
MAX_SAMPLES=100
LATENT_STEPS=3
NUM_PATHS=30
TEMPERATURE=0.7
```

## Integration with Existing System

The fact-checking task integrates seamlessly with:

- **Multi-GPU Training**: Use `run_multi_gpu.py` for distributed inference
- **PRM Data Collection**: Enable with `--collect_prm_data` flag
- **Configuration Presets**: Use `--config_preset` for predefined settings
- **Custom Questions**: Can be adapted for custom fact-checking questions

## Future Enhancements

Potential improvements:

1. **Semantic Similarity**: Use embedding-based similarity instead of exact match
2. **Answer Scoring**: Implement partial credit for partially correct answers
3. **Multi-Reference Answers**: Support multiple valid answer formats
4. **Fact Verification**: Add external knowledge base verification
5. **Confidence Scores**: Report model confidence for each prediction

## References

- Main inference script: `run.py`
- Data loader: `data.py`
- Inference methods: `methods/`
- Configuration: `config.py`
- Utilities: `utils.py`

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Enable DEBUG logging for more information
3. Review the dataset format and ensure compatibility
4. Verify model compatibility with the task

---

**Last Updated**: December 29, 2025
**Version**: 1.0

