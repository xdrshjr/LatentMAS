# Multi-GPU Data Collection Implementation Guide

## Overview

This document describes the multi-GPU parallel data collection feature implemented for the LatentMAS project. The feature enables distributed data collection across multiple GPUs, significantly accelerating the PRM training data collection process.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  collect_training_data.sh                    │
│              (Entry Point - Mode Selection)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────────┐
│ Single-GPU    │         │  Multi-GPU       │
│ Mode          │         │  Mode            │
│               │         │                  │
│ run.py        │         │ run_multi_gpu.py │
│ (Direct)      │         │ (Orchestrator)   │
└───────────────┘         └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
              ┌─────────┐    ┌─────────┐    ┌─────────┐
              │ GPU 0   │    │ GPU 1   │    │ GPU N   │
              │ run.py  │    │ run.py  │    │ run.py  │
              │ Process │    │ Process │    │ Process │
              └────┬────┘    └────┬────┘    └────┬────┘
                   │              │              │
                   ▼              ▼              ▼
              ┌─────────┐    ┌─────────┐    ┌─────────┐
              │gpu_0.pt │    │gpu_1.pt │    │gpu_N.pt │
              └────┬────┘    └────┬────┘    └────┬────┘
                   │              │              │
                   └──────────────┼──────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Data Aggregator │
                        │  (utils.py)      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  merged.pt       │
                        │  (Final Result)  │
                        └──────────────────┘
```

## Implementation Details

### 1. Data Distribution (`utils.py`)

#### Functions Added:

**`split_dataset_for_multi_gpu(dataset, num_gpus, gpu_id)`**
- Divides dataset evenly across GPUs
- Returns the subset assigned to a specific GPU
- Handles uneven splits by distributing remainder to first GPUs

**`get_gpu_data_range(total_samples, num_gpus, gpu_id)`**
- Calculates start and end indices for each GPU
- Ensures balanced distribution
- Returns tuple of (start_idx, end_idx)

**`validate_gpu_availability(gpu_ids)`**
- Checks if specified GPUs are available
- Validates CUDA availability
- Tests GPU accessibility

Example:
```python
# 100 samples, 4 GPUs
# GPU 0: samples [0:25]   (25 samples)
# GPU 1: samples [25:50]  (25 samples)
# GPU 2: samples [50:75]  (25 samples)
# GPU 3: samples [75:100] (25 samples)
```

### 2. Result Aggregation (`utils.py`)

#### Functions Added:

**`aggregate_prm_data(output_dir, num_gpus, batch_name)`**
- Loads data from all GPU-specific files
- Combines question records and tree structures
- Saves merged data to a single file
- Returns path to merged file

**`merge_prm_metadata(metadata_list, batch_name)`**
- Aggregates statistics from all GPUs
- Calculates overall accuracy
- Preserves individual GPU metadata for reference

### 3. Multi-GPU Orchestrator (`run_multi_gpu.py`)

New file that coordinates multiple GPU processes.

#### Key Features:

1. **Process Management**
   - Launches separate subprocess for each GPU
   - Sets isolated CUDA_VISIBLE_DEVICES for each process
   - Monitors process completion
   - Handles errors and cleanup

2. **Command Construction**
   - Builds appropriate command for each GPU
   - Distributes data ranges
   - Assigns unique seeds for diversity
   - Adds GPU-specific output suffixes

3. **Logging**
   - Creates dedicated log directory for each run
   - Separate stdout/stderr logs per GPU
   - Orchestrator log for overall coordination

#### Usage:
```bash
python run_multi_gpu.py \
  --num_gpus 4 \
  --gpu_ids "0,1,2,3" \
  --method latent_mas_multipath \
  --model_name <model_path> \
  --task gsm8k \
  --max_samples 100 \
  --collect_prm_data \
  --prm_output_dir prm_data \
  ...
```

### 4. Run Script Updates (`run.py`)

#### New Arguments:

- `--gpu_id`: GPU ID for this process (used internally)
- `--data_start_idx`: Starting index in dataset
- `--output_suffix`: Suffix for output files (e.g., "gpu_0")

#### Modified Behavior:

1. **Dataset Loading**
   - Converts iterator to list for slicing
   - Applies data range slicing for multi-GPU mode
   - Respects `data_start_idx` parameter

2. **Output File Naming**
   - Appends output_suffix to batch names
   - Example: `gsm8k_latent_mas_multipath_20231226_120000_gpu_0.pt`

3. **Logging**
   - Logs GPU assignment and data range
   - Indicates multi-GPU mode when active

### 5. Shell Script Enhancement (`collect_training_data.sh`)

#### New Configuration Section:

```bash
# Multi-GPU Configuration
ENABLE_MULTI_GPU=false    # Enable/disable multi-GPU mode
NUM_GPUS=4                # Number of GPUs to use
GPU_IDS="0,1,2,3"        # Comma-separated GPU IDs
```

#### Mode Selection Logic:

**Single-GPU Mode** (ENABLE_MULTI_GPU=false):
- Uses existing CUDA_VISIBLE_DEVICES setting
- Calls `run.py` directly
- Backward compatible with existing workflow

**Multi-GPU Mode** (ENABLE_MULTI_GPU=true):
- Validates GPU configuration
- Calls `run_multi_gpu.py` orchestrator
- Displays multi-GPU specific information
- Verifies merged output file

#### Post-Execution Verification:

- Checks for data files in output directory
- Counts .pt and .json files
- In multi-GPU mode, verifies merged file exists
- Provides appropriate next steps

## Usage Examples

### Single-GPU Mode (Default)

```bash
# Edit collect_training_data.sh
ENABLE_MULTI_GPU=false
MAX_SAMPLES=100

# Run
bash collect_training_data.sh
```

Output:
```
prm_data/
├── gsm8k_latent_mas_multipath_20231226_120000.pt
└── gsm8k_latent_mas_multipath_20231226_120000_metadata.json
```

### Multi-GPU Mode (4 GPUs)

```bash
# Edit collect_training_data.sh
ENABLE_MULTI_GPU=true
NUM_GPUS=4
GPU_IDS="0,1,2,3"
MAX_SAMPLES=400

# Run
bash collect_training_data.sh
```

Output:
```
prm_data/
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_0.pt
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_0_metadata.json
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_1.pt
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_1_metadata.json
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_2.pt
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_2_metadata.json
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_3.pt
├── gsm8k_latent_mas_multipath_20231226_120000_gpu_3_metadata.json
├── gsm8k_latent_mas_multipath_20231226_120000_merged.pt
└── gsm8k_latent_mas_multipath_20231226_120000_merged_metadata.json

output/multi_gpu_logs/20231226_120000/
├── orchestrator_20231226_120000.log
├── gpu_0_stdout.log
├── gpu_0_stderr.log
├── gpu_1_stdout.log
├── gpu_1_stderr.log
├── gpu_2_stdout.log
├── gpu_2_stderr.log
├── gpu_3_stdout.log
└── gpu_3_stderr.log
```

### Using Specific GPUs

```bash
# Use only GPUs 1, 3, 5, 7
ENABLE_MULTI_GPU=true
NUM_GPUS=4
GPU_IDS="1,3,5,7"
```

## Logging

### Log Levels

All components use comprehensive logging with appropriate levels:

**INFO Level:**
- GPU assignment and data range
- Process start/completion
- Data aggregation progress
- Final statistics

**DEBUG Level:**
- Detailed data split calculations
- Subprocess command construction
- File I/O operations
- Memory usage per GPU

### Log Files

1. **Orchestrator Log**: `output/multi_gpu_logs/<timestamp>/orchestrator_<timestamp>.log`
   - Overall coordination
   - Process management
   - Aggregation results

2. **GPU Process Logs**: `output/multi_gpu_logs/<timestamp>/gpu_<id>_stdout.log`
   - Individual GPU execution
   - Model loading and inference
   - Data collection progress

3. **Error Logs**: `output/multi_gpu_logs/<timestamp>/gpu_<id>_stderr.log`
   - Errors and warnings per GPU
   - Stack traces for debugging

## Performance Considerations

### Speedup

With N GPUs, expected speedup is approximately N×, assuming:
- Sufficient GPU memory per device
- No I/O bottlenecks
- Balanced dataset distribution

### Memory Usage

Each GPU process:
- Loads its own model instance
- Processes its data subset independently
- No shared memory between processes

### Best Practices

1. **GPU Selection**: Use GPUs with similar specifications
2. **Batch Size**: Adjust `BATCH_SIZE` based on GPU memory
3. **Sample Distribution**: Ensure `MAX_SAMPLES` is divisible by `NUM_GPUS` for balanced load
4. **Monitoring**: Check individual GPU logs for any failures

## Error Handling

### GPU Validation Failures

If GPU validation fails:
```
✗ ERROR: GPU 2 is not available. Valid range: [0, 1]
```

**Solution**: Adjust `GPU_IDS` to use only available GPUs

### Process Failures

If a GPU process fails:
```
[Multi-GPU] GPU 1 process failed with return code 1
```

**Solution**: Check `gpu_1_stderr.log` for error details

### Partial Results

If some GPUs succeed but others fail:
- Individual GPU data files are preserved
- Aggregation may fail if critical data is missing
- Review logs to identify failed GPU

### Data Aggregation Failures

If aggregation fails:
```
[Multi-GPU Aggregation] No data collected from any GPU
```

**Solution**: 
1. Check if GPU processes completed successfully
2. Verify output directory contains GPU-specific files
3. Review orchestrator log for process errors

## Troubleshooting

### Issue: "No GPU IDs provided"

**Cause**: `GPU_IDS` is empty or malformed

**Solution**: Set `GPU_IDS="0,1,2,3"` with valid GPU IDs

### Issue: "Number of GPU IDs does not match num_gpus"

**Cause**: Mismatch between `NUM_GPUS` and count of IDs in `GPU_IDS`

**Solution**: Ensure `NUM_GPUS=4` and `GPU_IDS="0,1,2,3"` have matching counts

### Issue: "CUDA out of memory"

**Cause**: GPU memory insufficient for model + batch size

**Solution**: 
- Reduce `BATCH_SIZE`
- Reduce `NUM_PATHS`
- Use smaller model

### Issue: "Merged file not found"

**Cause**: Aggregation failed or was skipped

**Solution**:
1. Check orchestrator log for aggregation errors
2. Verify all GPU processes completed successfully
3. Manually run aggregation if needed

## Testing

### Smoke Test (2 GPUs, Small Dataset)

```bash
# Edit collect_training_data.sh
ENABLE_MULTI_GPU=true
NUM_GPUS=2
GPU_IDS="0,1"
MAX_SAMPLES=10
BATCH_SIZE=1

bash collect_training_data.sh
```

Expected: 2 GPU files + 1 merged file

### Full Test (4 GPUs, Large Dataset)

```bash
ENABLE_MULTI_GPU=true
NUM_GPUS=4
GPU_IDS="0,1,2,3"
MAX_SAMPLES=1000

bash collect_training_data.sh
```

Expected: 4 GPU files + 1 merged file with 1000 total samples

## Migration Guide

### From Single-GPU to Multi-GPU

1. **No Code Changes Required**: All changes are configuration-based
2. **Update Script**: Set `ENABLE_MULTI_GPU=true` in `collect_training_data.sh`
3. **Configure GPUs**: Set `NUM_GPUS` and `GPU_IDS`
4. **Run**: Execute script as before
5. **Verify**: Check for merged file in output directory

### Backward Compatibility

- Single-GPU mode remains default
- Existing workflows unaffected
- All previous arguments supported
- Output format unchanged (merged file has same structure)

## File Summary

### Modified Files:

1. **`utils.py`**
   - Added: Data splitting functions
   - Added: Result aggregation functions
   - Added: GPU validation function
   - Lines added: ~350

2. **`run.py`**
   - Added: Multi-GPU command-line arguments
   - Modified: Dataset loading with slicing support
   - Modified: Output file naming with suffix
   - Lines modified: ~30

3. **`collect_training_data.sh`**
   - Added: Multi-GPU configuration section
   - Added: Mode selection logic
   - Enhanced: Post-execution verification
   - Lines added: ~150

### New Files:

1. **`run_multi_gpu.py`**
   - Multi-GPU orchestrator
   - Process management
   - Command construction
   - Result aggregation
   - Lines: ~550

2. **`MULTI_GPU_IMPLEMENTATION.md`**
   - This documentation file

### Unmodified Files:

- `methods/latent_prm/data_storage.py` (already supports custom batch names)
- All other project files

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic GPU Allocation**: Automatically detect and use all available GPUs
2. **Load Balancing**: Adjust distribution based on GPU capabilities
3. **Fault Tolerance**: Continue with remaining GPUs if one fails
4. **Progress Monitoring**: Real-time progress dashboard across all GPUs
5. **Distributed Training**: Extend to distributed model training (not just data collection)

## Conclusion

The multi-GPU data collection feature provides:

✅ **Scalability**: Linear speedup with number of GPUs  
✅ **Simplicity**: Configuration-based, no code changes needed  
✅ **Reliability**: Comprehensive error handling and logging  
✅ **Compatibility**: Fully backward compatible with single-GPU mode  
✅ **Flexibility**: Support for any GPU configuration  

The implementation follows best practices:
- Clear separation of concerns
- Comprehensive logging at appropriate levels
- Robust error handling
- Detailed documentation
- Backward compatibility

For questions or issues, refer to the troubleshooting section or review the detailed logs in `output/multi_gpu_logs/`.

