# Checkpoint Control and Loss Curve Visualization Implementation

## Overview

This document describes the implementation of two key features for the Latent PRM training pipeline:
1. **Configurable checkpoint saving** - Control whether to save checkpoints during training
2. **Loss curve visualization** - Automatic generation of training loss curves as PNG images

## Implementation Date

December 26, 2025

## Changes Made

### 1. Shell Script Modifications (`train_latent_prm.sh`)

#### Added Configuration Parameter
- **Line 59**: Added `SAVE_CHECKPOINTS=true` parameter to control checkpoint saving
  - Default: `true` (maintains backward compatibility)
  - When `false`, no checkpoints are saved during training (saves disk space)
  - Comment updated to clarify that `SAVE_STEPS` only applies when checkpoints are enabled

#### Updated Configuration Display
- **Line 117**: Added display of `SAVE_CHECKPOINTS` value in training configuration output

#### Command Argument Passing
- **Lines 163-165**: Added conditional logic to pass `--no_save_checkpoints` flag when `SAVE_CHECKPOINTS=false`

### 2. Trainer Module Modifications (`methods/latent_prm/trainer.py`)

#### Import Additions
- **Lines 9, 20-22**: Added imports for visualization:
  - `List` type hint for loss tracking lists
  - `matplotlib` with non-interactive backend ('Agg')
  - `matplotlib.pyplot` for plotting

#### Class Constructor Updates
- **Line 121**: Added `save_checkpoints: bool = True` parameter
- **Line 143**: Store `save_checkpoints` as instance variable
- **Lines 230-232**: Initialize loss tracking lists:
  - `self.step_losses`: Loss value at each optimization step
  - `self.step_numbers`: Step numbers for x-axis
  - `self.epoch_losses`: Average loss per epoch

#### Training Loop Modifications

##### Loss Tracking (Lines 453-456)
```python
# Track loss for visualization
self.step_losses.append(batch_loss)
self.step_numbers.append(self.global_step)
logger.debug(f"[Trainer] Tracked loss for step {self.global_step}: {batch_loss:.4f}")
```

##### Conditional Checkpoint Saving (Lines 479-483)
```python
# Save checkpoint (only if enabled)
if self.save_checkpoints and self.global_step % self.save_steps == 0:
    logger.info(f"[Trainer] Saving checkpoint at step {self.global_step}")
    self._save_checkpoint(...)
```

##### Epoch Checkpoint Control (Lines 506-515)
- Checkpoints only saved when `self.save_checkpoints` is `True`
- Appropriate debug logging when checkpoint saving is disabled

##### Final Checkpoint Control (Lines 527-537)
- Final checkpoint only saved when `self.save_checkpoints` is `True`
- Info-level logging for both enabled and disabled states

##### Loss Curve Generation (Lines 539-541)
```python
# Generate and save loss curve visualization
logger.info("[Trainer] Generating loss curve visualization...")
self._save_loss_curve()
```

#### New Method: `_save_loss_curve()` (Lines 637-729)

**Purpose**: Generate and save comprehensive loss curve visualization

**Features**:
1. **Automatic directory creation**: Creates `{output_dir}/img/` directory
2. **Data validation**: Checks for available loss data before plotting
3. **Comprehensive visualization**:
   - Step-wise training loss (blue line, `#2E86AB`)
   - Epoch boundaries (purple dashed vertical lines, `#A23B72`)
   - Epoch average losses (orange horizontal segments, `#F18F01`)
4. **Professional styling**:
   - 12x6 inch figure at 100 DPI
   - Bold axis labels and title
   - Grid with subtle styling
   - Legend with transparency
   - Statistics text box showing:
     - Total steps
     - Number of epochs
     - Best loss
     - Final loss
5. **Robust error handling**: Catches exceptions and logs warnings without failing training
6. **Detailed logging**:
   - INFO: Major events (directory creation, plot generation, save completion)
   - DEBUG: Detailed statistics (min, max, mean loss values, steps per epoch)

**Output**: `{output_dir}/img/loss_curve.png`

#### Statistics Tracking Updates (Lines 544-551)
Added loss tracking data to training statistics:
```python
stats = {
    "total_steps": self.global_step,
    "num_epochs": self.num_epochs,
    "best_loss": self.best_loss,
    "final_loss": self.epoch_losses[-1],
    "epoch_losses": self.epoch_losses,
    "step_losses": self.step_losses,      # NEW
    "step_numbers": self.step_numbers,    # NEW
}
```

#### Command-Line Argument Addition
- **Lines 801-803**: Added `--no_save_checkpoints` argument
  - Type: `action="store_true"`
  - Help text: "Disable checkpoint saving during training"
  - Default: `False` (checkpoints enabled by default)

#### Trainer Instantiation Update
- **Line 851**: Pass `save_checkpoints=not args.no_save_checkpoints` to trainer

## Usage Examples

### Example 1: Default Behavior (Checkpoints Enabled)
```bash
# In train_latent_prm.sh
SAVE_CHECKPOINTS=true
SAVE_STEPS=1000

# Result:
# - Checkpoints saved every 1000 steps
# - Epoch checkpoints saved
# - Final checkpoint saved
# - Loss curve saved to checkpoints/{model_name}/img/loss_curve.png
```

### Example 2: Disable Checkpoints (Save Disk Space)
```bash
# In train_latent_prm.sh
SAVE_CHECKPOINTS=false
SAVE_STEPS=1000  # Ignored when SAVE_CHECKPOINTS=false

# Result:
# - No checkpoints saved during training
# - No epoch checkpoints saved
# - No final checkpoint saved
# - Loss curve still saved to checkpoints/{model_name}/img/loss_curve.png
```

### Example 3: Direct Python Invocation
```bash
# With checkpoints (default)
python -m methods.latent_prm.trainer \
    --model_path /path/to/model \
    --data_dir prm_data \
    --output_dir checkpoints/my_model \
    --num_epochs 5 \
    --save_steps 500

# Without checkpoints
python -m methods.latent_prm.trainer \
    --model_path /path/to/model \
    --data_dir prm_data \
    --output_dir checkpoints/my_model \
    --num_epochs 5 \
    --no_save_checkpoints
```

## Output Structure

```
checkpoints/
└── {model_name}_{timestamp}/
    ├── img/
    │   └── loss_curve.png          # NEW: Loss visualization
    ├── checkpoint-{step}/          # Only if SAVE_CHECKPOINTS=true
    │   ├── pytorch_model.bin
    │   ├── optimizer.pt
    │   ├── scheduler.pt
    │   ├── training_state.json
    │   └── config.json
    ├── epoch_{n}/                  # Only if SAVE_CHECKPOINTS=true
    │   └── ...
    ├── best/                       # Only if SAVE_CHECKPOINTS=true
    │   └── ...
    ├── final/                      # Only if SAVE_CHECKPOINTS=true
    │   └── ...
    ├── training_stats.json         # Always saved (includes loss arrays)
    └── training.log                # Always saved
```

## Loss Curve Visualization Details

### Plot Components

1. **Training Loss Line**
   - Color: Blue (`#2E86AB`)
   - Style: Solid line, 1.5pt width, 70% opacity
   - Data: Loss at each optimization step

2. **Epoch Boundaries**
   - Color: Purple (`#A23B72`)
   - Style: Dashed vertical lines, 1.0pt width, 50% opacity
   - Position: At the end of each epoch

3. **Epoch Average Loss**
   - Color: Orange (`#F18F01`)
   - Style: Horizontal line segments, 2.0pt width, 80% opacity
   - Position: Spans each epoch at the average loss value

4. **Statistics Box**
   - Position: Top-left corner
   - Background: Wheat color with 50% opacity
   - Contents: Total steps, epochs, best loss, final loss

### Styling
- Figure size: 12x6 inches
- DPI: 100 (high quality)
- Grid: Enabled with subtle dotted lines
- Title: "Training Loss Curve - Latent PRM Fine-tuning"
- X-axis: "Training Step"
- Y-axis: "Loss (MSE)"

## Logging Levels

### INFO Level
- Checkpoint saving status (enabled/disabled)
- Checkpoint save operations
- Loss curve generation start/completion
- Training completion messages

### DEBUG Level
- Loss tracking at each step
- Image directory creation
- Steps per epoch calculation
- Loss statistics (min, max, mean)
- Checkpoint component saves (optimizer, scheduler, etc.)

## Error Handling

### Loss Curve Generation
- Wrapped in try-except block
- Failures logged as ERROR with full traceback
- Warning issued but training not interrupted
- Graceful degradation: training completes successfully even if visualization fails

### Checkpoint Saving
- Controlled by boolean flag
- Clear logging when disabled
- No errors or warnings when intentionally disabled

## Backward Compatibility

- **Default behavior unchanged**: Checkpoints are saved by default
- **Existing scripts**: Continue to work without modification
- **New features**: Opt-in via configuration parameters

## Testing Recommendations

1. **Test with checkpoints enabled** (default)
   - Verify checkpoints are saved at correct intervals
   - Verify loss curve is generated
   - Check output directory structure

2. **Test with checkpoints disabled**
   - Verify no checkpoint directories are created
   - Verify loss curve is still generated
   - Verify training completes successfully

3. **Test loss curve visualization**
   - Verify PNG file is created in img/ directory
   - Verify plot contains all expected components
   - Verify statistics box shows correct values

4. **Test error handling**
   - Simulate matplotlib import failure
   - Verify training completes despite visualization failure

## Performance Considerations

### Disk Space Savings
- Disabling checkpoints can save significant disk space
- Typical checkpoint size: 1-5 GB per checkpoint
- For 5 epochs with checkpoints every 1000 steps: potentially 20-100 GB saved

### Memory Overhead
- Loss tracking adds minimal memory overhead
- Each loss value: 8 bytes (float)
- For 10,000 steps: ~80 KB additional memory
- Negligible compared to model size (typically several GB)

### Visualization Generation Time
- Typically < 1 second for plots with < 10,000 points
- Generated only once at the end of training
- No impact on training speed

## Dependencies

- **matplotlib**: Already in requirements.txt
- **numpy**: Implicitly used by matplotlib (already in requirements.txt)
- No new dependencies added

## Future Enhancements (Not Implemented)

Potential future improvements:
1. Multiple loss curve formats (PDF, SVG)
2. Separate plots for training/validation loss
3. Learning rate schedule visualization
4. Gradient norm tracking and visualization
5. Configurable plot styling via config file
6. Interactive plots with plotly
7. Real-time loss curve updates during training

## Related Files

- `train_latent_prm.sh`: Training script with configuration
- `methods/latent_prm/trainer.py`: Core training implementation
- `requirements.txt`: Dependencies (matplotlib already included)

## Conclusion

This implementation provides:
1. ✅ Configurable checkpoint saving to manage disk space
2. ✅ Automatic loss curve visualization for training analysis
3. ✅ Comprehensive logging at appropriate levels
4. ✅ Robust error handling
5. ✅ Backward compatibility
6. ✅ Professional visualization quality

All requirements have been successfully implemented with production-ready code quality.

