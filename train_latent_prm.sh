#!/bin/bash

# ==================================================================================================
# Latent PRM Fine-tuning Script
# ==================================================================================================
# This script fine-tunes the Qwen-0.6B model on collected latent reasoning path data
# for Process Reward Model (PRM) training. It performs full-parameter fine-tuning
# to learn to predict path quality scores from latent sequences.
#
# Key features:
# - Full-parameter fine-tuning (not LoRA)
# - Trains on latent sequences from collected data
# - Uses prm_score as training target
# - Progress bars and loss tracking in console
# - Configurable checkpoint saving frequency
# ==================================================================================================

# ==================================Cloud Compute==================================
export CUDA_VISIBLE_DEVICES=0
export https_proxy=http://127.0.0.1:7890;
export http_proxy=http://127.0.0.1:7890;
export all_proxy=socks5://127.0.0.1:7890;
export HF_HOME=/autodl-fs/data/models
MODEL_NAME="/autodl-fs/data/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Local Compute==================================
#export CUDA_VISIBLE_DEVICES=1
#export https_proxy=http://127.0.0.1:7897
#export http_proxy=http://127.0.0.1:7897
#export all_proxy=socks5://127.0.0.1:7897
#export HF_HOME=/mnt/mydisk/models
#MODEL_NAME="/home/xdrshjr/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Configuration==================================
# Data configuration
DATA_DIR="prm_data"                # Directory with collected training data (.pt files)

# Output configuration
OUTPUT_DIR="checkpoints/qwen_prm"  # Directory to save checkpoints
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

# Training hyperparameters
NUM_EPOCHS=5                       # Number of training epochs
BATCH_SIZE=4                       # Batch size (adjust based on GPU memory)
LEARNING_RATE=2e-5                 # Learning rate for full fine-tuning
WEIGHT_DECAY=0.01                  # Weight decay for regularization
WARMUP_RATIO=0.1                   # Warmup ratio (10% of total steps)
GRADIENT_ACCUMULATION_STEPS=2      # Gradient accumulation for effective larger batch size
MAX_GRAD_NORM=1.0                  # Maximum gradient norm for clipping

# Model configuration
POOLING_STRATEGY="mean"            # Pooling strategy: mean/last/max
DROPOUT_PROB=0.1                   # Dropout probability
MAX_SEQ_LENGTH=None                 # Maximum sequence length (None = no limit)
USE_PRM_SCORE=true                 # Use prm_score (true) or score (false) as target

# Logging and checkpointing
SAVE_CHECKPOINTS=false              # Whether to save checkpoints during training
SAVE_STEPS=1000000                 # Save checkpoint every N steps (only if SAVE_CHECKPOINTS=true)
LOGGING_STEPS=10                   # Log metrics every N steps
LOG_LEVEL="INFO"                   # Logging level (DEBUG/INFO/WARNING/ERROR)

# Device configuration
DEVICE="cuda"                      # Training device (cuda/cpu)
MIXED_PRECISION=true               # Use mixed precision training (fp16)
SEED=42                            # Random seed for reproducibility

# ==================================Validation==================================
echo "========================================"
echo "Latent PRM Fine-tuning"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "✗ ERROR: Data directory does not exist: ${DATA_DIR}"
    echo "Please run collect_training_data.sh first to collect training data"
    exit 1
fi

# Check if data directory contains .pt files
PT_FILES=$(find "${DATA_DIR}" -name "*.pt" | wc -l)
if [ ${PT_FILES} -eq 0 ]; then
    echo "✗ ERROR: No .pt files found in ${DATA_DIR}"
    echo "Please run collect_training_data.sh first to collect training data"
    exit 1
fi

echo "✓ Found ${PT_FILES} .pt files in ${DATA_DIR}"
echo ""

# ==================================Training Configuration==================================
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Training hyperparameters:"
echo "  - Num epochs: ${NUM_EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  - Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  - Weight decay: ${WEIGHT_DECAY}"
echo "  - Warmup ratio: ${WARMUP_RATIO}"
echo "  - Max gradient norm: ${MAX_GRAD_NORM}"
echo ""
echo "Model configuration:"
echo "  - Pooling strategy: ${POOLING_STRATEGY}"
echo "  - Dropout probability: ${DROPOUT_PROB}"
echo "  - Max sequence length: ${MAX_SEQ_LENGTH}"
echo "  - Use prm_score: ${USE_PRM_SCORE}"
echo ""
echo "Logging and checkpointing:"
echo "  - Save checkpoints: ${SAVE_CHECKPOINTS}"
echo "  - Save steps: ${SAVE_STEPS}"
echo "  - Logging steps: ${LOGGING_STEPS}"
echo "  - Log level: ${LOG_LEVEL}"
echo ""
echo "Device configuration:"
echo "  - Device: ${DEVICE}"
echo "  - Mixed precision: ${MIXED_PRECISION}"
echo "  - Random seed: ${SEED}"
echo "========================================"
echo ""

# ==================================Training==================================
echo "Starting training..."
echo ""

# Build command arguments
CMD_ARGS=(
    --model_path "${MODEL_NAME}"
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --num_epochs ${NUM_EPOCHS}
    --batch_size ${BATCH_SIZE}
    --learning_rate ${LEARNING_RATE}
    --weight_decay ${WEIGHT_DECAY}
    --warmup_ratio ${WARMUP_RATIO}
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    --max_grad_norm ${MAX_GRAD_NORM}
    --save_steps ${SAVE_STEPS}
    --logging_steps ${LOGGING_STEPS}
    --pooling_strategy ${POOLING_STRATEGY}
    --dropout_prob ${DROPOUT_PROB}
    --device ${DEVICE}
    --seed ${SEED}
    --log_level ${LOG_LEVEL}
)

# Add optional arguments
if [ "${USE_PRM_SCORE}" = true ]; then
    CMD_ARGS+=(--use_prm_score)
fi

if [ "${MIXED_PRECISION}" = false ]; then
    CMD_ARGS+=(--no_mixed_precision)
fi

if [ "${SAVE_CHECKPOINTS}" = false ]; then
    CMD_ARGS+=(--no_save_checkpoints)
fi

if [ ! -z "${MAX_SEQ_LENGTH}" ] && [ "${MAX_SEQ_LENGTH}" != "None" ]; then
    CMD_ARGS+=(--max_seq_length ${MAX_SEQ_LENGTH})
fi

# Run training
python -m methods.latent_prm.trainer "${CMD_ARGS[@]}"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training Completed Successfully!"
    echo "========================================"
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    echo "Checkpoints saved:"
    ls -lh "${OUTPUT_DIR}"
    echo ""
    echo "To use the fine-tuned model:"
    echo "1. Check the best checkpoint: ${OUTPUT_DIR}/best/"
    echo "2. Check the final checkpoint: ${OUTPUT_DIR}/final/"
    echo "3. Check training statistics: ${OUTPUT_DIR}/training_stats.json"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ Training Failed!"
    echo "========================================"
    echo "Please check the logs for errors"
    echo "Log file: ${OUTPUT_DIR}/training.log"
    echo "========================================"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Evaluate the fine-tuned model on validation data"
echo "2. Use the model for improved path scoring in inference"
echo "3. Compare performance with baseline model"
echo "========================================"

