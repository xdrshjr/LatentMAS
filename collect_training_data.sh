#!/bin/bash

# ==================================================================================================
# Latent PRM Training Data Collection Script
# ==================================================================================================
# This script collects multi-path latent reasoning data for training a Process Reward Model (PRM)
# on the Qwen-0.6B model. It runs the latent_mas_multipath method in data collection mode.
#
# Key differences from inference.sh:
# - Enables PRM data collection mode (--collect_prm_data)
# - Disables pruning and merging to maximize path diversity
# - Saves all reasoning paths with tree structure and scores
# - Data is saved to output/prm_data/ directory
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
# Dataset configuration
TASK="gsm8k"                    # Dataset to use (gsm8k, aime2024, etc.)
MAX_SAMPLES=2                 # Number of questions to collect data for (-1 for all)
SEED=42                         # Random seed for reproducibility

# Model configuration
MAX_NEW_TOKENS=2048             # Maximum tokens for generation
TEMPERATURE=0.7                 # Baseline temperature for diversity
TOP_P=0.95                      # Nucleus sampling parameter

# Multi-path configuration (for data collection)
NUM_PATHS=10                    # Number of paths to generate per agent (increased for diversity)
LATENT_STEPS=5                  # Number of latent thinking steps per path
DIVERSITY_STRATEGY="hybrid"     # Diversity strategy (temperature/noise/hybrid)

# Data collection configuration
OUTPUT_DIR="prm_data"           # Directory to save collected data (project root)
BATCH_SIZE=1                    # Process one question at a time for data collection

# Logging configuration
LOG_LEVEL="INFO"                # Logging level (DEBUG/INFO/WARNING/ERROR)

# ==================================Data Collection Mode==================================
echo "========================================"
echo "Latent PRM Training Data Collection"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Task: ${TASK}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "Num Paths: ${NUM_PATHS}"
echo "Latent Steps: ${LATENT_STEPS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""

python run.py \
  --method latent_mas_multipath \
  --model_name ${MODEL_NAME} \
  --task ${TASK} \
  --prompt sequential \
  --max_samples ${MAX_SAMPLES} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --seed ${SEED} \
  --generate_bs ${BATCH_SIZE} \
  --latent_steps ${LATENT_STEPS} \
  --num_paths ${NUM_PATHS} \
  --diversity_strategy ${DIVERSITY_STRATEGY} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --latent_space_realign \
  --latent_consistency_metric 'cosine' \
  --disable_visualization \
  --log_level ${LOG_LEVEL} \
  --collect_prm_data \
  --prm_output_dir ${OUTPUT_DIR} \
  --prm_disable_pruning \
  --prm_disable_merging

echo ""
echo "========================================"
echo "Data Collection Complete!"
echo "========================================"
echo "Data saved to: ${OUTPUT_DIR}"
echo ""
echo "Verifying saved data..."
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR})" ]; then
    echo "✓ Data directory exists and contains files"
    echo "Files in ${OUTPUT_DIR}:"
    ls -lh ${OUTPUT_DIR}
else
    echo "✗ WARNING: Data directory is empty or does not exist!"
    echo "Please check the logs for errors"
fi
echo ""
echo "Next steps:"
echo "1. Check the collected data in ${OUTPUT_DIR}"
echo "2. Use the data to train a PRM model on Qwen-0.6B"
echo "3. Fine-tune the latent reasoning process"
echo "========================================"

