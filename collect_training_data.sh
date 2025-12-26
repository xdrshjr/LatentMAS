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
#
# Multi-GPU Support:
# - Set ENABLE_MULTI_GPU=true to enable parallel data collection across multiple GPUs
# - Each GPU processes a distinct subset of the dataset
# - Results are automatically aggregated at the end
# ==================================================================================================




# ==================================Single-GPU Configuration==================================
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1

# ==================================Multi-GPU Configuration==================================
# Set to true to enable multi-GPU parallel data collection
export CUDA_VISIBLE_DEVICES=0,1
ENABLE_MULTI_GPU=true

# Number of GPUs to use (only effective when ENABLE_MULTI_GPU=true)
NUM_GPUS=2

# Comma-separated GPU IDs to use (e.g., "0,1,2,3")
# Must match NUM_GPUS count
GPU_IDS="0,1"



# ==================================Cloud Compute==================================
# These settings are used when ENABLE_MULTI_GPU=false

#export https_proxy=http://127.0.0.1:7890;
#export http_proxy=http://127.0.0.1:7890;
#export all_proxy=socks5://127.0.0.1:7890;
#export HF_HOME=/autodl-fs/data/models
#MODEL_NAME="/autodl-fs/data/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Local Compute==================================
# Uncomment these for local development (when ENABLE_MULTI_GPU=false)

export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
export HF_HOME=/mnt/mydisk/models
MODEL_NAME="/home/xdrshjr/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Configuration==================================
# Dataset configuration
TASK="gsm8k"                    # Dataset to use (gsm8k, aime2024, etc.)
MAX_SAMPLES=100                 # Number of questions to collect data for (-1 for all)
SEED=42                         # Random seed for reproducibility

# Model configuration
MAX_NEW_TOKENS=2048             # Maximum tokens for generation
TEMPERATURE=0.7                 # Baseline temperature for diversity
TOP_P=0.95                      # Nucleus sampling parameter

# Multi-path configuration (for data collection)
NUM_PATHS=20                    # Number of paths to generate per agent (increased for diversity)
LATENT_STEPS=5                  # Number of latent thinking steps per path
DIVERSITY_STRATEGY="hybrid"     # Diversity strategy (temperature/noise/hybrid)

# Data collection configuration
OUTPUT_DIR="prm_data"           # Directory to save collected data (project root)
BATCH_SIZE=1                    # Process one question at a time for data collection

# Logging configuration
LOG_LEVEL="INFO"                # Logging level (DEBUG/INFO/WARNING/ERROR)

# ==================================Execution Mode Selection==================================
if [ "$ENABLE_MULTI_GPU" = true ]; then
    # ========================================
    # MULTI-GPU MODE
    # ========================================
    echo "========================================"
    echo "Latent PRM Training Data Collection"
    echo "========================================"
    echo "MODE: Multi-GPU Parallel Processing"
    echo "Model: ${MODEL_NAME}"
    echo "Task: ${TASK}"
    echo "Max Samples: ${MAX_SAMPLES}"
    echo "Num Paths: ${NUM_PATHS}"
    echo "Latent Steps: ${LATENT_STEPS}"
    echo "Output Directory: ${OUTPUT_DIR}"
    echo "----------------------------------------"
    echo "Multi-GPU Configuration:"
    echo "  Number of GPUs: ${NUM_GPUS}"
    echo "  GPU IDs: ${GPU_IDS}"
    echo "  Samples per GPU: ~$((MAX_SAMPLES / NUM_GPUS))"
    echo "========================================"
    echo ""
    
    # Validate GPU configuration
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    if [ ${#GPU_ARRAY[@]} -ne $NUM_GPUS ]; then
        echo "✗ ERROR: Number of GPU IDs (${#GPU_ARRAY[@]}) does not match NUM_GPUS ($NUM_GPUS)"
        echo "Please check GPU_IDS configuration"
        exit 1
    fi
    
    echo "Launching multi-GPU orchestrator..."
    echo ""
    
    # Launch multi-GPU orchestrator
    # Note: CUDA_VISIBLE_DEVICES is managed by the orchestrator for each subprocess
    python run_multi_gpu.py \
      --num_gpus ${NUM_GPUS} \
      --gpu_ids ${GPU_IDS} \
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
      --log_level ${LOG_LEVEL} \
      --collect_prm_data \
      --prm_output_dir ${OUTPUT_DIR} \
      --prm_disable_pruning \
      --prm_disable_merging
    
    EXECUTION_STATUS=$?
    
    echo ""
    echo "========================================"
    if [ $EXECUTION_STATUS -eq 0 ]; then
        echo "Multi-GPU Data Collection Complete!"
    else
        echo "Multi-GPU Data Collection Failed!"
        echo "Exit code: $EXECUTION_STATUS"
    fi
    echo "========================================"
    
else
    # ========================================
    # SINGLE-GPU MODE
    # ========================================
    echo "========================================"
    echo "Latent PRM Training Data Collection"
    echo "========================================"
    echo "MODE: Single-GPU Processing"
    echo "Model: ${MODEL_NAME}"
    echo "Task: ${TASK}"
    echo "Max Samples: ${MAX_SAMPLES}"
    echo "Num Paths: ${NUM_PATHS}"
    echo "Latent Steps: ${LATENT_STEPS}"
    echo "Output Directory: ${OUTPUT_DIR}"
    echo "----------------------------------------"
    echo "GPU Configuration:"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
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
    
    EXECUTION_STATUS=$?
    
    echo ""
    echo "========================================"
    if [ $EXECUTION_STATUS -eq 0 ]; then
        echo "Data Collection Complete!"
    else
        echo "Data Collection Failed!"
        echo "Exit code: $EXECUTION_STATUS"
    fi
    echo "========================================"
fi

# ==================================Post-Execution Verification==================================
echo "Data saved to: ${OUTPUT_DIR}"
echo ""
echo "Verifying saved data..."
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR})" ]; then
    echo "✓ Data directory exists and contains files"
    echo ""
    echo "Files in ${OUTPUT_DIR}:"
    ls -lh ${OUTPUT_DIR}
    echo ""
    
    # Count data files
    NUM_PT_FILES=$(find ${OUTPUT_DIR} -name "*.pt" | wc -l)
    NUM_JSON_FILES=$(find ${OUTPUT_DIR} -name "*.json" | wc -l)
    echo "Summary:"
    echo "  - Data files (.pt): ${NUM_PT_FILES}"
    echo "  - Metadata files (.json): ${NUM_JSON_FILES}"
    
    if [ "$ENABLE_MULTI_GPU" = true ]; then
        # Check for merged file in multi-GPU mode
        MERGED_FILES=$(find ${OUTPUT_DIR} -name "*_merged.pt" | wc -l)
        if [ $MERGED_FILES -gt 0 ]; then
            echo "  - Merged files: ${MERGED_FILES}"
            echo "✓ Multi-GPU data aggregation successful"
        else
            echo "✗ WARNING: No merged file found in multi-GPU mode"
            echo "  Data may not have been aggregated properly"
        fi
    fi
else
    echo "✗ WARNING: Data directory is empty or does not exist!"
    echo "Please check the logs for errors"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Check the collected data in ${OUTPUT_DIR}"
if [ "$ENABLE_MULTI_GPU" = true ]; then
    echo "2. Review multi-GPU logs in output/multi_gpu_logs/"
    echo "3. Verify the merged data file (*_merged.pt)"
    echo "4. Use the merged data to train a PRM model"
else
    echo "2. Use the data to train a PRM model on Qwen-0.6B"
    echo "3. Fine-tune the latent reasoning process"
fi
echo "========================================"

# Exit with the same status as the main execution
exit $EXECUTION_STATUS

