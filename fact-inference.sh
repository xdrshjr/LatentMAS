#!/bin/bash

# ==================================================================================================
# Fact-Checking Dataset Inference Script
# ==================================================================================================
# This script performs inference on the CoT fact-checking dataset (cot-fact-wiki)
# Dataset location: data/cot-fact-wiki/test-00000-of-00001.json
# ==================================================================================================

# ==================================Cloud Compute==================================
#export CUDA_VISIBLE_DEVICES=0
#export https_proxy=http://127.0.0.1:7890;
#export http_proxy=http://127.0.0.1:7890;
#export all_proxy=socks5://127.0.0.1:7890;
#export HF_HOME=/autodl-fs/data/models
#MODEL_NAME="/autodl-fs/data/models/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Local Compute==================================
export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
export HF_HOME=/mnt/mydisk/models
MODEL_NAME="/home/xdrshjr/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

# ==================================Configuration Parameters==================================
# Number of samples to process from the dataset (-1 for all samples)
MAX_SAMPLES=5

# Output directories (automatically created by the Python script)
# - Question-answer records: output/res/
# - Result logs: output/simple_res/
# - CSV results: output/csv_res/
OUTPUT_DIR_RECORDS="output/res"
OUTPUT_DIR_LOGS="output/simple_res"
OUTPUT_DIR_CSV="output/csv_res"

# Model configuration
# You can change this to use different models
# MODEL_NAME="Qwen/Qwen3-0.6B"  # Use HuggingFace model name
# Or use local model path (as set above)

# Inference method: baseline, text_mas, latent_mas, latent_mas_multipath
METHOD="latent_mas_multipath"

# Generation parameters
MAX_NEW_TOKENS=2048
TEMPERATURE=0.3
TOP_P=0.95
GENERATE_BS=1
SEED=42

# Latent reasoning parameters (for latent_mas and latent_mas_multipath)
LATENT_STEPS=10

# Multi-path parameters (for latent_mas_multipath only)
NUM_PATHS=30
NUM_PARENT_PATHS=10
DIVERSITY_STRATEGY="temperature"
PRUNING_STRATEGY="adaptive"
TOPK_K=3
MERGE_THRESHOLD=0.9
BRANCH_THRESHOLD=0.5
LATENT_CONSISTENCY_METRIC="cosine"

# Visualization control
DISABLE_VISUALIZATION="--disable_visualization"  # Comment out to enable visualization

# ==================================================================================================

echo "=========================================="
echo "Fact-Checking Dataset Inference"
echo "=========================================="
echo "Dataset: data/cot-fact-wiki/test-00000-of-00001.json"
echo "Method: ${METHOD}"
echo "Model: ${MODEL_NAME}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "Output Directories:"
echo "  - Records: ${OUTPUT_DIR_RECORDS}"
echo "  - Logs: ${OUTPUT_DIR_LOGS}"
echo "  - CSV: ${OUTPUT_DIR_CSV}"
echo "=========================================="

# Create output directories if they don't exist
mkdir -p "${OUTPUT_DIR_RECORDS}"
mkdir -p "${OUTPUT_DIR_LOGS}"
mkdir -p "${OUTPUT_DIR_CSV}"

# Run inference based on selected method
if [ "${METHOD}" == "baseline" ]; then
    echo "Running baseline method..."
    python run.py \
      --method baseline \
      --model_name "${MODEL_NAME}" \
      --task cot_fact_wiki \
      --max_samples ${MAX_SAMPLES} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature ${TEMPERATURE} \
      --top_p ${TOP_P} \
      --generate_bs ${GENERATE_BS} \
      --seed ${SEED} \
      ${DISABLE_VISUALIZATION}

elif [ "${METHOD}" == "text_mas" ]; then
    echo "Running text-based multi-agent system..."
    python run.py \
      --method text_mas \
      --model_name "${MODEL_NAME}" \
      --task cot_fact_wiki \
      --prompt sequential \
      --max_samples ${MAX_SAMPLES} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature ${TEMPERATURE} \
      --top_p ${TOP_P} \
      --generate_bs ${GENERATE_BS} \
      --seed ${SEED} \
      ${DISABLE_VISUALIZATION}

elif [ "${METHOD}" == "latent_mas" ]; then
    echo "Running latent multi-agent system..."
    python run.py \
      --method latent_mas \
      --model_name "${MODEL_NAME}" \
      --task cot_fact_wiki \
      --prompt sequential \
      --max_samples ${MAX_SAMPLES} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --temperature ${TEMPERATURE} \
      --top_p ${TOP_P} \
      --generate_bs ${GENERATE_BS} \
      --latent_steps ${LATENT_STEPS} \
      --seed ${SEED} \
      ${DISABLE_VISUALIZATION}

elif [ "${METHOD}" == "latent_mas_multipath" ]; then
    echo "Running multi-path latent reasoning..."
    python run.py \
      --method latent_mas_multipath \
      --model_name "${MODEL_NAME}" \
      --task cot_fact_wiki \
      --prompt sequential \
      --max_samples ${MAX_SAMPLES} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --seed ${SEED} \
      --generate_bs ${GENERATE_BS} \
      --latent_steps ${LATENT_STEPS} \
      --num_paths ${NUM_PATHS} \
      --num_parent_paths ${NUM_PARENT_PATHS} \
      --diversity_strategy "${DIVERSITY_STRATEGY}" \
      --temperature ${TEMPERATURE} \
      --top_p ${TOP_P} \
      --enable_branching \
      --enable_merging \
      --pruning_strategy "${PRUNING_STRATEGY}" \
      --topk_k ${TOPK_K} \
      --merge_threshold ${MERGE_THRESHOLD} \
      --branch_threshold ${BRANCH_THRESHOLD} \
      --latent_space_realign \
      --latent_consistency_metric "${LATENT_CONSISTENCY_METRIC}" \
      ${DISABLE_VISUALIZATION}

else
    echo "Error: Unknown method '${METHOD}'"
    echo "Available methods: baseline, text_mas, latent_mas, latent_mas_multipath"
    exit 1
fi

echo "=========================================="
echo "Inference completed!"
echo "Results saved to:"
echo "  - Question-answer records: ${OUTPUT_DIR_RECORDS}/*.jsonl"
echo "  - Result logs: ${OUTPUT_DIR_LOGS}/*.log"
echo "  - CSV results: ${OUTPUT_DIR_CSV}/results.csv"
echo "=========================================="

