#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
export HF_HOME=/mnt/mydisk/models


INPUT_FILE="output/res/cot_fact_wiki_latent_mas_multipath_c1899de289a04d12100db370d81485cdf75e47ca_20251230_152200.jsonl"
OUTPUT_DIR="output/fact-res"

# Model configuration
# You can override this by editing the script or passing arguments if extended
CHECK_MODEL="jdqqjr/Qwen2.5-0.5B-Instruct_8epoch_Fact_Checker"

echo "Starting fact check accuracy evaluation..."
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $CHECK_MODEL"

# Ensure python is in path, or specify python path if needed. 
# Assuming 'python' is the correct interpreter.
python scripts/check_fact_accuracy.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --check_model_name "$CHECK_MODEL"

echo "Fact check completed."
