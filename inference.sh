
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

# ==================================Basic latent reasoning==================================
#python run.py \
#  --method latent_mas \
#  --model_name Qwen/Qwen3-0.6B \
#  --task gsm8k \
#  --prompt sequential \
#  --temperature 0.5 \
#  --max_samples 50 \
#  --disable_visualization \
#  --generate_bs 1 \
#  --max_new_tokens 2048


# ==================================MultiPATH conservative latent reasoning==================================
python run.py \
  --method latent_mas_multipath \
  --model_name ${MODEL_NAME} \
  --task gsm8k \
  --prompt sequential \
  --max_samples 50 \
  --max_new_tokens 2048 \
  --seed 42 \
  --generate_bs 1 \
  --latent_steps 2 \
  --num_paths 20 \
  --diversity_strategy 'hybrid' \
  --temperature 0.5 \
  --top_p 0.95 \
  --enable_branching \
  --enable_merging \
  --pruning_strategy 'adaptive' \
  --merge_threshold 0.9 \
  --branch_threshold 0.5 \
  --latent_space_realign \
  --latent_consistency_metric 'kl_divergence' \
  --disable_visualization