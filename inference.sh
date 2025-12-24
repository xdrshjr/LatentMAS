
# ==================================Cloud Compute==================================
#export CUDA_VISIBLE_DEVICES=0
#export https_proxy=http://127.0.0.1:7890;
#export http_proxy=http://127.0.0.1:7890;
#export all_proxy=socks5://127.0.0.1:7890;
#export HF_HOME=/autodl-fs/data/models

# ==================================Local Compute==================================
export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
export HF_HOME=/mnt/mydisk/models

# Basic latent reasoning
#python run.py \
#--method latent_mas \
#--model_name Qwen/Qwen3-4B \
#--task gsm8k \
#--prompt sequential \
#--max_samples -1 \
#--max_new_tokens 2048


# MultiPATH conservative latent reasoning
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-0.6B \
  --task gsm8k \
  --prompt sequential \
  --config_preset balanced \
  --max_samples 20 \
  --max_new_tokens 2048 \
  --seed 42 \
  --generate_bs 1 \
  --latent_steps 5 \
  --num_paths 20 \
  --diversity_strategy 'temperature' \
  --temperature 0.5 \
  --enable_branching \
  --enable_merging \
  --latent_consistency_metric cosine \
  --disable_visualization