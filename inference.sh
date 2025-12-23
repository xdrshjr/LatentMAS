export CUDA_VISIBLE_DEVICES=0
export https_proxy=http://127.0.0.1:7890;
export http_proxy=http://127.0.0.1:7890;
export all_proxy=socks5://127.0.0.1:7890;
export HF_HOME="/autodl-fs/data/models"

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
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --config_preset balanced \
  --max_samples -1 \
  --max_new_tokens 512 \
  --seed 42 \
  --latent_steps 10 \
  --enable_visualization
  # Add --disable_visualization to skip visualization generation