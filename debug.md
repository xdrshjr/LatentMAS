
```shell


# cloud compute
PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0;https_proxy=http://127.0.0.1:7890;http_proxy=http://127.0.0.1:7890;all_proxy=socks5://127.0.0.1:7890;HF_HOME=/autodl-fs/data/models

# local compute
PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=1;https_proxy=http://127.0.0.1:7897;http_proxy=http://127.0.0.1:7897;all_proxy=socks5://127.0.0.1:7897;HF_HOME=/mnt/mydisk/models


```


```shell
# inference 1
python run.py
--method latent_mas
--model_name Qwen/Qwen3-4B
--task gsm8k
--prompt sequential
--max_samples -1
--max_new_tokens 2048


# collection
python run_multi_gpu.py
--num_gpus 1
--gpu_ids 0
--method latent_mas_multipath
--model_name /home/xdrshjr/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
--task gsm8k
--prompt sequential
--max_samples 10
--max_new_tokens 2048
--seed 42
--generate_bs 1
--latent_steps 10
--num_paths 5
--diversity_strategy hybrid
--temperature 0.5
--top_p 0.95
--latent_space_realign
--latent_consistency_metric cosine
--log_level INFO
--collect_prm_data
--prm_output_dir ./output/debug
--prm_disable_pruning
--prm_disable_merging

```

