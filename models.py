import os
import csv
import torch
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

# Logger setup
logger = logging.getLogger(__name__)


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:

            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas":
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # Set padding_side to 'left' for decoder-only models
            self.tokenizer.padding_side = 'left'

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval()
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Set padding_side to 'left' for decoder-only models
        self.tokenizer.padding_side = 'left'
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt)) # use tokenizer transfer into model template format
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None # 输入词嵌入, [vocab_size, emb_size][151669, 2560]
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None # 输出词嵌入, [2560, 151669]
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)  # detach是为了断开梯度，防止影响模型
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight) # 衡量嵌入之间的相似度
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) # 添加正则项
        gram = gram + reg   # 方程的左侧项
        rhs = torch.matmul(output_weight.T, input_weight)   # 衡量输入和输出嵌入之间的关系， 方程的右侧项
        realign_matrix = torch.linalg.solve(gram, rhs)  # 求解方程： GM = R，寻找一个M，M = G^-1 R
        target_norm = input_weight.norm(dim=1).mean().detach()  # 计算向量的平均长度，作为缩放基准

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args) # 获取初始化的latent realigin metrix
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)  # 设置模型对应的realign metrix，和模型一一对应

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix) # 对齐hidden到输入空间, H:[B, D] x M:[D, D] = A:[B, D]

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values  # get the kv-cache hidden states

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D], 获取第一层的所有batch的最后一个 token 的 hidden_states, embedding层
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D], 获取最后一层的所有batch的最后一个 token 的 hidden_states, hidden层
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())    # 所有隐藏层向量

        for step in range(latent_steps):    # 使用latent思考多步

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)  # 取出hidden_vec，然后对齐到输入空间, [B, D], [4, 2560]

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()

            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)   # kv-cache的长度， 121, 获取已经缓存的kv-cache长度
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )   # 把输出latent，连同kv-cache一起交给LLM
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = []
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding


        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings
    
    @torch.no_grad()
    def generate_diverse_latent_paths(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        num_paths: int = 5,
        latent_steps: int = 10,
        diversity_strategy: Optional[Any] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple diverse latent reasoning paths.
        
        This method generates multiple reasoning paths with diversity to explore
        different reasoning trajectories in the latent space.
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            num_paths: Number of diverse paths to generate
            latent_steps: Number of latent thinking steps per path
            diversity_strategy: Strategy for generating diversity (optional)
            past_key_values: Optional past KV cache to continue from
            
        Returns:
            List of dictionaries, each containing:
                - 'path_id': Path identifier
                - 'latent_history': List of latent vectors
                - 'hidden_states': Final hidden states
                - 'kv_cache': Final KV cache
                - 'metadata': Additional information about the path
        """
        logger.info(f"[ModelWrapper.generate_diverse_latent_paths] Generating {num_paths} diverse paths "
                   f"with {latent_steps} latent steps each")
        
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        
        # Import diversity strategy here to avoid circular imports
        if diversity_strategy is None:
            from methods.diversity_strategies import HybridDiversityStrategy
            diversity_strategy = HybridDiversityStrategy()
            logger.debug("[ModelWrapper.generate_diverse_latent_paths] Using default HybridDiversityStrategy")
        
        paths = []
        
        for path_idx in range(num_paths):
            logger.info(f"[Path Generation] Starting path {path_idx + 1}/{num_paths}")
            
            # Get temperature for this path
            temperature = diversity_strategy.get_temperature(path_idx, num_paths)
            logger.debug(f"[Path Generation] Path {path_idx + 1} temperature: {temperature:.4f}")
            
            # Generate initial hidden states
            if past_key_values is not None:
                past_len = _past_length(past_key_values)
                if past_len > 0:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    full_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                else:
                    full_attention_mask = attention_mask
            else:
                full_attention_mask = attention_mask
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # Apply diversity strategy to initial hidden state
            last_hidden = diversity_strategy.apply(
                last_hidden,
                path_idx,
                num_paths
            )
            
            # Track latent history for this path
            latent_history = []
            
            # Generate latent steps
            for step in range(latent_steps):
                logger.debug(f"[Path Generation] Path {path_idx + 1}, latent step {step + 1}/{latent_steps}")
                
                # Apply latent realignment
                source_model = self.HF_model if hasattr(self, "HF_model") else self.model
                latent_vec = self._apply_latent_realignment(last_hidden, source_model)
                
                # Log latent vector statistics
                latent_norm = latent_vec.norm(dim=-1).mean().item()
                logger.debug(f"[Path Generation] Path {path_idx + 1}, step {step + 1}: latent norm = {latent_norm:.4f}")
                
                # Store latent vector
                latent_history.append(latent_vec.detach().clone())
                
                # Continue generation with latent vector
                latent_embed = latent_vec.unsqueeze(1)
                past_len = _past_length(past)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # Store path information
            path_info = {
                'path_id': path_idx,
                'latent_history': latent_history,
                'hidden_states': last_hidden.detach().clone(),
                'kv_cache': past,
                'metadata': {
                    'temperature': temperature,
                    'latent_steps': latent_steps,
                    'diversity_strategy': diversity_strategy.__class__.__name__,
                }
            }
            paths.append(path_info)
            
            # Log path completion with statistics
            hidden_norm = last_hidden.norm(dim=-1).mean().item()
            logger.info(f"[Path Generation] Completed path {path_idx + 1}/{num_paths} - final hidden norm: {hidden_norm:.4f}")
            logger.debug(f"[Path Generation] Path {path_idx + 1} metadata: {path_info['metadata']}")
        
        logger.info(f"[ModelWrapper.generate_diverse_latent_paths] Generated {len(paths)} diverse paths")
        return paths
    
    @torch.no_grad()
    def generate_latent_with_branching(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        num_branches: int = 3,
        latent_steps: int = 10,
        diversity_strategy: Optional[Any] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple latent continuations from an existing state.
        
        This method branches from an existing KV cache state to generate
        multiple diverse continuations. Useful for adaptive branching during
        multi-path reasoning.
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            num_branches: Number of branches to create
            latent_steps: Number of latent steps for each branch
            diversity_strategy: Strategy for generating diversity
            past_key_values: KV cache to branch from (required)
            
        Returns:
            List of branch dictionaries with latent histories and states
        """
        logger.info(f"[ModelWrapper.generate_latent_with_branching] Creating {num_branches} branches "
                   f"with {latent_steps} steps each")
        
        if past_key_values is None:
            logger.warning("[ModelWrapper.generate_latent_with_branching] No past_key_values provided, "
                         "falling back to generate_diverse_latent_paths")
            return self.generate_diverse_latent_paths(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_paths=num_branches,
                latent_steps=latent_steps,
                diversity_strategy=diversity_strategy,
                past_key_values=None,
            )
        
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        
        # Import diversity strategy
        if diversity_strategy is None:
            from methods.diversity_strategies import NoiseDiversityStrategy
            diversity_strategy = NoiseDiversityStrategy(noise_scale=0.1)
            logger.debug("[ModelWrapper.generate_latent_with_branching] Using default NoiseDiversityStrategy")
        
        # Get current hidden state from the branching point
        past_len = _past_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        else:
            full_attention_mask = attention_mask
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        
        base_hidden = outputs.hidden_states[-1][:, -1, :]
        base_past = outputs.past_key_values
        
        branches = []
        
        for branch_idx in range(num_branches):
            logger.info(f"[Branching] Creating branch {branch_idx + 1}/{num_branches}")
            
            # Apply diversity to create different branch starting points
            branch_hidden = diversity_strategy.apply(
                base_hidden.clone(),
                branch_idx,
                num_branches
            )
            logger.debug(f"[Branching] Branch {branch_idx + 1} divergence applied")
            
            # Note: KV cache cannot be easily cloned, so we'll reuse base_past
            # In practice, each branch will diverge as it generates new tokens
            past = base_past
            last_hidden = branch_hidden
            
            latent_history = []
            
            # Generate latent steps for this branch
            for step in range(latent_steps):
                logger.debug(f"[Branching] Branch {branch_idx + 1}, latent step {step + 1}/{latent_steps}")
                
                # Apply latent realignment
                source_model = self.HF_model if hasattr(self, "HF_model") else self.model
                latent_vec = self._apply_latent_realignment(last_hidden, source_model)
                
                # Log latent statistics
                latent_norm = latent_vec.norm(dim=-1).mean().item()
                logger.debug(f"[Branching] Branch {branch_idx + 1}, step {step + 1}: latent norm = {latent_norm:.4f}")
                
                latent_history.append(latent_vec.detach().clone())
                
                # Continue generation
                latent_embed = latent_vec.unsqueeze(1)
                past_len = _past_length(past)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # Store branch information
            branch_info = {
                'branch_id': branch_idx,
                'latent_history': latent_history,
                'hidden_states': last_hidden.detach().clone(),
                'kv_cache': past,
                'metadata': {
                    'latent_steps': latent_steps,
                    'diversity_strategy': diversity_strategy.__class__.__name__,
                    'branched_from_past_length': _past_length(base_past),
                }
            }
            branches.append(branch_info)
            
            # Log branch completion
            hidden_norm = last_hidden.norm(dim=-1).mean().item()
            logger.info(f"[Branching] Completed branch {branch_idx + 1}/{num_branches} - final hidden norm: {hidden_norm:.4f}")
            logger.debug(f"[Branching] Branch {branch_idx + 1} metadata: {branch_info['metadata']}")
        
        logger.info(f"[ModelWrapper.generate_latent_with_branching] Created {len(branches)} branches")
        return branches

