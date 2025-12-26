"""Model architecture for Latent PRM fine-tuning.

This module implements a wrapper around Qwen model that processes latent
sequences and predicts path quality scores.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QwenLatentPRM(nn.Module):
    """Qwen model fine-tuned for latent sequence scoring.
    
    This model takes latent reasoning sequences as input (bypassing the
    embedding layer) and predicts a quality score for the reasoning path.
    
    Architecture:
    1. Input: Latent sequences [batch, seq_len, hidden_dim]
    2. Process through Qwen transformer layers
    3. Pooling: Mean pooling over sequence (with attention mask)
    4. Regression head: Linear layer → Sigmoid → Score [0, 1]
    
    Attributes:
        config: Model configuration
        transformer: Qwen transformer layers
        score_head: Regression head for score prediction
        pooling_strategy: How to pool sequence ('mean', 'last', 'max')
    """
    
    def __init__(
        self,
        model_path: str,
        pooling_strategy: str = "mean",
        dropout_prob: float = 0.1,
        freeze_transformer: bool = False
    ):
        """Initialize the QwenLatentPRM model.
        
        Args:
            model_path: Path to pretrained Qwen model
            pooling_strategy: Pooling method ('mean', 'last', 'max')
            dropout_prob: Dropout probability for regression head
            freeze_transformer: Whether to freeze transformer layers (False = full fine-tuning)
        """
        super().__init__()
        
        logger.info(f"[QwenLatentPRM] Initializing model from: {model_path}")
        logger.info(f"[QwenLatentPRM] Pooling strategy: {pooling_strategy}")
        logger.info(f"[QwenLatentPRM] Dropout probability: {dropout_prob}")
        logger.info(f"[QwenLatentPRM] Freeze transformer: {freeze_transformer}")
        
        self.pooling_strategy = pooling_strategy
        
        # Load Qwen model configuration
        try:
            logger.debug(f"[QwenLatentPRM] Loading model config from {model_path}")
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            logger.debug(f"[QwenLatentPRM] Config loaded: hidden_size={self.config.hidden_size}")
        except Exception as e:
            logger.error(f"[QwenLatentPRM] Failed to load config: {e}", exc_info=True)
            raise
        
        # Load Qwen transformer model
        try:
            logger.info(f"[QwenLatentPRM] Loading transformer model...")
            self.transformer = AutoModel.from_pretrained(
                model_path,
                config=self.config,
                trust_remote_code=True
            )
            logger.info(f"[QwenLatentPRM] ✓ Transformer model loaded successfully")
            
            # Log model size
            num_params = sum(p.numel() for p in self.transformer.parameters())
            num_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"[QwenLatentPRM] Model parameters: {num_params:,} total, "
                       f"{num_trainable:,} trainable")
            
        except Exception as e:
            logger.error(f"[QwenLatentPRM] Failed to load transformer: {e}", exc_info=True)
            raise
        
        # Optionally freeze transformer layers
        if freeze_transformer:
            logger.info(f"[QwenLatentPRM] Freezing transformer parameters")
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info(f"[QwenLatentPRM] ✓ Transformer frozen (only head will be trained)")
        else:
            logger.info(f"[QwenLatentPRM] Transformer parameters are trainable (full fine-tuning)")
        
        # Create regression head
        hidden_size = self.config.hidden_size
        logger.debug(f"[QwenLatentPRM] Creating regression head with hidden_size={hidden_size}")
        
        self.score_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        logger.info(f"[QwenLatentPRM] ✓ Regression head created")
        logger.debug(f"[QwenLatentPRM] Head architecture: {self.score_head}")
        
        # Log total trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[QwenLatentPRM] Total model parameters: {total_params:,}")
        logger.info(f"[QwenLatentPRM] Trainable parameters: {trainable_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(
        self,
        latent_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            latent_sequences: Latent sequences [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return pooled hidden states
            
        Returns:
            Tuple of (predicted_scores, pooled_hidden_states)
            - predicted_scores: Tensor [batch_size, 1] with scores in [0, 1]
            - pooled_hidden_states: Optional tensor [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = latent_sequences.shape
        
        logger.debug(f"[QwenLatentPRM] Forward pass: batch_size={batch_size}, "
                    f"seq_len={seq_len}, hidden_dim={hidden_dim}")
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=latent_sequences.device)
            logger.debug(f"[QwenLatentPRM] Created default attention mask (all ones)")
        
        # Pass latent sequences through transformer
        # Note: We're feeding latents directly, bypassing the embedding layer
        try:
            logger.debug(f"[QwenLatentPRM] Passing through transformer...")
            
            # For Qwen models, we can pass inputs_embeds directly
            outputs = self.transformer(
                inputs_embeds=latent_sequences,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False
            )
            
            # Get last hidden states
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            logger.debug(f"[QwenLatentPRM] Transformer output shape: {hidden_states.shape}")
            
        except Exception as e:
            logger.error(f"[QwenLatentPRM] Error in transformer forward pass: {e}", exc_info=True)
            raise
        
        # Pool hidden states
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        logger.debug(f"[QwenLatentPRM] Pooled hidden states shape: {pooled.shape}")
        
        # Pass through regression head
        scores = self.score_head(pooled)  # [batch_size, 1]
        logger.debug(f"[QwenLatentPRM] Predicted scores shape: {scores.shape}")
        logger.debug(f"[QwenLatentPRM] Score range: [{scores.min().item():.4f}, "
                    f"{scores.max().item():.4f}]")
        
        if return_hidden_states:
            return scores, pooled
        else:
            return scores, None
    
    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool hidden states according to pooling strategy.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled tensor [batch_size, hidden_dim]
        """
        if self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            # Expand mask to match hidden_states dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Sum over sequence length, weighted by mask
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            
            # Divide by number of non-masked tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
            
            logger.debug(f"[QwenLatentPRM] Applied mean pooling")
            
        elif self.pooling_strategy == "last":
            # Take last non-masked token
            # Find last non-masked position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, seq_lengths]
            
            logger.debug(f"[QwenLatentPRM] Applied last-token pooling")
            
        elif self.pooling_strategy == "max":
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Set masked positions to very negative value
            hidden_states_masked = hidden_states.clone()
            hidden_states_masked[mask_expanded == 0] = -1e9
            
            # Max over sequence length
            pooled = torch.max(hidden_states_masked, dim=1)[0]
            
            logger.debug(f"[QwenLatentPRM] Applied max pooling")
            
        else:
            logger.error(f"[QwenLatentPRM] Unknown pooling strategy: {self.pooling_strategy}")
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def predict_scores(
        self,
        latent_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict scores for latent sequences (inference mode).
        
        Args:
            latent_sequences: Latent sequences [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Predicted scores [batch_size] in range [0, 1]
        """
        self.eval()
        with torch.no_grad():
            scores, _ = self.forward(latent_sequences, attention_mask)
            return scores.squeeze(-1)  # [batch_size]
    
    def get_model_info(self) -> dict:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "QwenLatentPRM",
            "hidden_size": self.config.hidden_size,
            "pooling_strategy": self.pooling_strategy,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        }


def create_model(
    model_path: str,
    pooling_strategy: str = "mean",
    dropout_prob: float = 0.1,
    freeze_transformer: bool = False,
    device: str = "cuda"
) -> QwenLatentPRM:
    """Create and initialize a QwenLatentPRM model.
    
    Args:
        model_path: Path to pretrained Qwen model
        pooling_strategy: Pooling method ('mean', 'last', 'max')
        dropout_prob: Dropout probability
        freeze_transformer: Whether to freeze transformer (False = full fine-tuning)
        device: Device to load model on
        
    Returns:
        Initialized QwenLatentPRM model
    """
    logger.info(f"[create_model] Creating QwenLatentPRM model")
    logger.info(f"[create_model] Model path: {model_path}")
    logger.info(f"[create_model] Device: {device}")
    
    # Create model
    model = QwenLatentPRM(
        model_path=model_path,
        pooling_strategy=pooling_strategy,
        dropout_prob=dropout_prob,
        freeze_transformer=freeze_transformer
    )
    
    # Move to device
    logger.info(f"[create_model] Moving model to device: {device}")
    model = model.to(device)
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"[create_model] Model created successfully:")
    logger.info(f"  - Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  - Trainable parameters: {model_info['trainable_parameters']:,} "
               f"({model_info['trainable_percentage']:.2f}%)")
    
    return model

