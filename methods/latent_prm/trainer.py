"""Trainer for Latent PRM fine-tuning.

This module implements the training loop for fine-tuning Qwen model
on latent reasoning path scoring with full-parameter updates.
"""

import logging
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from datetime import datetime
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .dataset import create_dataloader
from .model import create_model
from progress_utils import ProgressBarManager

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that writes through tqdm to avoid progress bar conflicts.
    
    This handler ensures that log messages appear above the progress bar instead of
    disrupting it, maintaining a clean console output with the progress bar fixed at the bottom.
    """
    
    def __init__(self, progress_manager: Optional[ProgressBarManager] = None):
        """Initialize the handler.
        
        Args:
            progress_manager: Optional ProgressBarManager instance for writing logs
        """
        super().__init__()
        self.progress_manager = progress_manager
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through tqdm.
        
        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            if self.progress_manager and self.progress_manager.main_bar:
                # Write through tqdm to keep progress bar at bottom
                self.progress_manager.main_bar.write(msg, file=sys.stderr)
            else:
                # Fallback to standard output if no progress bar
                sys.stderr.write(msg + '\n')
                sys.stderr.flush()
        except Exception:
            self.handleError(record)


class LatentPRMTrainer:
    """Trainer for fine-tuning Qwen on latent PRM scoring.
    
    Features:
    - Full-parameter fine-tuning
    - MSE loss for score regression
    - Gradient accumulation for memory efficiency
    - Mixed precision training (fp16/bf16)
    - Progress bars with tqdm
    - Comprehensive logging
    - Checkpoint saving
    
    Attributes:
        model: QwenLatentPRM model
        dataloader: DataLoader for training data
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function (MSE)
        device: Training device
        args: Training arguments
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str,
        num_epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_steps: int = 100,
        logging_steps: int = 10,
        use_prm_score: bool = True,
        pooling_strategy: str = "mean",
        dropout_prob: float = 0.1,
        max_seq_length: Optional[int] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        save_checkpoints: bool = True,
        seed: int = 42
    ):
        """Initialize the trainer.
        
        Args:
            model_path: Path to pretrained Qwen model
            data_dir: Directory containing training data (.pt files)
            output_dir: Directory to save checkpoints and logs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for AdamW
            weight_decay: Weight decay for AdamW
            warmup_ratio: Ratio of warmup steps
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            use_prm_score: Use prm_score (True) or score (False) as target
            pooling_strategy: Pooling method for model
            dropout_prob: Dropout probability
            max_seq_length: Maximum sequence length
            device: Training device
            mixed_precision: Use mixed precision training
            save_checkpoints: Whether to save checkpoints during training
            seed: Random seed
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.device = device
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.save_checkpoints = save_checkpoints
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Trainer] Output directory: {self.output_dir}")
        
        # Set random seed
        self._set_seed(seed)
        
        # Log configuration
        self._log_config()
        
        # Initialize progress bar manager
        self.progress_manager = ProgressBarManager()
        logger.debug(f"[Trainer] Progress bar manager initialized")
        
        # Create dataloader
        logger.info(f"[Trainer] Creating dataloader...")
        self.dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            use_prm_score=use_prm_score,
            max_seq_length=max_seq_length
        )
        logger.info(f"[Trainer] ✓ Dataloader created: {len(self.dataloader)} batches")
        
        # Print a data example for user reference
        self._print_data_example()
        
        # Create model
        logger.info(f"[Trainer] Creating model...")
        self.model = create_model(
            model_path=model_path,
            pooling_strategy=pooling_strategy,
            dropout_prob=dropout_prob,
            freeze_transformer=False,  # Full fine-tuning
            device=device
        )
        logger.info(f"[Trainer] ✓ Model created and moved to {device}")
        
        # Create optimizer
        logger.info(f"[Trainer] Creating optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"[Trainer] ✓ AdamW optimizer created (lr={learning_rate}, wd={weight_decay})")
        
        # Create learning rate scheduler
        total_steps = len(self.dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        logger.info(f"[Trainer] Creating LR scheduler...")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)")
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"[Trainer] ✓ LR scheduler created")
        
        # Create loss function
        self.criterion = nn.MSELoss()
        logger.info(f"[Trainer] ✓ Loss function: MSE")
        
        # Create gradient scaler for mixed precision
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info(f"[Trainer] ✓ Mixed precision training enabled (fp16)")
        else:
            self.scaler = None
            logger.info(f"[Trainer] Mixed precision training disabled")
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Loss tracking for visualization
        self.step_losses: List[float] = []  # Loss at each optimization step
        self.step_numbers: List[int] = []   # Step numbers for x-axis
        self.epoch_losses: List[float] = [] # Average loss per epoch
        
        logger.info(f"[Trainer] ✓ Trainer initialized successfully")
        logger.info("=" * 80)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"[Trainer] Random seed set to: {seed}")
    
    def _log_config(self) -> None:
        """Log training configuration."""
        logger.info("=" * 80)
        logger.info("[Trainer] Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"")
        logger.info(f"Training hyperparameters:")
        logger.info(f"  - Num epochs: {self.num_epochs}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  - Max gradient norm: {self.max_grad_norm}")
        logger.info(f"  - Mixed precision: {self.mixed_precision}")
        logger.info(f"")
        logger.info(f"Logging and checkpointing:")
        logger.info(f"  - Save checkpoints: {self.save_checkpoints}")
        logger.info(f"  - Logging steps: {self.logging_steps}")
        logger.info(f"  - Save steps: {self.save_steps}")
        logger.info("=" * 80)
    
    def _print_data_example(self) -> None:
        """Print a data example for user reference before training starts.
        
        This helps users understand the data format and verify data loading is correct.
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("[Trainer] Data Example for Reference")
        logger.info("=" * 80)
        
        try:
            # Get the first sample from the dataset
            if len(self.dataloader.dataset) == 0:
                logger.warning("[Trainer] Dataset is empty, cannot show example")
                return
            
            # Get first sample
            latent_sequence, target_score, metadata = self.dataloader.dataset[0]
            
            # Log sample information
            logger.info(f"Sample metadata:")
            logger.info(f"  - Question ID: {metadata.get('question_id', 'N/A')}")
            logger.info(f"  - Path ID: {metadata.get('path_id', 'N/A')}")
            logger.info(f"  - Agent name: {metadata.get('agent_name', 'N/A')}")
            logger.info(f"  - Agent index: {metadata.get('agent_idx', 'N/A')}")
            logger.info(f"")
            logger.info(f"Latent sequence information:")
            logger.info(f"  - Sequence length: {len(latent_sequence)} steps")
            logger.info(f"  - Hidden dimension: {latent_sequence.shape[-1]}")
            logger.info(f"  - Tensor shape: {list(latent_sequence.shape)}")
            logger.info(f"  - Data type: {latent_sequence.dtype}")
            logger.info(f"")
            logger.info(f"Target score information:")
            logger.info(f"  - Target score: {target_score:.6f}")
            logger.info(f"  - Original score: {metadata.get('original_score', 'N/A')}")
            logger.info(f"  - PRM score: {metadata.get('prm_score', 'N/A')}")
            logger.info(f"")
            
            # Show first few latent vectors (first 3 steps)
            num_steps_to_show = min(3, len(latent_sequence))
            logger.info(f"First {num_steps_to_show} latent vectors (showing first 10 dimensions):")
            for i in range(num_steps_to_show):
                vector = latent_sequence[i]
                # Show first 10 dimensions
                vector_preview = vector[:10].tolist()
                vector_str = ", ".join([f"{v:.4f}" for v in vector_preview])
                logger.info(f"  Step {i}: [{vector_str}, ...]")
            
            logger.info(f"")
            logger.info(f"Statistics:")
            logger.info(f"  - Min value: {latent_sequence.min().item():.6f}")
            logger.info(f"  - Max value: {latent_sequence.max().item():.6f}")
            logger.info(f"  - Mean value: {latent_sequence.mean().item():.6f}")
            logger.info(f"  - Std value: {latent_sequence.std().item():.6f}")
            
            logger.info("=" * 80)
            logger.info("")
            
        except Exception as e:
            logger.error(f"[Trainer] Error printing data example: {e}", exc_info=True)
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop.
        
        Returns:
            Dictionary with training statistics
        """
        logger.info("[Trainer] Starting training...")
        logger.info("=" * 80)
        
        # Setup tqdm logging handler to write logs above progress bar
        tqdm_handler = TqdmLoggingHandler(self.progress_manager)
        tqdm_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Remove existing console handlers and add tqdm handler
        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        for handler in console_handlers:
            root_logger.removeHandler(handler)
        root_logger.addHandler(tqdm_handler)
        
        logger.debug("[Trainer] Tqdm logging handler configured")
        
        # Training statistics
        total_loss = 0.0
        total_steps = 0
        
        # Calculate total training steps for progress bar
        total_training_steps = len(self.dataloader) * self.num_epochs
        
        # Create main progress bar that stays at the bottom
        self.progress_manager.create_main_progress(
            total=total_training_steps,
            desc="Training Progress",
            unit="batch"
        )
        
        logger.info(f"[Trainer] Total training steps: {total_training_steps}")
        logger.info(f"[Trainer] Batches per epoch: {len(self.dataloader)}")
        logger.info("")
        
        try:
            # Training loop
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                logger.info("=" * 80)
                
                # Set model to training mode
                self.model.train()
                
                epoch_loss = 0.0
                num_batches = 0
                
                # Batch loop
                for batch_idx, batch in enumerate(self.dataloader):
                    # Move batch to device
                    latent_sequences = batch["latent_sequences"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    target_scores = batch["target_scores"].to(self.device)
                    
                    logger.debug(f"[Trainer] Processing batch {batch_idx + 1}/{len(self.dataloader)}: "
                               f"batch_size={latent_sequences.shape[0]}, "
                               f"seq_len={latent_sequences.shape[1]}")
                    
                    # Forward pass with mixed precision
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred_scores, _ = self.model(latent_sequences, attention_mask)
                            pred_scores = pred_scores.squeeze(-1)  # [batch_size]
                            loss = self.criterion(pred_scores, target_scores)
                            loss = loss / self.gradient_accumulation_steps
                    else:
                        pred_scores, _ = self.model(latent_sequences, attention_mask)
                        pred_scores = pred_scores.squeeze(-1)  # [batch_size]
                        loss = self.criterion(pred_scores, target_scores)
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Accumulate loss
                    batch_loss = loss.item() * self.gradient_accumulation_steps
                    epoch_loss += batch_loss
                    total_loss += batch_loss
                    num_batches += 1
                    
                    # Update weights after accumulation steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        logger.debug(f"[Trainer] Gradient norm before clipping: {grad_norm:.4f}")
                        
                        # Optimizer step
                        if self.mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        # Scheduler step
                        self.scheduler.step()
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        # Increment global step
                        self.global_step += 1
                        total_steps += 1
                        
                        # Track loss for visualization
                        self.step_losses.append(batch_loss)
                        self.step_numbers.append(self.global_step)
                        logger.debug(f"[Trainer] Tracked loss for step {self.global_step}: {batch_loss:.4f}")
                        
                        # Update progress bar with current metrics
                        current_lr = self.scheduler.get_last_lr()[0]
                        avg_loss = total_loss / total_steps
                        
                        self.progress_manager.set_main_postfix(
                            epoch=f"{epoch + 1}/{self.num_epochs}",
                            loss=f"{batch_loss:.4f}",
                            avg_loss=f"{avg_loss:.4f}",
                            lr=f"{current_lr:.2e}"
                        )
                        self.progress_manager.update_main_progress(1)
                        
                        # Logging
                        if self.global_step % self.logging_steps == 0:
                            logger.info(f"[Trainer] Step {self.global_step}: "
                                       f"loss={batch_loss:.4f}, "
                                       f"avg_loss={avg_loss:.4f}, "
                                       f"lr={current_lr:.2e}, "
                                       f"grad_norm={grad_norm:.4f}")
                        
                        # Save checkpoint (only if enabled)
                        if self.save_checkpoints and self.global_step % self.save_steps == 0:
                            logger.info(f"[Trainer] Saving checkpoint at step {self.global_step}")
                            self._save_checkpoint(
                                step=self.global_step,
                                loss=batch_loss,
                                is_best=False
                            )
                    else:
                        # Still update progress bar even without optimizer step
                        self.progress_manager.update_main_progress(1)
            
                # End of epoch
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                self.epoch_losses.append(avg_epoch_loss)
                
                logger.info("")
                logger.info(f"[Trainer] Epoch {epoch + 1} completed:")
                logger.info(f"  - Average loss: {avg_epoch_loss:.4f}")
                logger.info(f"  - Total steps: {self.global_step}")
                logger.info(f"  - Batches processed: {num_batches}")
                
                # Save epoch checkpoint (only if enabled)
                is_best = avg_epoch_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_epoch_loss
                    logger.info(f"[Trainer] ✓ New best loss: {self.best_loss:.4f}")
                
                if self.save_checkpoints:
                    logger.info(f"[Trainer] Saving epoch {epoch + 1} checkpoint")
                    self._save_checkpoint(
                        step=self.global_step,
                        loss=avg_epoch_loss,
                        is_best=is_best,
                        epoch=epoch + 1
                    )
                else:
                    logger.debug(f"[Trainer] Checkpoint saving disabled, skipping epoch checkpoint")
            
            # Training completed
            logger.info("")
            logger.info("=" * 80)
            logger.info("[Trainer] Training completed!")
            logger.info("=" * 80)
            logger.info(f"Total steps: {self.global_step}")
            logger.info(f"Best loss: {self.best_loss:.4f}")
            logger.info(f"Final loss: {self.epoch_losses[-1]:.4f}")
            logger.info("=" * 80)
            
            # Save final checkpoint (only if enabled)
            if self.save_checkpoints:
                logger.info("[Trainer] Saving final checkpoint")
                self._save_checkpoint(
                    step=self.global_step,
                    loss=self.epoch_losses[-1],
                    is_best=False,
                    is_final=True
                )
            else:
                logger.info("[Trainer] Checkpoint saving disabled, skipping final checkpoint")
            
            # Generate and save loss curve visualization
            logger.info("[Trainer] Generating loss curve visualization...")
            self._save_loss_curve()
            
            # Save training statistics
            stats = {
                "total_steps": self.global_step,
                "num_epochs": self.num_epochs,
                "best_loss": self.best_loss,
                "final_loss": self.epoch_losses[-1],
                "epoch_losses": self.epoch_losses,
                "step_losses": self.step_losses,
                "step_numbers": self.step_numbers,
            }
            self._save_training_stats(stats)
            
            return stats
            
        finally:
            # Clean up progress bar
            self.progress_manager.close_all()
            logger.debug("[Trainer] Progress bar cleaned up")
            
            # Restore original logging handlers
            root_logger.removeHandler(tqdm_handler)
            for handler in console_handlers:
                root_logger.addHandler(handler)
            logger.debug("[Trainer] Original logging handlers restored")
    
    def _save_checkpoint(
        self,
        step: int,
        loss: float,
        is_best: bool = False,
        is_final: bool = False,
        epoch: Optional[int] = None
    ) -> None:
        """Save model checkpoint.
        
        Args:
            step: Current training step
            loss: Current loss value
            is_best: Whether this is the best checkpoint
            is_final: Whether this is the final checkpoint
            epoch: Current epoch (optional)
        """
        # Create checkpoint directory
        if is_final:
            checkpoint_dir = self.output_dir / "final"
            logger.info(f"[Trainer] Saving final checkpoint...")
        elif is_best:
            checkpoint_dir = self.output_dir / "best"
            logger.info(f"[Trainer] Saving best checkpoint...")
        elif epoch is not None:
            checkpoint_dir = self.output_dir / f"epoch_{epoch}"
            logger.info(f"[Trainer] Saving epoch {epoch} checkpoint...")
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
            logger.info(f"[Trainer] Saving checkpoint at step {step}...")
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(self.model.state_dict(), model_path)
        logger.debug(f"[Trainer] Model state saved to: {model_path}")
        
        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        logger.debug(f"[Trainer] Optimizer state saved to: {optimizer_path}")
        
        # Save scheduler state
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(self.scheduler.state_dict(), scheduler_path)
        logger.debug(f"[Trainer] Scheduler state saved to: {scheduler_path}")
        
        # Save training state
        training_state = {
            "global_step": step,
            "epoch": self.current_epoch,
            "loss": loss,
            "best_loss": self.best_loss,
        }
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        logger.debug(f"[Trainer] Training state saved to: {state_path}")
        
        # Save model config
        config_path = checkpoint_dir / "config.json"
        model_info = self.model.get_model_info()
        with open(config_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.debug(f"[Trainer] Model config saved to: {config_path}")
        
        logger.info(f"[Trainer] ✓ Checkpoint saved to: {checkpoint_dir}")
    
    def _save_loss_curve(self) -> None:
        """Generate and save loss curve visualization as PNG.
        
        Creates a comprehensive loss curve plot showing:
        - Training loss at each optimization step
        - Average loss per epoch (as vertical lines)
        
        The plot is saved to {output_dir}/img/loss_curve.png
        """
        try:
            # Create img directory
            img_dir = self.output_dir / "img"
            img_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[Trainer] Created img directory: {img_dir}")
            
            # Check if we have data to plot
            if not self.step_losses or not self.epoch_losses:
                logger.warning("[Trainer] No loss data to plot, skipping loss curve generation")
                return
            
            logger.info(f"[Trainer] Plotting loss curve with {len(self.step_losses)} step losses and {len(self.epoch_losses)} epoch losses")
            
            # Create figure with high DPI for better quality
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            # Plot step-wise training loss
            ax.plot(self.step_numbers, self.step_losses, 
                   label='Training Loss (per step)', 
                   color='#2E86AB', 
                   linewidth=1.5, 
                   alpha=0.7)
            
            # Calculate and plot epoch boundaries and average losses
            steps_per_epoch = len(self.dataloader) // self.gradient_accumulation_steps
            logger.debug(f"[Trainer] Steps per epoch: {steps_per_epoch}")
            
            for epoch_idx, epoch_loss in enumerate(self.epoch_losses):
                epoch_step = (epoch_idx + 1) * steps_per_epoch
                # Draw vertical line at epoch boundary
                ax.axvline(x=epoch_step, 
                          color='#A23B72', 
                          linestyle='--', 
                          linewidth=1.0, 
                          alpha=0.5)
                # Add epoch average loss as horizontal line segment
                start_step = epoch_idx * steps_per_epoch if epoch_idx > 0 else 0
                ax.hlines(y=epoch_loss, 
                         xmin=start_step, 
                         xmax=epoch_step,
                         color='#F18F01', 
                         linewidth=2.0, 
                         alpha=0.8,
                         label=f'Epoch {epoch_idx + 1} Avg' if epoch_idx == 0 else '')
            
            # Customize plot
            ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
            ax.set_title('Training Loss Curve - Latent PRM Fine-tuning', 
                        fontsize=14, 
                        fontweight='bold',
                        pad=20)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Add statistics text box
            stats_text = (
                f'Total Steps: {self.global_step}\n'
                f'Epochs: {self.num_epochs}\n'
                f'Best Loss: {self.best_loss:.4f}\n'
                f'Final Loss: {self.epoch_losses[-1]:.4f}'
            )
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Tight layout to prevent label cutoff
            plt.tight_layout()
            
            # Save figure
            output_path = img_dir / "loss_curve.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            logger.info(f"[Trainer] ✓ Loss curve saved to: {output_path}")
            logger.debug(f"[Trainer] Loss curve statistics - Min: {min(self.step_losses):.4f}, "
                        f"Max: {max(self.step_losses):.4f}, "
                        f"Mean: {sum(self.step_losses)/len(self.step_losses):.4f}")
            
        except Exception as e:
            logger.error(f"[Trainer] Failed to generate loss curve: {e}", exc_info=True)
            logger.warning("[Trainer] Training completed successfully, but loss curve generation failed")
    
    def _save_training_stats(self, stats: Dict[str, Any]) -> None:
        """Save training statistics.
        
        Args:
            stats: Dictionary with training statistics
        """
        stats_path = self.output_dir / "training_stats.json"
        
        # Add metadata
        stats["timestamp"] = datetime.now().isoformat()
        stats["model_path"] = str(self.model_path)
        stats["data_dir"] = str(self.data_dir)
        stats["num_samples"] = len(self.dataloader.dataset)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"[Trainer] Training statistics saved to: {stats_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Latent PRM model")
    
    # Model and data
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pretrained Qwen model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing training data (.pt files)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save checkpoints and logs")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Ratio of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Steps to accumulate gradients")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Logging and checkpointing
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log metrics every N steps")
    
    # Model configuration
    parser.add_argument("--use_prm_score", action="store_true", default=True,
                       help="Use prm_score as target (default: True)")
    parser.add_argument("--pooling_strategy", type=str, default="mean",
                       choices=["mean", "last", "max"],
                       help="Pooling strategy for sequence")
    parser.add_argument("--dropout_prob", type=float, default=0.1,
                       help="Dropout probability")
    parser.add_argument("--max_seq_length", type=int, default=None,
                       help="Maximum sequence length")
    
    # Device and precision
    parser.add_argument("--device", type=str, default="cuda",
                       help="Training device (cuda/cpu)")
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--no_save_checkpoints", action="store_true",
                       help="Disable checkpoint saving during training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log")
        ]
    )
    
    logger.info("=" * 80)
    logger.info("Latent PRM Training")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = LatentPRMTrainer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_prm_score=args.use_prm_score,
        pooling_strategy=args.pooling_strategy,
        dropout_prob=args.dropout_prob,
        max_seq_length=args.max_seq_length,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        save_checkpoints=not args.no_save_checkpoints,
        seed=args.seed
    )
    
    # Train
    stats = trainer.train()
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Best loss: {stats['best_loss']:.4f}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

