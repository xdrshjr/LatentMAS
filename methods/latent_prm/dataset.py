"""Dataset loader for Latent PRM training data.

This module loads collected PRM training data from .pt files and prepares
it for fine-tuning the Qwen model on latent reasoning path scoring.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class LatentPRMDataset(Dataset):
    """Dataset for loading latent PRM training data.
    
    Loads .pt files containing reasoning paths with latent sequences and scores.
    Each sample consists of:
    - latent_sequence: Tensor of shape [num_steps, hidden_dim]
    - target_score: Float in range [0, 1] (prm_score or score)
    - metadata: Additional information about the path
    
    Attributes:
        data_dir: Directory containing .pt files
        samples: List of (latent_sequence, target_score, metadata) tuples
        use_prm_score: Whether to use prm_score (True) or score (False)
        max_seq_length: Maximum sequence length (for truncation)
    """
    
    def __init__(
        self,
        data_dir: str,
        use_prm_score: bool = True,
        max_seq_length: Optional[int] = None,
        min_seq_length: int = 1
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing .pt files with collected data
            use_prm_score: If True, use prm_score; if False, use score field
            max_seq_length: Maximum sequence length (None = no limit)
            min_seq_length: Minimum sequence length to include
        """
        self.data_dir = Path(data_dir)
        self.use_prm_score = use_prm_score
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.samples: List[Tuple[torch.Tensor, float, Dict[str, Any]]] = []
        
        logger.info(f"[LatentPRMDataset] Initializing dataset from: {self.data_dir}")
        logger.info(f"[LatentPRMDataset] Using {'prm_score' if use_prm_score else 'score'} as target")
        logger.debug(f"[LatentPRMDataset] Max sequence length: {max_seq_length}")
        logger.debug(f"[LatentPRMDataset] Min sequence length: {min_seq_length}")
        
        # Load all data
        self._load_data()
        
        logger.info(f"[LatentPRMDataset] Loaded {len(self.samples)} samples")
        if len(self.samples) > 0:
            self._log_statistics()
    
    def _load_data(self) -> None:
        """Load all .pt files from data directory."""
        # Find all .pt files
        pt_files = sorted(self.data_dir.glob("*.pt"))
        
        if not pt_files:
            logger.warning(f"[LatentPRMDataset] No .pt files found in {self.data_dir}")
            return
        
        logger.info(f"[LatentPRMDataset] Found {len(pt_files)} .pt files")
        
        # Load each file
        num_files_loaded = 0
        num_paths_loaded = 0
        num_paths_skipped = 0
        
        for pt_file in pt_files:
            try:
                logger.debug(f"[LatentPRMDataset] Loading file: {pt_file.name}")
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                
                # Check if this is a batch file or single question file
                if "questions" in data:
                    # Batch file
                    questions = data["questions"]
                    logger.debug(f"[LatentPRMDataset] Batch file with {len(questions)} questions")
                elif "paths" in data:
                    # Single question file
                    questions = [data]
                    logger.debug(f"[LatentPRMDataset] Single question file")
                else:
                    logger.warning(f"[LatentPRMDataset] Invalid file format: {pt_file.name}")
                    continue
                
                # Process each question
                for question_data in questions:
                    paths = question_data.get("paths", [])
                    question_id = question_data.get("question_id", "unknown")
                    
                    logger.debug(f"[LatentPRMDataset] Processing question {question_id} "
                               f"with {len(paths)} paths")
                    
                    # Process each path
                    for path_data in paths:
                        path_id = path_data.get("path_id", -1)
                        
                        # Extract latent sequence
                        latent_history = path_data.get("latent_history")
                        if latent_history is None or len(latent_history) == 0:
                            logger.debug(f"[LatentPRMDataset] Skipping path {path_id}: "
                                       f"empty latent_history")
                            num_paths_skipped += 1
                            continue
                        
                        # Check sequence length constraints
                        seq_len = len(latent_history)
                        if seq_len < self.min_seq_length:
                            logger.debug(f"[LatentPRMDataset] Skipping path {path_id}: "
                                       f"sequence too short ({seq_len} < {self.min_seq_length})")
                            num_paths_skipped += 1
                            continue
                        
                        # Truncate if necessary
                        if self.max_seq_length is not None and seq_len > self.max_seq_length:
                            logger.debug(f"[LatentPRMDataset] Truncating path {path_id}: "
                                       f"{seq_len} -> {self.max_seq_length}")
                            latent_history = latent_history[:self.max_seq_length]
                        
                        # Extract target score
                        if self.use_prm_score:
                            target_score = path_data.get("prm_score")
                            # Fallback to score if prm_score is None
                            if target_score is None:
                                target_score = path_data.get("score", 0.5)
                                logger.debug(f"[LatentPRMDataset] Path {path_id}: "
                                           f"prm_score is None, using score={target_score:.4f}")
                        else:
                            target_score = path_data.get("score", 0.5)
                        
                        # Validate score
                        if target_score is None:
                            logger.warning(f"[LatentPRMDataset] Path {path_id}: "
                                         f"no valid score, skipping")
                            num_paths_skipped += 1
                            continue
                        
                        # Ensure score is in [0, 1] range
                        target_score = float(target_score)
                        if not (0.0 <= target_score <= 1.0):
                            logger.warning(f"[LatentPRMDataset] Path {path_id}: "
                                         f"score {target_score:.4f} out of range [0,1], clipping")
                            target_score = max(0.0, min(1.0, target_score))
                        
                        # Create metadata
                        metadata = {
                            "question_id": question_id,
                            "path_id": path_id,
                            "agent_name": path_data.get("agent_name", "unknown"),
                            "agent_idx": path_data.get("agent_idx", -1),
                            "num_latent_steps": len(latent_history),
                            "original_score": path_data.get("score"),
                            "prm_score": path_data.get("prm_score"),
                        }
                        
                        # Add sample
                        self.samples.append((latent_history, target_score, metadata))
                        num_paths_loaded += 1
                
                num_files_loaded += 1
                logger.debug(f"[LatentPRMDataset] Successfully loaded file: {pt_file.name}")
                
            except Exception as e:
                logger.error(f"[LatentPRMDataset] Error loading file {pt_file.name}: {e}",
                           exc_info=True)
                continue
        
        logger.info(f"[LatentPRMDataset] Loaded {num_files_loaded}/{len(pt_files)} files")
        logger.info(f"[LatentPRMDataset] Loaded {num_paths_loaded} paths, "
                   f"skipped {num_paths_skipped} paths")
    
    def _log_statistics(self) -> None:
        """Log statistics about the loaded dataset."""
        if len(self.samples) == 0:
            return
        
        # Collect statistics
        seq_lengths = [len(sample[0]) for sample in self.samples]
        target_scores = [sample[1] for sample in self.samples]
        
        # Compute statistics
        stats = {
            "num_samples": len(self.samples),
            "seq_length_min": min(seq_lengths),
            "seq_length_max": max(seq_lengths),
            "seq_length_mean": np.mean(seq_lengths),
            "seq_length_std": np.std(seq_lengths),
            "score_min": min(target_scores),
            "score_max": max(target_scores),
            "score_mean": np.mean(target_scores),
            "score_std": np.std(target_scores),
        }
        
        logger.info(f"[LatentPRMDataset] Dataset statistics:")
        logger.info(f"  - Number of samples: {stats['num_samples']}")
        logger.info(f"  - Sequence length: min={stats['seq_length_min']}, "
                   f"max={stats['seq_length_max']}, "
                   f"mean={stats['seq_length_mean']:.2f}±{stats['seq_length_std']:.2f}")
        logger.info(f"  - Target scores: min={stats['score_min']:.4f}, "
                   f"max={stats['score_max']:.4f}, "
                   f"mean={stats['score_mean']:.4f}±{stats['score_std']:.4f}")
        
        # Log hidden dimension
        if len(self.samples) > 0:
            hidden_dim = self.samples[0][0].shape[-1]
            logger.info(f"  - Hidden dimension: {hidden_dim}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (latent_sequence, target_score, metadata)
        """
        latent_history, target_score, metadata = self.samples[idx]
        
        # Convert latent_history to tensor if it's a list
        if isinstance(latent_history, list):
            # Stack list of tensors into a single tensor
            latent_sequence = torch.stack([
                torch.as_tensor(latent).squeeze() if torch.is_tensor(latent) 
                else torch.as_tensor(latent).squeeze()
                for latent in latent_history
            ])
        else:
            # Already a tensor, just ensure it's 2D [seq_len, hidden_dim]
            latent_sequence = torch.as_tensor(latent_history)
            if latent_sequence.dim() == 3:
                # Remove extra dimension if present [seq_len, 1, hidden_dim] -> [seq_len, hidden_dim]
                latent_sequence = latent_sequence.squeeze(1)
        
        return latent_sequence, target_score, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if len(self.samples) == 0:
            return {"num_samples": 0}
        
        seq_lengths = [len(sample[0]) for sample in self.samples]
        target_scores = [sample[1] for sample in self.samples]
        
        return {
            "num_samples": len(self.samples),
            "seq_length_min": min(seq_lengths),
            "seq_length_max": max(seq_lengths),
            "seq_length_mean": float(np.mean(seq_lengths)),
            "seq_length_std": float(np.std(seq_lengths)),
            "score_min": min(target_scores),
            "score_max": max(target_scores),
            "score_mean": float(np.mean(target_scores)),
            "score_std": float(np.std(target_scores)),
            "hidden_dim": self.samples[0][0].shape[-1] if len(self.samples) > 0 else 0,
        }


def collate_fn(batch: List[Tuple[torch.Tensor, float, Dict[str, Any]]]) -> Dict[str, Any]:
    """Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of (latent_sequence, target_score, metadata) tuples
        
    Returns:
        Dictionary with batched tensors:
        - latent_sequences: Padded tensor [batch_size, max_seq_len, hidden_dim]
        - attention_mask: Mask tensor [batch_size, max_seq_len]
        - target_scores: Tensor [batch_size]
        - seq_lengths: Tensor [batch_size] with original sequence lengths
        - metadata: List of metadata dicts
    """
    # Separate components
    latent_sequences = [item[0] for item in batch]
    target_scores = [item[1] for item in batch]
    metadata_list = [item[2] for item in batch]
    
    # Get sequence lengths
    seq_lengths = [len(seq) for seq in latent_sequences]
    max_seq_len = max(seq_lengths)
    batch_size = len(batch)
    hidden_dim = latent_sequences[0].shape[-1]
    
    # Create padded tensor
    padded_sequences = torch.zeros(batch_size, max_seq_len, hidden_dim, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    
    # Fill in sequences and mask
    for i, (seq, seq_len) in enumerate(zip(latent_sequences, seq_lengths)):
        padded_sequences[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
    
    # Convert target scores to tensor
    target_scores_tensor = torch.tensor(target_scores, dtype=torch.float32)
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
    
    return {
        "latent_sequences": padded_sequences,
        "attention_mask": attention_mask,
        "target_scores": target_scores_tensor,
        "seq_lengths": seq_lengths_tensor,
        "metadata": metadata_list,
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    use_prm_score: bool = True,
    max_seq_length: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for latent PRM training data.
    
    Args:
        data_dir: Directory containing .pt files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        use_prm_score: Whether to use prm_score as target
        max_seq_length: Maximum sequence length
        **kwargs: Additional arguments for Dataset
        
    Returns:
        DataLoader instance
    """
    logger.info(f"[create_dataloader] Creating DataLoader")
    logger.info(f"  - data_dir: {data_dir}")
    logger.info(f"  - batch_size: {batch_size}")
    logger.info(f"  - shuffle: {shuffle}")
    logger.info(f"  - use_prm_score: {use_prm_score}")
    logger.debug(f"  - num_workers: {num_workers}")
    logger.debug(f"  - max_seq_length: {max_seq_length}")
    
    # Create dataset
    dataset = LatentPRMDataset(
        data_dir=data_dir,
        use_prm_score=use_prm_score,
        max_seq_length=max_seq_length,
        **kwargs
    )
    
    if len(dataset) == 0:
        logger.error(f"[create_dataloader] Dataset is empty! Check data directory: {data_dir}")
        raise ValueError(f"No valid samples found in {data_dir}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(f"[create_dataloader] DataLoader created with {len(dataset)} samples, "
               f"{len(dataloader)} batches")
    
    return dataloader

