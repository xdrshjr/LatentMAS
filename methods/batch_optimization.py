"""Batch processing optimization module for multi-path reasoning.

This module provides utilities to optimize batch processing of multiple reasoning paths,
including dynamic batching, path grouping, and tensor operation optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class BatchGroup:
    """Represents a group of paths to be processed together.
    
    Attributes:
        group_id: Unique identifier for this batch group
        path_ids: List of path IDs in this group
        sequence_length: Common sequence length for this group
        batch_size: Number of paths in this group
        metadata: Additional metadata
    """
    group_id: int
    path_ids: List[int]
    sequence_length: int
    batch_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Get the number of paths in this group."""
        return self.batch_size


class DynamicBatcher:
    """Dynamic batching utility for multi-path processing.
    
    This class groups paths with similar lengths together to minimize padding
    overhead and maximize GPU utilization.
    
    Attributes:
        max_batch_size: Maximum number of paths per batch
        length_tolerance: Maximum length difference within a batch
        padding_token_id: Token ID to use for padding
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        length_tolerance: int = 32,
        padding_token_id: int = 0
    ):
        """Initialize the dynamic batcher.
        
        Args:
            max_batch_size: Maximum paths per batch
            length_tolerance: Maximum length difference within batch
            padding_token_id: Token ID for padding
        """
        self.max_batch_size = max_batch_size
        self.length_tolerance = length_tolerance
        self.padding_token_id = padding_token_id
        
        logger.info(
            f"[DynamicBatcher] Initialized with max_batch_size={max_batch_size}, "
            f"length_tolerance={length_tolerance}"
        )
    
    def create_batches(
        self,
        input_ids_list: List[torch.Tensor],
        path_ids: Optional[List[int]] = None
    ) -> List[BatchGroup]:
        """Create optimal batches from a list of input sequences.
        
        Args:
            input_ids_list: List of input token ID tensors
            path_ids: Optional list of path IDs corresponding to inputs
            
        Returns:
            List of BatchGroup objects
        """
        if not input_ids_list:
            return []
        
        if path_ids is None:
            path_ids = list(range(len(input_ids_list)))
        
        # Get sequence lengths
        lengths = [ids.shape[1] for ids in input_ids_list]
        
        # Sort by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        batches = []
        current_batch_indices = []
        current_length_range = None
        
        for idx in sorted_indices:
            length = lengths[idx]
            
            # Check if we should start a new batch
            if current_length_range is None:
                # First item in new batch
                current_length_range = (length, length)
                current_batch_indices = [idx]
            elif (length - current_length_range[0] <= self.length_tolerance and
                  len(current_batch_indices) < self.max_batch_size):
                # Add to current batch
                current_batch_indices.append(idx)
                current_length_range = (current_length_range[0], length)
            else:
                # Create batch from current group
                batch_group = self._create_batch_group(
                    current_batch_indices,
                    path_ids,
                    lengths,
                    len(batches)
                )
                batches.append(batch_group)
                
                # Start new batch
                current_length_range = (length, length)
                current_batch_indices = [idx]
        
        # Add final batch
        if current_batch_indices:
            batch_group = self._create_batch_group(
                current_batch_indices,
                path_ids,
                lengths,
                len(batches)
            )
            batches.append(batch_group)
        
        logger.info(
            f"[DynamicBatcher] Created {len(batches)} batches from {len(input_ids_list)} sequences"
        )
        
        # Log batch statistics
        for i, batch in enumerate(batches):
            logger.debug(
                f"[DynamicBatcher] Batch {i}: {batch.batch_size} sequences, "
                f"length={batch.sequence_length}"
            )
        
        return batches
    
    def _create_batch_group(
        self,
        indices: List[int],
        path_ids: List[int],
        lengths: List[int],
        group_id: int
    ) -> BatchGroup:
        """Create a BatchGroup from indices.
        
        Args:
            indices: Indices of sequences in this batch
            path_ids: List of all path IDs
            lengths: List of all sequence lengths
            group_id: ID for this batch group
            
        Returns:
            BatchGroup object
        """
        batch_path_ids = [path_ids[i] for i in indices]
        max_length = max(lengths[i] for i in indices)
        
        return BatchGroup(
            group_id=group_id,
            path_ids=batch_path_ids,
            sequence_length=max_length,
            batch_size=len(indices),
            metadata={'original_indices': indices}
        )
    
    def pad_batch(
        self,
        tensors: List[torch.Tensor],
        target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of tensors to the same length.
        
        Args:
            tensors: List of tensors to pad
            target_length: Target length (None for max length in batch)
            
        Returns:
            Tuple of (padded_tensor, attention_mask)
        """
        if not tensors:
            raise ValueError("Cannot pad empty tensor list")
        
        # Determine target length
        if target_length is None:
            target_length = max(t.shape[1] for t in tensors)
        
        batch_size = len(tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype
        
        # Create padded tensor
        if len(tensors[0].shape) == 2:
            # Token IDs: [batch, seq_len]
            padded = torch.full(
                (batch_size, target_length),
                self.padding_token_id,
                dtype=dtype,
                device=device
            )
        else:
            # Hidden states: [batch, seq_len, hidden_dim]
            hidden_dim = tensors[0].shape[2]
            padded = torch.zeros(
                (batch_size, target_length, hidden_dim),
                dtype=dtype,
                device=device
            )
        
        # Create attention mask
        attention_mask = torch.zeros(
            (batch_size, target_length),
            dtype=torch.long,
            device=device
        )
        
        # Fill in actual values
        total_padding = 0
        for i, tensor in enumerate(tensors):
            seq_len = tensor.shape[1]
            padded[i, :seq_len] = tensor[0]  # Assuming batch size 1 inputs
            attention_mask[i, :seq_len] = 1
            total_padding += (target_length - seq_len)
        
        logger.debug(
            f"[DynamicBatcher] Padded {batch_size} tensors to length {target_length}, "
            f"total padding tokens: {total_padding}"
        )
        
        return padded, attention_mask


class PathGrouper:
    """Groups similar paths for efficient batch processing.
    
    This class analyzes path characteristics and groups similar paths together
    to enable more efficient batch processing.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize the path grouper.
        
        Args:
            similarity_threshold: Minimum similarity for grouping paths
        """
        self.similarity_threshold = similarity_threshold
        logger.info(
            f"[PathGrouper] Initialized with similarity_threshold={similarity_threshold}"
        )
    
    def group_by_similarity(
        self,
        hidden_states_list: List[torch.Tensor],
        path_ids: Optional[List[int]] = None
    ) -> List[List[int]]:
        """Group paths by hidden state similarity.
        
        Args:
            hidden_states_list: List of hidden state tensors
            path_ids: Optional list of path IDs
            
        Returns:
            List of groups, where each group is a list of path indices
        """
        if not hidden_states_list:
            return []
        
        if path_ids is None:
            path_ids = list(range(len(hidden_states_list)))
        
        n = len(hidden_states_list)
        groups = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new group
            current_group = [i]
            assigned.add(i)
            
            # Find similar paths
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                similarity = self._compute_similarity(
                    hidden_states_list[i],
                    hidden_states_list[j]
                )
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
                    assigned.add(j)
                    logger.debug(
                        f"[PathGrouper] Grouped path {j} with path {i}, "
                        f"similarity={similarity:.3f}"
                    )
            
            groups.append(current_group)
        
        logger.info(
            f"[PathGrouper] Grouped {n} paths into {len(groups)} groups by similarity"
        )
        
        return groups
    
    def _compute_similarity(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two hidden states.
        
        Args:
            hidden1: First hidden state tensor
            hidden2: Second hidden state tensor
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Flatten and normalize
        h1 = hidden1.flatten()
        h2 = hidden2.flatten()
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))
        
        return similarity.item()
    
    def group_by_length(
        self,
        lengths: List[int],
        max_length_diff: int = 32
    ) -> List[List[int]]:
        """Group paths by sequence length.
        
        Args:
            lengths: List of sequence lengths
            max_length_diff: Maximum length difference within a group
            
        Returns:
            List of groups, where each group is a list of path indices
        """
        if not lengths:
            return []
        
        # Sort by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        groups = []
        current_group = [sorted_indices[0]]
        base_length = lengths[sorted_indices[0]]
        
        for idx in sorted_indices[1:]:
            length = lengths[idx]
            
            if length - base_length <= max_length_diff:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
                base_length = length
        
        if current_group:
            groups.append(current_group)
        
        logger.info(
            f"[PathGrouper] Grouped {len(lengths)} paths into {len(groups)} groups by length"
        )
        
        return groups


class TensorOptimizer:
    """Optimizes tensor operations for multi-path processing.
    
    This class provides utilities to reduce redundant computations and
    optimize tensor operations across multiple paths.
    """
    
    def __init__(self):
        """Initialize the tensor optimizer."""
        logger.info("[TensorOptimizer] Initialized")
    
    def deduplicate_computations(
        self,
        tensors: List[torch.Tensor],
        tolerance: float = 1e-6
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Identify and remove duplicate tensors.
        
        Args:
            tensors: List of tensors to deduplicate
            tolerance: Tolerance for considering tensors equal
            
        Returns:
            Tuple of (unique_tensors, mapping_indices)
        """
        if not tensors:
            return [], []
        
        unique_tensors = []
        mapping = []
        
        for i, tensor in enumerate(tensors):
            # Check if similar to any existing unique tensor
            found_match = False
            for j, unique_tensor in enumerate(unique_tensors):
                if self._tensors_equal(tensor, unique_tensor, tolerance):
                    mapping.append(j)
                    found_match = True
                    logger.debug(
                        f"[TensorOptimizer] Tensor {i} is duplicate of unique tensor {j}"
                    )
                    break
            
            if not found_match:
                unique_tensors.append(tensor)
                mapping.append(len(unique_tensors) - 1)
        
        num_duplicates = len(tensors) - len(unique_tensors)
        if num_duplicates > 0:
            logger.info(
                f"[TensorOptimizer] Deduplicated {num_duplicates} tensors, "
                f"{len(unique_tensors)} unique tensors remain"
            )
        else:
            logger.debug("[TensorOptimizer] No duplicate tensors found")
        
        return unique_tensors, mapping
    
    def _tensors_equal(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        tolerance: float
    ) -> bool:
        """Check if two tensors are equal within tolerance.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            tolerance: Tolerance for equality
            
        Returns:
            True if tensors are equal within tolerance
        """
        if tensor1.shape != tensor2.shape:
            return False
        
        diff = torch.abs(tensor1 - tensor2).max().item()
        return diff <= tolerance
    
    def batch_matrix_multiply(
        self,
        matrices: List[torch.Tensor],
        vector: torch.Tensor
    ) -> List[torch.Tensor]:
        """Efficiently multiply multiple matrices by the same vector.
        
        Args:
            matrices: List of matrices to multiply
            vector: Vector to multiply with
            
        Returns:
            List of result tensors
        """
        if not matrices:
            return []
        
        # Stack matrices for batch processing
        stacked = torch.stack(matrices, dim=0)
        
        # Batch multiply
        results = torch.matmul(stacked, vector.unsqueeze(-1))
        
        # Unstack results
        result_list = [results[i].squeeze(-1) for i in range(len(matrices))]
        
        logger.debug(
            f"[TensorOptimizer] Batch multiplied {len(matrices)} matrices with vector"
        )
        
        return result_list
    
    def optimize_attention_computation(
        self,
        query_list: List[torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor
    ) -> List[torch.Tensor]:
        """Optimize attention computation for multiple queries with shared K, V.
        
        Args:
            query_list: List of query tensors
            key: Shared key tensor
            value: Shared value tensor
            
        Returns:
            List of attention output tensors
        """
        if not query_list:
            return []
        
        # Stack queries
        queries_stacked = torch.stack(query_list, dim=0)
        
        # Compute attention scores
        scores = torch.matmul(queries_stacked, key.transpose(-2, -1))
        scores = scores / (key.shape[-1] ** 0.5)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        outputs = torch.matmul(attn_weights, value)
        
        # Unstack
        result_list = [outputs[i] for i in range(len(query_list))]
        
        logger.debug(
            f"[TensorOptimizer] Optimized attention computation for {len(query_list)} queries"
        )
        
        return result_list


class BatchProcessingCoordinator:
    """Coordinates batch processing optimization for multi-path reasoning.
    
    This class combines dynamic batching, path grouping, and tensor optimization
    to maximize throughput and minimize computational overhead.
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        length_tolerance: int = 32,
        similarity_threshold: float = 0.8,
        padding_token_id: int = 0
    ):
        """Initialize the batch processing coordinator.
        
        Args:
            max_batch_size: Maximum paths per batch
            length_tolerance: Maximum length difference within batch
            similarity_threshold: Minimum similarity for grouping
            padding_token_id: Token ID for padding
        """
        self.batcher = DynamicBatcher(max_batch_size, length_tolerance, padding_token_id)
        self.grouper = PathGrouper(similarity_threshold)
        self.optimizer = TensorOptimizer()
        
        logger.info(
            f"[BatchProcessingCoordinator] Initialized with max_batch_size={max_batch_size}, "
            f"length_tolerance={length_tolerance}, similarity_threshold={similarity_threshold}"
        )
    
    def optimize_batch_processing(
        self,
        input_ids_list: List[torch.Tensor],
        hidden_states_list: Optional[List[torch.Tensor]] = None,
        path_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Optimize batch processing for a set of paths.
        
        Args:
            input_ids_list: List of input token ID tensors
            hidden_states_list: Optional list of hidden state tensors
            path_ids: Optional list of path IDs
            
        Returns:
            Dictionary with optimization results and statistics
        """
        logger.info(
            f"[BatchProcessingCoordinator] Optimizing batch processing for "
            f"{len(input_ids_list)} paths"
        )
        
        # Create batches by length
        batches = self.batcher.create_batches(input_ids_list, path_ids)
        
        # Group by similarity if hidden states provided
        similarity_groups = None
        if hidden_states_list is not None:
            similarity_groups = self.grouper.group_by_similarity(
                hidden_states_list,
                path_ids
            )
        
        # Deduplicate tensors if provided
        unique_tensors = None
        tensor_mapping = None
        if hidden_states_list is not None:
            unique_tensors, tensor_mapping = self.optimizer.deduplicate_computations(
                hidden_states_list
            )
        
        # Compile statistics
        stats = {
            'num_paths': len(input_ids_list),
            'num_batches': len(batches),
            'avg_batch_size': sum(len(b) for b in batches) / len(batches) if batches else 0,
            'num_similarity_groups': len(similarity_groups) if similarity_groups else 0,
            'num_unique_tensors': len(unique_tensors) if unique_tensors else len(input_ids_list),
            'deduplication_ratio': (
                len(unique_tensors) / len(input_ids_list)
                if unique_tensors else 1.0
            ),
        }
        
        logger.info(
            f"[BatchProcessingCoordinator] Optimization complete: "
            f"{stats['num_batches']} batches, "
            f"avg_batch_size={stats['avg_batch_size']:.1f}, "
            f"deduplication_ratio={stats['deduplication_ratio']:.2f}"
        )
        
        return {
            'batches': batches,
            'similarity_groups': similarity_groups,
            'unique_tensors': unique_tensors,
            'tensor_mapping': tensor_mapping,
            'statistics': stats,
        }

