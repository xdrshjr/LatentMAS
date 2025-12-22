"""KV-Cache optimization module for multi-path reasoning.

This module provides utilities to optimize KV-cache usage across multiple reasoning paths,
including shared prefix detection, cache reuse, eviction strategies, and serialization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import torch
import pickle
from pathlib import Path
import hashlib

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached KV-cache entry.
    
    Attributes:
        cache_id: Unique identifier for this cache entry
        kv_cache: The actual KV-cache tuple
        prefix_hash: Hash of the input prefix that generated this cache
        access_count: Number of times this cache has been accessed
        last_access_step: Step number of last access
        memory_size: Estimated memory size in bytes
        metadata: Additional metadata
    """
    cache_id: str
    kv_cache: Tuple
    prefix_hash: str
    access_count: int = 0
    last_access_step: int = 0
    memory_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self, step: int) -> None:
        """Record an access to this cache entry.
        
        Args:
            step: Current step number
        """
        self.access_count += 1
        self.last_access_step = step
        logger.debug(f"[CacheEntry] Accessed cache {self.cache_id}, count: {self.access_count}")


class KVCacheManager:
    """Manages KV-cache optimization for multi-path reasoning.
    
    This class provides methods to detect shared prefixes, reuse caches,
    implement eviction strategies, and track memory usage.
    
    Attributes:
        max_cache_size: Maximum cache size in bytes (None for unlimited)
        eviction_strategy: Strategy for cache eviction ('lru', 'lfu', 'fifo')
        cache_entries: Dictionary of cache entries by cache_id
        prefix_to_cache: Mapping from prefix hash to cache_id
        current_memory: Current total memory usage in bytes
        current_step: Current step counter for LRU tracking
        hit_count: Number of cache hits
        miss_count: Number of cache misses
    """
    
    def __init__(
        self,
        max_cache_size: Optional[int] = None,
        eviction_strategy: str = "lru"
    ):
        """Initialize the KV-cache manager.
        
        Args:
            max_cache_size: Maximum cache size in bytes (None for unlimited)
            eviction_strategy: Eviction strategy ('lru', 'lfu', 'fifo')
        """
        self.max_cache_size = max_cache_size
        self.eviction_strategy = eviction_strategy
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.prefix_to_cache: Dict[str, str] = {}
        self.current_memory: int = 0
        self.current_step: int = 0
        self.hit_count: int = 0
        self.miss_count: int = 0
        
        logger.info(
            f"[KVCacheManager] Initialized with max_size={max_cache_size}, "
            f"eviction_strategy={eviction_strategy}"
        )
    
    def compute_prefix_hash(self, input_ids: torch.Tensor) -> str:
        """Compute a hash for an input prefix.
        
        Args:
            input_ids: Input token IDs tensor
            
        Returns:
            Hash string representing the prefix
        """
        # Convert tensor to bytes and compute hash
        input_bytes = input_ids.cpu().numpy().tobytes()
        hash_obj = hashlib.sha256(input_bytes)
        return hash_obj.hexdigest()
    
    def estimate_cache_size(self, kv_cache: Tuple) -> int:
        """Estimate memory size of a KV-cache.
        
        Args:
            kv_cache: KV-cache tuple
            
        Returns:
            Estimated size in bytes
        """
        if kv_cache is None:
            return 0
        
        total_size = 0
        try:
            for layer_cache in kv_cache:
                if layer_cache is not None:
                    # Each layer has (key, value) tensors
                    key_tensor, value_tensor = layer_cache
                    total_size += key_tensor.element_size() * key_tensor.nelement()
                    total_size += value_tensor.element_size() * value_tensor.nelement()
        except Exception as e:
            logger.debug(f"[KVCacheManager] Error estimating cache size: {e}")
            return 0
        
        return total_size
    
    def get_cache(
        self,
        input_ids: torch.Tensor,
        prefix_length: Optional[int] = None
    ) -> Optional[Tuple]:
        """Retrieve cached KV-cache for a given input prefix.
        
        Args:
            input_ids: Input token IDs
            prefix_length: Length of prefix to consider (None for full sequence)
            
        Returns:
            Cached KV-cache if found, None otherwise
        """
        self.current_step += 1
        
        # Use prefix if specified
        if prefix_length is not None:
            prefix_ids = input_ids[:, :prefix_length]
        else:
            prefix_ids = input_ids
        
        prefix_hash = self.compute_prefix_hash(prefix_ids)
        
        # Check if we have a cached entry
        if prefix_hash in self.prefix_to_cache:
            cache_id = self.prefix_to_cache[prefix_hash]
            if cache_id in self.cache_entries:
                entry = self.cache_entries[cache_id]
                entry.access(self.current_step)
                self.hit_count += 1
                
                logger.debug(
                    f"[KVCacheManager] Cache HIT for prefix_hash={prefix_hash[:8]}..., "
                    f"cache_id={cache_id}"
                )
                logger.info(
                    f"[KVCacheManager] Cache hit rate: "
                    f"{self.hit_count}/{self.hit_count + self.miss_count} "
                    f"({100 * self.hit_count / (self.hit_count + self.miss_count):.1f}%)"
                )
                
                return entry.kv_cache
        
        self.miss_count += 1
        logger.debug(
            f"[KVCacheManager] Cache MISS for prefix_hash={prefix_hash[:8]}..."
        )
        
        return None
    
    def put_cache(
        self,
        input_ids: torch.Tensor,
        kv_cache: Tuple,
        prefix_length: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a KV-cache for a given input prefix.
        
        Args:
            input_ids: Input token IDs
            kv_cache: KV-cache to store
            prefix_length: Length of prefix to consider (None for full sequence)
            metadata: Additional metadata
            
        Returns:
            Cache ID of the stored entry
        """
        # Use prefix if specified
        if prefix_length is not None:
            prefix_ids = input_ids[:, :prefix_length]
        else:
            prefix_ids = input_ids
        
        prefix_hash = self.compute_prefix_hash(prefix_ids)
        
        # Check if already cached
        if prefix_hash in self.prefix_to_cache:
            cache_id = self.prefix_to_cache[prefix_hash]
            logger.debug(
                f"[KVCacheManager] Cache already exists for prefix_hash={prefix_hash[:8]}..., "
                f"cache_id={cache_id}"
            )
            return cache_id
        
        # Estimate size
        cache_size = self.estimate_cache_size(kv_cache)
        
        # Evict if necessary
        if self.max_cache_size is not None:
            while self.current_memory + cache_size > self.max_cache_size and self.cache_entries:
                self._evict_one()
        
        # Create new entry
        cache_id = f"cache_{len(self.cache_entries)}_{prefix_hash[:8]}"
        entry = CacheEntry(
            cache_id=cache_id,
            kv_cache=kv_cache,
            prefix_hash=prefix_hash,
            access_count=1,
            last_access_step=self.current_step,
            memory_size=cache_size,
            metadata=metadata or {}
        )
        
        self.cache_entries[cache_id] = entry
        self.prefix_to_cache[prefix_hash] = cache_id
        self.current_memory += cache_size
        
        logger.info(
            f"[KVCacheManager] Stored cache {cache_id}, size={cache_size / (1024**2):.2f} MB, "
            f"total_memory={self.current_memory / (1024**2):.2f} MB"
        )
        logger.debug(f"[KVCacheManager] Total cache entries: {len(self.cache_entries)}")
        
        return cache_id
    
    def _evict_one(self) -> None:
        """Evict one cache entry based on the eviction strategy."""
        if not self.cache_entries:
            return
        
        if self.eviction_strategy == "lru":
            # Evict least recently used
            victim_id = min(
                self.cache_entries.keys(),
                key=lambda cid: self.cache_entries[cid].last_access_step
            )
        elif self.eviction_strategy == "lfu":
            # Evict least frequently used
            victim_id = min(
                self.cache_entries.keys(),
                key=lambda cid: self.cache_entries[cid].access_count
            )
        else:  # fifo
            # Evict first in (oldest)
            victim_id = next(iter(self.cache_entries.keys()))
        
        victim = self.cache_entries[victim_id]
        self.current_memory -= victim.memory_size
        
        # Remove from mappings
        del self.cache_entries[victim_id]
        if victim.prefix_hash in self.prefix_to_cache:
            del self.prefix_to_cache[victim.prefix_hash]
        
        logger.info(
            f"[KVCacheManager] Evicted cache {victim_id} using {self.eviction_strategy} strategy, "
            f"freed {victim.memory_size / (1024**2):.2f} MB"
        )
    
    def detect_shared_prefix(
        self,
        input_ids_list: List[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Detect the longest shared prefix among multiple input sequences.
        
        Args:
            input_ids_list: List of input token ID tensors
            
        Returns:
            Tuple of (shared_prefix_tensor, shared_length)
        """
        if not input_ids_list:
            return None, 0
        
        if len(input_ids_list) == 1:
            return input_ids_list[0], input_ids_list[0].shape[1]
        
        # Find minimum length
        min_length = min(ids.shape[1] for ids in input_ids_list)
        
        # Find shared prefix length
        shared_length = 0
        for i in range(min_length):
            tokens = [ids[0, i].item() for ids in input_ids_list]
            if len(set(tokens)) == 1:  # All same
                shared_length = i + 1
            else:
                break
        
        if shared_length > 0:
            shared_prefix = input_ids_list[0][:, :shared_length]
            logger.info(
                f"[KVCacheManager] Detected shared prefix of length {shared_length} "
                f"among {len(input_ids_list)} sequences"
            )
            logger.debug(
                f"[KVCacheManager] Shared prefix can save "
                f"{(len(input_ids_list) - 1) * shared_length} token computations"
            )
            return shared_prefix, shared_length
        
        logger.debug("[KVCacheManager] No shared prefix detected")
        return None, 0
    
    def clear(self) -> None:
        """Clear all cached entries."""
        num_entries = len(self.cache_entries)
        memory_freed = self.current_memory
        
        self.cache_entries.clear()
        self.prefix_to_cache.clear()
        self.current_memory = 0
        
        logger.info(
            f"[KVCacheManager] Cleared {num_entries} cache entries, "
            f"freed {memory_freed / (1024**2):.2f} MB"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_accesses = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_accesses if total_accesses > 0 else 0.0
        
        stats = {
            'num_entries': len(self.cache_entries),
            'total_memory_mb': self.current_memory / (1024**2),
            'max_memory_mb': self.max_cache_size / (1024**2) if self.max_cache_size else None,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'eviction_strategy': self.eviction_strategy,
        }
        
        logger.info(
            f"[KVCacheManager] Statistics: {len(self.cache_entries)} entries, "
            f"{stats['total_memory_mb']:.2f} MB, hit_rate={hit_rate:.2%}"
        )
        
        return stats
    
    def save_to_file(self, filepath: str) -> None:
        """Serialize cache manager state to file.
        
        Args:
            filepath: Path to save the cache state
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable state (excluding actual tensors)
        state = {
            'max_cache_size': self.max_cache_size,
            'eviction_strategy': self.eviction_strategy,
            'current_memory': self.current_memory,
            'current_step': self.current_step,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'prefix_to_cache': self.prefix_to_cache,
            'cache_metadata': {
                cid: {
                    'cache_id': entry.cache_id,
                    'prefix_hash': entry.prefix_hash,
                    'access_count': entry.access_count,
                    'last_access_step': entry.last_access_step,
                    'memory_size': entry.memory_size,
                    'metadata': entry.metadata,
                }
                for cid, entry in self.cache_entries.items()
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(
            f"[KVCacheManager] Saved cache state to {filepath}, "
            f"{len(self.cache_entries)} entries"
        )
    
    def load_from_file(self, filepath: str) -> None:
        """Load cache manager state from file.
        
        Note: This loads metadata only, not the actual KV-cache tensors.
        
        Args:
            filepath: Path to load the cache state from
        """
        load_path = Path(filepath)
        if not load_path.exists():
            logger.warning(f"[KVCacheManager] Cache file not found: {filepath}")
            return
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.max_cache_size = state['max_cache_size']
        self.eviction_strategy = state['eviction_strategy']
        self.current_memory = state['current_memory']
        self.current_step = state['current_step']
        self.hit_count = state['hit_count']
        self.miss_count = state['miss_count']
        self.prefix_to_cache = state['prefix_to_cache']
        
        logger.info(
            f"[KVCacheManager] Loaded cache state from {filepath}, "
            f"{len(state['cache_metadata'])} entries (metadata only)"
        )
        logger.debug(
            f"[KVCacheManager] Note: Actual KV-cache tensors not loaded, "
            f"will be regenerated on demand"
        )


class SharedPrefixOptimizer:
    """Optimizer for detecting and exploiting shared prefixes across paths.
    
    This class provides utilities to identify common prefixes in multiple reasoning
    paths and optimize computation by reusing cached results.
    """
    
    def __init__(self, cache_manager: KVCacheManager):
        """Initialize the shared prefix optimizer.
        
        Args:
            cache_manager: KV-cache manager to use for caching
        """
        self.cache_manager = cache_manager
        logger.info("[SharedPrefixOptimizer] Initialized")
    
    def optimize_batch(
        self,
        input_ids_list: List[torch.Tensor],
        generate_fn: callable
    ) -> List[Tuple]:
        """Optimize batch processing by detecting shared prefixes.
        
        Args:
            input_ids_list: List of input token ID tensors
            generate_fn: Function to generate KV-cache for inputs
            
        Returns:
            List of KV-caches for each input
        """
        logger.info(
            f"[SharedPrefixOptimizer] Optimizing batch of {len(input_ids_list)} sequences"
        )
        
        # Detect shared prefix
        shared_prefix, shared_length = self.cache_manager.detect_shared_prefix(input_ids_list)
        
        results = []
        
        if shared_prefix is not None and shared_length > 0:
            # Try to get cached KV for shared prefix
            shared_kv = self.cache_manager.get_cache(shared_prefix, prefix_length=shared_length)
            
            if shared_kv is None:
                # Generate and cache shared prefix
                logger.debug(
                    f"[SharedPrefixOptimizer] Generating KV-cache for shared prefix "
                    f"of length {shared_length}"
                )
                shared_kv = generate_fn(shared_prefix)
                self.cache_manager.put_cache(
                    shared_prefix,
                    shared_kv,
                    prefix_length=shared_length,
                    metadata={'type': 'shared_prefix'}
                )
            
            # Generate remaining parts
            for input_ids in input_ids_list:
                if input_ids.shape[1] > shared_length:
                    # Has additional tokens beyond shared prefix
                    logger.debug(
                        f"[SharedPrefixOptimizer] Generating continuation from position {shared_length}"
                    )
                    full_kv = generate_fn(input_ids, past_key_values=shared_kv)
                    results.append(full_kv)
                else:
                    # Only shared prefix
                    results.append(shared_kv)
        else:
            # No shared prefix, process individually
            logger.debug("[SharedPrefixOptimizer] No shared prefix, processing individually")
            for input_ids in input_ids_list:
                kv = generate_fn(input_ids)
                results.append(kv)
        
        logger.info(
            f"[SharedPrefixOptimizer] Completed batch optimization, "
            f"generated {len(results)} KV-caches"
        )
        
        return results

