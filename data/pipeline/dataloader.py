"""
Training DataLoader Module

Efficient data loading for distributed training:
- Streaming from shards
- Prefetching and buffering
- Dynamic batching
- Multi-worker loading
- JAX/NumPy compatible
"""

import os
import logging
import threading
import queue
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any, Iterator, Tuple
from dataclasses import dataclass, field
from collections import deque
import random

import numpy as np
import jax.numpy as jnp

from .config import PipelineConfig, ShardConfig
from .sharding import ShardReader, ShardIndex

logger = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    """A batch of training data."""
    # Core tensors (JAX arrays)
    input_ids: jnp.ndarray  # Shape: [batch_size, seq_len]
    attention_mask: jnp.ndarray  # Shape: [batch_size, seq_len]
    labels: jnp.ndarray  # Shape: [batch_size, seq_len]
    
    # Optional embeddings for multimodal
    embeddings: Optional[jnp.ndarray] = None  # Shape: [batch_size, num_emb, embed_dim]
    
    # Batch metadata
    batch_size: int = 0
    sequence_length: int = 0
    num_tokens: int = 0
    
    @classmethod
    def from_numpy(
        cls,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> "TrainingBatch":
        """Create batch from NumPy arrays."""
        batch_size, seq_len = input_ids.shape
        
        return cls(
            input_ids=jnp.array(input_ids),
            attention_mask=jnp.array(attention_mask),
            labels=jnp.array(labels),
            embeddings=jnp.array(embeddings) if embeddings is not None else None,
            batch_size=batch_size,
            sequence_length=seq_len,
            num_tokens=int(attention_mask.sum()),
        )
    
    def to_dict(self) -> Dict[str, jnp.ndarray]:
        """Convert to dictionary for model input."""
        result = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }
        if self.embeddings is not None:
            result["embeddings"] = self.embeddings
        return result


class ShardBuffer:
    """
    Buffer that loads and shuffles samples from shards.
    
    Implements reservoir sampling for memory-efficient shuffling.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        seed: Optional[int] = None,
    ):
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.rng = random.Random(seed)
    
    def add(self, sample: Dict[str, np.ndarray]):
        """Add a sample to the buffer."""
        self.buffer.append(sample)
    
    def add_batch(self, samples: List[Dict[str, np.ndarray]]):
        """Add multiple samples."""
        for sample in samples:
            self.add(sample)
    
    def get(self) -> Optional[Dict[str, np.ndarray]]:
        """Get a random sample from the buffer."""
        if not self.buffer:
            return None
        
        idx = self.rng.randint(0, len(self.buffer) - 1)
        
        # Swap with last and pop for O(1) removal
        self.buffer[idx], self.buffer[-1] = self.buffer[-1], self.buffer[idx]
        return self.buffer.pop()
    
    def get_batch(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Get a batch of random samples."""
        samples = []
        for _ in range(min(batch_size, len(self.buffer))):
            sample = self.get()
            if sample is not None:
                samples.append(sample)
        return samples
    
    def is_ready(self, min_samples: int = 1000) -> bool:
        """Check if buffer has enough samples for sampling."""
        return len(self.buffer) >= min_samples
    
    def __len__(self) -> int:
        return len(self.buffer)


class ShardedDataLoader:
    """
    Production-ready data loader for training.
    
    Features:
    - Streams data from shards
    - Shuffles across shards and within buffer
    - Prefetches next shard in background
    - Supports multiple workers
    - Memory-efficient for large datasets
    """
    
    def __init__(
        self,
        index_path: str,
        batch_size: int = 32,
        split: str = "train",
        shuffle: bool = True,
        buffer_size: int = 10000,
        prefetch_shards: int = 2,
        seed: Optional[int] = None,
        drop_last: bool = True,
    ):
        """
        Initialize the data loader.
        
        Args:
            index_path: Path to shard index file
            batch_size: Batch size
            split: Which split to load ("train", "val", "test")
            shuffle: Whether to shuffle data
            buffer_size: Size of shuffle buffer
            prefetch_shards: Number of shards to prefetch
            seed: Random seed for reproducibility
            drop_last: Whether to drop incomplete last batch
        """
        self.reader = ShardReader(index_path)
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Get shard paths
        self.shard_paths = self.reader.get_shard_paths(split)
        
        # Buffer for shuffling
        self.buffer = ShardBuffer(buffer_size, seed)
        
        # Prefetching
        self.prefetch_shards = prefetch_shards
        self.prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_shards)
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_prefetch = threading.Event()
        
        # State
        self.current_epoch = 0
        self.samples_seen = 0
        self.batches_yielded = 0
        
        # Calculate total samples
        stats = self.reader.get_statistics()
        self.total_samples = stats["total_samples"]
        self.total_batches = self.total_samples // batch_size
        
        logger.info(f"Initialized DataLoader: {len(self.shard_paths)} shards, "
                   f"{self.total_samples} samples, batch_size={batch_size}")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.total_batches
    
    def __iter__(self) -> Iterator[TrainingBatch]:
        """Iterate over batches."""
        return self._iterate_epoch()
    
    def _iterate_epoch(self) -> Generator[TrainingBatch, None, None]:
        """Generate batches for one epoch."""
        self.current_epoch += 1
        self.samples_seen = 0
        self.batches_yielded = 0
        
        # Get shard order
        shard_paths = list(self.shard_paths)
        if self.shuffle:
            rng = random.Random(self.seed + self.current_epoch if self.seed else None)
            rng.shuffle(shard_paths)
        
        # Start prefetching
        if self.prefetch_shards > 0:
            self._start_prefetch(shard_paths)
        
        try:
            # Process shards
            for shard_path in shard_paths:
                # Load shard (from prefetch queue or directly)
                if self.prefetch_shards > 0:
                    try:
                        shard_data = self.prefetch_queue.get(timeout=60)
                    except queue.Empty:
                        logger.warning("Prefetch timeout, loading directly")
                        shard_data = self.reader.read_shard(shard_path)
                else:
                    shard_data = self.reader.read_shard(shard_path)
                
                # Add samples to buffer
                num_samples = len(shard_data["input_ids"])
                for i in range(num_samples):
                    sample = {key: arr[i] for key, arr in shard_data.items()}
                    self.buffer.add(sample)
                
                # Yield batches when buffer is ready
                while self.buffer.is_ready(self.batch_size * 2):
                    batch = self._create_batch()
                    if batch is not None:
                        yield batch
            
            # Drain remaining buffer
            while len(self.buffer) >= self.batch_size:
                batch = self._create_batch()
                if batch is not None:
                    yield batch
            
            # Handle last incomplete batch
            if not self.drop_last and len(self.buffer) > 0:
                batch = self._create_batch(allow_smaller=True)
                if batch is not None:
                    yield batch
        
        finally:
            self._stop_prefetch()
    
    def _create_batch(self, allow_smaller: bool = False) -> Optional[TrainingBatch]:
        """Create a batch from the buffer."""
        samples = self.buffer.get_batch(self.batch_size)
        
        if len(samples) < self.batch_size and not allow_smaller:
            # Put samples back
            for s in samples:
                self.buffer.add(s)
            return None
        
        if not samples:
            return None
        
        # Stack samples
        input_ids = np.stack([s["input_ids"] for s in samples])
        
        # Handle attention mask
        if "attention_mask" in samples[0]:
            attention_mask = np.stack([s["attention_mask"] for s in samples])
        else:
            # Create from input_ids (non-zero = attended)
            attention_mask = (input_ids != 0).astype(np.int32)
        
        # Handle labels
        if "labels" in samples[0]:
            labels = np.stack([s["labels"] for s in samples])
        else:
            # Shift input_ids for language modeling
            labels = np.roll(input_ids, -1, axis=1)
            labels[:, -1] = 0  # Pad last position
        
        # Handle embeddings
        embeddings = None
        if "embeddings" in samples[0]:
            embeddings = np.stack([s["embeddings"] for s in samples])
        
        self.samples_seen += len(samples)
        self.batches_yielded += 1
        
        return TrainingBatch.from_numpy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            embeddings=embeddings,
        )
    
    def _start_prefetch(self, shard_paths: List[Path]):
        """Start background prefetching."""
        self.stop_prefetch.clear()
        
        def prefetch_worker():
            for shard_path in shard_paths:
                if self.stop_prefetch.is_set():
                    break
                try:
                    data = self.reader.read_shard(shard_path)
                    self.prefetch_queue.put(data, timeout=60)
                except Exception as e:
                    logger.error(f"Prefetch error: {e}")
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _stop_prefetch(self):
        """Stop background prefetching."""
        self.stop_prefetch.set()
        
        # Clear queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=5)
    
    def get_state(self) -> Dict[str, Any]:
        """Get loader state for checkpointing."""
        return {
            "epoch": self.current_epoch,
            "samples_seen": self.samples_seen,
            "batches_yielded": self.batches_yielded,
            "seed": self.seed,
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore loader state from checkpoint."""
        self.current_epoch = state.get("epoch", 0)
        self.samples_seen = state.get("samples_seen", 0)
        self.batches_yielded = state.get("batches_yielded", 0)


class MultiModalDataLoader(ShardedDataLoader):
    """
    Extended data loader for multimodal training.
    
    Handles text + image/audio/video embeddings.
    """
    
    def __init__(
        self,
        index_path: str,
        batch_size: int = 16,  # Smaller batches for multimodal
        max_text_tokens: int = 1024,
        max_embedding_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(index_path, batch_size, **kwargs)
        
        self.max_text_tokens = max_text_tokens
        self.max_embedding_tokens = max_embedding_tokens
    
    def _create_batch(self, allow_smaller: bool = False) -> Optional[TrainingBatch]:
        """Create multimodal batch with proper padding."""
        samples = self.buffer.get_batch(self.batch_size)
        
        if len(samples) < self.batch_size and not allow_smaller:
            for s in samples:
                self.buffer.add(s)
            return None
        
        if not samples:
            return None
        
        # Separate text and embeddings
        # Text tokens are truncated/padded to max_text_tokens
        # Embeddings are truncated/padded to max_embedding_tokens
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_embeddings = []
        
        for sample in samples:
            input_ids = sample["input_ids"]
            
            # Truncate/pad text
            if len(input_ids) > self.max_text_tokens:
                input_ids = input_ids[:self.max_text_tokens]
            else:
                padding = np.zeros(self.max_text_tokens - len(input_ids), dtype=input_ids.dtype)
                input_ids = np.concatenate([input_ids, padding])
            
            batch_input_ids.append(input_ids)
            
            # Attention mask
            if "attention_mask" in sample:
                mask = sample["attention_mask"]
                if len(mask) > self.max_text_tokens:
                    mask = mask[:self.max_text_tokens]
                else:
                    padding = np.zeros(self.max_text_tokens - len(mask), dtype=mask.dtype)
                    mask = np.concatenate([mask, padding])
            else:
                mask = (input_ids != 0).astype(np.int32)
            batch_attention_mask.append(mask)
            
            # Labels
            if "labels" in sample:
                labels = sample["labels"]
                if len(labels) > self.max_text_tokens:
                    labels = labels[:self.max_text_tokens]
                else:
                    padding = np.zeros(self.max_text_tokens - len(labels), dtype=labels.dtype)
                    labels = np.concatenate([labels, padding])
            else:
                labels = np.roll(input_ids, -1)
                labels[-1] = 0
            batch_labels.append(labels)
            
            # Embeddings
            if "embeddings" in sample:
                emb = sample["embeddings"]
                embed_dim = emb.shape[-1] if len(emb.shape) > 1 else 768
                
                if len(emb) > self.max_embedding_tokens:
                    emb = emb[:self.max_embedding_tokens]
                elif len(emb) < self.max_embedding_tokens:
                    padding = np.zeros(
                        (self.max_embedding_tokens - len(emb), embed_dim),
                        dtype=emb.dtype
                    )
                    emb = np.concatenate([emb, padding], axis=0)
                
                batch_embeddings.append(emb)
        
        self.samples_seen += len(samples)
        self.batches_yielded += 1
        
        result = TrainingBatch.from_numpy(
            input_ids=np.stack(batch_input_ids),
            attention_mask=np.stack(batch_attention_mask),
            labels=np.stack(batch_labels),
            embeddings=np.stack(batch_embeddings) if batch_embeddings else None,
        )
        
        return result


def create_dataloader(
    shard_dir: str,
    batch_size: int = 32,
    split: str = "train",
    multimodal: bool = False,
    **kwargs,
) -> ShardedDataLoader:
    """
    Factory function to create appropriate data loader.
    
    Args:
        shard_dir: Directory containing shards and index
        batch_size: Batch size
        split: Which split to load
        multimodal: Whether to use multimodal loader
        **kwargs: Additional arguments passed to loader
        
    Returns:
        Configured data loader
    """
    index_path = os.path.join(shard_dir, "shard_index.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Shard index not found: {index_path}")
    
    if multimodal:
        return MultiModalDataLoader(
            index_path=index_path,
            batch_size=batch_size,
            split=split,
            **kwargs,
        )
    else:
        return ShardedDataLoader(
            index_path=index_path,
            batch_size=batch_size,
            split=split,
            **kwargs,
        )
