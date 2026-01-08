"""
Data Sharding Module

Converts tokenized data into efficient binary shards for training:
- Sequence packing for efficiency
- Multiple output formats (SafeTensors, NumPy, Arrow)
- Compression support
- Shard indexing for distributed training
- Train/val/test splitting
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import struct

import numpy as np

from .config import PipelineConfig, ShardConfig
from .extraction import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class TokenizedSample:
    """A tokenized sample ready for sharding."""
    # Token IDs
    token_ids: np.ndarray  # Shape: [seq_len]
    
    # Optional embeddings for multimodal
    embeddings: Optional[np.ndarray] = None  # Shape: [num_embeddings, embed_dim]
    
    # Attention mask
    attention_mask: Optional[np.ndarray] = None
    
    # Labels for training (shifted token_ids for language modeling)
    labels: Optional[np.ndarray] = None
    
    # Metadata
    source_id: Optional[str] = None
    modalities: List[str] = field(default_factory=list)


@dataclass
class ShardMetadata:
    """Metadata for a single shard."""
    shard_id: str
    filename: str
    split: str  # "train", "val", "test"
    
    # Size info
    num_samples: int = 0
    num_tokens: int = 0
    size_bytes: int = 0
    
    # Content info
    modalities: List[str] = field(default_factory=list)
    has_embeddings: bool = False
    
    # Sequence info
    sequence_length: int = 0
    packed_sequences: bool = False
    
    # Checksums
    checksum: Optional[str] = None
    
    # Timestamps
    created_at: Optional[str] = None


@dataclass
class ShardIndex:
    """Index of all shards in a dataset."""
    version: str = "1.0"
    name: str = ""
    created_at: str = ""
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Shards by split
    train_shards: List[ShardMetadata] = field(default_factory=list)
    val_shards: List[ShardMetadata] = field(default_factory=list)
    test_shards: List[ShardMetadata] = field(default_factory=list)
    
    # Statistics
    total_samples: int = 0
    total_tokens: int = 0
    total_size_bytes: int = 0
    
    # Vocabulary info
    vocab_size: int = 0
    
    def add_shard(self, metadata: ShardMetadata):
        """Add a shard to the index."""
        if metadata.split == "train":
            self.train_shards.append(metadata)
        elif metadata.split == "val":
            self.val_shards.append(metadata)
        else:
            self.test_shards.append(metadata)
        
        self.total_samples += metadata.num_samples
        self.total_tokens += metadata.num_tokens
        self.total_size_bytes += metadata.size_bytes
    
    def save(self, path: str):
        """Save index to JSON file."""
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> "ShardIndex":
        """Load index from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Convert shard metadata
        for split in ["train_shards", "val_shards", "test_shards"]:
            data[split] = [ShardMetadata(**s) for s in data.get(split, [])]
        
        return cls(**data)


class SequencePacker:
    """
    Packs multiple short sequences into fixed-length sequences.
    
    This maximizes GPU utilization by avoiding padding waste.
    """
    
    def __init__(
        self, 
        sequence_length: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
    ):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Current packing buffer
        self.buffer: List[int] = []
        self.buffer_sources: List[str] = []
    
    def pack(
        self, 
        sequences: Iterator[Tuple[List[int], str]],
    ) -> Generator[TokenizedSample, None, None]:
        """
        Pack sequences into fixed-length chunks.
        
        Args:
            sequences: Iterator of (token_ids, source_id) tuples
            
        Yields:
            TokenizedSample with packed sequences
        """
        for token_ids, source_id in sequences:
            # Add EOS token between sequences
            if self.buffer:
                self.buffer.append(self.eos_token_id)
            
            self.buffer.extend(token_ids)
            self.buffer_sources.append(source_id)
            
            # Emit full sequences
            while len(self.buffer) >= self.sequence_length:
                chunk = self.buffer[:self.sequence_length]
                self.buffer = self.buffer[self.sequence_length:]
                
                yield self._create_sample(chunk, self.buffer_sources.copy())
                self.buffer_sources = []
        
        # Emit remaining buffer with padding
        if self.buffer:
            yield self._create_sample(self.buffer, self.buffer_sources)
            self.buffer = []
            self.buffer_sources = []
    
    def _create_sample(
        self, 
        token_ids: List[int], 
        sources: List[str],
    ) -> TokenizedSample:
        """Create a tokenized sample with padding."""
        # Pad to sequence length
        padded = token_ids[:self.sequence_length]
        padding_length = self.sequence_length - len(padded)
        padded = padded + [self.pad_token_id] * padding_length
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        attention_mask = attention_mask[:self.sequence_length]
        
        # Create labels (shifted input for language modeling)
        labels = padded[1:] + [self.pad_token_id]
        
        return TokenizedSample(
            token_ids=np.array(padded, dtype=np.int32),
            attention_mask=np.array(attention_mask, dtype=np.int32),
            labels=np.array(labels, dtype=np.int32),
            source_id=";".join(sources[:5]),  # Track up to 5 sources
            modalities=["text"],
        )


class ShardWriter:
    """
    Writes tokenized samples to shards.
    
    Supports multiple formats:
    - SafeTensors: Fast, secure, framework-agnostic
    - NumPy: Simple, widely compatible
    - Arrow: Columnar, memory-mapped, streaming
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.shard_config = config.sharding
        
        # Create output directory
        self.output_dir = Path(self.shard_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index
        self.index = ShardIndex(
            name=config.name,
            created_at=datetime.now().isoformat(),
            config={"sequence_length": self.shard_config.sequence_length},
            vocab_size=config.tokenization.vocab_size,
        )
        
        # Current shard state
        self.current_split = "train"
        self.current_samples: List[TokenizedSample] = []
        self.current_tokens = 0
        self.current_bytes = 0
        self.shard_counter = defaultdict(int)
        
        # Sequence packer
        self.packer = SequencePacker(
            sequence_length=self.shard_config.sequence_length,
            pad_token_id=config.tokenization.pad_token_id,
            eos_token_id=config.tokenization.eos_token_id,
        )
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "total_tokens": 0,
            "shards_written": 0,
        }
    
    def set_split(self, split: str):
        """Set current split (train/val/test)."""
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        # Flush current shard if switching splits
        if self.current_samples and self.current_split != split:
            self._write_shard()
        
        self.current_split = split
    
    def add_sample(self, sample: TokenizedSample):
        """Add a sample to the current shard."""
        self.current_samples.append(sample)
        self.current_tokens += len(sample.token_ids)
        self.current_bytes += sample.token_ids.nbytes
        
        if sample.embeddings is not None:
            self.current_bytes += sample.embeddings.nbytes
        
        # Check if shard is full
        if self._should_flush():
            self._write_shard()
    
    def add_tokens(self, token_ids: List[int], source_id: str = ""):
        """Add raw token IDs (will be packed into sequences)."""
        # This is a simplified version - in production, use the packer
        sample = TokenizedSample(
            token_ids=np.array(token_ids, dtype=np.int32),
            source_id=source_id,
        )
        self.add_sample(sample)
    
    def _should_flush(self) -> bool:
        """Check if current shard should be written."""
        return (
            self.current_bytes >= self.shard_config.max_shard_size_bytes or
            self.current_tokens >= self.shard_config.max_tokens_per_shard or
            len(self.current_samples) >= self.shard_config.max_samples_per_shard
        )
    
    def _write_shard(self):
        """Write current samples to a shard file."""
        if not self.current_samples:
            return
        
        # Generate shard ID
        shard_num = self.shard_counter[self.current_split]
        shard_id = f"{self.current_split}_{shard_num:05d}"
        
        # Choose format
        if self.shard_config.format == "safetensors":
            filename = self._write_safetensors(shard_id)
        elif self.shard_config.format == "arrow":
            filename = self._write_arrow(shard_id)
        else:  # numpy
            filename = self._write_numpy(shard_id)
        
        # Calculate checksum
        filepath = self.output_dir / filename
        checksum = self._calculate_checksum(filepath)
        
        # Create metadata
        metadata = ShardMetadata(
            shard_id=shard_id,
            filename=filename,
            split=self.current_split,
            num_samples=len(self.current_samples),
            num_tokens=self.current_tokens,
            size_bytes=filepath.stat().st_size,
            sequence_length=self.shard_config.sequence_length,
            packed_sequences=self.shard_config.pack_sequences,
            checksum=checksum,
            created_at=datetime.now().isoformat(),
            modalities=list(set(
                m for s in self.current_samples for m in s.modalities
            )),
            has_embeddings=any(s.embeddings is not None for s in self.current_samples),
        )
        
        # Add to index
        self.index.add_shard(metadata)
        
        # Update counters
        self.shard_counter[self.current_split] += 1
        self.stats["total_samples"] += len(self.current_samples)
        self.stats["total_tokens"] += self.current_tokens
        self.stats["shards_written"] += 1
        
        # Reset current shard
        self.current_samples = []
        self.current_tokens = 0
        self.current_bytes = 0
        
        logger.info(f"Written shard {shard_id}: {metadata.num_samples} samples, {metadata.num_tokens} tokens")
    
    def _write_safetensors(self, shard_id: str) -> str:
        """Write shard using SafeTensors format."""
        try:
            from safetensors.numpy import save_file
        except ImportError:
            logger.warning("safetensors not installed, falling back to numpy")
            return self._write_numpy(shard_id)
        
        filename = f"{shard_id}.safetensors"
        filepath = self.output_dir / filename
        
        # Stack samples into arrays
        token_ids = np.stack([s.token_ids for s in self.current_samples])
        
        tensors = {"input_ids": token_ids}
        
        # Add attention masks if available
        if self.current_samples[0].attention_mask is not None:
            attention_masks = np.stack([s.attention_mask for s in self.current_samples])
            tensors["attention_mask"] = attention_masks
        
        # Add labels if available
        if self.current_samples[0].labels is not None:
            labels = np.stack([s.labels for s in self.current_samples])
            tensors["labels"] = labels
        
        # Add embeddings if present (variable length, so store separately)
        embedding_samples = [s for s in self.current_samples if s.embeddings is not None]
        if embedding_samples:
            # For now, pad embeddings to max length
            max_emb_len = max(s.embeddings.shape[0] for s in embedding_samples)
            embed_dim = embedding_samples[0].embeddings.shape[1]
            
            padded_embeddings = []
            for s in self.current_samples:
                if s.embeddings is not None:
                    pad_len = max_emb_len - s.embeddings.shape[0]
                    if pad_len > 0:
                        padding = np.zeros((pad_len, embed_dim), dtype=s.embeddings.dtype)
                        padded = np.concatenate([s.embeddings, padding], axis=0)
                    else:
                        padded = s.embeddings
                else:
                    padded = np.zeros((max_emb_len, embed_dim), dtype=np.float32)
                padded_embeddings.append(padded)
            
            tensors["embeddings"] = np.stack(padded_embeddings)
        
        save_file(tensors, str(filepath))
        return filename
    
    def _write_numpy(self, shard_id: str) -> str:
        """Write shard using NumPy format."""
        filename = f"{shard_id}.npz"
        filepath = self.output_dir / filename
        
        # Stack samples
        token_ids = np.stack([s.token_ids for s in self.current_samples])
        
        arrays = {"input_ids": token_ids}
        
        if self.current_samples[0].attention_mask is not None:
            arrays["attention_mask"] = np.stack([s.attention_mask for s in self.current_samples])
        
        if self.current_samples[0].labels is not None:
            arrays["labels"] = np.stack([s.labels for s in self.current_samples])
        
        # Optionally compress
        if self.shard_config.compression:
            np.savez_compressed(filepath, **arrays)
        else:
            np.savez(filepath, **arrays)
        
        return filename
    
    def _write_arrow(self, shard_id: str) -> str:
        """Write shard using Arrow/Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed, falling back to numpy")
            return self._write_numpy(shard_id)
        
        filename = f"{shard_id}.parquet"
        filepath = self.output_dir / filename
        
        # Create table
        arrays = {
            "input_ids": [s.token_ids.tolist() for s in self.current_samples],
        }
        
        if self.current_samples[0].attention_mask is not None:
            arrays["attention_mask"] = [s.attention_mask.tolist() for s in self.current_samples]
        
        if self.current_samples[0].labels is not None:
            arrays["labels"] = [s.labels.tolist() for s in self.current_samples]
        
        if self.current_samples[0].source_id:
            arrays["source_id"] = [s.source_id for s in self.current_samples]
        
        table = pa.table(arrays)
        
        # Write with optional compression
        compression = self.shard_config.compression or "snappy"
        pq.write_table(table, filepath, compression=compression)
        
        return filename
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate MD5 checksum of file."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def flush(self):
        """Flush any remaining samples to disk."""
        if self.current_samples:
            self._write_shard()
    
    def finalize(self) -> ShardIndex:
        """Finalize writing and save index."""
        self.flush()
        
        # Save index
        index_path = self.output_dir / self.shard_config.index_file
        self.index.save(str(index_path))
        
        logger.info(f"Finalized: {self.stats['shards_written']} shards, "
                   f"{self.stats['total_samples']} samples, "
                   f"{self.stats['total_tokens']} tokens")
        
        return self.index


class ShardReader:
    """
    Reads samples from shards.
    
    Supports streaming and random access.
    """
    
    def __init__(self, index_path: str):
        self.index = ShardIndex.load(index_path)
        self.shard_dir = Path(index_path).parent
    
    def get_shard_paths(self, split: str = "train") -> List[Path]:
        """Get paths to all shards for a split."""
        if split == "train":
            shards = self.index.train_shards
        elif split == "val":
            shards = self.index.val_shards
        else:
            shards = self.index.test_shards
        
        return [self.shard_dir / s.filename for s in shards]
    
    def read_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        """Read a single shard file."""
        suffix = shard_path.suffix
        
        if suffix == ".safetensors":
            return self._read_safetensors(shard_path)
        elif suffix == ".parquet":
            return self._read_arrow(shard_path)
        else:  # .npz
            return self._read_numpy(shard_path)
    
    def _read_safetensors(self, path: Path) -> Dict[str, np.ndarray]:
        """Read SafeTensors shard."""
        from safetensors.numpy import load_file
        return load_file(str(path))
    
    def _read_numpy(self, path: Path) -> Dict[str, np.ndarray]:
        """Read NumPy shard."""
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    
    def _read_arrow(self, path: Path) -> Dict[str, np.ndarray]:
        """Read Arrow/Parquet shard."""
        import pyarrow.parquet as pq
        
        table = pq.read_table(path)
        
        result = {}
        for col in table.column_names:
            arr = table[col].to_pylist()
            result[col] = np.array(arr)
        
        return result
    
    def iterate_samples(
        self, 
        split: str = "train",
        shuffle_shards: bool = True,
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Iterate over all samples in a split.
        
        Args:
            split: Which split to iterate
            shuffle_shards: Whether to shuffle shard order
            
        Yields:
            Dictionary with input_ids, attention_mask, labels, etc.
        """
        shard_paths = self.get_shard_paths(split)
        
        if shuffle_shards:
            np.random.shuffle(shard_paths)
        
        for shard_path in shard_paths:
            data = self.read_shard(shard_path)
            num_samples = len(data["input_ids"])
            
            for i in range(num_samples):
                sample = {key: arr[i] for key, arr in data.items()}
                yield sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_samples": self.index.total_samples,
            "total_tokens": self.index.total_tokens,
            "total_size_bytes": self.index.total_size_bytes,
            "train_shards": len(self.index.train_shards),
            "val_shards": len(self.index.val_shards),
            "test_shards": len(self.index.test_shards),
            "vocab_size": self.index.vocab_size,
        }
