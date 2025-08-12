import faiss
import jax
import logging
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Enhanced memory item with metadata."""
    key: np.ndarray
    value: np.ndarray
    timestamp: float
    access_count: int
    importance_score: float
    context_tags: List[str]
    emotional_valence: float  # -1 to 1
    consolidation_level: int  # 0=STM, 1=MTM, 2=LTM


class AdaptiveForgettingCurve:
    """Implements sophisticated forgetting mechanisms."""
    
    def __init__(self, base_decay: float = 0.1, importance_weight: float = 0.5):
        self.base_decay = base_decay
        self.importance_weight = importance_weight
    
    def calculate_retention_probability(self, memory_item: MemoryItem, 
                                     current_time: float) -> float:
        """Calculate probability of retaining a memory."""
        # Time-based decay
        time_delta = current_time - memory_item.timestamp
        time_decay = np.exp(-self.base_decay * time_delta)
        
        # Importance-based retention
        importance_boost = 1.0 + self.importance_weight * memory_item.importance_score
        
        # Access frequency boost
        access_boost = 1.0 + 0.1 * np.log1p(memory_item.access_count)
        
        # Emotional significance boost
        emotional_boost = 1.0 + 0.2 * abs(memory_item.emotional_valence)
        
        # Combined retention probability
        retention_prob = time_decay * importance_boost * access_boost * emotional_boost
        return min(1.0, retention_prob)


class ContextualMemoryIndex:
    """Enhanced memory indexing with contextual clustering."""
    
    def __init__(self, embedding_dim: int, num_clusters: int = 8):
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.context_clusters = {}
        self.memory_items: List[MemoryItem] = []
        
    def add_memory(self, memory_item: MemoryItem):
        """Add memory with contextual clustering."""
        self.memory_items.append(memory_item)
        key_array = memory_item.key.reshape(1, -1).astype(np.float32)
        self.index.add(key_array)
        
        # Update context clusters
        for tag in memory_item.context_tags:
            if tag not in self.context_clusters:
                self.context_clusters[tag] = []
            self.context_clusters[tag].append(len(self.memory_items) - 1)
    
    def retrieve_contextual(self, query: np.ndarray, context_tags: List[str], 
                          k: int = 5) -> List[MemoryItem]:
        """Retrieve memories with contextual filtering."""
        if self.index.ntotal == 0:
            return []
            
        # Get candidate memories
        query_array = query.reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(query_array, min(k * 3, self.index.ntotal))
        
        candidates = []
        for idx in indices[0]:
            if idx < len(self.memory_items):
                memory = self.memory_items[idx]
                # Calculate context relevance
                context_overlap = len(set(memory.context_tags) & set(context_tags))
                relevance_score = context_overlap / (len(context_tags) + 1e-8)
                candidates.append((memory, relevance_score))
        
        # Sort by relevance and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in candidates[:k]]

class MemoryBank:
    def __init__(self, memory_size: int, embedding_dim: int, retrieval_k: int):
        """
        Memory Bank using FAISS for efficient retrieval.
        """
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.retrieval_k = retrieval_k
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.values = []
        self.feedback_scores = []

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)

    def store(self, keys, values, feedback_scores=None, spike_threshold=0.1, epsilon=1e-8):
        """
        Stores key-value pairs in the memory bank.
        :param keys: Batch of mean-pooled embeddings (shape: [batch_size, embedding_dim])
        :param values: Corresponding transformer activations (shape: [batch_size, embedding_dim])
        """
        # **Ensure computation is complete before conversion**
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)

        # **Detach from JAX before passing to FAISS -- Apply Spiking Attention to keys before storage**
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)

        # **Ensure 2D shape**
        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)

        assert keys_np.shape[1] == self.embedding_dim, f"Expected dim {self.embedding_dim}, got {keys_np.shape[1]}"
        if len(self.values) + len(keys_np) > self.memory_size:
            remove_count = len(self.values) + len(keys_np) - self.memory_size
            self.values = self.values[remove_count:]
            self.index.reset()
            self.index.add(np.asarray(self.values, dtype=np.float32))

        self.values.extend(values_np.tolist())
        self.feedback_scores.extend([feedback_scores] * len(keys_np) if feedback_scores else [0.0] * len(keys_np))
        self.index.add(keys_np)

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        """
        Retrieves closest memory values.
        :param queries: Query embeddings (shape: [batch_size, embedding_dim])
        :return: Retrieved memory activations (shape: [batch_size, embedding_dim])
        """
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if self.index.ntotal == 0:
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        distances, indices = self.index.search(queries_np, self.retrieval_k)
        retrieved_values = np.array([self.values[idx] for idx in indices.flatten()]).reshape(
            indices.shape[0], indices.shape[1], self.embedding_dim
        )
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        noise = np.random.normal(0, 0.1, retrieved_values.shape).astype(np.float32)
        retrieved_values = retrieved_values + noise
        final_norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / final_norms
        #logger.info(f"[MemoryBank] Retrieved norm: {np.linalg.norm(retrieved_values.mean(axis=1)):.4f}")
        return np.mean(retrieved_values, axis=1)
    
class ShortTermMemory:
    def __init__(self, buffer_size: int, embedding_dim: int):
        """Short-Term Memory as a per-batch buffer."""
        self.buffer_size = buffer_size  
        self.embedding_dim = embedding_dim
        self.buffer = []
        self.feedback_scores = []  

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def store(self, keys, values, spike_threshold=0.1, epsilon=1e-8):
        """Store embeddings for the current batch, resetting each call."""
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)

        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)

        assert keys_np.shape[1] == self.embedding_dim
        self.buffer = list(values_np[:self.buffer_size])  

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        """Retrieve all embeddings from the current batch buffer."""
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if not self.buffer:
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        retrieved_values = np.mean(self.buffer, axis=0, keepdims=True)
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        return np.repeat(retrieved_values, queries_np.shape[0], axis=0)
    
class MidTermMemory:
    def __init__(self, buffer_size: int, embedding_dim: int, retention_steps: int):
        self.buffer_size = buffer_size  
        self.embedding_dim = embedding_dim
        self.buffer = []
        self.feedback_scores = []  
        self.step_count = 0
        self.retention_steps = retention_steps  

    def apply_spiking_attention_jnp(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to JAX array.
        """
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def store(self, keys, values, spike_threshold=0.1, epsilon=1e-8):
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
        keys = self.apply_spiking_attention_jnp(keys, spike_threshold, epsilon)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)
        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)
        assert keys_np.shape[1] == self.embedding_dim

        self.buffer.extend(values_np)
        self.buffer = self.buffer[-self.buffer_size:]
        self.step_count += 1

    def retrieve(self, queries_np, spike_threshold=0.1, epsilon=1e-8):
        queries_np = jax.block_until_ready(queries_np)
        queries_np = self.apply_spiking_attention_jnp(queries_np, spike_threshold, epsilon)
        queries_np = np.asarray(jax.device_get(queries_np), dtype=np.float32)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)
        assert queries_np.shape[1] == self.embedding_dim
        if not self.buffer or self.step_count % self.retention_steps == 0:
            self.buffer = []
            return np.zeros((queries_np.shape[0], self.embedding_dim), dtype=np.float32)
        retrieved_values = np.mean(self.buffer, axis=0, keepdims=True)
        norms = np.linalg.norm(retrieved_values, axis=-1, keepdims=True) + epsilon
        retrieved_values = retrieved_values / norms
        return np.repeat(retrieved_values, queries_np.shape[0], axis=0)