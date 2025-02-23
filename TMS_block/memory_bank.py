import faiss
import jax
import logging
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    def store(self, keys, values):
        """
        Stores key-value pairs in the memory bank.
        :param keys: Batch of mean-pooled embeddings (shape: [batch_size, embedding_dim])
        :param values: Corresponding transformer activations (shape: [batch_size, embedding_dim])
        """
        # **Ensure computation is complete before conversion**
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)

        # **Detach from JAX before passing to FAISS**
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

        self.values.extend(values_np)
        self.index.add(keys_np)

    def retrieve(self, queries_np, epsilon=1e-8):
        """
        Retrieves closest memory values.
        :param queries: Query embeddings (shape: [batch_size, embedding_dim])
        :return: Retrieved memory activations (shape: [batch_size, embedding_dim])
        """
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
        logger.info(f"[MemoryBank] Retrieved norm: {np.linalg.norm(retrieved_values.mean(axis=1)):.4f}")
        return np.mean(retrieved_values, axis=1)
    
class ShortTermMemory:
    def __init__(self, buffer_size: int, embedding_dim: int):
        """Short-Term Memory as a per-batch buffer."""
        self.buffer_size = buffer_size  
        self.embedding_dim = embedding_dim
        self.buffer = []  

    def store(self, keys, values):
        """Store embeddings for the current batch, resetting each call."""
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
        keys_np = np.asarray(jax.device_get(keys), dtype=np.float32)
        values_np = np.asarray(jax.device_get(values), dtype=np.float32)

        if keys_np.ndim == 1:
            keys_np = keys_np.reshape(1, -1)
        if values_np.ndim == 1:
            values_np = values_np.reshape(1, -1)

        assert keys_np.shape[1] == self.embedding_dim
        self.buffer = list(values_np[:self.buffer_size])  

    def retrieve(self, queries_np, epsilon=1e-8):
        """Retrieve all embeddings from the current batch buffer."""
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
        self.step_count = 0
        self.retention_steps = retention_steps  

    def store(self, keys, values):
        keys = jax.block_until_ready(keys)
        values = jax.block_until_ready(values)
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

    def retrieve(self, queries_np, epsilon=1e-8):
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