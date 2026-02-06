"""
Training Integration for Retrieval Augmentation

Provides utilities for integrating retrieval into the training loop:
- Retrieval-augmented batch preparation
- Contrastive loss for retrieval alignment
- Training hooks and callbacks
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.config.retrieval_config import RetrievalConfig
from src.modules.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class AugmentedBatch:
    """
    Batch with optional retrieval augmentation.
    
    Attributes:
        input_ids: Original input token IDs
        retrieved_texts: Retrieved document texts
        retrieved_embeddings: Embeddings of retrieved docs
        retrieval_mask: Mask for valid retrieved docs
        augmentation_applied: Whether augmentation was applied
    """
    input_ids: jnp.ndarray
    retrieved_texts: Optional[List[List[str]]] = None
    retrieved_embeddings: Optional[jnp.ndarray] = None
    retrieval_mask: Optional[jnp.ndarray] = None
    augmentation_applied: bool = False


class RetrievalContrastiveLoss:
    """
    Contrastive loss for retrieval alignment.
    
    Encourages the model to:
    - Pull query embeddings close to relevant document embeddings
    - Push query embeddings away from irrelevant documents
    
    This helps the model learn to use retrieved information effectively.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        hard_negative_weight: float = 1.0,
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for softmax (lower = sharper)
            margin: Margin for triplet loss
            hard_negative_weight: Weight for hard negatives
        """
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
    
    def __call__(
        self,
        query_embeddings: jnp.ndarray,
        positive_embeddings: jnp.ndarray,
        negative_embeddings: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute contrastive loss.
        
        Args:
            query_embeddings: Query embeddings [batch, d_model]
            positive_embeddings: Positive (relevant) doc embeddings [batch, d_model]
            negative_embeddings: Negative doc embeddings [batch, num_neg, d_model]
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        metrics = {}
        
        # Normalize embeddings
        query_norm = query_embeddings / (jnp.linalg.norm(query_embeddings, axis=-1, keepdims=True) + 1e-8)
        pos_norm = positive_embeddings / (jnp.linalg.norm(positive_embeddings, axis=-1, keepdims=True) + 1e-8)
        
        # Positive similarity
        pos_sim = jnp.sum(query_norm * pos_norm, axis=-1)  # [batch]
        metrics["positive_similarity"] = jnp.mean(pos_sim)
        
        if negative_embeddings is not None:
            # Negative similarities
            neg_norm = negative_embeddings / (jnp.linalg.norm(negative_embeddings, axis=-1, keepdims=True) + 1e-8)
            neg_sim = jnp.sum(query_norm[:, None, :] * neg_norm, axis=-1)  # [batch, num_neg]
            
            metrics["negative_similarity"] = jnp.mean(neg_sim)
            
            # InfoNCE loss (contrastive)
            # Logits: positive vs all negatives
            all_sim = jnp.concatenate([pos_sim[:, None], neg_sim], axis=-1)  # [batch, 1 + num_neg]
            # Target: positive is at index 0
            loss = -jax.nn.log_softmax(all_sim / self.temperature, axis=-1)[:, 0]
            loss = jnp.mean(loss)
            
            # Also add triplet loss for hard negatives
            hardest_neg = jnp.max(neg_sim, axis=-1)  # [batch]
            triplet_loss = jax.nn.relu(hardest_neg - pos_sim + self.margin)
            triplet_loss = jnp.mean(triplet_loss) * self.hard_negative_weight
            
            loss = loss + triplet_loss
            metrics["triplet_loss"] = triplet_loss
            
        else:
            # No negatives - use in-batch negatives
            # All other samples in batch are negatives
            sim_matrix = jnp.matmul(query_norm, pos_norm.T)  # [batch, batch]
            # Diagonal elements are positives (label = sample index)
            loss = -jax.nn.log_softmax(sim_matrix / self.temperature, axis=-1)
            loss = jnp.mean(jnp.diag(loss))
            
        metrics["contrastive_loss"] = loss
        
        return loss, metrics


class RetrievalAugmentedTraining:
    """
    Training utilities for retrieval augmentation.
    
    Handles:
    - Probabilistic augmentation (not every batch)
    - Retrieval during training
    - Loss computation with retrieval components
    """
    
    def __init__(
        self,
        config: RetrievalConfig,
        retriever: Optional[HybridRetriever] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        """
        Initialize training integration.
        
        Args:
            config: Retrieval configuration
            retriever: HybridRetriever instance
            embedding_fn: Function to embed text
        """
        self.config = config
        self.retriever = retriever
        self.embedding_fn = embedding_fn
        
        # Contrastive loss (if enabled)
        self.contrastive_loss = None
        if config.use_contrastive_loss:
            self.contrastive_loss = RetrievalContrastiveLoss()
            
        # Stats tracking
        self.augmentation_count = 0
        self.total_batches = 0
        
    def set_retriever(self, retriever: HybridRetriever) -> None:
        """Set or update the retriever."""
        self.retriever = retriever
        logger.info(f"Retriever configured with {retriever.num_documents} documents")
        
    def set_embedding_fn(self, embedding_fn: Callable) -> None:
        """Set or update the embedding function."""
        self.embedding_fn = embedding_fn
        
    def should_augment(self, rng_key: jax.Array) -> bool:
        """
        Determine if this batch should be augmented.
        
        Uses the configured augmentation probability.
        """
        if not self.config.enabled:
            return False
            
        if self.retriever is None or self.retriever.num_documents == 0:
            return False
            
        # Random decision based on probability
        uniform = jax.random.uniform(rng_key)
        return float(uniform) < self.config.augmentation_probability
    
    def prepare_augmented_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
    ) -> AugmentedBatch:
        """
        Prepare a batch with optional retrieval augmentation.
        
        Args:
            batch: Original batch dict with 'input_ids'
            rng_key: JAX random key
            
        Returns:
            AugmentedBatch with retrieval information
        """
        self.total_batches += 1
        
        input_ids = batch["input_ids"]
        
        if not self.should_augment(rng_key):
            return AugmentedBatch(
                input_ids=input_ids,
                augmentation_applied=False,
            )
            
        self.augmentation_count += 1
        
        # Get query representations
        # In practice, you'd use the model's embeddings
        # Here we use a simple approach: first N tokens as query
        batch_size = input_ids.shape[0]
        
        # Retrieve for each sample in batch
        retrieved_texts = []
        retrieved_embeddings = []
        
        for i in range(batch_size):
            # Create query from input
            # This is simplified - in practice you'd decode and use actual text
            query_text = f"query_{i}"  # Placeholder
            
            # Get query embedding if we have embedding function
            query_embedding = None
            if self.embedding_fn is not None:
                # This would be called outside JIT
                pass
                
            # Retrieve documents
            results = self.retriever.search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=self.config.top_k,
            )
            
            texts = [r.text for r in results]
            retrieved_texts.append(texts)
            
            # Get embeddings for retrieved docs
            if self.embedding_fn is not None and texts:
                embeds = self.embedding_fn(texts)
                retrieved_embeddings.append(embeds)
                
        # Pad/stack embeddings
        retrieved_emb_array = None
        retrieval_mask = None
        
        if retrieved_embeddings:
            # Pad to same length
            max_retrieved = max(len(e) for e in retrieved_embeddings)
            padded = []
            masks = []
            
            for embeds in retrieved_embeddings:
                orig_len = len(embeds)
                if orig_len < max_retrieved:
                    pad_shape = (max_retrieved - orig_len, embeds.shape[-1])
                    embeds = np.concatenate([embeds, np.zeros(pad_shape)], axis=0)
                    mask = np.array([True] * orig_len + [False] * (max_retrieved - orig_len))
                else:
                    mask = np.ones(max_retrieved, dtype=bool)
                padded.append(embeds)
                masks.append(mask)
                
            retrieved_emb_array = jnp.array(np.stack(padded))
            retrieval_mask = jnp.array(np.stack(masks))
            
        return AugmentedBatch(
            input_ids=input_ids,
            retrieved_texts=retrieved_texts,
            retrieved_embeddings=retrieved_emb_array,
            retrieval_mask=retrieval_mask,
            augmentation_applied=True,
        )
    
    def compute_retrieval_loss(
        self,
        query_embeddings: jnp.ndarray,
        augmented_batch: AugmentedBatch,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute retrieval-related loss components.
        
        Args:
            query_embeddings: Model's query embeddings [batch, d_model]
            augmented_batch: Augmented batch with retrieval info
            
        Returns:
            Tuple of (loss, metrics)
        """
        metrics = {}
        loss = jnp.array(0.0)
        
        if not augmented_batch.augmentation_applied:
            return loss, metrics
            
        if not self.config.use_contrastive_loss:
            return loss, metrics
            
        if augmented_batch.retrieved_embeddings is None:
            return loss, metrics
            
        # Use first retrieved doc as positive
        positive_embeddings = augmented_batch.retrieved_embeddings[:, 0, :]
        
        # Use rest as negatives
        negative_embeddings = augmented_batch.retrieved_embeddings[:, 1:, :]
        
        # Compute contrastive loss
        contrastive_loss, contrastive_metrics = self.contrastive_loss(
            query_embeddings=query_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
        )
        
        loss = contrastive_loss * self.config.contrastive_weight
        metrics.update(contrastive_metrics)
        metrics["retrieval_loss"] = loss
        
        return loss, metrics
    
    @property
    def augmentation_rate(self) -> float:
        """Actual augmentation rate during training."""
        if self.total_batches == 0:
            return 0.0
        return self.augmentation_count / self.total_batches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_batches": self.total_batches,
            "augmented_batches": self.augmentation_count,
            "augmentation_rate": self.augmentation_rate,
            "config_probability": self.config.augmentation_probability,
            "retriever_docs": self.retriever.num_documents if self.retriever else 0,
        }


def create_retrieval_train_step(
    base_train_step: Callable,
    retrieval_training: RetrievalAugmentedTraining,
) -> Callable:
    """
    Wrap a training step with retrieval augmentation.
    
    Args:
        base_train_step: Original training step function
        retrieval_training: RetrievalAugmentedTraining instance
        
    Returns:
        Augmented training step function
    """
    def augmented_train_step(
        params: Dict,
        opt_state: Any,
        batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
    ) -> Tuple[Dict, Any, Dict[str, Any]]:
        """
        Training step with optional retrieval augmentation.
        """
        # Split RNG
        rng_key, augment_key = jax.random.split(rng_key)
        
        # Prepare augmented batch (outside JIT)
        augmented_batch = retrieval_training.prepare_augmented_batch(
            batch, 
            augment_key
        )
        
        # Add retrieved context to batch
        if augmented_batch.augmentation_applied:
            batch["retrieved_context"] = augmented_batch.retrieved_embeddings
            batch["retrieval_mask"] = augmented_batch.retrieval_mask
            
        # Run base training step
        new_params, new_opt_state, metrics = base_train_step(
            params, opt_state, batch, rng_key
        )
        
        # Add retrieval stats to metrics
        metrics["retrieval_augmented"] = augmented_batch.augmentation_applied
        
        return new_params, new_opt_state, metrics
    
    return augmented_train_step


# =============================================================================
# Integration with AGIConfig
# =============================================================================

def add_retrieval_config_to_agi(agi_config: Any) -> None:
    """
    Add retrieval configuration to existing AGIConfig.
    
    This extends AGIConfig with retrieval parameters without modifying
    the original class.
    
    Args:
        agi_config: AGIConfig instance to extend
    """
    # Default retrieval settings
    defaults = {
        "retrieval_enabled": False,
        "retrieval_top_k": 5,
        "retrieval_augmentation_probability": 0.2,
        "retrieval_use_hybrid": True,
        "retrieval_sparse_weight": 0.3,
        "retrieval_dense_weight": 0.7,
        "retrieval_use_contrastive_loss": False,
        "retrieval_contrastive_weight": 0.05,
    }
    
    for key, value in defaults.items():
        if not hasattr(agi_config, key):
            setattr(agi_config, key, value)
            
    logger.info("Retrieval configuration added to AGIConfig")


def create_retrieval_config_from_agi(agi_config: Any) -> RetrievalConfig:
    """
    Create RetrievalConfig from AGIConfig parameters.
    
    Args:
        agi_config: AGIConfig instance
        
    Returns:
        RetrievalConfig instance
    """
    return RetrievalConfig(
        enabled=getattr(agi_config, "retrieval_enabled", False),
        top_k=getattr(agi_config, "retrieval_top_k", 5),
        augmentation_probability=getattr(agi_config, "retrieval_augmentation_probability", 0.2),
        use_hybrid=getattr(agi_config, "retrieval_use_hybrid", True),
        sparse_weight=getattr(agi_config, "retrieval_sparse_weight", 0.3),
        dense_weight=getattr(agi_config, "retrieval_dense_weight", 0.7),
        use_contrastive_loss=getattr(agi_config, "retrieval_use_contrastive_loss", False),
        contrastive_weight=getattr(agi_config, "retrieval_contrastive_weight", 0.05),
        embedding_dim=getattr(agi_config, "d_model", 384),
    )
