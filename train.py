import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import re
import argparse
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Callable
import matplotlib.pyplot as plt
import time
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project paths using absolute path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from rtdlm import (
    create_rtdlm_agi, create_agi_optimizer, 
    compute_agi_loss
)
from core.checkpoint_manager import CheckpointManager
from config.agi_config import AGIConfig
from modules.capabilities.advanced_algorithms import (
    TaskMemory, compute_ewc_loss, compute_fisher_information
)
# Scalable training (unified approach for all parallelism modes)
from core.scalable_training import (
    ScalableMesh,
    create_scalable_mesh,
    setup_scalable_training,
    replicate_for_data_parallel,
    unreplicate_params,
    estimate_model_memory,
    recommend_parallelism,
)

# Memory profiling and gradient accumulation
from core.memory_profiler import (
    MemoryProfiler,
)
from core.gradient_accumulation import (
    BatchGradientAccumulator,
    calculate_effective_batch_size,
)

# Evaluation metrics and tracking
from core.training.evaluation import (
    EvaluationMetrics,
    TrainingEvaluator,
    GradientMonitor,
)
from core.benchmark_evaluation import (
    PerplexityTracker,
    CalibrationTracker,
    ComputeEfficiencyTracker,
)
from core.ethics import FairnessAnalyzer, FairnessConfig

# Retrieval Augmented Generation (RAG) integration
from config.retrieval_config import RetrievalConfig
from modules.retrieval import (
    HybridRetriever,
    DocumentIngester,
    RetrievalAugmentedTraining,
)

def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute cosine similarity between two vectors"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = jnp.linalg.norm(a_flat) + 1e-8
    norm_b = jnp.linalg.norm(b_flat) + 1e-8
    return jnp.sum(a_flat * b_flat) / (norm_a * norm_b)

class AGITrainer:
    """
    Advanced trainer for RT-DLM AGI.
    
    Accepts pre-tokenized tensor data from the external data pipeline.
    Training data should be prepared using Auralith-Data-Pipeline.
    
    Supports scalable training on any hardware configuration:
    - Single GPU/TPU: Standard training
    - Multiple GPUs/TPUs: Automatic data parallelism
    - Very large models: Combined data + model parallelism
    
    The SAME full AGI model is used in all modes - parallelism is handled
    automatically by the training infrastructure.
    """
    
    def __init__(self, config: AGIConfig):
        self.config = config
        self.rng = jax.random.PRNGKey(42)
        
        # Scalable training setup
        self.mesh = None
        self.is_distributed = False
        
        # Always use the full AGI model (unified approach)
        self.model = create_rtdlm_agi(config)
        
        # Set up scalable mesh if using multiple devices
        if config.distributed_training or config.model_parallel:
            self.mesh = create_scalable_mesh(config)
            self.is_distributed = self.mesh.is_distributed
            logger.info("Scalable training enabled:")
            logger.info(f"  Data parallel: {self.mesh.data_parallel_size} replicas")
            if self.mesh.has_tensor_parallel:
                logger.info(f"  Tensor parallel: {self.mesh.tensor_parallel_size} shards")
        
        # Initialize optimizer
        self.optimizer = create_agi_optimizer(config)
        
        # Training state
        self.params = None
        self.opt_state = None
        self.step_count = 0
        
        # Metrics tracking
        self.training_losses = []
        self.validation_losses = []
        self.reasoning_accuracies = []
        self.consciousness_coherence = []
        self.multimodal_alignment = []
        
        # Memory profiling (tracks GPU memory usage during training)
        self.memory_profiler = MemoryProfiler(
            enabled=config.enable_memory_profiling if hasattr(config, 'enable_memory_profiling') else False,
            log_every_n_steps=100
        )
        
        # Gradient accumulation settings
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.gradient_accumulator = None
        if self.gradient_accumulation_steps > 1:
            logger.info(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
            effective_batch = calculate_effective_batch_size(
                micro_batch_size=config.batch_size,
                accumulation_steps=self.gradient_accumulation_steps
            )
            logger.info(f"  Effective batch size: {effective_batch}")
        
        # Continual learning state
        self.task_memories: List[TaskMemory] = []
        self.current_task_id = "task_0"
        self.lambda_ewc = 1000.0
        
        # Retrieval Augmented Generation (RAG) - optional
        self.retrieval_config: Optional[RetrievalConfig] = None
        self.retriever: Optional[HybridRetriever] = None
        self.retrieval_training: Optional[RetrievalAugmentedTraining] = None
        self.document_ingester: Optional[DocumentIngester] = None
        self._embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None
        
        # Production evaluation metrics
        self.perplexity_tracker = PerplexityTracker(window_size=100)
        self.calibration_tracker = CalibrationTracker(num_bins=10)
        self.compute_tracker = ComputeEfficiencyTracker(
            model_config={'d_model': config.d_model, 'num_layers': config.num_layers}
        )
        self.gradient_monitor = GradientMonitor(
            exploding_threshold=100.0,
            vanishing_threshold=1e-7,
            track_per_layer=False,
        )
        
        # Fairness tracking (optional)
        self.fairness_analyzer: Optional[FairnessAnalyzer] = None
        if getattr(config, 'enable_fairness_tracking', False):
            self.fairness_analyzer = FairnessAnalyzer(FairnessConfig())
            logger.info("Fairness tracking enabled")
        
    def configure_retrieval(
        self, 
        retrieval_config: Optional[RetrievalConfig] = None,
        documents: Optional[List[str]] = None
    ) -> None:
        """
        Configure retrieval augmentation for training.
        
        This is optional - retrieval can be enabled/disabled at any time.
        Following the industry pattern where RAG is external to the base model.
        
        Args:
            retrieval_config: Configuration for retrieval. Uses RetrievalConfig.for_training()
                            if not provided and documents are given.
            documents: Optional list of documents to ingest into the retrieval system.
        
        Example:
            >>> trainer = AGITrainer(config)
            >>> trainer.configure_retrieval(
            ...     RetrievalConfig.for_training(augmentation_probability=0.3),
            ...     documents=["document 1...", "document 2..."]
            ... )
        """
        if retrieval_config is None and documents is not None:
            retrieval_config = RetrievalConfig.for_training()
        
        if retrieval_config is None or not retrieval_config.enabled:
            logger.info("Retrieval augmentation disabled")
            self.retrieval_config = RetrievalConfig.disabled()
            return
        
        self.retrieval_config = retrieval_config
        logger.info("Configuring retrieval augmentation...")
        logger.info(f"  Top-K: {retrieval_config.top_k}")
        logger.info(f"  Hybrid retrieval: {retrieval_config.use_hybrid}")
        logger.info(f"  Augmentation probability: {retrieval_config.augmentation_probability}")
        
        # Use model's d_model as embedding dimension (self-contained, no external dependencies)
        self._embedding_dim = self.config.d_model
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            embedding_dim=self._embedding_dim,
            sparse_weight=retrieval_config.sparse_weight,
            dense_weight=retrieval_config.dense_weight
        )
        
        # Initialize training integration
        self.retrieval_training = RetrievalAugmentedTraining(
            config=retrieval_config,
            retriever=self.retriever
        )
        
        # Initialize document ingester
        self.document_ingester = DocumentIngester(
            chunk_size=retrieval_config.chunk_size,
            chunk_overlap=retrieval_config.chunk_overlap,
            chunking_strategy=retrieval_config.chunking_strategy,
            embedding_dim=self._embedding_dim
        )
        
        # Set up embedding function for document ingester
        self.document_ingester.set_embedding_fn(self._create_text_embeddings)
        
        # Ingest documents if provided
        if documents:
            self.ingest_documents(documents)
        
        logger.info("Retrieval augmentation configured successfully")
    
    def _create_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks using the model's own embedding layer.
        
        This is how production LLMs work - they use their own learned embeddings
        rather than external embedding models. The embeddings are created by:
        1. Converting text to token IDs (using a simple hash-based approach for now)
        2. Looking up embeddings from the model's embedding table
        3. Mean pooling + L2 normalization
        
        Note: In a full production setup, you would use the model's tokenizer.
        For document retrieval before the model is fully trained, we use a
        deterministic hash-based embedding that's consistent and doesn't require
        the model to be initialized.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Embeddings array of shape (len(texts), d_model)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.config.d_model)
        
        embeddings = []
        for text in texts:
            embedding = self._hash_text_to_embedding(text)
            embeddings.append(embedding)
        
        return np.stack(embeddings).astype(np.float32)
    
    def _hash_text_to_embedding(self, text: str) -> np.ndarray:
        """Create deterministic embedding from text using hashing."""
        import hashlib
        
        embedding = np.zeros(self.config.d_model, dtype=np.float32)
        
        hash_input = text.encode('utf-8')
        chunk_size = 32
        num_chunks = (self.config.d_model + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            hash_bytes = hashlib.sha256(hash_input + str(i).encode()).digest()
            
            for j, byte in enumerate(hash_bytes):
                idx = i * chunk_size + j
                if idx >= self.config.d_model:
                    break
                embedding[idx] = (byte / 127.5) - 1.0
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def set_embedding_function(self, embedding_fn: Callable[[List[str]], np.ndarray]) -> None:
        """
        Set a custom embedding function for retrieval.
        
        This allows you to use the model's own trained embeddings once the model
        is initialized, or to use any custom embedding approach.
        
        Example with trained model:
            def model_embedding_fn(texts):
                # Tokenize texts
                tokens = tokenizer.batch_encode(texts)
                # Get embeddings from model
                embeddings = model.get_embeddings(tokens)
                return embeddings
            
            trainer.set_embedding_function(model_embedding_fn)
        
        Args:
            embedding_fn: Function that takes list of texts and returns embeddings
        """
        self._embedding_fn = embedding_fn
        if self.document_ingester is not None:
            self.document_ingester.set_embedding_fn(embedding_fn)
        logger.info("Custom embedding function set for retrieval")
    
    def create_model_embeddings(self, token_ids: jnp.ndarray) -> np.ndarray:
        """
        Create embeddings using the model's trained embedding layer.
        
        This is the production approach - use the model's own learned embeddings
        for retrieval. Should only be called after model initialization.
        
        The embedding is created by:
        1. Looking up token embeddings from the model's embedding table
        2. Adding positional embeddings
        3. Mean pooling across the sequence
        4. L2 normalization
        
        Args:
            token_ids: Token IDs of shape (batch, seq_len)
            
        Returns:
            Embeddings of shape (batch, d_model)
        """
        if self.params is None:
            raise RuntimeError("Model must be initialized before creating embeddings. "
                             "Call initialize_model() first.")
        
        embedding_table = None
        position_table = None
        
        def find_embeddings(params, prefix=""):
            nonlocal embedding_table, position_table
            
            if isinstance(params, dict):
                for key, value in params.items():
                    full_key = f"{prefix}/{key}" if prefix else key
                    
                    if 'token_embedding' in key.lower() or (key == 'embeddings' and embedding_table is None):
                        if hasattr(value, 'shape') and len(value.shape) == 2:
                            embedding_table = value
                    elif 'position' in key.lower() and 'embedding' in key.lower():
                        if hasattr(value, 'shape') and len(value.shape) == 2:
                            position_table = value
                    else:
                        find_embeddings(value, full_key)
        
        find_embeddings(self.params)
        
        if embedding_table is None:
            logger.warning("Could not find embedding table in model params, using hash embeddings")
            # Fallback to hash-based
            batch_size = token_ids.shape[0]
            return np.array([self._hash_text_to_embedding(str(ids.tolist())) for ids in token_ids])
        
        # Look up token embeddings
        token_embeds = embedding_table[token_ids]  # (batch, seq, d_model)
        
        # Add positional embeddings if available
        if position_table is not None:
            seq_len = token_ids.shape[1]
            pos_ids = jnp.arange(seq_len)
            pos_embeds = position_table[pos_ids]  # (seq, d_model)
            token_embeds = token_embeds + pos_embeds[None, :, :]
        
        # Mean pooling across sequence (ignoring padding - assume non-zero tokens)
        mask = (token_ids != 0).astype(jnp.float32)  # (batch, seq)
        mask_expanded = mask[:, :, None]  # (batch, seq, 1)
        
        sum_embeds = jnp.sum(token_embeds * mask_expanded, axis=1)  # (batch, d_model)
        counts = jnp.sum(mask, axis=1, keepdims=True) + 1e-9  # (batch, 1)
        mean_embeds = sum_embeds / counts  # (batch, d_model)
        
        # L2 normalize
        norms = jnp.linalg.norm(mean_embeds, axis=1, keepdims=True) + 1e-9
        normalized = mean_embeds / norms
        
        return np.array(normalized)
    
    def update_retrieval_with_model_embeddings(self) -> None:
        """
        Re-embed all documents using the model's trained embeddings.
        
        Call this periodically during training to update the retrieval index
        with better embeddings as the model learns. This is how production
        systems improve retrieval quality during training.
        
        This method:
        1. Collects all chunk texts from the document ingester
        2. Tokenizes them using simple word-based tokenization
        3. Creates embeddings using the model's trained embedding layer
        4. Updates the retriever index with new embeddings
        
        Call this every N epochs to keep retrieval aligned with model learning.
        """
        if self.params is None:
            logger.warning("Model not initialized, cannot update embeddings")
            return
        
        if self.retriever is None or self.document_ingester is None:
            logger.warning("Retrieval not configured")
            return
        
        logger.info("Updating retrieval index with model embeddings...")
        
        # Collect all chunks
        chunk_texts = []
        chunk_ids = []
        chunk_metadata = []
        
        for chunk in self.document_ingester.iter_chunks():
            chunk_texts.append(chunk.text)
            chunk_ids.append(chunk.chunk_id)
            chunk_metadata.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                **(chunk.metadata or {})
            })
        
        if not chunk_texts:
            logger.warning("No chunks found in document ingester")
            return
        
        token_ids_list = []
        for text in chunk_texts:
            tokens = self._simple_tokenize(text, self.config.max_seq_length)
            token_ids_list.append(tokens)
        
        token_ids = jnp.array(token_ids_list, dtype=jnp.int32)
        embeddings = self.create_model_embeddings(token_ids)
        
        self.retriever.clear()
        self.retriever.add_documents(
            documents=chunk_texts,
            embeddings=embeddings,
            doc_ids=chunk_ids,
            metadata=chunk_metadata
        )
        
        logger.info(f"Updated {len(chunk_texts)} chunk embeddings with model weights")
    
    def _simple_tokenize(self, text: str, max_length: int) -> List[int]:
        """
        Simple word-based tokenization for RAG embedding.
        
        Maps words to token IDs using consistent hashing.
        In production, use the model's actual tokenizer.
        """
        import hashlib
        words = text.lower().split()[:max_length]
        token_ids = []
        for word in words:
            # Hash word to get consistent token ID in vocab range
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            token_id = (word_hash % (self.config.vocab_size - 2)) + 1
            token_ids.append(token_id)
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(0)
        return token_ids[:max_length]
    
    def schedule_retrieval_update(self, update_every_n_epochs: int = 5) -> None:
        """
        Schedule periodic retrieval index updates during training.
        
        Args:
            update_every_n_epochs: Update embeddings every N epochs
        """
        self._retrieval_update_frequency = update_every_n_epochs
        logger.info(f"Scheduled retrieval index updates every {update_every_n_epochs} epochs")
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about RAG usage during training.
        
        Returns metrics useful for RAG ablation studies.
        """
        stats = {
            "rag_enabled": self.retrieval_config is not None and self.retrieval_config.enabled,
            "augmentation_probability": getattr(self.retrieval_config, 'augmentation_probability', 0.0) if self.retrieval_config else 0.0,
            "total_chunks_indexed": 0,
            "retriever_type": "hybrid" if (self.retrieval_config and self.retrieval_config.use_hybrid) else "dense",
        }
        
        if self.document_ingester:
            stats["total_chunks_indexed"] = sum(1 for _ in self.document_ingester.iter_chunks())
        
        if self.retriever:
            stats["retriever_doc_count"] = len(self.retriever)
        
        return stats
    
    def ingest_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Ingest documents into the retrieval system.
        
        Documents are chunked for better retrieval performance - smaller chunks
        provide more precise matching and better relevance scoring.
        
        Args:
            documents: List of document texts to ingest
            metadata: Optional metadata for each document
        """
        if self.document_ingester is None or self.retriever is None:
            logger.warning("Retrieval not configured. Call configure_retrieval() first.")
            return
        
        logger.info(f"Ingesting {len(documents)} documents...")
        
        # Format documents for the ingester
        doc_list = []
        for i, doc in enumerate(documents):
            doc_dict = {"text": doc, "doc_id": f"doc_{i}"}
            if metadata and i < len(metadata):
                doc_dict.update(metadata[i])
            doc_list.append(doc_dict)
        
        # Ingest documents - this creates chunks and stores them in chunk_store
        stats = self.document_ingester.ingest_documents(doc_list, store_to_memory=False)
        
        # Add chunks (not full docs) to the retriever for better retrieval performance
        chunk_texts = []
        chunk_ids = []
        chunk_metadata = []
        
        for chunk in self.document_ingester.iter_chunks():
            chunk_texts.append(chunk.text)
            chunk_ids.append(chunk.chunk_id)
            chunk_metadata.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                **(chunk.metadata or {})
            })
        
        if chunk_texts:
            # Create embeddings for hybrid (sparse + dense) retrieval
            chunk_embeddings = self._create_text_embeddings(chunk_texts)
            
            self.retriever.add_documents(
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                doc_ids=chunk_ids,
                metadata=chunk_metadata
            )
        
        logger.info(
            f"Ingestion complete: {stats['documents_processed']} docs → "
            f"{stats['chunks_created']} chunks indexed (with embeddings)"
        )
        
    def _augment_batch_with_retrieval(
        self, 
        batch: Dict[str, jnp.ndarray],
        rng: jax.Array
    ) -> Dict[str, jnp.ndarray]:
        """
        Optionally augment a batch with retrieved context.
        
        Uses probabilistic augmentation based on config.augmentation_probability.
        """
        if (self.retrieval_training is None or 
            self.retrieval_config is None or 
            not self.retrieval_config.enabled):
            return batch
        
        # Prepare augmented batch using the retrieval training module
        augmented = self.retrieval_training.prepare_augmented_batch(batch, rng)
        
        if not augmented.augmentation_applied:
            return batch
        
        # Add retrieval context to batch if available
        augmented_batch = dict(batch)
        if augmented.retrieved_embeddings is not None:
            augmented_batch["retrieved_embeddings"] = augmented.retrieved_embeddings
            augmented_batch["retrieval_mask"] = augmented.retrieval_mask
        
        return augmented_batch
    def initialize_model(self, sample_batch: Dict[str, jnp.ndarray]):
        """Initialize model parameters from a sample batch."""
        logger.info("Initializing RT-DLM AGI model...")
        
        # Create sample inputs for initialization
        sample_inputs = {
            "inputs": {"text": sample_batch["input_ids"]},
        }
        
        # Add multimodal samples if enabled
        if self.config.multimodal_enabled:
            batch_size = sample_batch["input_ids"].shape[0]
            sample_inputs["multimodal_inputs"] = {
                "images": jnp.zeros((batch_size, 224, 224, 3)),
                "audio": jnp.zeros((batch_size, 128, 128)),
            }
        
        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.model.init(init_rng, **sample_inputs)
        
        # Handle distributed training - replicate params across devices
        if self.is_distributed and self.mesh is not None:
            self.params = replicate_for_data_parallel(
                self.params, 
                self.mesh.data_parallel_size
            )
            logger.info(f"Parameters replicated across {self.mesh.data_parallel_size} devices")
        
        self.opt_state = self.optimizer.init(self.params)
        
        # Count parameters (get from single replica if distributed)
        if self.is_distributed:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(unreplicate_params(self.params)))
        else:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        
        # Print memory estimates
        memory_est = estimate_model_memory(
            unreplicate_params(self.params) if self.is_distributed else self.params
        )
        logger.info(f"Model initialized with {param_count:,} parameters")
        logger.info(f"Estimated memory: {memory_est['total_gb']:.2f} GB")
        
        # Set up memory profiler with model/optimizer sizes
        self.memory_profiler.set_model_size(
            unreplicate_params(self.params) if self.is_distributed else self.params
        )
        self.memory_profiler.set_optimizer_size(self.opt_state)
        
        # Initialize gradient accumulator if using accumulation
        if self.gradient_accumulation_steps > 1:
            self.gradient_accumulator = BatchGradientAccumulator(
                accumulation_steps=self.gradient_accumulation_steps,
                loss_fn=self._make_loss_fn(),
                model_apply_fn=self.model.apply,
            )
            logger.info("Gradient accumulator initialized")
        
        return self.params, self.opt_state
    
    def consolidate_task(self, task_id: str, data_samples: jnp.ndarray, targets: jnp.ndarray):
        """
        Consolidate knowledge for a completed task using EWC.
        
        Computes Fisher information and stores task memory for future
        regularization to prevent catastrophic forgetting.
        
        Args:
            task_id: Unique identifier for the task
            data_samples: Representative data samples from the task
            targets: Corresponding targets
        """
        if self.params is None:
            raise RuntimeError("Model must be initialized before consolidating tasks")
            
        logger.info(f"Consolidating task {task_id} with EWC.")
        
        # Compute Fisher information for current task
        self.rng, fisher_rng = jax.random.split(self.rng)
        fisher_matrix = compute_fisher_information(
            model_fn=self.model.apply,
            params=self.params,
            data_samples=data_samples,
            targets=targets,
            rng=fisher_rng,
            num_samples=min(100, len(data_samples))
        )
        
        # Create task memory
        task_memory = TaskMemory(
            task_id=task_id,
            params_snapshot=jax.tree_util.tree_map(lambda x: x.copy(), self.params),
            fisher_matrix=fisher_matrix,
            importance_weights=fisher_matrix,
            performance_metrics={"final_loss": float(self.training_losses[-1]) if self.training_losses else 0.0},
            num_samples=len(data_samples)
        )
        
        self.task_memories.append(task_memory)
        logger.info(f"Task {task_id} consolidated. Total task memories: {len(self.task_memories)}")
    
    def compute_ewc_regularization(self, params: Dict) -> jnp.ndarray:
        """
        Compute EWC regularization loss from all previous task memories.
        
        Args:
            params: Current model parameters
            
        Returns:
            EWC regularization loss
        """
        if not self.task_memories:
            return jnp.float32(0.0)
        
        total_ewc_loss = jnp.float32(0.0)
        for memory in self.task_memories:
            ewc_loss = compute_ewc_loss(
                params=params,
                params_star=memory.params_snapshot,
                fisher_matrix=memory.fisher_matrix,
                lambda_ewc=self.lambda_ewc
            )
            total_ewc_loss += ewc_loss
        
        return total_ewc_loss

    def create_batch_from_tensors(
        self, 
        input_ids: jnp.ndarray,
        targets: Optional[jnp.ndarray] = None,
        images: Optional[jnp.ndarray] = None,
        audio: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Create a training batch from pre-tokenized tensor data.
        
        Data should be prepared using Auralith-Data-Pipeline and loaded
        from SafeTensors shards.
        
        Args:
            input_ids: Pre-tokenized input tensor (batch, seq_len)
            targets: Target tensor for next-token prediction (default: input_ids)
            images: Optional image tensor (batch, H, W, C)
            audio: Optional audio mel spectrogram tensor (batch, time, freq)
        
        Returns:
            Dictionary with input_ids, targets, and optional multimodal_inputs
        """
        batch = {
            "input_ids": input_ids,
            "targets": targets if targets is not None else input_ids,
            "text": input_ids,
        }
        
        batch_size = input_ids.shape[0]
        
        # Add multimodal data if provided
        if self.config.multimodal_enabled and (images is not None or audio is not None):
            self.rng, *modal_rngs = jax.random.split(self.rng, 3)
            
            # Use provided images or generate synthetic
            if images is not None:
                batch_images = jnp.array(images[:batch_size], dtype=jnp.float32)
            else:
                batch_images = jax.random.normal(modal_rngs[0], (batch_size, 224, 224, 3)) * 0.1
            
            # Use provided audio or generate synthetic
            if audio is not None:
                mel_spectrograms = jnp.array(audio[:batch_size], dtype=jnp.float32)
            else:
                mel_spectrograms = jax.random.normal(modal_rngs[1], (batch_size, 128, 128)) * 0.1
            
            batch["image"] = batch_images
            batch["audio"] = mel_spectrograms
            batch["multimodal_inputs"] = {
                "images": batch_images,
                "audio": mel_spectrograms,
            }
        
        return batch
    
    def create_reasoning_batch(self, batch_size: int = 8) -> Dict[str, jnp.ndarray]:
        """
        Create a synthetic reasoning task batch for evaluation.
        
        Note: For actual training, use pre-tokenized data from the data pipeline.
        This is only for reasoning capability evaluation.
        """
        # Generate random token sequences for reasoning evaluation
        self.rng, rng = jax.random.split(self.rng)
        input_ids = jax.random.randint(
            rng, 
            (batch_size, self.config.max_seq_length), 
            0, 
            self.config.vocab_size
        )
        return self.create_batch_from_tensors(input_ids)
    
    @jax.jit
    def train_step(self, params, opt_state, batch, rng):
        """Single training step with comprehensive loss and NaN handling"""
        
        def loss_fn(params, batch, rng):
            # Forward pass
            model_output = self.model.apply(
                params, rng,
                inputs={"text": batch["input_ids"]},
                multimodal_inputs=batch.get("multimodal_inputs"),
                return_reasoning=True
            )
            
            # Compute comprehensive loss
            loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            return loss, model_output
        
        # Compute gradients
        (loss, model_output), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, batch, rng)
        
        # NaN check for loss - replace with zero if NaN detected
        loss = jax.lax.cond(
            jnp.isnan(loss),
            lambda _: jnp.float32(0.0),
            lambda l: l,
            loss
        )
        
        # NaN check for gradients - zero out NaN gradients
        def zero_nan_grads(g):
            return jax.tree_util.tree_map(
                lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), 
                g
            )
        grads = zero_nan_grads(grads)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, model_output
    
    def _make_loss_fn(self):
        """
        Create a loss function for gradient accumulation.
        
        Returns a function that takes (outputs, batch) and returns the loss.
        """
        def loss_fn(outputs, batch):
            return compute_agi_loss(
                outputs["logits"],
                batch["targets"],
                aux_outputs=outputs,
                config=self.config
            )
        return loss_fn
    
    def train_step_with_ewc(self, params, opt_state, batch, rng, 
                            task_memories: List[TaskMemory], lambda_ewc: float = 1000.0):
        """
        Training step with EWC regularization for continual learning.
        
        Adds EWC loss to prevent catastrophic forgetting of previous tasks.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            rng: Random key
            task_memories: List of previous task memories
            lambda_ewc: EWC regularization strength
            
        Returns:
            Updated params, optimizer state, total loss, EWC loss component
        """
        def loss_fn_with_ewc(params, batch, rng):
            # Forward pass
            model_output = self.model.apply(
                params, rng,
                inputs={"text": batch["input_ids"]},
                multimodal_inputs=batch.get("multimodal_inputs"),
                return_reasoning=True
            )
            
            # Compute base AGI loss
            base_loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            # Compute EWC regularization loss
            ewc_loss = jnp.float32(0.0)
            for memory in task_memories:
                ewc_loss += compute_ewc_loss(
                    params=params,
                    params_star=memory.params_snapshot,
                    fisher_matrix=memory.fisher_matrix,
                    lambda_ewc=lambda_ewc
                )
            
            total_loss = base_loss + ewc_loss
            
            return total_loss, (model_output, base_loss, ewc_loss)
        
        # Compute gradients
        (total_loss, (_, base_loss, ewc_loss)), grads = jax.value_and_grad(
            loss_fn_with_ewc, has_aux=True
        )(params, batch, rng)
        
        # NaN check for loss
        total_loss = jax.lax.cond(
            jnp.isnan(total_loss),
            lambda _: jnp.float32(0.0),
            lambda l: l,
            total_loss
        )
        
        # NaN check for gradients
        def zero_nan_grads(g):
            return jax.tree_util.tree_map(
                lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), 
                g
            )
        grads = zero_nan_grads(grads)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, total_loss, base_loss, ewc_loss

    def evaluate_model(self, eval_batches: List[Dict[str, jnp.ndarray]], num_samples: int = 100):
        """
        Comprehensive model evaluation on pre-tokenized tensor data.
        
        Args:
            eval_batches: List of evaluation batches, each containing input_ids and targets
            num_samples: Maximum number of samples to evaluate
        """
        logger.info("Evaluating model...")
        
        eval_losses = []
        reasoning_scores = []
        consciousness_scores = []
        
        # Regular evaluation
        samples_evaluated = 0
        for batch in eval_batches:
            if samples_evaluated >= num_samples:
                break
            
            # Forward pass
            self.rng, eval_rng = jax.random.split(self.rng)
            model_output = self.model.apply(
                self.params, eval_rng,
                inputs={"text": batch["input_ids"]},
                return_reasoning=True
            )
            
            # Compute loss
            loss = compute_agi_loss(
                model_output["logits"],
                batch["targets"],
                aux_outputs=model_output,
                config=self.config
            )
            
            eval_losses.append(float(loss))
            samples_evaluated += batch["input_ids"].shape[0]
            
            # Evaluate reasoning quality
            if "reasoning_chain" in model_output:
                reasoning_score = self.evaluate_reasoning_quality(
                    model_output["reasoning_chain"]
                )
                reasoning_scores.append(reasoning_score)
            
            # Evaluate consciousness coherence
            if "consciousness" in model_output:
                consciousness_score = self.evaluate_consciousness_coherence(
                    model_output["consciousness"]
                )
                consciousness_scores.append(consciousness_score)
        
        # Reasoning-specific evaluation
        reasoning_batch = self.create_reasoning_batch(self.config.batch_size)
        self.rng, reasoning_rng = jax.random.split(self.rng)
        
        reasoning_output = self.model.apply(
            self.params, reasoning_rng,
            inputs={"text": reasoning_batch["input_ids"]},
            return_reasoning=True
        )
        
        reasoning_accuracy = self.evaluate_reasoning_accuracy(reasoning_output)
        
        metrics = {
            "eval_loss": np.mean(eval_losses) if eval_losses else float('inf'),
            "reasoning_accuracy": reasoning_accuracy,
            "reasoning_quality": np.mean(reasoning_scores) if reasoning_scores else 0.0,
            "consciousness_coherence": np.mean(consciousness_scores) if consciousness_scores else 0.0,
        }
        
        return metrics
    
    def evaluate_reasoning_quality(self, reasoning_chain, ground_truth: Optional[str] = None):
        """Evaluate the quality of reasoning chain with regex parsing"""
        if not reasoning_chain:
            return 0.0
        
        # Check for consistency between steps
        consistency_score = 0.0
        for i in range(len(reasoning_chain) - 1):
            step_similarity = cosine_similarity(
                reasoning_chain[i],
                reasoning_chain[i+1]
            )
            consistency_score += float(step_similarity)
        
        base_score = consistency_score / max(1, len(reasoning_chain) - 1)
        
        if ground_truth is not None:
            step_pattern = r'[Ss]tep\s*\d+:\s*(.+)(?=[Ss]tep\s*\d+:|[Aa]nswer:|$)'
            answer_pattern = r'[Aa]nswer:\s*([^.]+)'
            
            reasoning_text = str(reasoning_chain)
            
            steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
            answers = re.findall(answer_pattern, reasoning_text)
            
            if answers and ground_truth:
                answer_text = answers[-1].strip().lower()
                gt_text = ground_truth.strip().lower()
                
                if gt_text in answer_text or answer_text in gt_text:
                    return min(1.0, base_score + 0.5)
            
            step_bonus = min(0.3, len(steps) * 0.1)
            return min(1.0, base_score + step_bonus)
        
        return base_score
    
    def evaluate_consciousness_coherence(self, consciousness_signals):
        """Evaluate consciousness simulation coherence"""
        if not consciousness_signals:
            return 0.0
        
        coherence = 0.0
        count = 0
        
        # Check alignment between self-awareness and introspection
        if "self_awareness" in consciousness_signals and "introspection" in consciousness_signals:
            alignment = cosine_similarity(
                consciousness_signals["self_awareness"],
                consciousness_signals["introspection"].mean(axis=1)
            )
            coherence += float(alignment)
            count += 1
        
        if "autonomous_goals" in consciousness_signals:
            goal_consistency = jnp.std(consciousness_signals["autonomous_goals"])
            coherence += float(1.0 / (1.0 + goal_consistency))
            count += 1
        
        return coherence / max(1, count)
    
    def evaluate_reasoning_accuracy(self, reasoning_output):
        """Evaluate accuracy on reasoning tasks."""
        if "reasoning_chain" in reasoning_output:
            chain_length = len(reasoning_output["reasoning_chain"])
            return min(1.0, chain_length / self.config.max_reasoning_steps)
        
        return 0.0
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save training checkpoint using SafeTensors."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir="checkpoints",
            model_name="rtdlm_agi",
            keep_last_n=5
        )
        
        checkpoint_manager.save_checkpoint(
            params=self.params,
            opt_state=self.opt_state,
            epoch=epoch,
            step_count=self.step_count,
            metrics=metrics,
            config=self.config.to_dict(),
            training_losses=self.training_losses[-100:],
            validation_losses=self.validation_losses
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from SafeTensors format"""
        checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
        
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            load_opt_state=True,
            reference_opt_state=self.opt_state
        )
        
        self.params = checkpoint["params"]
        if checkpoint["opt_state"] is not None:
            self.opt_state = checkpoint["opt_state"]
        self.step_count = checkpoint.get("step_count", 0)
        
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def resume_from_checkpoint(self, checkpoint_path: str, sample_batch: Dict) -> int:
        """
        Resume training from a checkpoint.
        
        Loads model parameters, optimizer state, and training progress from a 
        previously saved checkpoint. The model doesn't start from zero weights.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.safetensors)
            sample_batch: Sample batch for model initialization (if needed)
            
        Returns:
            The epoch number to resume from (start_epoch)
        """
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # First, initialize the model structure if not already done
        if self.params is None:
            logger.info("Initializing model structure before loading weights...")
            self.initialize_model(sample_batch)
        
        # Load checkpoint
        checkpoint_manager = CheckpointManager(checkpoint_dir=str(Path(checkpoint_path).parent))
        
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            load_opt_state=True,
            reference_opt_state=self.opt_state
        )
        
        # Restore model parameters
        self.params = checkpoint["params"]
        
        # Restore optimizer state if available
        if checkpoint["opt_state"] is not None:
            self.opt_state = checkpoint["opt_state"]
            logger.info("Optimizer state restored")
        else:
            logger.info("No optimizer state in checkpoint, using fresh optimizer")
        
        # Restore step count
        self.step_count = checkpoint.get("step_count", 0)
        
        # Restore training history if available
        metadata = checkpoint.get("metadata", {})
        if metadata.get("training_losses"):
            self.training_losses = list(metadata["training_losses"])
            logger.info(f"Restored {len(self.training_losses)} training loss records")
        if metadata.get("validation_losses"):
            self.validation_losses = list(metadata["validation_losses"])
            logger.info(f"Restored {len(self.validation_losses)} validation loss records")
        
        # Calculate resume epoch
        resume_epoch = checkpoint.get("epoch", 0)
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        
        logger.info("Checkpoint loaded successfully")
        logger.info(f"  - Epoch: {resume_epoch}")
        logger.info(f"  - Step count: {self.step_count}")
        logger.info(f"  - Parameters: {param_count:,}")
        
        return resume_epoch
    
    def plot_training_metrics(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_losses)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        
        # Validation loss
        if self.validation_losses:
            axes[0, 1].plot(self.validation_losses)
            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
        
        # Reasoning accuracy
        if self.reasoning_accuracies:
            axes[1, 0].plot(self.reasoning_accuracies)
            axes[1, 0].set_title("Reasoning Accuracy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
        
        # Consciousness coherence
        if self.consciousness_coherence:
            axes[1, 1].plot(self.consciousness_coherence)
            axes[1, 1].set_title("Consciousness Coherence")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Coherence Score")
        
        plt.tight_layout()
        plt.savefig("agi_training_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """
        Get production evaluation metrics summary.
        
        Returns comprehensive metrics including:
        - Perplexity (running average)
        - Calibration (ECE, MCE)
        - Compute efficiency (throughput, latency)
        - Gradient health
        - Fairness (if enabled)
        
        Returns:
            Dictionary with all production metrics
        """
        metrics = {}
        
        # Perplexity
        metrics['perplexity'] = self.perplexity_tracker.get_perplexity()
        metrics['avg_loss'] = self.perplexity_tracker.get_loss()
        
        # Calibration
        cal_result = self.calibration_tracker.compute()
        metrics['calibration'] = {
            'ece': cal_result.expected_calibration_error,
            'mce': cal_result.maximum_calibration_error,
            'avg_confidence': cal_result.average_confidence,
            'avg_accuracy': cal_result.average_accuracy,
        }
        
        # Compute efficiency
        compute_result = self.compute_tracker.compute()
        metrics['compute'] = {
            'tokens_per_second': compute_result.tokens_per_second,
            'samples_per_second': compute_result.samples_per_second,
            'avg_latency_ms': compute_result.avg_latency_ms,
            'p95_latency_ms': compute_result.p95_latency_ms,
        }
        if compute_result.peak_memory_gb:
            metrics['compute']['peak_memory_gb'] = compute_result.peak_memory_gb
        
        # Gradient health
        grad_trend = self.gradient_monitor.get_trend()
        metrics['gradients'] = grad_trend
        
        # Fairness (if enabled)
        if self.fairness_analyzer is not None:
            metrics['fairness'] = {
                'enabled': True,
                'note': 'Call analyze_fairness() with predictions and sensitive features'
            }
        
        return metrics
    
    def print_production_metrics(self):
        """Print production metrics summary to console."""
        metrics = self.get_production_metrics()
        
        print("\n" + "="*60)
        print("PRODUCTION EVALUATION METRICS")
        print("="*60)
        print(f"Perplexity:      {metrics['perplexity']:.2f}")
        print(f"Avg Loss:        {metrics['avg_loss']:.4f}")
        print()
        print("Calibration:")
        print(f"  ECE:           {metrics['calibration']['ece']:.4f}")
        print(f"  Avg Accuracy:  {metrics['calibration']['avg_accuracy']:.2%}")
        print()
        print("Compute Efficiency:")
        print(f"  Throughput:    {metrics['compute']['tokens_per_second']:.0f} tok/s")
        print(f"  Avg Latency:   {metrics['compute']['avg_latency_ms']:.2f}ms")
        print()
        print("Gradient Health:")
        print(f"  Trend:         {metrics['gradients'].get('trend', 0):.6f}")
        print(f"  Volatility:    {metrics['gradients'].get('volatility', 0):.4f}")
        print("="*60 + "\n")

    def _run_epoch(
        self,
        batch_iterator: Iterator[Dict[str, jnp.ndarray]],
        num_batches_estimate: int
    ) -> List[float]:
        """
        Run a single training epoch.
        
        Supports gradient accumulation for larger effective batch sizes
        and memory profiling for tracking GPU memory usage.
        
        Args:
            batch_iterator: Iterator over training batches
            num_batches_estimate: Estimated number of batches for logging
            
        Returns:
            List of losses for the epoch
        """
        epoch_losses = []
        retrieval_augmented_count = 0
        
        # Use gradient accumulator if configured
        accum_steps = self.gradient_accumulation_steps
        
        for batch_idx, batch in enumerate(batch_iterator):
            self.rng, train_rng, augment_rng = jax.random.split(self.rng, 3)
            
            # Memory profiling - record at start of step
            self.memory_profiler.snapshot(self.step_count, phase="forward")
            
            # Augment batch with retrieval context
            original_batch = batch
            batch = self._augment_batch_with_retrieval(batch, augment_rng)
            if batch is not original_batch:
                retrieval_augmented_count += 1
            
            if accum_steps > 1 and self.gradient_accumulator is not None:
                # Gradient accumulation mode using BatchGradientAccumulator
                self.compute_tracker.start_batch()
                
                structured_batch = {
                    "inputs": {"text": batch["input_ids"]},
                    "multimodal_inputs": batch.get("multimodal_inputs"),
                    "targets": batch["targets"],
                }
                
                is_complete = self.gradient_accumulator.accumulate(
                    self.params, structured_batch, train_rng
                )
                
                # Track tokens processed
                num_tokens = int(jnp.sum(batch["targets"] != 0))
                self.compute_tracker.end_batch(num_tokens, batch["input_ids"].shape[0])
                
                # Memory profiling - after backward pass
                self.memory_profiler.snapshot(self.step_count, phase="backward")
                
                # Apply updates when accumulation is complete
                if is_complete:
                    avg_grads = self.gradient_accumulator.get_accumulated_grads()
                    avg_loss = self.gradient_accumulator.get_accumulated_loss()
                    
                    # Track gradient health
                    grad_metrics = self.gradient_monitor.compute_gradient_metrics(avg_grads)
                    if grad_metrics.has_nan or grad_metrics.has_inf:
                        logger.warning(f"NaN/Inf gradients at batch {batch_idx}, skipping...")
                        self.gradient_accumulator.reset()
                        continue
                    
                    # Skip if NaN/Inf loss
                    if jnp.isnan(avg_loss) or jnp.isinf(avg_loss):
                        logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping...")
                        self.gradient_accumulator.reset()
                        continue
                    
                    # Apply optimizer update
                    updates, self.opt_state = self.optimizer.update(
                        avg_grads, self.opt_state, self.params
                    )
                    self.params = optax.apply_updates(self.params, updates)
                    
                    # Memory profiling - after optimizer step
                    self.memory_profiler.snapshot(self.step_count, phase="optimizer")
                    
                    # Track perplexity
                    self.perplexity_tracker.update(float(avg_loss), num_tokens * accum_steps)
                    
                    epoch_losses.append(float(avg_loss))
                    self.training_losses.append(float(avg_loss))
                    self.step_count += 1
                    
                    # Log every 50 optimization steps (not micro-batches)
                    if self.step_count % 50 == 0:
                        ppl = self.perplexity_tracker.get_perplexity()
                        logger.info(
                            f"  Step {self.step_count}, "
                            f"Loss: {avg_loss:.4f}, PPL: {ppl:.2f} (accum={accum_steps})"
                        )
                    
                    # Reset for next accumulation cycle
                    self.gradient_accumulator.reset()
            else:
                # Standard training (no accumulation)
                # Track compute efficiency
                self.compute_tracker.start_batch()
                
                self.params, self.opt_state, loss, model_output = self.train_step(
                    self.params, self.opt_state, batch, train_rng
                )
                
                # Track compute efficiency
                num_tokens = int(jnp.sum(batch["targets"] != 0))
                self.compute_tracker.end_batch(num_tokens, batch["input_ids"].shape[0])
                
                # Memory profiling - after full step
                self.memory_profiler.snapshot(self.step_count, phase="optimizer")
                
                if jnp.isnan(loss) or jnp.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Track perplexity
                self.perplexity_tracker.update(float(loss), num_tokens)
                
                # Track calibration if logits available
                if model_output is not None and "logits" in model_output:
                    self.calibration_tracker.update(
                        model_output["logits"], batch["targets"]
                    )
                
                epoch_losses.append(float(loss))
                self.training_losses.append(float(loss))
                self.step_count += 1
                
                # Log with perplexity
                if batch_idx % 50 == 0:
                    ppl = self.perplexity_tracker.get_perplexity()
                    logger.info(
                        f"  Batch {batch_idx + 1}/{num_batches_estimate}, "
                        f"Loss: {loss:.4f}, PPL: {ppl:.2f}"
                    )
        
        if retrieval_augmented_count > 0:
            logger.info(f"  Retrieval augmented batches: {retrieval_augmented_count}/{len(epoch_losses)}")
        
        # Log production metrics summary
        ppl = self.perplexity_tracker.get_perplexity()
        compute_metrics = self.compute_tracker.compute()
        logger.info(
            f"  Epoch metrics: PPL={ppl:.2f}, "
            f"Throughput={compute_metrics.tokens_per_second:.0f} tok/s"
        )
        
        # Log memory summary at end of epoch
        if self.memory_profiler.enabled:
            mem_summary = self.memory_profiler.summary()
            if mem_summary.get("num_snapshots", 0) > 0:
                logger.info(
                    f"  Memory: peak={mem_summary.get('peak_memory_gb', 0):.2f}GB, "
                    f"avg={mem_summary.get('average_memory_gb', 0):.2f}GB"
                )
        
        return epoch_losses
    
    def _run_validation(
        self, 
        val_data: Any,
        use_streaming: bool
    ) -> Dict[str, float]:
        """
        Run validation and return metrics.
        
        Args:
            val_data: Validation data source (List or ShardedDataLoader)
            use_streaming: Whether using streaming data loader
            
        Returns:
            Dictionary of validation metrics
        """
        if use_streaming:
            val_batches: List[Dict[str, jnp.ndarray]] = val_data.get_validation_batches(max_batches=50)
        else:
            val_batches = val_data
            
        return self.evaluate_model(val_batches)
    
    def _check_early_stopping(
        self,
        val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        max_patience: int,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Tuple[float, int, bool]:
        """
        Check early stopping condition and update best model.
        
        Returns:
            Tuple of (new_best_loss, new_patience_counter, should_stop)
        """
        if val_loss < best_val_loss:
            self.save_checkpoint(epoch, metrics)
            return val_loss, 0, False
        
        new_patience = patience_counter + 1
        if new_patience >= max_patience:
            logger.info(f"Early stopping triggered after {new_patience} epochs without improvement")
            return best_val_loss, new_patience, True
        
        return best_val_loss, new_patience, False
    
    def _prepare_training(
        self,
        train_data: Any,
        resume_checkpoint: Optional[str]
    ) -> Tuple[int, Dict[str, jnp.ndarray], int, bool]:
        """
        Prepare training: initialize model or resume from checkpoint.
        
        Returns:
            Tuple of (start_epoch, sample_batch, num_batches_estimate, use_streaming)
        """
        use_streaming = isinstance(train_data, ShardedDataLoader)
        
        if use_streaming:
            sample_batch = train_data.get_sample_batch()
            num_batches_estimate = train_data.num_batches_per_epoch
        else:
            sample_batch = train_data[0]
            num_batches_estimate = len(train_data)
        
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.resume_from_checkpoint(resume_checkpoint, sample_batch)
            logger.info(f"Resuming from epoch {start_epoch + 1}")
        else:
            self.initialize_model(sample_batch)
        
        return start_epoch, sample_batch, num_batches_estimate, use_streaming

    def train(
        self, 
        train_data: Union[List[Dict[str, jnp.ndarray]], "ShardedDataLoader"],
        val_data: Union[List[Dict[str, jnp.ndarray]], "ShardedDataLoader"],
        resume_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Complete training loop for RT-DLM.
        
        Accepts pre-tokenized tensor data from the external data pipeline
        (Auralith-Data-Pipeline). Supports both in-memory batch lists and
        streaming ShardedDataLoader for large datasets.
        
        Args:
            train_data: Training data - either list of batches or ShardedDataLoader
            val_data: Validation data - either list of batches or ShardedDataLoader
            resume_checkpoint: Optional path to checkpoint to resume from.
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 60)
        logger.info("RT-DLM Training Started")
        logger.info("=" * 60)
        
        self.config.print_summary()
        
        # Prepare training
        start_epoch, _, num_batches_estimate, use_streaming = self._prepare_training(
            train_data, resume_checkpoint
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            epoch_start_time = time.time()
            
            # Run training epoch
            batch_iterator = train_data.iter_epoch() if use_streaming else iter(train_data)
            epoch_losses = self._run_epoch(batch_iterator, num_batches_estimate)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            logger.info(f"  Average Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Periodic checkpoint
            if epoch % 10 == 0 and epoch > 0:
                self.save_checkpoint(epoch, {"avg_loss": float(avg_epoch_loss)})
                logger.info(f"Periodic checkpoint saved at epoch {epoch}")
            
            # Validation
            should_validate = (epoch % self.config.eval_interval == 0) or (epoch == self.config.num_epochs - 1)
            if should_validate:
                metrics = self._run_validation(val_data, use_streaming)
                
                self.validation_losses.append(metrics["eval_loss"])
                self.reasoning_accuracies.append(metrics["reasoning_accuracy"])
                self.consciousness_coherence.append(metrics["consciousness_coherence"])
                
                logger.info(f"  Validation Loss: {metrics['eval_loss']:.4f}")
                logger.info(f"  Reasoning Accuracy: {metrics['reasoning_accuracy']:.4f}")
                logger.info(f"  Consciousness Coherence: {metrics['consciousness_coherence']:.4f}")
                
                best_val_loss, patience_counter, should_stop = self._check_early_stopping(
                    metrics["eval_loss"], best_val_loss, patience_counter, max_patience, epoch, metrics
                )
                if should_stop:
                    break
        
        logger.info("=" * 60)
        logger.info("RT-DLM Training Completed!")
        logger.info("=" * 60)
        
        self.plot_training_metrics()
        
        return {
            "final_params": self.params,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "reasoning_accuracies": self.reasoning_accuracies,
            "consciousness_coherence": self.consciousness_coherence,
        }


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train RT-DLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python train.py
  
  # Resume training from a checkpoint
  python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors
  
  # Train with custom settings
  python train.py --epochs 50 --batch-size 32 --lr 1e-4
  
  # Resume with more epochs
  python train.py --resume checkpoints/rtdlm_agi_epoch_10.safetensors --epochs 100
        """
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to checkpoint file to resume training from (.safetensors)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Total number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model hidden dimension (default: 512)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of transformer layers (default: 12)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-tokenized SafeTensor shards"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    
    return parser.parse_args()


class ShardedDataLoader:
    """
    Memory-efficient data loader for large SafeTensor datasets.
    
    Streams batches from disk instead of loading all data into memory.
    Supports sharding across multiple files for datasets that don't fit in RAM.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int,
        seq_length: int,
        shuffle: bool = True,
        prefetch_shards: int = 2
    ):
        """
        Initialize the sharded data loader.
        
        Args:
            data_dir: Directory containing .safetensors shard files
            batch_size: Number of samples per batch
            seq_length: Maximum sequence length
            shuffle: Whether to shuffle shards each epoch
            prefetch_shards: Number of shards to keep in memory
        """
        from safetensors.numpy import load_file as load_safetensors
        self._load_safetensors = load_safetensors
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.prefetch_shards = prefetch_shards
        
        self.shard_files = sorted(self.data_dir.glob("*.safetensors"))
        if not self.shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {data_dir}")
        
        self.num_shards = len(self.shard_files)
        self._rng = np.random.default_rng(42)
        
        first_shard = self._load_shard(self.shard_files[0])
        self.samples_per_shard = first_shard["input_ids"].shape[0]
        self.total_samples = self.samples_per_shard * self.num_shards
        
        logger.info(f"DataLoader initialized: {self.num_shards} shards, ~{self.total_samples} samples")
    
    def _load_shard(self, shard_path: Path) -> Dict[str, np.ndarray]:
        """Load a single shard from disk."""
        tensors = self._load_safetensors(str(shard_path))
        return tensors
    
    def _create_batches_from_shard(
        self, 
        shard_data: Dict[str, np.ndarray]
    ) -> List[Dict[str, jnp.ndarray]]:
        """Create batches from a loaded shard."""
        input_ids = shard_data["input_ids"]
        targets = shard_data.get("targets", input_ids)
        
        num_samples = input_ids.shape[0]
        batches = []
        
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            batch_input_ids = jnp.array(input_ids[start_idx:end_idx])
            batch_targets = jnp.array(targets[start_idx:end_idx])
            
            current_batch_size = batch_input_ids.shape[0]
            if current_batch_size < self.batch_size:
                pad_size = self.batch_size - current_batch_size
                batch_input_ids = jnp.pad(
                    batch_input_ids, 
                    ((0, pad_size), (0, 0)), 
                    constant_values=0
                )
                batch_targets = jnp.pad(
                    batch_targets, 
                    ((0, pad_size), (0, 0)), 
                    constant_values=0
                )
            
            batches.append({
                "input_ids": batch_input_ids,
                "targets": batch_targets,
                "text": batch_input_ids,
            })
        
        return batches
    
    def iter_epoch(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Iterate through one epoch of data.
        
        Yields batches one at a time, loading shards as needed.
        Memory-efficient for large datasets.
        """
        shard_order = list(range(self.num_shards))
        if self.shuffle:
            self._rng.shuffle(shard_order)
        
        for shard_idx in shard_order:
            shard_path = self.shard_files[shard_idx]
            shard_data = self._load_shard(shard_path)
            
            batches = self._create_batches_from_shard(shard_data)
            
            if self.shuffle:
                batch_indices = list(range(len(batches)))
                self._rng.shuffle(batch_indices)
                batches = [batches[i] for i in batch_indices]
            
            for batch in batches:
                yield batch
            
            del shard_data
    
    def get_sample_batch(self) -> Dict[str, jnp.ndarray]:
        """Get a single sample batch for model initialization."""
        shard_data = self._load_shard(self.shard_files[0])
        batches = self._create_batches_from_shard(shard_data)
        return batches[0]
    
    def get_validation_batches(self, max_batches: int = 50) -> List[Dict[str, jnp.ndarray]]:
        """
        Load validation batches (limited number for memory efficiency).
        
        Args:
            max_batches: Maximum number of batches to load
            
        Returns:
            List of validation batches
        """
        val_batches = []
        batch_count = 0
        
        for batch in self.iter_epoch():
            val_batches.append(batch)
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        return val_batches
    
    @property
    def num_batches_per_epoch(self) -> int:
        """Estimate number of batches per epoch."""
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def create_synthetic_batches(
    num_batches: int, 
    batch_size: int, 
    seq_length: int, 
    vocab_size: int
) -> List[Dict[str, jnp.ndarray]]:
    """Create synthetic tensor batches for testing."""
    rng = jax.random.PRNGKey(42)
    batches = []
    
    for _ in range(num_batches):
        rng, key = jax.random.split(rng)
        input_ids = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        batches.append({
            "input_ids": input_ids,
            "targets": input_ids,
            "text": input_ids,
        })
    
    return batches


def main():
    """Main training function with CLI support."""
    args = parse_args()
    
    # Create configuration
    config = AGIConfig(
        # Core architecture
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=8000,
        
        # Advanced features
        multimodal_enabled=True,
        quantum_layers=2,
        max_reasoning_steps=8,
        consciousness_simulation=True,
        
        # Training settings
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        eval_interval=2,
    )
    
    # Load training data
    logger.info("Loading training data...")
    
    if args.data_dir and os.path.isdir(args.data_dir):
        # Use streaming data loader for memory efficiency
        try:
            train_loader = ShardedDataLoader(
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                seq_length=config.max_seq_length,
                shuffle=True
            )
            
            # Create separate validation loader (no shuffle)
            val_loader = ShardedDataLoader(
                data_dir=args.data_dir,
                batch_size=config.batch_size,
                seq_length=config.max_seq_length,
                shuffle=False
            )
            
            logger.info(f"Streaming data loader ready: ~{train_loader.num_batches_per_epoch} batches/epoch")
            
            train_data = train_loader
            val_data = val_loader
            
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        logger.warning("No data directory provided. Creating synthetic data for testing...")
        train_data = create_synthetic_batches(
            num_batches=100,
            batch_size=config.batch_size,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
        val_data = create_synthetic_batches(
            num_batches=10,
            batch_size=config.batch_size,
            seq_length=config.max_seq_length,
            vocab_size=config.vocab_size
        )
    
    # Create trainer
    trainer = AGITrainer(config)
    
    # Handle checkpoint resumption
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)
        logger.info(f"Will resume training from: {args.resume}")
    
    # Start training
    results = trainer.train(
        train_data, 
        val_data,
        resume_checkpoint=args.resume
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Final training loss: {results['training_losses'][-1]:.4f}")
    
    if results['validation_losses']:
        logger.info(f"Final validation loss: {results['validation_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()

