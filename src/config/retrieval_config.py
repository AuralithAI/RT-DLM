"""
RT-DLM Retrieval Augmentation Configuration

Centralized configuration for retrieval-augmented generation (RAG) capabilities.
This config follows the same pattern as AGIConfig and other RT-DLM configs.

Design Philosophy:
    - Retrieval should be optional (like GPT-4, Claude, Gemini)
    - Support multiple backends (flexibility)
    - Integrate with existing memory systems (MemoryBank, PersistentLTMStorage)
    - Enable/disable per-forward pass or per-training phase

Usage:
    from src.config.retrieval_config import RetrievalConfig, RetrievalProvider
    
    # Training with retrieval
    config = RetrievalConfig.for_training()
    
    # Inference with retrieval
    config = RetrievalConfig.for_inference()
    
    # Custom configuration
    config = RetrievalConfig(
        enabled=True,
        provider=RetrievalProvider.INTERNAL,
        top_k=5,
        use_hybrid=True,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class RetrievalProvider(str, Enum):
    """
    Supported retrieval backends.
    
    INTERNAL: Use RT-DLM's built-in MemoryBank/PersistentLTMStorage
    FAISS: Standalone FAISS index (local)
    PINECONE: Pinecone vector database (cloud)
    WEAVIATE: Weaviate vector database (cloud/local)
    CHROMADB: ChromaDB (local/cloud)
    """
    INTERNAL = "internal"
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMADB = "chromadb"


class ChunkingStrategy(str, Enum):
    """
    Document chunking strategies for ingestion.
    
    FIXED: Fixed character/token count chunks
    SENTENCE: Split on sentence boundaries
    SEMANTIC: Split based on semantic similarity
    RECURSIVE: LangChain-style recursive splitting (recommended)
    SLIDING_WINDOW: Overlapping windows
    """
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval augmentation.
    
    This config controls all aspects of retrieval-augmented generation:
    - Whether retrieval is enabled
    - Which backend to use
    - How documents are chunked and embedded
    - How retrieval integrates with training
    - Cross-attention settings for the forward pass
    
    Attributes are organized into logical sections for clarity.
    
    Example:
        # Simple usage
        config = RetrievalConfig(enabled=True, top_k=5)
        
        # From preset
        config = RetrievalConfig.for_training()
        
        # Integrate with AGIConfig
        from src.config.agi_config import AGIConfig
        agi_config = AGIConfig()
        retrieval_config = RetrievalConfig(embedding_dim=agi_config.d_model)
    """
    
    # ==========================================================================
    # Core Settings
    # ==========================================================================
    
    enabled: bool = False
    """Master switch for retrieval augmentation. Default off for backward compat."""
    
    provider: RetrievalProvider = RetrievalProvider.INTERNAL
    """Which retrieval backend to use. INTERNAL uses RT-DLM's MemoryBank."""
    
    # ==========================================================================
    # Retrieval Parameters
    # ==========================================================================
    
    top_k: int = 5
    """Number of documents to retrieve per query."""
    
    min_score_threshold: float = 0.0
    """Minimum similarity score to include a result (0-1)."""
    
    use_hybrid: bool = True
    """Use hybrid (dense + sparse/BM25) retrieval. Industry best practice."""
    
    sparse_weight: float = 0.3
    """Weight for sparse (BM25) retrieval in hybrid mode."""
    
    dense_weight: float = 0.7
    """Weight for dense (embedding) retrieval in hybrid mode."""
    
    rerank_results: bool = False
    """Apply cross-encoder reranking. Higher quality but higher latency."""
    
    # ==========================================================================
    # Document Processing
    # ==========================================================================
    
    chunk_size: int = 512
    """Target size for document chunks (in tokens/characters)."""
    
    chunk_overlap: int = 128
    """Overlap between consecutive chunks to maintain context."""
    
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    """How to split documents into chunks. RECURSIVE is recommended."""
    
    max_chunks_per_doc: int = 100
    """Maximum chunks to create from a single document."""
    
    # ==========================================================================
    # Embedding Settings
    # ==========================================================================
    
    embedding_dim: int = 384
    """Dimension of document embeddings. Should match AGIConfig.d_model."""
    
    normalize_embeddings: bool = True
    """L2 normalize embeddings for cosine similarity."""
    
    use_model_embeddings: bool = True
    """Use RT-DLM's own embeddings vs external embedding model."""
    
    # ==========================================================================
    # Training Integration
    # ==========================================================================
    
    augmentation_probability: float = 0.2
    """Probability of augmenting a batch during training. 0.2 is a good start."""
    
    use_contrastive_loss: bool = False
    """Add contrastive loss term for retrieval alignment during training."""
    
    contrastive_weight: float = 0.1
    """Weight of contrastive loss if enabled."""
    
    query_prefix: str = ""
    """Prefix to add to queries (some embedding models need this)."""
    
    document_prefix: str = ""
    """Prefix to add to documents during embedding."""
    
    # ==========================================================================
    # Cross-Attention Settings (for forward pass integration)
    # ==========================================================================
    
    cross_attention_layers: List[int] = field(default_factory=lambda: [0, 4, 8])
    """Which transformer layers should attend to retrieved documents."""
    
    cross_attention_heads: Optional[int] = None
    """Number of heads for cross-attention. None = same as model num_heads."""
    
    retrieved_context_length: int = 2048
    """Maximum tokens from retrieved documents to attend to."""
    
    # ==========================================================================
    # External Provider Settings (for cloud backends)
    # ==========================================================================
    
    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "rtdlm-retrieval"
    
    # Weaviate
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str = "RTDLMDocument"
    
    # ChromaDB
    chromadb_path: str = "./chroma_db"
    chromadb_collection: str = "rtdlm_docs"
    
    # ==========================================================================
    # Performance Settings
    # ==========================================================================
    
    cache_embeddings: bool = True
    """Cache document embeddings to avoid recomputation."""
    
    batch_retrieval: bool = True
    """Batch multiple queries for efficient retrieval."""
    
    async_retrieval: bool = False
    """Use async retrieval (useful for external providers)."""
    
    retrieval_timeout: float = 5.0
    """Timeout for external retrieval calls in seconds."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.top_k > 0, "top_k must be positive"
        assert 0 <= self.min_score_threshold <= 1, "min_score_threshold must be in [0, 1]"
        assert abs(self.sparse_weight + self.dense_weight - 1.0) < 1e-6, \
            "sparse_weight + dense_weight must equal 1.0"
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.chunk_overlap >= 0, "chunk_overlap must be non-negative"
        assert self.chunk_overlap < self.chunk_size, "chunk_overlap must be less than chunk_size"
        assert 0 <= self.augmentation_probability <= 1, "augmentation_probability must be in [0, 1]"
        
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        """Create config from dictionary."""
        # Convert enum strings back to enums
        if "provider" in data and isinstance(data["provider"], str):
            data["provider"] = RetrievalProvider(data["provider"])
        if "chunking_strategy" in data and isinstance(data["chunking_strategy"], str):
            data["chunking_strategy"] = ChunkingStrategy(data["chunking_strategy"])
        return cls(**data)
    
    @classmethod
    def for_training(cls) -> "RetrievalConfig":
        """
        Preset for training with retrieval augmentation.
        
        Uses conservative settings:
        - 20% augmentation probability
        - Contrastive loss enabled
        - 3 documents per query
        """
        return cls(
            enabled=True,
            provider=RetrievalProvider.INTERNAL,
            top_k=3,
            use_hybrid=True,
            augmentation_probability=0.2,
            use_contrastive_loss=True,
            contrastive_weight=0.05,
        )
    
    @classmethod
    def for_inference(cls) -> "RetrievalConfig":
        """
        Preset for inference with retrieval.
        
        Uses quality-focused settings:
        - Always retrieve (100% probability)
        - Reranking enabled
        - 5 documents per query
        """
        return cls(
            enabled=True,
            provider=RetrievalProvider.INTERNAL,
            top_k=5,
            use_hybrid=True,
            rerank_results=True,
            augmentation_probability=1.0,
        )
    
    @classmethod
    def disabled(cls) -> "RetrievalConfig":
        """Preset with retrieval disabled (default behavior)."""
        return cls(enabled=False)
    
    def sync_with_agi_config(self, agi_config) -> None:
        """
        Sync retrieval config with AGIConfig parameters.
        
        Ensures embedding_dim matches d_model, etc.
        
        Args:
            agi_config: AGIConfig instance
        """
        self.embedding_dim = agi_config.d_model
        if self.cross_attention_heads is None:
            self.cross_attention_heads = agi_config.num_heads
