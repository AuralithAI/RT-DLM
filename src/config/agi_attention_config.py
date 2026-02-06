"""
RT-DLM AGI Attention Configuration

Centralized configuration for AGI-scale attention mechanisms including:
- Ring Attention for distributed infinite context
- Cross-Memory Attention for LTM/STM/MTM interaction
- Hierarchical Memory Fusion
- Infinite Context via hierarchical compression

This config follows the same pattern as AGIConfig and RetrievalConfig.

Usage:
    from src.config.agi_attention_config import AGIAttentionConfig
    
    # Default configuration
    config = AGIAttentionConfig()
    
    # For distributed training
    config = AGIAttentionConfig.for_distributed(num_devices=8)
    
    # For long-context inference
    config = AGIAttentionConfig.for_long_context()
    
    # Sync with AGIConfig
    from src.config.agi_config import AGIConfig
    agi_config = AGIConfig()
    attention_config = AGIAttentionConfig.from_agi_config(agi_config)
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class MemoryFusionStrategy(str, Enum):
    """
    Strategy for fusing LTM/STM/MTM memory banks.
    
    WEIGHTED_SUM: Simple weighted sum (legacy behavior)
    CROSS_ATTENTION: Memory banks interact via cross-attention
    HIERARCHICAL: Multi-level attention-based fusion
    """
    WEIGHTED_SUM = "weighted_sum"
    CROSS_ATTENTION = "cross_attention"
    HIERARCHICAL = "hierarchical"


class ContextStrategy(str, Enum):
    """
    Strategy for handling long contexts.
    
    STANDARD: Standard attention (O(n²) complexity)
    SLIDING_WINDOW: Local attention window
    RING: Ring attention for distributed processing
    HIERARCHICAL: Chunked processing with compression
    """
    STANDARD = "standard"
    SLIDING_WINDOW = "sliding_window"
    RING = "ring"
    HIERARCHICAL = "hierarchical"


@dataclass
class AGIAttentionConfig:
    """
    Configuration for AGI-scale attention mechanisms.
    
    This config controls advanced attention features for AGI-level capabilities:
    - Ring Attention for infinite context distributed across devices
    - Cross-Memory Attention for deep memory bank interaction
    - Hierarchical Memory Fusion for multi-level integration
    - Infinite Context via hierarchical compression
    
    Attributes are organized into logical sections for clarity.
    
    Example:
        # Simple usage
        config = AGIAttentionConfig(enable_ring_attention=True)
        
        # From preset
        config = AGIAttentionConfig.for_distributed(num_devices=4)
        
        # Integrate with AGIConfig
        from src.config.agi_config import AGIConfig
        agi_config = AGIConfig(d_model=512, num_heads=8)
        attention_config = AGIAttentionConfig.from_agi_config(agi_config)
    """
    
    # ==========================================================================
    # Core Model Parameters
    # ==========================================================================
    
    d_model: int = 384
    """Model dimension. Should match AGIConfig.d_model."""
    
    num_heads: int = 8
    """Number of attention heads. Should match AGIConfig.num_heads."""
    
    head_dim: Optional[int] = None
    """Dimension per head. Derived from d_model // num_heads if None."""
    
    max_seq_length: int = 2048
    """Maximum sequence length. Should match AGIConfig.max_seq_length."""
    
    # ==========================================================================
    # Ring Attention Settings
    # ==========================================================================
    
    enable_ring_attention: bool = False
    """Enable Ring Attention for distributed infinite context processing."""
    
    ring_block_size: int = 512
    """Block size for Ring Attention chunking."""
    
    num_ring_devices: int = 1
    """Number of devices for distributed Ring Attention."""
    
    context_strategy: ContextStrategy = ContextStrategy.STANDARD
    """Strategy for handling long contexts."""
    
    # ==========================================================================
    # Memory Cross-Attention Settings
    # ==========================================================================
    
    enable_memory_cross_attention: bool = False
    """Enable cross-attention between LTM/STM/MTM memory banks."""
    
    memory_fusion_strategy: MemoryFusionStrategy = MemoryFusionStrategy.WEIGHTED_SUM
    """Strategy for fusing memory bank representations."""
    
    num_memory_heads: int = 4
    """Number of heads for memory cross-attention."""
    
    memory_dropout: float = 0.1
    """Dropout rate for memory attention layers."""
    
    # ==========================================================================
    # Infinite Context Settings
    # ==========================================================================
    
    enable_infinite_context: bool = False
    """Enable infinite context via hierarchical compression."""
    
    context_chunk_size: int = 1024
    """Chunk size for infinite context processing."""
    
    global_context_size: int = 256
    """Size of compressed global context buffer."""
    
    context_compression_ratio: int = 4
    """Compression ratio for chunk summarization."""
    
    # ==========================================================================
    # Position Encoding
    # ==========================================================================
    
    use_rope: bool = True
    """Use Rotary Position Embedding."""
    
    rope_theta: float = 10000.0
    """Base frequency for RoPE computation."""
    
    rope_scaling: Optional[float] = None
    """Scaling factor for extended context (e.g., 2.0 for 2x length)."""
    
    # ==========================================================================
    # Performance Settings
    # ==========================================================================
    
    use_flash_attention: bool = False
    """Use Flash Attention when available."""
    
    attention_dropout: float = 0.0
    """Dropout rate for attention weights."""
    
    def __post_init__(self):
        """Validate configuration and derive computed fields."""
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads
        
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.ring_block_size > 0, "ring_block_size must be positive"
        assert self.num_ring_devices > 0, "num_ring_devices must be positive"
        assert self.context_chunk_size > 0, "context_chunk_size must be positive"
        assert self.global_context_size > 0, "global_context_size must be positive"
        assert 0 <= self.memory_dropout <= 1, "memory_dropout must be in [0, 1]"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be in [0, 1]"
        
        # Validate memory fusion strategy consistency
        if self.memory_fusion_strategy != MemoryFusionStrategy.WEIGHTED_SUM:
            self.enable_memory_cross_attention = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AGIAttentionConfig":
        """Create config from dictionary."""
        if "memory_fusion_strategy" in data and isinstance(data["memory_fusion_strategy"], str):
            data["memory_fusion_strategy"] = MemoryFusionStrategy(data["memory_fusion_strategy"])
        if "context_strategy" in data and isinstance(data["context_strategy"], str):
            data["context_strategy"] = ContextStrategy(data["context_strategy"])
        return cls(**data)
    
    @classmethod
    def from_agi_config(cls, agi_config) -> "AGIAttentionConfig":
        """
        Create AGIAttentionConfig from an AGIConfig instance.
        
        Syncs model dimensions and attention parameters.
        
        Args:
            agi_config: AGIConfig instance
            
        Returns:
            AGIAttentionConfig with synced parameters
        """
        return cls(
            d_model=agi_config.d_model,
            num_heads=agi_config.num_heads,
            max_seq_length=agi_config.max_seq_length,
            enable_ring_attention=getattr(agi_config, 'enable_ring_attention', False),
            ring_block_size=getattr(agi_config, 'ring_block_size', 512),
            num_ring_devices=getattr(agi_config, 'num_ring_devices', 1),
            enable_memory_cross_attention=getattr(agi_config, 'enable_memory_cross_attention', False),
            num_memory_heads=getattr(agi_config, 'memory_attention_heads', 4),
            memory_dropout=getattr(agi_config, 'memory_dropout', 0.1),
            enable_infinite_context=getattr(agi_config, 'enable_infinite_context', False),
            context_chunk_size=getattr(agi_config, 'context_chunk_size', 1024),
            global_context_size=getattr(agi_config, 'global_context_size', 256),
            use_rope=(getattr(agi_config, 'position_encoding', 'rope') == 'rope'),
            rope_theta=getattr(agi_config, 'rope_theta', 10000.0),
        )
    
    @classmethod
    def for_distributed(cls, num_devices: int = 4) -> "AGIAttentionConfig":
        """
        Preset for distributed training with Ring Attention.
        
        Optimized for multi-device setups with infinite context.
        
        Args:
            num_devices: Number of devices for distributed attention
        """
        return cls(
            enable_ring_attention=True,
            ring_block_size=256,
            num_ring_devices=num_devices,
            context_strategy=ContextStrategy.RING,
            enable_memory_cross_attention=True,
            memory_fusion_strategy=MemoryFusionStrategy.CROSS_ATTENTION,
        )
    
    @classmethod
    def for_long_context(cls, max_length: int = 32768) -> "AGIAttentionConfig":
        """
        Preset for long-context inference.
        
        Uses hierarchical compression for efficient processing.
        
        Args:
            max_length: Target maximum context length
        """
        chunk_size = min(2048, max_length // 8)
        global_size = min(512, max_length // 16)
        
        return cls(
            max_seq_length=max_length,
            enable_infinite_context=True,
            context_chunk_size=chunk_size,
            global_context_size=global_size,
            context_strategy=ContextStrategy.HIERARCHICAL,
            enable_memory_cross_attention=True,
            memory_fusion_strategy=MemoryFusionStrategy.HIERARCHICAL,
        )
    
    @classmethod
    def for_memory_reasoning(cls) -> "AGIAttentionConfig":
        """
        Preset for enhanced memory-based reasoning.
        
        Enables cross-memory attention for deep LTM/STM/MTM interaction.
        """
        return cls(
            enable_memory_cross_attention=True,
            memory_fusion_strategy=MemoryFusionStrategy.HIERARCHICAL,
            num_memory_heads=8,
        )
    
    @classmethod
    def disabled(cls) -> "AGIAttentionConfig":
        """Preset with all advanced features disabled (legacy behavior)."""
        return cls(
            enable_ring_attention=False,
            enable_memory_cross_attention=False,
            enable_infinite_context=False,
            memory_fusion_strategy=MemoryFusionStrategy.WEIGHTED_SUM,
            context_strategy=ContextStrategy.STANDARD,
        )
    
    def sync_with_agi_config(self, agi_config) -> None:
        """
        Sync attention config with AGIConfig parameters.
        
        Ensures d_model, num_heads, etc. match.
        
        Args:
            agi_config: AGIConfig instance
        """
        self.d_model = agi_config.d_model
        self.num_heads = agi_config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.max_seq_length = agi_config.max_seq_length
