"""
Core Components Module

Provides reusable components for RT-DLM AGI system including:
- ReusableAttention: Unified attention wrapper with spiking/pruning
- SpikingMechanism: Modular spiking attention
- PruningManager: Centralized pruning utilities
- ReusableFeedForward: Shared FFN implementation
- ReusableTransformerBlock: Complete transformer layer
"""

from core.components.reusable_components import (
    # Configuration
    AttentionConfig,
    # Core modules
    ReusableAttention,
    ReusableFeedForward,
    ReusableTransformerBlock,
    # Mechanisms
    SpikingMechanism,
    PruningManager,
    # Factory functions
    create_attention,
    create_transformer_block,
    # Utility functions
    apply_shared_spiking,
    compute_attention_sparsity,
)

__all__ = [
    'AttentionConfig',
    'ReusableAttention',
    'ReusableFeedForward',
    'ReusableTransformerBlock',
    'SpikingMechanism',
    'PruningManager',
    'create_attention',
    'create_transformer_block',
    'apply_shared_spiking',
    'compute_attention_sparsity',
]
