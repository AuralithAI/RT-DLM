"""
Core Components Module

Provides reusable components for RT-DLM AGI system including:
- ReusableAttention: Unified attention wrapper with spiking/pruning
- SpikingMechanism: Modular spiking attention
- PruningManager: Centralized pruning utilities
- ReusableFeedForward: Shared FFN implementation
- ReusableTransformerBlock: Complete transformer layer
- GraphNeuron: Graph-based neural components for relational reasoning
- DynamicGraphBuilder: Build graphs from sequence embeddings
- MultiHopGraphReasoner: Multi-hop reasoning over graph structures
"""

from src.core.components.reusable_components import (
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

from src.core.components.graph_neurons import (
    # Configuration
    GraphConfig,
    GraphOutput,
    # Core modules
    GraphNeuron,
    GraphAttentionUnit,
    DynamicGraphBuilder,
    RelationalRouter,
    MultiHopGraphReasoner,
    GraphMoE,
    GraphIntegratedTransformerBlock,
    # Factory functions
    create_graph_neuron,
    create_multi_hop_reasoner,
    create_graph_moe,
    # Utility functions
    compute_graph_loss,
    compute_graph_accuracy,
)

__all__ = [
    # Attention components
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
    # Graph neural components
    'GraphConfig',
    'GraphOutput',
    'GraphNeuron',
    'GraphAttentionUnit',
    'DynamicGraphBuilder',
    'RelationalRouter',
    'MultiHopGraphReasoner',
    'GraphMoE',
    'GraphIntegratedTransformerBlock',
    'create_graph_neuron',
    'create_multi_hop_reasoner',
    'create_graph_moe',
    'compute_graph_loss',
    'compute_graph_accuracy',
]
