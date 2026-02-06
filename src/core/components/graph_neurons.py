"""
Graph-Based Neurons for RT-DLM AGI

This module implements graph-based neural components that enable relational
reasoning and multi-hop inference. Graph neurons process structured relationships
between concepts, enabling better reasoning for tasks like health diagnostics,
disaster prediction, and knowledge graph completion.

Key Components:
- GraphNeuron: Core graph attention unit with residual connections
- DynamicGraphBuilder: Builds graphs dynamically from embeddings
- RelationalRouter: Routes information based on graph structure
- GraphMoE: Mixture-of-Experts with graph-based routing
- MultiHopGraphReasoner: Multi-hop reasoning over graph structures

Design Philosophy:
- Integrates with existing transformer architecture (TMSModel)
- Uses GraphAttentionLayer from comprehension_modules.py as foundation
- Supports dynamic graph construction from sequence embeddings
- Enables relational reasoning alongside standard attention

References:
- Graph Attention Networks (Veličković et al., 2018)
- Relational Graph Convolutional Networks (Schlichtkrull et al., 2018)
- GraphFormers (Yang et al., 2021)
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """Configuration for graph-based neural components."""
    d_model: int = 384
    num_heads: int = 8
    max_nodes: int = 64
    edge_threshold: float = 0.3
    num_edge_types: int = 8
    num_hops: int = 3
    dropout_rate: float = 0.1
    use_edge_features: bool = True
    graph_residual: bool = True
    enable_relational_routing: bool = True
    

class GraphOutput(NamedTuple):
    """Output from graph neural operations."""
    node_features: jnp.ndarray  # [batch, num_nodes, d_model]
    adjacency: jnp.ndarray  # [batch, num_nodes, num_nodes]
    edge_weights: Optional[jnp.ndarray]  # [batch, num_nodes, num_nodes]
    attention_weights: Optional[jnp.ndarray]  # [batch, num_heads, num_nodes, num_nodes]
    reasoning_path: Optional[jnp.ndarray]  # [batch, num_hops, num_nodes]


class GraphAttentionUnit(hk.Module):
    """
    Basic Graph Attention Unit using multi-head attention.
    
    This is a streamlined version of GraphAttentionLayer optimized for
    integration with transformer architectures.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply graph attention.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            adjacency: [batch, num_nodes, num_nodes] - connectivity mask
            is_training: Whether in training mode
            
        Returns:
            updated_features: [batch, num_nodes, d_model]
            attention_weights: [batch, num_heads, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Q, K, V projections
        query = hk.Linear(self.d_model, name="query")(node_features)
        key = hk.Linear(self.d_model, name="key")(node_features)
        value = hk.Linear(self.d_model, name="value")(node_features)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose: [batch, num_heads, num_nodes, head_dim]
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))
        
        # Compute attention scores
        scores = jnp.einsum('bhid,bhjd->bhij', query, key) / jnp.sqrt(self.head_dim)
        
        # Apply graph structure mask
        adj_mask = adjacency[:, None, :, :]  # [batch, 1, num_nodes, num_nodes]
        scores = jnp.where(adj_mask > 0, scores, -1e9)
        
        # Softmax and optional dropout
        attention_weights = jax.nn.softmax(scores, axis=-1)
        if is_training and self.dropout_rate > 0:
            attention_weights = hk.dropout(
                hk.next_rng_key(), self.dropout_rate, attention_weights
            )
        
        # Aggregate
        aggregated = jnp.einsum('bhij,bhjd->bhid', attention_weights, value)
        aggregated = jnp.transpose(aggregated, (0, 2, 1, 3))
        aggregated = aggregated.reshape(batch_size, num_nodes, self.d_model)
        
        # Output projection
        output = hk.Linear(self.d_model, name="output")(aggregated)
        
        return output, attention_weights


class GraphNeuron(hk.Module):
    """
    Graph Neuron - Core building block for graph-based reasoning.
    
    Combines graph attention with residual connections and layer normalization
    for stable training. Can be inserted into transformer blocks or used
    independently for graph-based tasks.
    
    Example usage in TMSModel:
        # After MoE layer
        graph_neuron = GraphNeuron(d_model, num_heads)
        adj_matrix = graph_builder(x)  # Build graph from features
        x = graph_neuron(x, adj_matrix)  # Apply graph reasoning
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_ffn: bool = True,
        ffn_expansion: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_ffn = use_ffn
        self.ffn_expansion = ffn_expansion
        
    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        is_training: bool = True
    ) -> jnp.ndarray:
        """
        Apply graph neuron processing.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            adjacency: [batch, num_nodes, num_nodes]
            is_training: Whether in training mode
            
        Returns:
            Updated node features: [batch, num_nodes, d_model]
        """
        # Graph attention with residual
        graph_attn = GraphAttentionUnit(
            self.d_model, self.num_heads, self.dropout_rate, name="graph_attn"
        )
        attn_output, _ = graph_attn(node_features, adjacency, is_training)
        
        # Residual connection and layer norm
        norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm1")
        x = norm1(node_features + attn_output)
        
        # Optional FFN
        if self.use_ffn:
            ffn = hk.Sequential([
                hk.Linear(self.d_model * self.ffn_expansion, name="ffn_up"),
                jax.nn.gelu,
                hk.Linear(self.d_model, name="ffn_down"),
            ])
            ffn_output = ffn(x)
            if is_training and self.dropout_rate > 0:
                ffn_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ffn_output)
            
            norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm2")
            x = norm2(x + ffn_output)
        
        return x


class DynamicGraphBuilder(hk.Module):
    """
    Dynamically builds graphs from sequence embeddings.
    
    Uses learned edge prediction to construct adjacency matrices,
    enabling integration of graph reasoning into sequence models.
    """
    
    def __init__(
        self,
        d_model: int,
        max_nodes: int = 64,
        edge_threshold: float = 0.3,
        num_edge_types: int = 1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.edge_threshold = edge_threshold
        self.num_edge_types = num_edge_types
        
    def __call__(
        self,
        embeddings: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build graph from embeddings.
        
        Args:
            embeddings: [batch, seq_len, d_model]
            mask: Optional mask for valid positions [batch, seq_len]
            
        Returns:
            node_features: [batch, num_nodes, d_model]
            adjacency: [batch, num_nodes, num_nodes]
        """
        batch_size, seq_len, _ = embeddings.shape
        num_nodes = min(seq_len, self.max_nodes)
        
        # Extract node representations (could use pooling for longer sequences)
        if seq_len > self.max_nodes:
            # Pool sequence into fixed number of nodes
            pool_size = seq_len // self.max_nodes
            node_features = jnp.mean(
                embeddings[:, :self.max_nodes * pool_size].reshape(
                    batch_size, self.max_nodes, pool_size, self.d_model
                ),
                axis=2
            )
        else:
            node_features = embeddings[:, :num_nodes]
            # Pad if needed
            if seq_len < self.max_nodes:
                padding = jnp.zeros((batch_size, self.max_nodes - seq_len, self.d_model))
                node_features = jnp.concatenate([node_features, padding], axis=1)
        
        # Edge prediction via learned bilinear scoring
        edge_predictor = hk.Linear(self.d_model, name="edge_predictor")
        transformed = edge_predictor(node_features)
        
        # Bilinear edge scores: [batch, num_nodes, num_nodes]
        edge_scores = jnp.einsum('bid,bjd->bij', node_features, transformed)
        edge_scores = edge_scores / jnp.sqrt(self.d_model)
        
        # Apply threshold to create adjacency
        edge_probs = jax.nn.sigmoid(edge_scores)
        adjacency = (edge_probs > self.edge_threshold).astype(jnp.float32)
        
        # Add self-loops
        adjacency = adjacency + jnp.eye(self.max_nodes)[None, :, :]
        adjacency = jnp.clip(adjacency, 0, 1)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to node dimensions
            node_mask = mask[:, :self.max_nodes]
            if mask.shape[1] < self.max_nodes:
                pad_mask = jnp.zeros((batch_size, self.max_nodes - mask.shape[1]))
                node_mask = jnp.concatenate([node_mask, pad_mask], axis=1)
            # Mask invalid nodes
            adjacency = adjacency * node_mask[:, :, None] * node_mask[:, None, :]
        
        return node_features, adjacency


class RelationalRouter(hk.Module):
    """
    Routes information based on learned relational structure.
    
    Used in conjunction with MoE to enable graph-based expert routing,
    where related tokens can influence each other's expert selection.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_edge_types: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_edge_types = num_edge_types
        
    def __call__(
        self,
        features: jnp.ndarray,
        adjacency: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute routing scores influenced by graph structure.
        
        Args:
            features: [batch, num_tokens, d_model]
            adjacency: [batch, num_tokens, num_tokens]
            
        Returns:
            routing_scores: [batch, num_tokens, num_experts]
        """
        # Local routing scores
        local_router = hk.Linear(self.num_experts, name="local_router")
        local_scores = local_router(features)  # [batch, num_tokens, num_experts]
        
        # Relational routing: aggregate neighbor routing preferences
        neighbor_scores = jnp.einsum('bij,bje->bie', adjacency, local_scores)
        neighbor_counts = jnp.sum(adjacency, axis=-1, keepdims=True) + 1e-8
        neighbor_scores = neighbor_scores / neighbor_counts
        
        # Combine local and relational routing
        combine_gate = hk.Linear(1, name="combine_gate")
        gate = jax.nn.sigmoid(combine_gate(features))  # [batch, num_tokens, 1]
        
        routing_scores = gate * local_scores + (1 - gate) * neighbor_scores
        
        return routing_scores


class MultiHopGraphReasoner(hk.Module):
    """
    Performs multi-hop reasoning over graph structures.
    
    Enables chain-of-thought style reasoning by propagating information
    through multiple graph attention hops, tracking reasoning paths.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_hops: int = 3,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_hops = num_hops
        self.dropout_rate = dropout_rate
        
    def __call__(
        self,
        node_features: jnp.ndarray,
        adjacency: jnp.ndarray,
        query: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform multi-hop reasoning.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            adjacency: [batch, num_nodes, num_nodes]
            query: Optional query for guided reasoning [batch, d_model]
            is_training: Whether in training mode
            
        Returns:
            final_features: [batch, num_nodes, d_model]
            reasoning_paths: [batch, num_hops, num_nodes] - attention over hops
        """
        batch_size, num_nodes, _ = node_features.shape
        
        reasoning_paths = []
        current_features = node_features
        
        for hop in range(self.num_hops):
            # Graph attention hop
            graph_neuron = GraphNeuron(
                self.d_model, 
                self.num_heads, 
                self.dropout_rate,
                use_ffn=(hop == self.num_hops - 1),  # FFN only on last hop
                name=f"hop_{hop}"
            )
            current_features = graph_neuron(current_features, adjacency, is_training)
            
            # Track reasoning path (which nodes are attended)
            if query is not None:
                # Compute attention to query
                path_scores = jnp.einsum('bd,bnd->bn', query[:, :], current_features)
                path_attention = jax.nn.softmax(path_scores, axis=-1)
            else:
                # Uniform attention over nodes
                path_attention = jnp.ones((batch_size, num_nodes)) / num_nodes
            
            reasoning_paths.append(path_attention)
        
        reasoning_paths = jnp.stack(reasoning_paths, axis=1)  # [batch, num_hops, num_nodes]
        
        return current_features, reasoning_paths


class GraphMoE(hk.Module):
    """
    Mixture-of-Experts with graph-based routing.
    
    Extends SparseMoE with relational routing, allowing expert selection
    to consider relationships between tokens.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        use_relational_routing: bool = True,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_relational_routing = use_relational_routing
        self.dropout_rate = dropout_rate
        
    def __call__(
        self,
        features: jnp.ndarray,
        adjacency: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Apply graph-aware MoE.
        
        Args:
            features: [batch, seq_len, d_model]
            adjacency: Optional graph structure [batch, seq_len, seq_len]
            is_training: Whether in training mode
            
        Returns:
            output: [batch, seq_len, d_model]
            expert_indices: [batch, seq_len, top_k]
            aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, _ = features.shape
        
        # Compute routing scores
        if self.use_relational_routing and adjacency is not None:
            router = RelationalRouter(
                self.d_model, self.num_experts, name="relational_router"
            )
            routing_logits = router(features, adjacency)
        else:
            # Standard routing
            router = hk.Linear(self.num_experts, name="router")
            routing_logits = router(features)
        
        # Top-k expert selection
        routing_probs = jax.nn.softmax(routing_logits, axis=-1)
        top_k_probs, top_k_indices = jax.lax.top_k(routing_probs, self.top_k)
        
        # Normalize top-k probs
        top_k_probs = top_k_probs / (jnp.sum(top_k_probs, axis=-1, keepdims=True) + 1e-8)
        
        # Create experts
        experts = [
            hk.Sequential([
                hk.Linear(self.d_model * 4, name=f"expert_{i}_up"),
                jax.nn.gelu,
                hk.Linear(self.d_model, name=f"expert_{i}_down"),
            ])
            for i in range(self.num_experts)
        ]
        
        # Compute expert outputs (simplified - actual impl would use scatter/gather)
        all_expert_outputs = jnp.stack([expert(features) for expert in experts], axis=-2)
        # [batch, seq_len, num_experts, d_model]
        
        # Gather top-k expert outputs
        batch_indices = jnp.arange(batch_size)[:, None, None]
        seq_indices = jnp.arange(seq_len)[None, :, None]
        selected_outputs = all_expert_outputs[batch_indices, seq_indices, top_k_indices]
        # [batch, seq_len, top_k, d_model]
        
        # Weighted combination
        output = jnp.einsum('bstd,bst->bsd', selected_outputs, top_k_probs)
        
        # Load balancing loss
        expert_usage = jnp.sum(routing_probs, axis=(0, 1))
        expert_importance = expert_usage / (jnp.sum(expert_usage) + 1e-8)
        aux_loss = jnp.std(expert_importance) / (jnp.mean(expert_importance) + 1e-8)
        
        return output, top_k_indices, aux_loss


class GraphIntegratedTransformerBlock(hk.Module):
    """
    Transformer block with integrated graph reasoning.
    
    Combines standard self-attention with graph-based reasoning,
    enabling both sequence modeling and relational understanding.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout_rate: float = 0.1,
        use_graph_neurons: bool = True,
        max_nodes: int = 64,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.dropout_rate = dropout_rate
        self.use_graph_neurons = use_graph_neurons
        self.max_nodes = max_nodes
        
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Apply transformer block with graph integration.
        
        Args:
            x: Input features [batch, seq_len, d_model]
            mask: Optional attention mask
            is_training: Whether in training mode
            
        Returns:
            output: [batch, seq_len, d_model]
            aux_info: Dictionary with attention weights, graph structure, etc.
        """
        aux_info = {}
        
        # Standard self-attention
        attn_output = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            w_init=hk.initializers.VarianceScaling(1.0),
            name="self_attention"
        )(x, x, x, mask=mask)
        
        norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm1")
        x = norm1(x + attn_output)
        
        # Graph reasoning (if enabled)
        if self.use_graph_neurons:
            # Build dynamic graph from current features
            graph_builder = DynamicGraphBuilder(
                self.d_model, self.max_nodes, name="graph_builder"
            )
            node_features, adjacency = graph_builder(x)
            aux_info["adjacency"] = adjacency
            
            # Apply graph neuron
            graph_neuron = GraphNeuron(
                self.d_model, self.num_heads, self.dropout_rate,
                use_ffn=False,  # Use shared FFN below
                name="graph_neuron"
            )
            graph_output = graph_neuron(node_features, adjacency, is_training)
            
            # Project back to sequence length if needed
            if graph_output.shape[1] != x.shape[1]:
                # Simple upsampling via tiling (could use more sophisticated methods)
                graph_output = jnp.tile(
                    graph_output[:, :1, :], 
                    (1, x.shape[1] // graph_output.shape[1] + 1, 1)
                )[:, :x.shape[1], :]
            
            # Combine with residual
            combine_gate = hk.Linear(1, name="graph_combine_gate")
            gate = jax.nn.sigmoid(combine_gate(x))
            x = gate * x + (1 - gate) * graph_output
            
            norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm2")
            x = norm2(x)
        
        # FFN
        ffn = hk.Sequential([
            hk.Linear(self.d_model * self.ffn_expansion, name="ffn_up"),
            jax.nn.gelu,
            hk.Linear(self.d_model, name="ffn_down"),
        ])
        ffn_output = ffn(x)
        if is_training and self.dropout_rate > 0:
            ffn_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ffn_output)
        
        norm3 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="norm3")
        output = norm3(x + ffn_output)
        
        return output, aux_info


# Factory functions for easy creation
def create_graph_neuron(config: GraphConfig) -> GraphNeuron:
    """Factory to create GraphNeuron from src.config."""
    return GraphNeuron(
        d_model=config.d_model,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
    )


def create_multi_hop_reasoner(config: GraphConfig) -> MultiHopGraphReasoner:
    """Factory to create MultiHopGraphReasoner from src.config."""
    return MultiHopGraphReasoner(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_hops=config.num_hops,
        dropout_rate=config.dropout_rate,
    )


def create_graph_moe(config: GraphConfig, num_experts: int = 8, top_k: int = 2) -> GraphMoE:
    """Factory to create GraphMoE from src.config."""
    return GraphMoE(
        d_model=config.d_model,
        num_experts=num_experts,
        top_k=top_k,
        use_relational_routing=config.enable_relational_routing,
        dropout_rate=config.dropout_rate,
    )


# Utility functions for graph operations
def compute_graph_loss(
    predicted_adjacency: jnp.ndarray,
    target_adjacency: jnp.ndarray,
    edge_weight: float = 1.0
) -> float:
    """
    Compute loss for graph structure prediction.
    
    Args:
        predicted_adjacency: Predicted adjacency matrix
        target_adjacency: Ground truth adjacency
        edge_weight: Weight for edge prediction loss
        
    Returns:
        Graph structure loss
    """
    # Binary cross-entropy for edge prediction
    eps = 1e-8
    bce_loss = -(
        target_adjacency * jnp.log(predicted_adjacency + eps) +
        (1 - target_adjacency) * jnp.log(1 - predicted_adjacency + eps)
    )
    return edge_weight * jnp.mean(bce_loss)


def compute_graph_accuracy(
    predicted_adjacency: jnp.ndarray,
    target_adjacency: jnp.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute accuracy for graph structure prediction.
    
    Args:
        predicted_adjacency: Predicted adjacency (probabilities)
        target_adjacency: Ground truth adjacency (binary)
        threshold: Threshold for converting probs to binary
        
    Returns:
        Accuracy score
    """
    predicted_binary = (predicted_adjacency > threshold).astype(jnp.float32)
    correct = jnp.sum(predicted_binary == target_adjacency)
    total = jnp.prod(jnp.array(target_adjacency.shape))
    return correct / total
