"""
Tests for Graph-Based Neural Components

Tests the graph neurons module for relational reasoning capabilities.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from core.components.graph_neurons import (
    GraphConfig,
    GraphNeuron,
    GraphAttentionUnit,
    DynamicGraphBuilder,
    RelationalRouter,
    MultiHopGraphReasoner,
    GraphMoE,
    GraphIntegratedTransformerBlock,
    create_graph_neuron,
    create_multi_hop_reasoner,
    compute_graph_loss,
    compute_graph_accuracy,
)


class TestGraphConfig:
    """Test GraphConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GraphConfig()
        assert config.d_model == 384
        assert config.num_heads == 8
        assert config.max_nodes == 64
        assert config.num_hops == 3
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = GraphConfig(d_model=512, num_heads=16, max_nodes=128)
        assert config.d_model == 512
        assert config.num_heads == 16
        assert config.max_nodes == 128


class TestGraphAttentionUnit:
    """Test GraphAttentionUnit module."""
    
    def test_graph_attention_shapes(self):
        """Test that graph attention produces correct output shapes."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        num_nodes = 8
        
        def forward(node_features, adjacency):
            gat = GraphAttentionUnit(d_model, num_heads)
            return gat(node_features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (batch_size, num_nodes, d_model))
        adjacency = jax.random.uniform(rng, (batch_size, num_nodes, num_nodes)) > 0.5
        adjacency = adjacency.astype(jnp.float32)
        
        params = forward_fn.init(rng, node_features, adjacency)
        output, attention_weights = forward_fn.apply(params, rng, node_features, adjacency)
        
        assert output.shape == (batch_size, num_nodes, d_model)
        assert attention_weights.shape == (batch_size, num_heads, num_nodes, num_nodes)
        
    def test_graph_attention_respects_adjacency(self):
        """Test that attention only flows along edges in adjacency matrix."""
        d_model = 32
        num_heads = 2
        batch_size = 1
        num_nodes = 4
        
        def forward(node_features, adjacency):
            gat = GraphAttentionUnit(d_model, num_heads)
            return gat(node_features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (batch_size, num_nodes, d_model))
        
        # Create sparse adjacency (only self-loops)
        adjacency = jnp.eye(num_nodes)[None, :, :]
        
        params = forward_fn.init(rng, node_features, adjacency)
        output, attention_weights = forward_fn.apply(params, rng, node_features, adjacency)
        
        # With only self-loops, attention should be concentrated on diagonal
        # Check that off-diagonal attention is very low
        for h in range(num_heads):
            diag_attn = jnp.diag(attention_weights[0, h])
            off_diag_sum = jnp.sum(attention_weights[0, h]) - jnp.sum(diag_attn)
            assert off_diag_sum < 1e-3, "Off-diagonal attention should be near zero with only self-loops"


class TestGraphNeuron:
    """Test GraphNeuron module."""
    
    def test_graph_neuron_forward(self):
        """Test GraphNeuron forward pass."""
        d_model = 64
        num_heads = 4
        batch_size = 2
        num_nodes = 8
        
        def forward(node_features, adjacency):
            gn = GraphNeuron(d_model, num_heads, use_ffn=True)
            return gn(node_features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (batch_size, num_nodes, d_model))
        adjacency = (jax.random.uniform(rng, (batch_size, num_nodes, num_nodes)) > 0.3).astype(jnp.float32)
        
        params = forward_fn.init(rng, node_features, adjacency)
        output = forward_fn.apply(params, rng, node_features, adjacency)
        
        assert output.shape == node_features.shape
        
    def test_graph_neuron_without_ffn(self):
        """Test GraphNeuron without FFN layer."""
        d_model = 64
        
        def forward(node_features, adjacency):
            gn = GraphNeuron(d_model, use_ffn=False)
            return gn(node_features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (2, 8, d_model))
        adjacency = jnp.ones((2, 8, 8))
        
        params = forward_fn.init(rng, node_features, adjacency)
        output = forward_fn.apply(params, rng, node_features, adjacency)
        
        assert output.shape == node_features.shape


class TestDynamicGraphBuilder:
    """Test DynamicGraphBuilder module."""
    
    def test_graph_builder_shapes(self):
        """Test that graph builder produces correct shapes."""
        d_model = 64
        max_nodes = 16
        batch_size = 2
        seq_len = 32
        
        def forward(embeddings):
            builder = DynamicGraphBuilder(d_model, max_nodes=max_nodes)
            return builder(embeddings)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        embeddings = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = forward_fn.init(rng, embeddings)
        node_features, adjacency = forward_fn.apply(params, rng, embeddings)
        
        assert node_features.shape == (batch_size, max_nodes, d_model)
        assert adjacency.shape == (batch_size, max_nodes, max_nodes)
        
    def test_graph_builder_self_loops(self):
        """Test that graph builder adds self-loops."""
        d_model = 32
        max_nodes = 8
        
        def forward(embeddings):
            builder = DynamicGraphBuilder(d_model, max_nodes=max_nodes, edge_threshold=1.0)
            return builder(embeddings)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        embeddings = jax.random.normal(rng, (1, 8, d_model))
        
        params = forward_fn.init(rng, embeddings)
        _, adjacency = forward_fn.apply(params, rng, embeddings)
        
        # Check self-loops exist
        diagonal = jnp.diag(adjacency[0])
        assert jnp.all(diagonal == 1.0), "Self-loops should be present"


class TestMultiHopGraphReasoner:
    """Test MultiHopGraphReasoner module."""
    
    def test_multi_hop_reasoning(self):
        """Test multi-hop reasoning over graph."""
        d_model = 64
        num_hops = 3
        batch_size = 2
        num_nodes = 8
        
        def forward(node_features, adjacency):
            reasoner = MultiHopGraphReasoner(d_model, num_hops=num_hops)
            return reasoner(node_features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (batch_size, num_nodes, d_model))
        adjacency = (jax.random.uniform(rng, (batch_size, num_nodes, num_nodes)) > 0.3).astype(jnp.float32)
        
        params = forward_fn.init(rng, node_features, adjacency)
        final_features, reasoning_paths = forward_fn.apply(params, rng, node_features, adjacency)
        
        assert final_features.shape == node_features.shape
        assert reasoning_paths.shape == (batch_size, num_hops, num_nodes)
        
    def test_multi_hop_with_query(self):
        """Test multi-hop reasoning with query guidance."""
        d_model = 64
        num_hops = 2
        batch_size = 2
        num_nodes = 8
        
        def forward(node_features, adjacency, query):
            reasoner = MultiHopGraphReasoner(d_model, num_hops=num_hops)
            return reasoner(node_features, adjacency, query=query, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        node_features = jax.random.normal(rng, (batch_size, num_nodes, d_model))
        adjacency = jnp.ones((batch_size, num_nodes, num_nodes))
        query = jax.random.normal(rng, (batch_size, d_model))
        
        params = forward_fn.init(rng, node_features, adjacency, query)
        final_features, reasoning_paths = forward_fn.apply(params, rng, node_features, adjacency, query)
        
        # Reasoning paths should sum to 1 (probability distribution)
        path_sums = jnp.sum(reasoning_paths, axis=-1)
        assert jnp.allclose(path_sums, 1.0, atol=1e-5)


class TestGraphMoE:
    """Test GraphMoE module."""
    
    def test_graph_moe_forward(self):
        """Test GraphMoE forward pass."""
        d_model = 64
        num_experts = 4
        batch_size = 2
        seq_len = 16
        
        def forward(features, adjacency):
            moe = GraphMoE(d_model, num_experts=num_experts, top_k=2)
            return moe(features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        features = jax.random.normal(rng, (batch_size, seq_len, d_model))
        adjacency = (jax.random.uniform(rng, (batch_size, seq_len, seq_len)) > 0.5).astype(jnp.float32)
        
        params = forward_fn.init(rng, features, adjacency)
        output, expert_indices, aux_loss = forward_fn.apply(params, rng, features, adjacency)
        
        assert output.shape == features.shape
        assert expert_indices.shape == (batch_size, seq_len, 2)  # top_k=2
        assert aux_loss.shape == ()  # scalar
        
    def test_graph_moe_without_adjacency(self):
        """Test GraphMoE without graph structure (falls back to standard routing)."""
        d_model = 64
        num_experts = 4
        
        def forward(features):
            moe = GraphMoE(d_model, num_experts=num_experts, use_relational_routing=False)
            return moe(features, adjacency=None, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        features = jax.random.normal(rng, (2, 16, d_model))
        
        params = forward_fn.init(rng, features)
        output, _, _ = forward_fn.apply(params, rng, features)
        
        assert output.shape == features.shape


class TestGraphIntegratedTransformerBlock:
    """Test GraphIntegratedTransformerBlock."""
    
    def test_transformer_block_forward(self):
        """Test transformer block with graph integration."""
        d_model = 64
        batch_size = 2
        seq_len = 16
        
        def forward(x):
            block = GraphIntegratedTransformerBlock(
                d_model, num_heads=4, use_graph_neurons=True
            )
            return block(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = forward_fn.init(rng, x)
        output, aux_info = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        assert "adjacency" in aux_info
        
    def test_transformer_block_without_graph(self):
        """Test transformer block without graph neurons."""
        d_model = 64
        
        def forward(x):
            block = GraphIntegratedTransformerBlock(
                d_model, use_graph_neurons=False
            )
            return block(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (2, 16, d_model))
        
        params = forward_fn.init(rng, x)
        output, aux_info = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        assert "adjacency" not in aux_info


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_graph_neuron(self):
        """Test create_graph_neuron factory."""
        config = GraphConfig(d_model=128, num_heads=8)
        
        def forward(x, adj):
            gn = create_graph_neuron(config)
            return gn(x, adj, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (2, 8, 128))
        adj = jnp.ones((2, 8, 8))
        
        params = forward_fn.init(rng, x, adj)
        output = forward_fn.apply(params, rng, x, adj)
        
        assert output.shape == x.shape
        
    def test_create_multi_hop_reasoner(self):
        """Test create_multi_hop_reasoner factory."""
        config = GraphConfig(d_model=64, num_hops=2)
        
        def forward(x, adj):
            reasoner = create_multi_hop_reasoner(config)
            return reasoner(x, adj, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (2, 8, 64))
        adj = jnp.ones((2, 8, 8))
        
        params = forward_fn.init(rng, x, adj)
        output, paths = forward_fn.apply(params, rng, x, adj)
        
        assert output.shape == x.shape
        assert paths.shape == (2, 2, 8)


class TestUtilityFunctions:
    """Test utility functions for graph operations."""
    
    def test_compute_graph_loss(self):
        """Test graph structure loss computation."""
        predicted = jnp.array([[0.8, 0.2], [0.3, 0.9]])
        target = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        
        loss = compute_graph_loss(predicted, target)
        
        assert loss.shape == ()
        assert loss > 0  # Should have some loss since prediction != target
        
    def test_compute_graph_accuracy(self):
        """Test graph accuracy computation."""
        # Perfect prediction
        predicted = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        target = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        
        accuracy = compute_graph_accuracy(predicted, target, threshold=0.5)
        
        assert accuracy == 1.0
        
        # Imperfect prediction
        predicted_bad = jnp.array([[0.4, 0.6], [0.6, 0.4]])
        accuracy_bad = compute_graph_accuracy(predicted_bad, target, threshold=0.5)
        
        assert accuracy_bad < 1.0


class TestIntegration:
    """Integration tests for graph neural components."""
    
    def test_end_to_end_graph_reasoning(self):
        """Test complete graph reasoning pipeline."""
        d_model = 64
        batch_size = 2
        seq_len = 32
        
        def forward(embeddings):
            # Build graph from embeddings
            builder = DynamicGraphBuilder(d_model, max_nodes=16)
            node_features, adjacency = builder(embeddings)
            
            # Apply graph neurons
            gn = GraphNeuron(d_model)
            processed = gn(node_features, adjacency, is_training=False)
            
            # Multi-hop reasoning
            reasoner = MultiHopGraphReasoner(d_model, num_hops=2)
            final_features, paths = reasoner(processed, adjacency, is_training=False)
            
            return final_features, paths
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        embeddings = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = forward_fn.init(rng, embeddings)
        final_features, paths = forward_fn.apply(params, rng, embeddings)
        
        assert final_features.shape == (batch_size, 16, d_model)
        assert paths.shape == (batch_size, 2, 16)
        
    def test_graph_moe_with_dynamic_graph(self):
        """Test GraphMoE with dynamically constructed graph."""
        d_model = 64
        batch_size = 2
        seq_len = 16
        
        def forward(features):
            # Build dynamic graph
            builder = DynamicGraphBuilder(d_model, max_nodes=seq_len)
            _, adjacency = builder(features)
            
            # Apply graph MoE
            moe = GraphMoE(d_model, num_experts=4)
            return moe(features, adjacency, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        features = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = forward_fn.init(rng, features)
        output, indices, loss = forward_fn.apply(params, rng, features)
        
        assert output.shape == features.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
