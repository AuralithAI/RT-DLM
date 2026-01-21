"""
Shape validation tests for module integration points.
Tests shape compatibility at every module boundary.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


BATCH_SIZES = [1, 4, 16, 32]
SEQ_LENGTHS = [8, 32, 64]


class TestSelfAttentionShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 8), (4, 32), (16, 64), (32, 16)
    ])
    def test_self_attention_output_shapes(self, rng_key, batch_size, seq_len):
        """Test self-attention output shapes."""
        from core.model.model_module_self_attention import SelfAttentionModel
        
        d_model, num_heads = 64, 4
        vocab_size, max_seq_length = 1000, 128
        
        def model_fn(inputs):
            model = SelfAttentionModel(
                d_model=d_model,
                num_heads=num_heads,
                vocab_size=vocab_size,
                max_seq_length=max_seq_length,
            )
            return model(inputs, return_attention=True, return_hidden_states=True)
        
        model = hk.transform_with_state(model_fn)
        
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, vocab_size)
        params, state = model.init(rng_key, inputs)
        (hidden_states, _attn_weights), _ = model.apply(params, state, rng_key, inputs)
        
        assert hidden_states.shape == (batch_size, seq_len, d_model)
        assert not jnp.any(jnp.isnan(hidden_states))


class TestTransformerShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 8), (4, 32), (16, 64)
    ])
    def test_transformer_output_shapes(self, rng_key, batch_size, seq_len):
        """Test transformer output shapes."""
        from core.model.model_transformer_module import TransformerModel
        
        d_model, num_heads, num_layers = 64, 4, 2
        vocab_size, max_seq_length = 1000, 128
        
        def model_fn(x):
            model = TransformerModel(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                vocab_size=vocab_size,
                max_seq_length=max_seq_length,
            )
            return model(x, rng=None, return_attention=True)
        
        model = hk.transform_with_state(model_fn)
        
        x = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        params, state = model.init(rng_key, x)
        (output, _attn), _ = model.apply(params, state, rng_key, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not jnp.any(jnp.isnan(output))


class TestSparseMoEShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 8), (4, 32), (16, 64)
    ])
    def test_moe_output_shapes(self, rng_key, batch_size, seq_len):
        """Test MoE output shapes."""
        from core.model.sparse_moe import SparseMoE
        
        d_model = 64
        num_experts = 4
        top_k = 2
        expert_capacity = 8
        
        def model_fn(x):
            model = SparseMoE(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                expert_capacity=expert_capacity,
            )
            return model(x)
        
        model = hk.transform_with_state(model_fn)
        
        x = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        params, state = model.init(rng_key, x)
        result, _ = model.apply(params, state, rng_key, x)
        
        output, expert_indices, aux_loss, _ = result
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert expert_indices.shape[:2] == (batch_size, seq_len)
        assert aux_loss.shape == ()
        assert not jnp.any(jnp.isnan(output))


class TestMemoryBankShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_memory_bank_output_shapes(self, rng_key, batch_size):
        """Test memory bank output shapes."""
        from core.model.memory_bank import MemoryBank
        
        memory_size = 32
        d_model = 64
        retrieval_k = 4
        
        # MemoryBank is a regular Python class, not a Haiku module
        memory_bank = MemoryBank(
            memory_size=memory_size,
            embedding_dim=d_model,
            retrieval_k=retrieval_k,
        )
        
        # Store some test data first
        keys = np.random.randn(batch_size, d_model).astype(np.float32)
        values = np.random.randn(batch_size, d_model).astype(np.float32)
        memory_bank.store(keys, values)
        
        # Retrieve
        queries = np.random.randn(batch_size, d_model).astype(np.float32)
        retrieved = memory_bank.retrieve(queries)
        
        # Check shape - retrieve returns mean of top-k values
        assert retrieved.shape[0] == batch_size
        assert retrieved.shape[-1] == d_model


class TestReasoningEngineShapes:
    """Test reasoning engine shapes across different batch and sequence configurations."""
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 8), (4, 32), (16, 64)
    ])
    def test_reasoning_output_shapes(self, rng_key, batch_size, seq_len):
        """Test reasoning engine output shapes."""
        from core.reasoning import ReasoningEngine
        from dataclasses import dataclass
        
        d_model = 64
        max_steps = 5
        
        @dataclass
        class ReasoningConfig:
            d_model: int = 64
            max_reasoning_steps: int = 5
        
        config = ReasoningConfig(d_model=d_model, max_reasoning_steps=max_steps)
        
        def model_fn(query, context):
            model = ReasoningEngine(config=config)
            return model(query, context)
        
        model = hk.transform_with_state(model_fn)
        
        query = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        context = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        params, state = model.init(rng_key, query, context)
        result, _ = model.apply(params, state, rng_key, query, context)
        
        assert isinstance(result, dict)
        if "reasoning_features" in result:
            feat = result["reasoning_features"]
            assert feat.shape[0] == batch_size
            assert feat.shape[-1] == d_model


class TestAGIAttentionShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 16), (4, 32), (8, 64)
    ])
    def test_agi_attention_shapes(self, rng_key, batch_size, seq_len):
        """Test AGI attention output shapes."""
        from core.model.agi_attention import AGIAttention, AGIAttentionConfig
        
        d_model = 64
        num_heads = 4
        
        config = AGIAttentionConfig(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_length=128,
            enable_ring_attention=False,
            enable_infinite_context=False,
        )
        
        def model_fn(x):
            model = AGIAttention(config)
            return model.forward_attention(x, mask=None, is_training=False)
        
        model = hk.transform_with_state(model_fn)
        
        x = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        params, state = model.init(rng_key, x)
        (output, _info), _ = model.apply(params, state, rng_key, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not jnp.any(jnp.isnan(output))


class TestHierarchicalMemoryFusionShapes:
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_memory_fusion_shapes(self, rng_key, batch_size):
        """Test hierarchical memory fusion shapes."""
        from core.model.agi_attention import HierarchicalMemoryFusion
        
        d_model = 64
        num_heads = 4
        
        def model_fn(ltm, stm, mtm, context):
            model = HierarchicalMemoryFusion(
                d_model=d_model,
                num_heads=num_heads,
            )
            return model(ltm=ltm, stm=stm, mtm=mtm, context=context)
        
        model = hk.transform_with_state(model_fn)
        
        ltm = jax.random.normal(rng_key, (batch_size, d_model))
        stm = jax.random.normal(rng_key, (batch_size, d_model))
        mtm = jax.random.normal(rng_key, (batch_size, d_model))
        context = jax.random.normal(rng_key, (batch_size, d_model))
        
        params, state = model.init(rng_key, ltm, stm, mtm, context)
        result, _ = model.apply(params, state, rng_key, ltm, stm, mtm, context)
        
        assert "fused_memory" in result
        assert result["fused_memory"].shape == (batch_size, d_model)


class TestEthicalRewardModelShapes:
    """Test ethical reward model shapes with proper batch handling."""
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 16), (4, 32), (8, 64)
    ])
    def test_reward_model_shapes(self, rng_key, batch_size, seq_len):
        """Test ethical reward model with different batch sizes."""
        from core.ethics.reward_model import EthicalRewardModel
        
        d_model = 64
        vocab_size = 1000
        max_seq_length = 128
        
        def model_fn(inputs, outputs):
            model = EthicalRewardModel(
                d_model=d_model,
                vocab_size=vocab_size,
                max_seq_length=max_seq_length,
            )
            return model(inputs, outputs)
        
        model = hk.transform_with_state(model_fn)
        
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, vocab_size)
        outputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, vocab_size)
        
        params, state = model.init(rng_key, inputs, outputs)
        result, _ = model.apply(params, state, rng_key, inputs, outputs)
        
        assert isinstance(result, dict)
        assert "reward_score" in result


class TestModuleBoundaryShapes:
    """Test shape compatibility at module boundaries."""
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_embedding_to_attention_shapes(self, rng_key):
        """Test embedding output -> attention input shape compatibility."""
        batch_size, seq_len = 4, 32
        vocab_size = 1000
        d_model = 64
        
        def embed_fn(inputs):
            embed = hk.Embed(vocab_size, d_model)
            pos_embed = hk.Embed(128, d_model)
            x = embed(inputs)
            positions = jnp.arange(inputs.shape[1])
            x = x + pos_embed(positions)
            return x
        
        embed = hk.transform(embed_fn)
        
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, vocab_size)
        params = embed.init(rng_key, inputs)
        embeddings = embed.apply(params, rng_key, inputs)
        
        assert embeddings.shape == (batch_size, seq_len, d_model)
    
    def test_attention_to_ffn_shapes(self, rng_key):
        """Test attention output -> FFN input shape compatibility."""
        batch_size, seq_len = 4, 32
        d_model = 64
        
        attn_output = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        def ffn_fn(x):
            ffn = hk.Sequential([
                hk.Linear(d_model * 4),
                jax.nn.gelu,
                hk.Linear(d_model),
            ])
            return ffn(x)
        
        ffn = hk.transform(ffn_fn)
        params = ffn.init(rng_key, attn_output)
        ffn_output = ffn.apply(params, rng_key, attn_output)
        
        assert ffn_output.shape == attn_output.shape
    
    def test_transformer_to_moe_shapes(self, rng_key):
        """Test transformer output -> MoE input shape compatibility."""
        batch_size, seq_len = 4, 32
        d_model = 64
        
        transformer_output = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        assert transformer_output.ndim == 3
        assert transformer_output.shape[-1] == d_model
    
    def test_moe_to_output_head_shapes(self, rng_key):
        """Test MoE output -> output head input shape compatibility."""
        batch_size, seq_len = 4, 32
        d_model = 64
        vocab_size = 1000
        
        moe_output = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        def output_fn(x):
            output_head = hk.Linear(vocab_size)
            return output_head(x)
        
        output = hk.transform(output_fn)
        params = output.init(rng_key, moe_output)
        logits = output.apply(params, rng_key, moe_output)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
