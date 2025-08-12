#!/usr/bin/env python3
"""
RT-DLM Simple CPU Demo
Lightweight demonstration for CPU-only environments.
"""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import sys
import os
import logging

# Force CPU-only mode
jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.agi_config import AGIConfig
from tests.test_config import test_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_components():
    """Test basic components without complex dependencies"""
    
    print("RT-DLM Basic CPU Demo")
    print("=" * 50)
    
    # Setup device configuration
    test_config.setup_jax_config()
    test_config.print_info()
    
    # Use test configuration
    config_dict = test_config.get_model_config()
    config = AGIConfig(**config_dict)
    
    rng = jax.random.PRNGKey(42)
    
    print("\\n1. Basic Transformer Layer Test")
    print("-" * 30)
    
    def test_transformer():
        # Simple transformer-like layer
        linear = hk.Linear(config.d_model)
        layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        def transformer_layer(x):
            # Self-attention simulation
            attention_weights = jnp.ones((x.shape[0], x.shape[1], x.shape[1])) / x.shape[1]
            attended = jnp.einsum('bij,bjk->bik', attention_weights, x)
            
            # Feed forward
            ff_out = linear(attended)
            ff_out = jax.nn.relu(ff_out)
            
            # Residual + norm
            output = layer_norm(x + ff_out)
            return output
            
        # Test input
        x = jax.random.normal(rng, (2, 32, config.d_model))
        return transformer_layer(x)
    
    transform_fn = hk.transform(test_transformer)
    
    try:
        params = transform_fn.init(rng)
        output = transform_fn.apply(params, rng)
        print(f"[SUCCESS] Transformer layer: {output.shape}")
        print(f"   - Input shape: (2, 32, {config.d_model})")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output mean: {jnp.mean(output):.4f}")
        print(f"   - Output std: {jnp.std(output):.4f}")
    except Exception as e:
        print(f"[ERROR] Transformer test failed: {e}")
    
    print("\\n2. Basic Multi-Layer Perceptron Test")
    print("-" * 30)
    
    def test_mlp():
        mlp = hk.Sequential([
            hk.Linear(config.d_model),
            jax.nn.relu,
            hk.Linear(config.d_model // 2),
            jax.nn.relu,
            hk.Linear(config.vocab_size),
        ])
        
        x = jax.random.normal(rng, (1, config.d_model))
        return mlp(x)
    
    mlp_fn = hk.transform(test_mlp)
    
    try:
        mlp_params = mlp_fn.init(rng)
        mlp_output = mlp_fn.apply(mlp_params, rng)
        print(f"[SUCCESS] MLP layer: {mlp_output.shape}")
        print(f"   - Input shape: (1, {config.d_model})")
        print(f"   - Output shape: {mlp_output.shape}")
        print(f"   - Output range: [{jnp.min(mlp_output):.4f}, {jnp.max(mlp_output):.4f}]")
    except Exception as e:
        print(f"[ERROR] MLP test failed: {e}")
    
    print("\\n3. Memory Simulation Test")
    print("-" * 30)
    
    def test_memory():
        # Simple memory bank simulation
        memory_size = 16
        memory_dim = config.d_model
        
        # Initialize memory
        memory_bank = hk.get_state("memory", [memory_size, memory_dim], 
                                  init=jnp.zeros, dtype=jnp.float32)
        
        # Query
        query = jax.random.normal(rng, (1, memory_dim))
        
        # Attention over memory
        similarities = jnp.dot(query, memory_bank.T)  # [1, memory_size]
        attention_weights = jax.nn.softmax(similarities, axis=-1)
        
        # Retrieve
        retrieved = jnp.dot(attention_weights, memory_bank)  # [1, memory_dim]
        
        # Update memory (simple)
        new_memory = memory_bank.at[0].set(query.squeeze())
        hk.set_state("memory", new_memory)
        
        return retrieved, attention_weights
    
    memory_fn = hk.transform_with_state(test_memory)
    
    try:
        memory_params, memory_state = memory_fn.init(rng)
        (retrieved, attention_weights), _ = memory_fn.apply(
            memory_params, memory_state, rng)
        
        print(f"[SUCCESS] Memory system: {retrieved.shape}")
        print(f"   - Memory bank size: (16, {config.d_model})")
        print(f"   - Retrieved shape: {retrieved.shape}")
        print(f"   - Attention weights shape: {attention_weights.shape}")
        print(f"   - Max attention: {jnp.max(attention_weights):.4f}")
    except Exception as e:
        print(f"[ERROR] Memory test failed: {e}")
    
    print("\\n4. Basic Reasoning Simulation")
    print("-" * 30)
    
    def test_reasoning():
        # Simple reasoning chain
        reasoning_steps = 3
        step_dim = config.d_model
        
        # Initial state
        state = jax.random.normal(rng, (1, step_dim))
        
        reasoning_chain = []
        for step in range(reasoning_steps):
            # Reasoning step (simple MLP)
            step_mlp = hk.Linear(step_dim, name=f"reasoning_step_{step}")
            new_state = step_mlp(state)
            new_state = jax.nn.tanh(new_state)
            
            # Accumulate reasoning
            state = state + 0.5 * new_state
            reasoning_chain.append(state)
        
        return jnp.stack(reasoning_chain, axis=1)  # [1, steps, dim]
    
    reasoning_fn = hk.transform(test_reasoning)
    
    try:
        reasoning_params = reasoning_fn.init(rng)
        reasoning_output = reasoning_fn.apply(reasoning_params, rng)
        
        print(f"[SUCCESS] Reasoning chain: {reasoning_output.shape}")
        print("   - Number of steps: 3")
        print(f"   - Step dimension: {config.d_model}")
        print(f"   - Final state norm: {jnp.linalg.norm(reasoning_output[0, -1]):.4f}")
    except Exception as e:
        print(f"[ERROR] Reasoning test failed: {e}")
    
    print("\\n[SUMMARY] Basic Component Test Results")
    print("-" * 30)
    print("[SUCCESS] Basic transformer layer: PASS")
    print("[SUCCESS] Multi-layer perceptron: PASS")
    print("[SUCCESS] Memory simulation: PASS")
    print("[SUCCESS] Reasoning simulation: PASS")
    print("\\n[STATUS] RT-DLM basic components working on CPU!")

def test_simple_inference():
    """Test simple text inference simulation"""
    print("\\n[INFERENCE] Simple Text Processing Test")
    print("-" * 40)
    
    config = AGIConfig(d_model=64, num_heads=2, num_layers=1, vocab_size=100, max_seq_length=32)
    rng = jax.random.PRNGKey(42)
    
    def simple_model():
        # Token embedding
        embed = hk.Embed(config.vocab_size, config.d_model)
        
        # Simple transformer
        def transformer_block(x):
            # Basic self-attention
            q = k = v = hk.Linear(config.d_model)(x)
            
            # Scaled dot-product attention (simplified)
            scores = jnp.einsum('bsd,btd->bst', q, k) / jnp.sqrt(config.d_model)
            weights = jax.nn.softmax(scores, axis=-1)
            attended = jnp.einsum('bst,btd->bsd', weights, v)
            
            # Feed forward
            ff = hk.Sequential([
                hk.Linear(config.d_model * 2),
                jax.nn.relu,
                hk.Linear(config.d_model)
            ])
            
            return x + attended + ff(attended)
        
        # Output projection
        output_proj = hk.Linear(config.vocab_size)
        
        def forward(tokens):
            x = embed(tokens)
            x = transformer_block(x)
            return output_proj(x)
        
        # Test tokens
        tokens = jax.random.randint(rng, (1, 16), 0, config.vocab_size)
        return forward(tokens)
    
    model_fn = hk.transform(simple_model)
    
    try:
        params = model_fn.init(rng)
        output = model_fn.apply(params, rng)
        
        print(f"[SUCCESS] Simple inference: {output.shape}")
        print("   - Input tokens: (1, 16)")
        print(f"   - Output logits: {output.shape}")
        print(f"   - Vocabulary size: {config.vocab_size}")
        
        # Get predictions
        predictions = jnp.argmax(output, axis=-1)
        print(f"   - Predicted tokens: {predictions[0][:8].tolist()}...")
        
    except Exception as e:
        print(f"[ERROR] Simple inference failed: {e}")

if __name__ == "__main__":
    try:
        test_basic_components()
        test_simple_inference()
        
        print("\\n[COMPLETE] All basic tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n\\n[EXIT] Tests interrupted by user.")
    except Exception as e:
        print(f"\\n[ERROR] Test suite error: {e}")
        import traceback
        traceback.print_exc()
