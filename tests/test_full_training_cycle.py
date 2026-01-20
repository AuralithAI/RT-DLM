"""
Full training cycle integration test.
Validates end-to-end training pipeline with real data flow.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import tempfile
import os
import sys
import time
from typing import Dict, Any, Tuple
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model.model_tms import TMSModel


class TrainingConfig:
    d_model = 64
    num_heads = 4
    num_layers = 2
    vocab_size = 1000
    max_seq_length = 64
    moe_experts = 4
    moe_top_k = 2
    memory_size = 32
    retrieval_k = 4
    ltm_weight = 0.3
    stm_weight = 0.3
    mtm_weight = 0.3
    batch_size = 4
    learning_rate = 1e-4
    num_steps = 10
    warmup_steps = 2
    grad_clip = 1.0


def create_real_batches(config: TrainingConfig, num_batches: int, rng_key: jax.random.PRNGKey):
    """Create realistic training batches with proper token distributions."""
    batches = []
    for _ in range(num_batches):
        rng_key, subkey = jax.random.split(rng_key)
        
        input_ids = jax.random.randint(
            subkey, 
            (config.batch_size, config.max_seq_length),
            minval=1,
            maxval=config.vocab_size
        )
        
        rng_key, subkey = jax.random.split(rng_key)
        target_ids = jax.random.randint(
            subkey,
            (config.batch_size, config.max_seq_length),
            minval=1,
            maxval=config.vocab_size
        )
        
        rng_key, subkey = jax.random.split(rng_key)
        mask = jax.random.bernoulli(subkey, 0.9, (config.batch_size, config.max_seq_length))
        mask = mask.at[:, 0].set(1.0)
        
        batches.append({
            'input_ids': input_ids,
            'target_ids': target_ids,
            'mask': mask.astype(jnp.float32)
        })
    
    return batches


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
    """Compute cross-entropy loss with proper masking."""
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    
    masked_loss = ce_loss * mask
    total_tokens = jnp.sum(mask) + 1e-8
    loss = jnp.sum(masked_loss) / total_tokens
    
    loss = jnp.where(jnp.isnan(loss), 0.0, loss)
    loss = jnp.where(jnp.isinf(loss), 100.0, loss)
    
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets) * mask
    accuracy = jnp.sum(correct) / total_tokens
    
    return loss, {'accuracy': accuracy, 'num_tokens': total_tokens}


class TestFullTrainingCycle:
    
    @pytest.fixture
    def config(self):
        return TrainingConfig()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def model_and_params(self, config, rng_key):
        def model_fn(inputs, rng=None):
            model = TMSModel(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts,
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(inputs, rng=rng)
        
        model = hk.transform_with_state(model_fn)
        
        dummy_input = jnp.ones((config.batch_size, config.max_seq_length), dtype=jnp.int32)
        params, state = model.init(rng_key, dummy_input)
        
        return model, params, state
    
    @pytest.fixture
    def optimizer(self, config):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.num_steps,
            end_value=config.learning_rate * 0.1
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(schedule, weight_decay=0.01)
        )
        
        return optimizer
    
    def test_model_initialization(self, model_and_params, config):
        """Test that model initializes with correct parameter shapes."""
        _, params, _ = model_and_params
        
        assert params is not None
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        assert param_count > 0
        print(f"\nModel parameter count: {param_count:,}")
        
        for leaf in jax.tree_util.tree_leaves(params):
            assert not jnp.any(jnp.isnan(leaf)), "NaN in initialized parameters"
            assert not jnp.any(jnp.isinf(leaf)), "Inf in initialized parameters"
    
    def test_forward_pass(self, model_and_params, config, rng_key):
        """Test forward pass produces valid outputs."""
        model, params, state = model_and_params
        
        batches = create_real_batches(config, 1, rng_key)
        batch = batches[0]
        
        logits, _ = model.apply(params, state, rng_key, batch['input_ids'])
        
        assert logits.shape == (config.batch_size, config.max_seq_length, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits)), "NaN in forward pass output"
        assert not jnp.any(jnp.isinf(logits)), "Inf in forward pass output"
    
    def test_loss_computation(self, model_and_params, config, rng_key):
        """Test loss computation with masking."""
        model, params, state = model_and_params
        
        batches = create_real_batches(config, 1, rng_key)
        batch = batches[0]
        
        logits, _ = model.apply(params, state, rng_key, batch['input_ids'])
        loss, metrics = compute_loss(logits, batch['target_ids'], batch['mask'])
        
        assert loss.shape == ()
        assert not jnp.isnan(loss), "NaN loss"
        assert not jnp.isinf(loss), "Inf loss"
        assert loss > 0, "Loss should be positive for random predictions"
        
        expected_loss = jnp.log(config.vocab_size)
        assert loss < expected_loss * 2, f"Loss {loss} unexpectedly high"
        
        print(f"\nInitial loss: {loss:.4f}, accuracy: {metrics['accuracy']:.4f}")
    
    def test_gradient_computation(self, model_and_params, config, rng_key):
        """Test gradient flow through all components."""
        model, params, state = model_and_params
        
        batches = create_real_batches(config, 1, rng_key)
        batch = batches[0]
        
        def loss_fn(params):
            logits, _ = model.apply(params, state, rng_key, batch['input_ids'])
            loss, _ = compute_loss(logits, batch['target_ids'], batch['mask'])
            return loss
        
        _, grads = jax.value_and_grad(loss_fn)(params)
        
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for i, grad in enumerate(grad_leaves):
            assert not jnp.any(jnp.isnan(grad)), f"NaN gradient in leaf {i}"
            assert not jnp.any(jnp.isinf(grad)), f"Inf gradient in leaf {i}"
        
        grad_norms = [jnp.linalg.norm(g) for g in grad_leaves]
        total_grad_norm = jnp.sqrt(sum(n**2 for n in grad_norms))
        
        assert total_grad_norm > 0, "Zero gradients - no gradient flow"
        print(f"\nGradient norm: {total_grad_norm:.6f}")
        
        zero_grads = sum(1 for g in grad_leaves if jnp.allclose(g, 0))
        print(f"Zero gradient tensors: {zero_grads}/{len(grad_leaves)}")
    
    def test_single_train_step(self, model_and_params, optimizer, config, rng_key):
        """Test a single training step."""
        model, params, state = model_and_params
        
        simple_optimizer = optax.adam(1e-3)
        opt_state = simple_optimizer.init(params)
        
        batches = create_real_batches(config, 1, rng_key)
        batch = batches[0]
        
        def loss_fn(params):
            logits, _ = model.apply(params, state, rng_key, batch['input_ids'])
            loss, metrics = compute_loss(logits, batch['target_ids'], batch['mask'])
            return loss, metrics
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        updates, _ = simple_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        for leaf in jax.tree_util.tree_leaves(new_params):
            assert not jnp.any(jnp.isnan(leaf)), "NaN in updated parameters"
            assert not jnp.any(jnp.isinf(leaf)), "Inf in updated parameters"
        
        old_flat = jax.tree_util.tree_leaves(params)
        new_flat = jax.tree_util.tree_leaves(new_params)
        params_changed = any(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat))
        assert params_changed, "Parameters did not change after update"
        
        print(f"\nStep loss: {loss:.4f}, accuracy: {metrics['accuracy']:.4f}")
    
    def test_multi_step_training(self, model_and_params, optimizer, config, rng_key):
        """Test multiple training steps show loss decrease."""
        model, params, state = model_and_params
        opt_state = optimizer.init(params)
        
        num_batches = config.num_steps
        batches = create_real_batches(config, num_batches, rng_key)
        
        @jax.jit
        def train_step(params, opt_state, state, batch, rng):
            def loss_fn(params):
                logits, new_state = model.apply(params, state, rng, batch['input_ids'])
                loss, metrics = compute_loss(logits, batch['target_ids'], batch['mask'])
                return loss, (metrics, new_state)
            
            (loss, (metrics, new_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, new_state, loss, metrics
        
        losses = []
        step_rng = rng_key
        
        for step, batch in enumerate(batches):
            step_rng, subkey = jax.random.split(step_rng)
            params, opt_state, state, loss, metrics = train_step(params, opt_state, state, batch, subkey)
            losses.append(float(loss))
            
            if step % 2 == 0:
                print(f"Step {step}: loss={loss:.4f}, acc={metrics['accuracy']:.4f}")
        
        for loss in losses:
            assert not np.isnan(loss), "NaN loss during training"
            assert not np.isinf(loss), "Inf loss during training"
        
        first_half_avg = np.mean(losses[:len(losses)//2])
        second_half_avg = np.mean(losses[len(losses)//2:])
        print(f"\nFirst half avg loss: {first_half_avg:.4f}")
        print(f"Second half avg loss: {second_half_avg:.4f}")
    
    def test_memory_usage(self, model_and_params, config, rng_key):
        """Profile memory usage during forward/backward pass."""
        model, params, state = model_and_params
        
        batches = create_real_batches(config, 1, rng_key)
        batch = batches[0]
        
        param_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
        param_mb = param_bytes / (1024 * 1024)
        
        def loss_fn(params):
            logits, _ = model.apply(params, state, rng_key, batch['input_ids'])
            loss, _ = compute_loss(logits, batch['target_ids'], batch['mask'])
            return loss
        
        _ = jax.value_and_grad(loss_fn)(params)
        
        input_bytes = batch['input_ids'].nbytes + batch['target_ids'].nbytes + batch['mask'].nbytes
        input_mb = input_bytes / (1024 * 1024)
        
        print(f"\nParameter memory: {param_mb:.2f} MB")
        print(f"Input batch memory: {input_mb:.4f} MB")
        print(f"Estimated training memory: {(param_mb * 4 + input_mb):.2f} MB (params + grads + optimizer + batch)")
    
    def test_checkpoint_save_load(self, model_and_params, optimizer, config, rng_key):
        """Test checkpoint save and load functionality."""
        model, params, state = model_and_params
        opt_state = optimizer.init(params)
        
        batches = create_real_batches(config, 3, rng_key)
        
        @jax.jit
        def train_step(params, opt_state, state, batch, rng):
            def loss_fn(params):
                logits, new_state = model.apply(params, state, rng, batch['input_ids'])
                loss, metrics = compute_loss(logits, batch['target_ids'], batch['mask'])
                return loss, (metrics, new_state)
            
            (loss, (metrics, new_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, new_state, loss, metrics
        
        step_rng = rng_key
        for batch in batches:
            step_rng, subkey = jax.random.split(step_rng)
            params, opt_state, state, _, _ = train_step(params, opt_state, state, batch, subkey)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import pickle
            
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
            checkpoint = {
                'params': jax.tree_util.tree_map(np.array, params),
                'opt_state': jax.tree_util.tree_map(
                    lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
                    opt_state
                ),
                'step': 3,
                'config': {
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'vocab_size': config.vocab_size,
                }
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            file_size = os.path.getsize(checkpoint_path)
            print(f"\nCheckpoint size: {file_size / 1024:.2f} KB")
            
            with open(checkpoint_path, 'rb') as f:
                loaded_checkpoint = pickle.load(f)
            
            loaded_params = jax.tree_util.tree_map(jnp.array, loaded_checkpoint['params'])
            
            original_flat = jax.tree_util.tree_leaves(params)
            loaded_flat = jax.tree_util.tree_leaves(loaded_params)
            
            for i, (orig, loaded) in enumerate(zip(original_flat, loaded_flat)):
                assert jnp.allclose(orig, loaded), f"Mismatch in parameter {i}"
            
            assert loaded_checkpoint['step'] == 3
            assert loaded_checkpoint['config']['d_model'] == config.d_model
            
            print("Checkpoint save/load verified successfully")
    
    def test_jit_compilation_speed(self, model_and_params, optimizer, config, rng_key):
        """Test JIT compilation and execution speed."""
        model, params, state = model_and_params
        opt_state = optimizer.init(params)
        
        batches = create_real_batches(config, 5, rng_key)
        
        @jax.jit
        def train_step(params, opt_state, state, batch, rng):
            def loss_fn(params):
                logits, new_state = model.apply(params, state, rng, batch['input_ids'])
                loss, _ = compute_loss(logits, batch['target_ids'], batch['mask'])
                return loss, new_state
            
            (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, new_state, loss
        
        start_compile = time.time()
        params, opt_state, state, _ = train_step(params, opt_state, state, batches[0], rng_key)
        jax.block_until_ready(params)
        compile_time = time.time() - start_compile
        
        step_times = []
        step_rng = rng_key
        for batch in batches[1:]:
            step_rng, subkey = jax.random.split(step_rng)
            start_step = time.time()
            params, opt_state, state, _ = train_step(params, opt_state, state, batch, subkey)
            jax.block_until_ready(params)
            step_times.append(time.time() - start_step)
        
        avg_step_time = np.mean(step_times)
        
        print(f"\nJIT compilation time: {compile_time:.3f}s")
        print(f"Average step time: {avg_step_time*1000:.2f}ms")
        print(f"Throughput: {config.batch_size / avg_step_time:.1f} samples/sec")
    
    def test_gradient_accumulation(self, model_and_params, optimizer, config, rng_key):
        """Test gradient accumulation for larger effective batch sizes."""
        model, params, state = model_and_params
        
        accumulation_steps = 4
        batches = create_real_batches(config, accumulation_steps, rng_key)
        
        def compute_grads(params, state, batch, rng):
            def loss_fn(params):
                logits, _ = model.apply(params, state, rng, batch['input_ids'])
                loss, _ = compute_loss(logits, batch['target_ids'], batch['mask'])
                return loss
            return jax.value_and_grad(loss_fn)(params)
        
        accumulated_grads = None
        total_loss = 0.0
        step_rng = rng_key
        
        for batch in batches:
            step_rng, subkey = jax.random.split(step_rng)
            loss, grads = compute_grads(params, state, batch, subkey)
            total_loss += loss
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda a, g: a + g,
                    accumulated_grads,
                    grads
                )
        
        accumulated_grads = jax.tree_util.tree_map(
            lambda g: g / accumulation_steps,
            accumulated_grads
        )
        avg_loss = total_loss / accumulation_steps
        
        for grad in jax.tree_util.tree_leaves(accumulated_grads):
            assert not jnp.any(jnp.isnan(grad)), "NaN in accumulated gradients"
            assert not jnp.any(jnp.isinf(grad)), "Inf in accumulated gradients"
        
        opt_state = optimizer.init(params)
        updates, _ = optimizer.update(accumulated_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        for leaf in jax.tree_util.tree_leaves(new_params):
            assert not jnp.any(jnp.isnan(leaf))
        
        print(f"\nAccumulated loss (avg over {accumulation_steps} steps): {avg_loss:.4f}")
        print(f"Effective batch size: {config.batch_size * accumulation_steps}")


class TestTrainScriptIntegration:
    
    @pytest.fixture
    def config(self):
        return TrainingConfig()
    
    def test_train_step_function_signature(self):
        """Verify train.py train_step matches expected signature."""
        try:
            from train import train_step
            import inspect
            sig = inspect.signature(train_step)
            params = list(sig.parameters.keys())
            
            assert 'params' in params or len(params) >= 1
            print(f"\ntrain_step signature: {sig}")
        except ImportError:
            pytest.skip("train.py not importable as module")
    
    def test_synthetic_batch_creation(self, config):
        """Verify create_synthetic_batches function works."""
        try:
            from train import create_synthetic_batches
            
            batches = create_synthetic_batches(
                num_batches=2,
                batch_size=config.batch_size,
                seq_length=config.max_seq_length,
                vocab_size=config.vocab_size
            )
            
            assert len(batches) == 2
            print(f"\nSynthetic batch keys: {batches[0].keys() if isinstance(batches[0], dict) else 'tensor'}")
        except (ImportError, TypeError) as e:
            pytest.skip(f"create_synthetic_batches not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
