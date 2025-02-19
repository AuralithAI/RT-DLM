import haiku as hk
import jax
import jax.numpy as jnp
import optax
import os
import gc
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_tms import TMSModel
from data_utils import DataProcessor, load_data

def train_and_evaluate(config, losses):
    """ Trains the model and returns the final validation loss. """
    
    rng = jax.random.PRNGKey(42)
    # Load dataset
    dataset_path = "data/train_data.txt"
    processor = DataProcessor(vocab_size=config.vocab_size)
    raw_texts = load_data(dataset_path)
    tokenized_texts = [processor.tokenize(text) for text in raw_texts]

    inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=jnp.int32)
    targets = jnp.array(inputs, dtype=jnp.int32)

    # Initialize TMS Model
    def forward_fn(inputs, return_attention=False):
        model = TMSModel(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            moe_experts=config.moe_experts,
            moe_top_k=config.moe_top_k
        )
        return model(inputs, return_attention=return_attention)

    model = hk.transform(forward_fn)
    params = model.init(rng, inputs[:config.batch_size])

    # Optimizer
    warmup_steps = 5000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=2e-6,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=200000,
        end_value=2e-6
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(schedule, weight_decay=1e-3)
    )
    opt_state = optimizer.init(params)

    # Compute loss function
    def compute_loss(params, rng, inputs, targets):
        logits, attn_weights, expert_indices, aux_loss = model.apply(params, rng, inputs, return_attention=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        total_loss = loss + 0.01 * aux_loss
        return total_loss, attn_weights, expert_indices

    def loss_for_gradients(params, rng, inputs, targets):
        loss, _, _ = compute_loss(params, rng, inputs, targets)  
        return loss

    @jax.jit
    def train_step(params, opt_state, rng, inputs, targets):
        loss, attn_weights, expert_indices = compute_loss(params, rng, inputs, targets)
        grads = jax.grad(loss_for_gradients)(params, rng, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, attn_weights, expert_indices, params, opt_state

    # Training loop
    for epoch in range(config.num_epochs):
        gc.collect()
        jax.clear_caches()
        
        for step in range(len(inputs) // config.batch_size):
            batch_start = step * config.batch_size
            batch_end = batch_start + config.batch_size
            batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

            step_rng, rng = jax.random.split(rng)
            loss, _, _, params, opt_state = train_step(params, opt_state, step_rng, batch_inputs, batch_targets)
            losses.append(loss)
            
            print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")
            del batch_inputs, batch_targets
            gc.collect()

    return losses, params  # Return final loss for hyperparameter tuning