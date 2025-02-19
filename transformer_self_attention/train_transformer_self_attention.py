import haiku as hk
import jax
import jax.numpy as jnp
import optax
import os
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from transformer_self_attention.model import TransformerSelfAttentionModel
from data_utils import DataProcessor, load_data

# Load configuration
config = TrainConfig()
rng = jax.random.PRNGKey(42)
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
print("[INFO] JAX device: ", jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

# Load dataset
dataset_path = "data/dataset.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts = load_data(dataset_path)
tokenized_texts = [processor.tokenize(text) for text in raw_texts]

# Pad sequences to `max_seq_length`
inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts], dtype=jnp.int32)
targets = jnp.array(inputs, dtype=jnp.int32)

# Initialize Transformer-SelfAttention Model
def forward_fn(inputs, return_attention=False):
    model = TransformerSelfAttentionModel(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length
    )
    return model(inputs, return_attention=return_attention)

model = hk.transform(forward_fn)
params = model.init(rng, inputs[:config.batch_size])  

# Optimizer
warmup_steps = 5000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6,
    peak_value=5e-4, 
    warmup_steps=warmup_steps, 
    decay_steps=200000, 
    end_value=1e-6
)
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adamw(schedule, weight_decay=1e-3)
)
opt_state = optimizer.init(params)

# Compute loss function
def compute_loss(params, rng, inputs, targets):
    logits, attention_weights = model.apply(params, rng, inputs, return_attention=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    smoothed_loss = loss * 0.85 + 0.15 * jnp.mean(loss)
    return smoothed_loss, attention_weights

def loss_for_gradients(params, rng, inputs, targets):
    loss, _ = compute_loss(params, rng, inputs, targets)  
    return loss

@jax.jit
def train_step(params, opt_state, rng, inputs, targets):
    loss, attention_weights = compute_loss(params, rng, inputs, targets) 
    grads = jax.grad(loss_for_gradients)(params, rng, inputs, targets) 
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, attention_weights, params, opt_state

# Training loop
losses = []
attns_maps = []
for epoch in range(config.num_epochs):
    gc.collect()
    jax.clear_caches()
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        loss, attention_weights, params, opt_state = train_step(params, opt_state, step_rng, batch_inputs, batch_targets)
        losses.append(loss)
        attns_maps.append(attention_weights)
        print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")
        del batch_inputs, batch_targets
        gc.collect()

# Plot loss curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.title("Transformer-SelfAttention Training Loss")
plt.show()
plt.savefig("transformer_self_attention_loss.png")
print("[INFO] Loss plot saved as transformer_self_attention_loss.png")