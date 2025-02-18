import haiku as hk
import jax
import jax.numpy as jnp
import optax
import os
import gc
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor, load_data, preprocess_batch

# Load configuration
config = TrainConfig()
rng = jax.random.PRNGKey(42)

# Set JAX configurations
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
print("[INFO] JAX device: ", jax.devices())

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"

# Load dataset
dataset_path = "data/train_data.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts = load_data(dataset_path)
processor.build_vocab(raw_texts)
inputs, targets = preprocess_batch(raw_texts, processor, config.max_seq_length)

# Convert to JAX arrays
inputs = jnp.array(inputs, dtype=jnp.int32)
targets = jnp.array(targets, dtype=jnp.int32)

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
    
    # Task Loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    # Auxiliary Loss from MoE
    total_loss = loss + 0.01 * aux_loss  # Adjust weight of MoE loss
    
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
losses = []
attn_maps_self = []
attn_maps_transformer = []
expert_usage = []
for epoch in range(config.num_epochs):
    gc.collect()
    jax.clear_caches()
    
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        loss, attn_weights, expert_indices, params, opt_state = train_step(params, opt_state, step_rng, batch_inputs, batch_targets)
        losses.append(loss)
        if type(attn_weights) == tuple:
            attn_weights_self, attn_weights_transformer = attn_weights
            attn_maps_self.append(attn_weights_self)
            attn_maps_transformer.append(attn_weights_transformer)

        expert_usage.append(expert_indices)
        
        print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")
        del batch_inputs, batch_targets
        gc.collect()

# Plot loss curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.title("TMS Training Loss")
plt.show()
plt.savefig("tms_loss.png")
print("[INFO] Loss plot saved as tms_loss.png")