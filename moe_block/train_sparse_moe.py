import haiku as hk
import jax
import jax.numpy as jnp
import optax
import os
import gc
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sparse_moe import SparseMoE
from data_utils import DataProcessor, load_data

# Training Configuration
class TrainConfig:
    def __init__(self):
        self.vocab_size = 6145
        self.d_model = 128
        self.moe_experts = 8
        self.moe_top_k = 2
        self.expert_capacity = 3  # Ensure token overflow handling
        self.batch_size = 64
        self.max_seq_length = 64
        self.learning_rate = 3e-4
        self.num_epochs = 100
        self.eval_interval = 10

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

def project_input(inputs, d_model, vocab_size):
    embedding_layer = hk.Embed(vocab_size, d_model, name="token_embedding")
    return embedding_layer(inputs)

# Define MoE Model
def forward_fn(inputs):
    #embedding_layer = hk.Embed(config.vocab_size, config.d_model, name="token_embedding")
    embedded_inputs = project_input(inputs, config.d_model, config.vocab_size)
    model = SparseMoE(
        d_model=config.d_model,
        num_experts=config.moe_experts,
        top_k=config.moe_top_k,
        expert_capacity=config.expert_capacity
    )
    moe_output, top_k_indices, aux_loss = model(embedded_inputs)
    return moe_output, embedded_inputs, top_k_indices, aux_loss

# Transform using Haiku
model = hk.transform(forward_fn)

# Initialize parameters
params = model.init(rng, inputs[:config.batch_size])

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(config.learning_rate, weight_decay=5e-3)
)
opt_state = optimizer.init(params)

# Compute Loss
def compute_loss(params, rng, inputs, targets):
    moe_output, embedded_inputs, top_k_indices, aux_loss = model.apply(params, rng, inputs)
    # Task-specific loss (MSE as placeholder)
    task_loss = jnp.mean(jnp.square(moe_output - embedded_inputs))

    # Total loss: Task loss + Weighted Auxiliary loss
    total_loss = task_loss + 0.01 * aux_loss  

    return total_loss

@jax.jit
def train_step(params, opt_state, rng, inputs, targets):
    """
    Perform one training step.
    """
    loss, grads = jax.value_and_grad(compute_loss)(params, rng, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

# Training Loop
losses = []
for epoch in range(config.num_epochs):
    gc.collect()
    jax.clear_caches()
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        loss, params, opt_state = train_step(params, opt_state, step_rng, batch_inputs, batch_targets)
        losses.append(loss)
        print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")    

# Plot Loss Curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Sparse MoE Training Loss")
plt.grid(True)
plt.show()
plt.savefig("sparse_moe_loss.png")
print("[INFO] Loss plot saved as sparse_moe_loss.png")