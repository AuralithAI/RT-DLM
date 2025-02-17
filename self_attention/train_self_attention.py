import jax
import os
import sys
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from model_module_self_attention import SelfAttentionModel
from data_utils import DataProcessor, load_data, preprocess_batch
from logLevel.logLevel import Logging
from datetime import datetime

# Set log file
log_filename = f"logs/training_selfattention{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Load configuration
config = TrainConfig()
Logging.info("Training configuration loaded.")
rng = jax.random.PRNGKey(42)
Logging.info("Random seed initialized.")

# Load dataset
dataset_path = "data/dataset.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts = load_data(dataset_path)
processor.build_vocab(raw_texts)
Logging.info(f"Loaded dataset from {dataset_path} and built vocabulary.")
inputs, targets = preprocess_batch(raw_texts, processor, config.max_seq_length)

# Convert to JAX arrays
inputs = jnp.array(inputs, dtype=jnp.int32)
targets = jnp.array(targets, dtype=jnp.int32)
Logging.info("Converted dataset to JAX arrays.")

# Initialize model directly
def forward_fn(inputs, return_attention=False):
    model = SelfAttentionModel(
        d_model=config.d_model,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length
    )
    return model(inputs, return_attention=return_attention)

model = hk.without_apply_rng(hk.transform(forward_fn))
params = model.init(rng, inputs[:config.batch_size])
Logging.info("Model initialized successfully.")

# Optimizer
# Add learning rate decay schedule
warmup_steps = 1000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-7, peak_value=3e-4, warmup_steps=warmup_steps, decay_steps=5000, end_value=1e-6
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  
    optax.adamw(schedule)
)

opt_state = optimizer.init(params)
Logging.info("Optimizer initialized.")

# Compute loss
def compute_loss(params, inputs, targets):
    logits, attention_weights = model.apply(params, inputs, return_attention=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    smoothed_loss = loss * 0.9 + 0.1 * jnp.mean(loss)
    return smoothed_loss.mean(), attention_weights

def loss_for_gradients(params, inputs, targets):
    loss, _ = compute_loss(params, inputs, targets)  
    return loss

# Training step
def train_step(params, opt_state, inputs, targets):
    loss, attention_weights = compute_loss(params, inputs, targets) 
    grads = jax.grad(loss_for_gradients)(params, inputs, targets) 
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, attention_weights, params, opt_state

def plot_attention_maps(attn_maps):
    fig, axes = plt.subplots(1, min(4, len(attn_maps)), figsize=(15, 5))
    for i in range(len(axes)):
        sns.heatmap(attn_maps[i][0], cmap="viridis", ax=axes[i])  
        axes[i].set_title(f"Attention Map {i+1}")
    plt.show()
    plt.savefig("attention_maps.png")

# Training loop
losses = []
attns_maps = []
Logging.info("Starting training loop...")
for epoch in range(config.num_epochs):
    Logging.info(f"Epoch {epoch+1} started.")
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        loss, attention_weights, params, opt_state = train_step(params, opt_state, batch_inputs, batch_targets)
        losses.append(loss)
        attns_maps.append(attention_weights)

        Logging.info(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")

Logging.info("Training completed.")
# Plot loss curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Self-Attention Training Loss")
plt.show()
plt.savefig("self_attention_loss.png")
Logging.info("Loss plot saved as self_attention_loss.png")
plot_attention_maps(attns_maps)
Logging.info("Attention maps saved as attention_maps.png")