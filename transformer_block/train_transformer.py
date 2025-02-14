import jax
import os
import sys
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig
from model_transformer_module import TransformerModel
from data_utils import DataProcessor, load_data, preprocess_batch

# Load configuration
config = TrainConfig()
rng = jax.random.PRNGKey(42)

# Load dataset
dataset_path = "data/dataset.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts = load_data(dataset_path)
processor.build_vocab(raw_texts)
inputs, targets = preprocess_batch(raw_texts, processor, config.max_seq_length)

# Convert to JAX arrays
inputs = jnp.array(inputs, dtype=jnp.int32)
targets = jnp.array(targets, dtype=jnp.int32)

def forward_fn(inputs, rng):
    model = TransformerModel(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,  
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length
    )
    return model(inputs)

model = hk.transform(forward_fn)
params = model.init(rng, inputs[:config.batch_size], rng)

# Optimizer
warmup_steps = 2000
schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-7, peak_value=3e-4, warmup_steps=warmup_steps, decay_steps=10000, end_value=1e-7
)
optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.adamw(schedule)
)
opt_state = optimizer.init(params)

# Compute loss function
def compute_loss(params, rng, inputs, targets):
    logits = model.apply(params, rng, inputs, rng) 
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean().astype(jnp.float32)
    return loss

# Training step
def train_step(params, opt_state, rng, inputs, targets):
    loss, grads = jax.value_and_grad(compute_loss)(params, rng, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

# Training loop
losses = []
for epoch in range(config.num_epochs):
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        loss, params, opt_state = train_step(params, opt_state, rng, batch_inputs, batch_targets)
        losses.append(loss)

        print(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss:.4f}")

# Plot loss curve
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Transformer Training Loss")
plt.show()
