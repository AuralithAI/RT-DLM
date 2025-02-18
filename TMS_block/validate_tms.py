import jax
import jax.numpy as jnp
import haiku as hk
import optax
import os
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor, load_data, preprocess_batch

# Load configuration
config = TrainConfig()
rng = jax.random.PRNGKey(42)

# Load validation dataset
val_dataset_path = "data/validation_data.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts_val = load_data(val_dataset_path)
inputs_val, targets_val = preprocess_batch(raw_texts_val, processor, config.max_seq_length)

# Convert to JAX arrays
inputs_val = jnp.array(inputs_val, dtype=jnp.int32)
targets_val = jnp.array(targets_val, dtype=jnp.int32)

# Load trained model
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
params = model.init(rng, inputs_val[:config.batch_size]) 

# Loss computation function
def compute_loss(params, rng, inputs, targets):
    logits, attn_weights, expert_indices, aux_loss = model.apply(params, rng, inputs, return_attention=True)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    total_loss = loss + 0.01 * aux_loss  
    
    return total_loss, attn_weights, expert_indices

# Validation loop
def validate_model(params, inputs, targets):
    total_val_loss = []
    attn_maps = []
    
    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        val_loss, attn_weights, expert_indices = compute_loss(params, step_rng, batch_inputs, batch_targets)
        
        total_val_loss.append(val_loss)
        
        if isinstance(attn_weights, tuple):  
            _, attn_transformer = attn_weights
            attn_avg = jnp.mean(attn_transformer, axis=0)  
            attn_maps.append(attn_avg)
    
    avg_val_loss = np.mean(total_val_loss)
    print(f"[INFO] Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss, attn_maps

# Run validation
val_loss, attention_maps = validate_model(params, inputs_val, targets_val)

# Plot loss comparison
plt.plot(val_loss, label="Validation Loss", color='red')
plt.axhline(y=config.eval_interval, linestyle="--", color="blue", label="Training Eval Interval")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Validation Loss vs Training Evaluation Interval")
plt.savefig("validation_loss.png")
print("[INFO] Validation loss plot saved as validation_loss.png")
