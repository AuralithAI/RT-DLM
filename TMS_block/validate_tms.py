import jax
import jax.numpy as jnp
import haiku as hk
import optax
import os
import gc
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor, load_data

# Load configuration
config = TrainConfig()
rng = jax.random.PRNGKey(42)

# Load validation dataset
val_dataset_path = "data/validation_data.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts_val = load_data(val_dataset_path)
tokenized_texts_val = [processor.tokenize(text) for text in raw_texts_val]
inputs_val = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts_val], dtype=jnp.int32)
targets_val = jnp.array(inputs_val, dtype=jnp.int32)

# Load trained model
def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
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
        mtm_weight=config.mtm_weight
    )
    return model(inputs, return_attention=return_attention, retrieved_memory_ltm=retrieved_memory_ltm, 
                 retrieved_memory_stm=retrieved_memory_stm, retrieved_memory_mtm=retrieved_memory_mtm)

model = hk.transform_with_state(forward_fn)

# Load trained parameters, state, and memory banks
params_path = "TMS_block/tms_best_params.pkl"
state_path = "TMS_block/tms_best_state.pkl"
ltm_path = "TMS_block/ltm_bank.pkl"
stm_path = "TMS_block/stm_bank.pkl"
mtm_path = "TMS_block/mtm_bank.pkl"

with open(params_path, "rb") as f:
    params = pickle.load(f)
with open(state_path, "rb") as f:
    state = pickle.load(f)
with open(ltm_path, "rb") as f:
    ltm = pickle.load(f)
with open(stm_path, "rb") as f:
    stm = pickle.load(f)
with open(mtm_path, "rb") as f:
    mtm = pickle.load(f)
print(f"[INFO] Loaded trained parameters from {params_path}, state from {state_path}, and memory banks from {ltm_path}, {stm_path}, {mtm_path}")

# Loss computation function with all memory banks
def compute_loss(params, state, rng, inputs, targets):
    embeddings = jnp.mean(model.apply(params, state, rng, inputs, return_attention=False)[0], axis=1)
    query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
    ltm_memory = jnp.array(ltm.retrieve(query_key_np), dtype=jnp.float32)
    stm_memory = jnp.array(stm.retrieve(query_key_np), dtype=jnp.float32)
    mtm_memory = jnp.array(mtm.retrieve(query_key_np), dtype=jnp.float32)

    (logits, attn_weights, expert_indices, aux_loss), new_state = model.apply(
        params, state, rng, inputs, return_attention=True,
        retrieved_memory_ltm=ltm_memory, retrieved_memory_stm=stm_memory, retrieved_memory_mtm=mtm_memory
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    total_loss = loss + 0.01 * aux_loss
    return total_loss, new_state

# Validation loop
def validate_model(params, state, rng, inputs, targets):
    total_val_loss = []

    for step in range(len(inputs) // config.batch_size):
        batch_start = step * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_inputs, batch_targets = inputs[batch_start:batch_end], targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        val_loss, new_state = compute_loss(params, state, step_rng, batch_inputs, batch_targets)
        total_val_loss.append(float(val_loss))
        state = new_state

    avg_val_loss = np.mean(total_val_loss)
    print(f"[INFO] Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss, total_val_loss

# Run validation
avg_val_loss, val_losses = validate_model(params, state, rng, inputs_val, targets_val)

# Plot validation loss
plt.plot(val_losses, label="Validation Loss per Batch")
plt.axhline(y=avg_val_loss, linestyle="--", color="red", label=f"Avg Validation Loss: {avg_val_loss:.4f}")
plt.xlabel("Batch Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Validation Loss Across Batches")
plt.savefig("data/validation_loss.png")
print("[INFO] Validation loss plot saved as data/validation_loss.png")