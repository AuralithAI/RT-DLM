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

# Custom config matching Trial 0 from train.log
class Trial0Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 384          # Confirmed from log
        self.num_layers = 7         # Confirmed from log
        self.num_heads = 8          # Suggest_categorical("num_heads", [4, 6, 8, 12])
        self.moe_experts = 4        # Updated per your config
        self.moe_top_k = 2          # Suggest_categorical("moe_top_k", [2, 3])
        self.batch_size = 2         # Suggest_categorical("batch_size", [2, 4, 8])
        self.memory_size = 5000     # Suggest_categorical("memory_size", [1000, 5000, 10000, 20000])
        self.retrieval_k = 3        # Suggest_categorical("retrieval_k", [1, 3, 5, 7])
        self.stm_buffer_size = 32   # Suggest_categorical("stm_buffer_size", [8, 16, 32, 64, 128])
        self.mtm_buffer_size = 1000 # Suggest_categorical("mtm_buffer_size", [500, 1000, 2000, 4000])
        self.retention_steps = 100  # Suggest_int("retention_steps", 50, 200, step=50)
        self.ltm_weight = 0.5       # Suggest_float("ltm_weight", 0.0, 1.0)
        self.stm_weight = 0.5       # Suggest_float("stm_weight", 0.0, 1.0)
        self.mtm_weight = 0.5       # Suggest_float("mtm_weight", 0.0, 1.0)

config = Trial0Config()
rng = jax.random.PRNGKey(42)

# Load validation dataset
val_dataset_path = "data/validation_data.txt"
processor = DataProcessor(vocab_size=config.vocab_size)
raw_texts_val = load_data(val_dataset_path)
tokenized_texts_val = [processor.tokenize(text) for text in raw_texts_val]
inputs_val = jnp.array([processor.pad_sequence(tokens, config.max_seq_length) for tokens in tokenized_texts_val], dtype=jnp.int32)
targets_val = inputs_val  # Next-token prediction
print(f"[INFO] Loaded validation dataset with {len(inputs_val)} samples")

# Define forward function for full pass
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

# Define embeddings function (mirrors train_tms.py's get_embeddings)
def embeddings_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
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
    x = model.embedding(inputs) + model.position_enc(jnp.arange(inputs.shape[1]))
    if retrieved_memory_ltm is not None:
        x += model.ltm_weight * model.memory_projection_ltm(jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_stm is not None:
        x += model.stm_weight * model.memory_projection_stm(jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_mtm is not None:
        x += model.mtm_weight * model.memory_projection_mtm(jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1))
    x, _ = model.self_attention(inputs, return_attention=True)
    x, _ = model.transformer(x, None, return_attention=True)
    x, _, _ = model.moe(x)
    x = model.norm(x)  # Pre-proj embeddings: (batch_size, seq_len, d_model)
    return x

model = hk.transform_with_state(forward_fn)
embeddings_model = hk.transform_with_state(embeddings_fn)

# Load trained parameters, state, and memory banks
params_path = "TMS_block/tms_params_trial_0.pkl"
state_path = "TMS_block/tms_state_trial_0.pkl"
ltm_path = "TMS_block/ltm_bank_trial_0.pkl"
stm_path = "TMS_block/stm_bank_trial_0.pkl"
mtm_path = "TMS_block/mtm_bank_trial_0.pkl"
thought_log_path = "TMS_block/thought_log_trial_0.pkl"

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
with open(thought_log_path, "rb") as f:
    thought_log = pickle.load(f)
print(f"[INFO] Loaded trained parameters, state, thought logs, and memory banks from Trial 0")
print(f"[DEBUG] ltm.embedding_dim: {ltm.embedding_dim}")

# Initialize models to ensure params align (dummy init)
dummy_inputs = inputs_val[:config.batch_size]
params_check, state_check = model.init(rng, dummy_inputs)
embeddings_params_check, embeddings_state_check = embeddings_model.init(rng, dummy_inputs)
print("[INFO] Models initialized with dummy inputs to align parameters")

# Loss and metrics computation function
def compute_metrics(params, state, rng, inputs, targets):
    embeddings, _ = embeddings_model.apply(params, state, rng, inputs)
    print(f"[DEBUG] Raw embeddings.shape: {embeddings.shape}")  # Should be (2, 64, 384)
    embeddings = jnp.mean(embeddings, axis=1)  # Shape: (batch_size, d_model)
    print(f"[DEBUG] Mean embeddings.shape: {embeddings.shape}")  # Should be (2, 384)
    query_key_np = np.asarray(jax.device_get(embeddings), dtype=np.float32)
    
    assert query_key_np.shape[1] == ltm.embedding_dim, f"Query dim {query_key_np.shape[1]} != ltm.embedding_dim {ltm.embedding_dim}"
    ltm_memory = jnp.array(ltm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
    stm_memory = jnp.array(stm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
    mtm_memory = jnp.array(mtm.retrieve(query_key_np, config.EPSILON), dtype=jnp.float32)
    print(f"[DEBUG] ltm_memory.shape: {ltm_memory.shape}")  # Should be (2, 384)
    print(f"[DEBUG] stm_memory.shape: {stm_memory.shape}")
    print(f"[DEBUG] mtm_memory.shape: {mtm_memory.shape}")

    (logits, (attn_weights_self, attn_weights_transformer), expert_indices, aux_loss), new_state = model.apply(
        params, state, rng, inputs, return_attention=True,
        retrieved_memory_ltm=ltm_memory, retrieved_memory_stm=stm_memory, retrieved_memory_mtm=mtm_memory
    )

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    total_loss = loss + 0.001 * aux_loss
    
    perplexity = jnp.exp(loss)
    
    # Use embeddings directly as query_key (already averaged)
    query_key = embeddings  # Shape: (2, 384)
    ltm_norm = jnp.linalg.norm(ltm_memory, axis=-1, keepdims=True) + config.EPSILON
    stm_norm = jnp.linalg.norm(stm_memory, axis=-1, keepdims=True) + config.EPSILON
    mtm_norm = jnp.linalg.norm(mtm_memory, axis=-1, keepdims=True) + config.EPSILON
    query_norm = jnp.linalg.norm(query_key, axis=-1, keepdims=True) + config.EPSILON
    print(f"[DEBUG] query_key.shape: {query_key.shape}")
    print(f"[DEBUG] ltm_norm.shape: {ltm_norm.shape}")
    ltm_similarity = jnp.sum(query_key * ltm_memory, axis=-1) / (query_norm * ltm_norm)
    stm_similarity = jnp.sum(query_key * stm_memory, axis=-1) / (query_norm * stm_norm)
    mtm_similarity = jnp.sum(query_key * mtm_memory, axis=-1) / (query_norm * mtm_norm)
    similarity_score = jnp.stack([ltm_similarity, stm_similarity, mtm_similarity], axis=-1)
    similarity_score = jnp.nan_to_num(similarity_score, nan=0.0)
    avg_similarity = jnp.mean(similarity_score)

    return total_loss, perplexity, avg_similarity, aux_loss, new_state

# Validation loop
def validate_model(params, state, rng, inputs, targets, batch_size=config.batch_size):
    total_val_loss = []
    total_perplexity = []
    total_similarity = []
    total_aux_loss = []
    num_batches = (len(inputs) + batch_size - 1) // batch_size

    for step in range(num_batches):
        batch_start = step * batch_size
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        batch_targets = targets[batch_start:batch_end]

        step_rng, rng = jax.random.split(rng)
        val_loss, val_perplexity, val_similarity, val_aux_loss, new_state = compute_metrics(
            params, state, step_rng, batch_inputs, batch_targets
        )
        total_val_loss.append(float(val_loss))
        total_perplexity.append(float(val_perplexity))
        total_similarity.append(float(val_similarity))
        total_aux_loss.append(float(val_aux_loss))
        state = new_state

    avg_val_loss = np.mean(total_val_loss)
    avg_perplexity = np.mean(total_perplexity)
    avg_similarity = np.mean(total_similarity)
    avg_aux_loss = np.mean(total_aux_loss)

    print(f"[INFO] Validation Metrics:")
    print(f"  Average Total Loss: {avg_val_loss:.4f}")
    print(f"  Average Perplexity: {avg_perplexity:.4f}")
    print(f"  Average Similarity: {avg_similarity:.4f}")
    print(f"  Average Aux Loss: {avg_aux_loss:.4f}")

    return avg_val_loss, avg_perplexity, avg_similarity, avg_aux_loss, total_val_loss

# Run validation
avg_val_loss, avg_perplexity, avg_similarity, avg_aux_loss, val_losses = validate_model(
    params, state, rng, inputs_val, targets_val
)

# Plot validation metrics
plt.figure(figsize=(10, 6))
plt.plot(val_losses, label="Total Loss per Batch")
plt.axhline(y=avg_val_loss, linestyle="--", color="red", label=f"Avg Total Loss: {avg_val_loss:.4f}")
plt.xlabel("Batch Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Validation Loss Across Batches - Trial 0")
plt.savefig("data/validation_loss_trial_0.png")
print("[INFO] Validation loss plot saved as data/validation_loss_trial_0.png")

# Clean up
gc.collect()
jax.clear_caches()
print("[INFO] Validation complete, memory cleared")