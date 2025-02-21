import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory

# Load config
config = TrainConfig()

# Load vocabulary & processor
processor = DataProcessor(vocab_size=config.vocab_size)

# Load trained model and state
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

# Load trained parameters and state
params_path = "TMS_block/tms_best_params.pkl"
state_path = "TMS_block/tms_best_state.pkl"
ltm_path = "TMS_block/ltm_bank.pkl"
stm_path = "TMS_block/stm_bank.pkl"
mtm_path = "TMS_block/mtm_bank.pkl"

with open(params_path, "rb") as f:
    params = pickle.load(f)
with open(state_path, "rb") as f:
    state = pickle.load(f)
print(f"[INFO] Loaded trained parameters from {params_path} and state from {state_path}")

# Load memory banks
if os.path.exists(ltm_path):
    with open(ltm_path, "rb") as f:
        ltm = pickle.load(f)
    print(f"[INFO] Loaded LTM from {ltm_path}")
else:
    ltm = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    print("[INFO] Initialized new LTM (no saved LTM found)")

if os.path.exists(stm_path):
    with open(stm_path, "rb") as f:
        stm = pickle.load(f)
    print(f"[INFO] Loaded STM from {stm_path}")
else:
    stm = ShortTermMemory(buffer_size=config.stm_buffer_size, embedding_dim=config.d_model)
    print("[INFO] Initialized new STM (no saved STM found)")

if os.path.exists(mtm_path):
    with open(mtm_path, "rb") as f:
        mtm = pickle.load(f)
    print(f"[INFO] Loaded MTM from {mtm_path}")
else:
    mtm = MidTermMemory(buffer_size=config.mtm_buffer_size, embedding_dim=config.d_model, retention_steps=config.retention_steps)
    print("[INFO] Initialized new MTM (no saved MTM found)")

# Function to generate text iteratively
def generate_text(prompt, max_length=50):
    tokens = processor.tokenize(prompt)
    print(f"Tokenized Prompt: {tokens}")
    
    tokens = processor.pad_sequence(tokens, config.max_seq_length)
    input_ids = jnp.array([tokens], dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)

    generated_ids = input_ids[0].tolist()

    for _ in range(max_length):
        # Get embeddings for retrieval
        embeddings, _ = model.apply(params, state, rng, input_ids, return_attention=False)
        query_key = jnp.mean(embeddings, axis=1)

        # Retrieve from all three memory banks
        ltm_memory_np = ltm.retrieve(np.asarray(jax.device_get(query_key), dtype=np.float32))
        stm_memory_np = stm.retrieve(np.asarray(jax.device_get(query_key), dtype=np.float32))
        mtm_memory_np = mtm.retrieve(np.asarray(jax.device_get(query_key), dtype=np.float32))

        ltm_memory = jnp.array(ltm_memory_np, dtype=jnp.float32)
        stm_memory = jnp.array(stm_memory_np, dtype=jnp.float32)
        mtm_memory = jnp.array(mtm_memory_np, dtype=jnp.float32)

        # Predict next token with all memories
        logits, new_state = model.apply(params, state, rng, input_ids, 
                                       retrieved_memory_ltm=ltm_memory, 
                                       retrieved_memory_stm=stm_memory, 
                                       retrieved_memory_mtm=mtm_memory)
        next_token_id = jnp.argmax(logits[:, -1, :], axis=-1)

        # Append to sequence and update STM/MTM
        generated_ids.append(int(next_token_id[0]))
        input_ids = jnp.array([generated_ids[-config.max_seq_length:]], dtype=jnp.float32)
        state = new_state

        # Update STM and MTM with new embeddings (optional during inference)
        embeddings = jnp.mean(model.apply(params, state, rng, input_ids, return_attention=False)[0], axis=1)
        stm.store(embeddings, embeddings)
        mtm.store(embeddings, embeddings)

        # Stop if EOS token (adjust '1' to your vocab’s EOS token)
        if next_token_id[0] == 1:
            break

    generated_text = processor.decode_tokens(generated_ids)
    print(f"Generated Token IDs: {generated_ids}")
    print(f"Generated Text: {generated_text}")
    return generated_text

if __name__ == "__main__":
    prompt = "The future of AI is"
    generated_text = generate_text(prompt)
    print("\nGenerated Output:\n", generated_text)