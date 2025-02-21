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
from memory_bank import MemoryBank

# Load config
config = TrainConfig()

# Load vocabulary & processor
processor = DataProcessor(vocab_size=config.vocab_size)

# Load trained model and state
def forward_fn(inputs, return_attention=False, retrieved_memory=None):
    model = TMSModel(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        moe_experts=config.moe_experts,
        moe_top_k=config.moe_top_k,
        memory_size=config.memory_size,
        retrieval_k=config.retrieval_k
    )
    return model(inputs, return_attention=return_attention, retrieved_memory=retrieved_memory)

model = hk.transform_with_state(forward_fn)

# Load trained parameters and state
params_path = "TMS_block/tms_best_params.pkl"  
state_path = "TMS_block/tms_best_state.pkl"  
with open(params_path, "rb") as f:
    params = pickle.load(f)
with open(state_path, "rb") as f:
    state = pickle.load(f)
print(f"[INFO] Loaded trained parameters from {params_path} and state from {state_path}")

# Load or initialize memory bank
memory_path = "TMS_block/memory_bank.pkl"
if os.path.exists(memory_path):
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    print(f"[INFO] Loaded memory bank from {memory_path}")
else:
    memory = MemoryBank(memory_size=config.memory_size, embedding_dim=config.d_model, retrieval_k=config.retrieval_k)
    print("[INFO] Initialized new memory bank (no saved memory found)")

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
        embeddings = model.apply(params, state, rng, input_ids, return_attention=False)[0]  
        query_key = jnp.mean(embeddings, axis=1)  
        retrieved_memory_np = memory.retrieve(np.asarray(jax.device_get(query_key), dtype=np.float32))
        retrieved_memory = jnp.array(retrieved_memory_np, dtype=jnp.float32)  

        # Predict next token
        logits, new_state = model.apply(params, state, rng, input_ids, retrieved_memory=retrieved_memory)
        next_token_id = jnp.argmax(logits[:, -1, :], axis=-1)  

        # Append to sequence
        generated_ids.append(int(next_token_id[0]))
        input_ids = jnp.array([generated_ids[-config.max_seq_length:]], dtype=jnp.int32)  
        state = new_state  

        # Stop if EOS token (assuming 1 is EOS; adjust based on your vocab)
        if next_token_id[0] == 1:
            break

    # Decode generated sequence
    generated_text = processor.decode_tokens(generated_ids)
    print(f"Generated Token IDs: {generated_ids}")
    print(f"Generated Text: {generated_text}")
    return generated_text

# Run inference
if __name__ == "__main__":
    prompt = "The future of AI is"
    generated_text = generate_text(prompt)
    print("\nGenerated Output:\n", generated_text)