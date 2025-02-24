import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import sys
import os
import numpy as np

# Re-enable JIT for production (remove if debugging persists)
jax.config.update('jax_disable_jit', False)

# Optionally enable stricter type checking (uncomment if dtype issue persists)
# jax.config.update('jax_enable_x64', True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory

# Load config specific to Trial 0
class Trial0Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 384          # Confirmed from log and train_config
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
        self.vocab_size = 8000      # Ensure this matches DataProcessor
        self.max_seq_length = 64    # Ensure this matches training

config = Trial0Config()

# Load vocabulary & processor
processor = DataProcessor(vocab_size=config.vocab_size)

# Custom function to force int32 before hk.Embed
def force_int32(inputs):
    if inputs.dtype != jnp.int32 or not all(isinstance(v, int) for v in inputs.flatten()):
        print(f"[CRITICAL] Forcing inputs to int32: original dtype {inputs.dtype}, values: {inputs.flatten()[:10]}...")
        # Ensure inputs are integers, no floating-point values
        inputs_list = [int(float(v)) if isinstance(v, (float, np.floating)) else int(v) for v in inputs.flatten()]
        return jnp.array(inputs_list, dtype=jnp.int32, copy=True).reshape(inputs.shape)
    print(f"[DEBUG] Inputs already int32: shape {inputs.shape}, dtype {inputs.dtype}, values {inputs.flatten()[:10]}...")
    return inputs

# Define forward function for full pass (pre-proj, d_model output for now)
def forward_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
    print(f"[DEBUG] Forward input shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
    # Force inputs to int32 if not already, to handle potential Haiku/JAX coercion
    if inputs.dtype != jnp.int32:
        print(f"[WARNING] Forcing Forward input to int32: original dtype {inputs.dtype}, values: {inputs.flatten()[:10]}...")
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)  # Force deep copy to prevent implicit coercion
    print(f"[DEBUG] Forward input after asarray shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
    # Use TMSModel's embedding with forced int32
    inputs = force_int32(inputs)
    print(f"[DEBUG] Inputs after force_int32 before embedding shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
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
    print(f"[DEBUG] Embedding + PosEnc shape: {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if retrieved_memory_ltm is not None:
        print(f"[DEBUG] LTM memory shape: {retrieved_memory_ltm.shape}, dtype: {retrieved_memory_ltm.dtype}")  # Should be (1, 384), float32
        x += model.ltm_weight * model.memory_projection_ltm(jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_stm is not None:
        print(f"[DEBUG] STM memory shape: {retrieved_memory_stm.shape}, dtype: {retrieved_memory_stm.dtype}")  # Should be (1, 384), float32
        x += model.stm_weight * model.memory_projection_stm(jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_mtm is not None:
        print(f"[DEBUG] MTM memory shape: {retrieved_memory_mtm.shape}, dtype: {retrieved_memory_mtm.dtype}")  # Should be (1, 384), float32
        x += model.mtm_weight * model.memory_projection_mtm(jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1))
    # Ensure self_attention outputs d_model=384, not vocab_size
    x, attn_self = model.self_attention(x, return_attention=True)
    print(f"[DEBUG] Self-Attention shape: {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] Self-Attention output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x, attn_transformer = model.transformer(x, attn_self, return_attention=True)
    print(f"[DEBUG] Transformer shape: {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] Transformer output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x, _, _ = model.moe(x)
    print(f"[DEBUG] MoE shape: {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] MoE output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x = model.norm(x)  # Stop here—pre-proj, ensuring (batch_size, seq_len, d_model)
    print(f"[DEBUG] Norm shape (pre-proj): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if return_attention:
        return x, (attn_self, attn_transformer)  # Return attention weights as placeholder
    return x  # Output (batch_size, seq_len, d_model) for now

# Define embeddings function (mirrors train_tms.py's get_embeddings, pre-proj)
def embeddings_fn(inputs, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None):
    print(f"[DEBUG] Embeddings input shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
    # Force inputs to int32 if not already, to handle potential Haiku/JAX coercion
    if inputs.dtype != jnp.int32:
        print(f"[WARNING] Forcing Embeddings input to int32: original dtype {inputs.dtype}, values: {inputs.flatten()[:10]}...")
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)  # Force deep copy to prevent implicit coercion
    print(f"[DEBUG] Embeddings input after asarray shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
    # Use TMSModel's embedding with forced int32
    inputs = force_int32(inputs)
    print(f"[DEBUG] Inputs after force_int32 before embedding shape: {inputs.shape}, dtype: {inputs.dtype}, values: {inputs.flatten()[:10]}...")  # Use flatten() instead of tolist()
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
    print(f"[DEBUG] Embedding + PosEnc shape (embeddings): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if retrieved_memory_ltm is not None:
        print(f"[DEBUG] LTM memory shape: {retrieved_memory_ltm.shape}, dtype: {retrieved_memory_ltm.dtype}")  # Should be (1, 384), float32
        x += model.ltm_weight * model.memory_projection_ltm(jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_stm is not None:
        print(f"[DEBUG] STM memory shape: {retrieved_memory_stm.shape}, dtype: {retrieved_memory_stm.dtype}")  # Should be (1, 384), float32
        x += model.stm_weight * model.memory_projection_stm(jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1))
    if retrieved_memory_mtm is not None:
        print(f"[DEBUG] MTM memory shape: {retrieved_memory_mtm.shape}, dtype: {retrieved_memory_mtm.dtype}")  # Should be (1, 384), float32
        x += model.mtm_weight * model.memory_projection_mtm(jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1))
    # Ensure self_attention outputs d_model=384, not vocab_size
    x, _ = model.self_attention(x, return_attention=True)
    print(f"[DEBUG] Self-Attention shape (embeddings): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] Self-Attention output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x, _ = model.transformer(x, None, return_attention=True)
    print(f"[DEBUG] Transformer shape (embeddings): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] Transformer output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x, _, _ = model.moe(x)
    print(f"[DEBUG] MoE shape (embeddings): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    if x.shape[-1] != config.d_model:
        print(f"[WARNING] MoE output shape {x.shape} mismatches d_model={config.d_model}")
        x = hk.Linear(config.d_model)(x)  # Force d_model if mismatched
    x = model.norm(x)  # Stop here—pre-proj, ensuring (batch_size, seq_len, d_model)
    print(f"[DEBUG] Norm shape (embeddings, pre-proj): {x.shape}, dtype: {x.dtype}")  # Should be (1, 64, 384), float32
    return x

# Load models
model = hk.transform_with_state(forward_fn)
embeddings_model = hk.transform_with_state(embeddings_fn)

# Load trained parameters, state, memory banks, and optionally thought log
params_path = "TMS_block/tms_params_trial_0.pkl"
state_path = "TMS_block/tms_state_trial_0.pkl"
ltm_path = "TMS_block/ltm_bank_trial_0.pkl"
stm_path = "TMS_block/stm_bank_trial_0.pkl"
mtm_path = "TMS_block/mtm_bank_trial_0.pkl"
thought_log_path = "TMS_block/thought_log_trial_0.pkl"

try:
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    print(f"[INFO] Loaded trained parameters from {params_path} and state from {state_path}")
    # Verify state is a mapping (dictionary)
    if not isinstance(state, dict):
        print(f"[WARNING] Loaded state is not a dictionary: {type(state)}, values: {state}")
        state = {}  # Fallback to empty state if invalid
    print(f"[DEBUG] Initial state type: {type(state)}, content: {state}")

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

    # Optionally load thought log for analysis (not used in generation)
    thought_log = None
    if os.path.exists(thought_log_path):
        try:
            with open(thought_log_path, "rb") as f:
                thought_log = pickle.load(f)
            print(f"[INFO] Loaded thought log from {thought_log_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load thought log: {e}")
    else:
        print("[INFO] No thought log found at {thought_log_path}")

except FileNotFoundError as e:
    print(f"[ERROR] Failed to load files: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error loading model or memory: {e}")
    sys.exit(1)

# Function to generate text iteratively with sampling option
def generate_text(prompt, max_length=50, temperature=1.0, top_k=5, use_memory=True, analyze_thoughts=False, initial_state=None):
    try:
        # Tokenize and pad prompt
        tokens = processor.tokenize(prompt)
        print(f"Tokenized Prompt: {tokens}")
        tokens = processor.pad_sequence(tokens, config.max_seq_length)
        input_ids = jnp.array([tokens], dtype=jnp.int32, copy=True)  # Ensure int32 with deep copy
        print(f"[DEBUG] Initial input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")  # Use flatten() instead of tolist()

        rng = jax.random.PRNGKey(42)
        generated_ids = input_ids[0].tolist()

        # Use initial_state if provided, otherwise use global state
        current_state = initial_state if initial_state is not None else state
        if not isinstance(current_state, dict):
            print(f"[WARNING] Initial state is not a dictionary: {type(current_state)}, values: {current_state}")
            current_state = {}  # Fallback to empty state if invalid

        for step in range(max_length):
            print(f"[DEBUG] Step {step} input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")  # Use flatten() instead of tolist()
            # Ensure input_ids is int32 before applying with deep copy
            if input_ids.dtype != jnp.int32:
                print(f"[WARNING] Forcing input_ids to int32: original dtype {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")
                input_ids = jnp.asarray(input_ids, dtype=jnp.int32, copy=True)  # Force deep copy to prevent implicit coercion
            print(f"[DEBUG] Step {step} input_ids after check shape: {input_ids.shape}, dtype: {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")  # Use flatten() instead of tolist()
            # Get embeddings for memory retrieval (pre-proj, d_model)
            embeddings, new_state = embeddings_model.apply(params, current_state, rng, input_ids, return_attention=False)  # Correct order: params, state, rng, inputs
            print(f"[DEBUG] Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")  # Should be (1, 64, 384), float32
            print(f"[DEBUG] New state after embeddings: {type(new_state)}, content: {new_state}")
            query_key = jnp.mean(embeddings, axis=1)  # Shape: (1, d_model)

            # Retrieve from memory banks if enabled
            ltm_memory = None
            stm_memory = None
            mtm_memory = None
            if use_memory:
                query_key_np = np.asarray(jax.device_get(query_key), dtype=np.float32)
                try:
                    ltm_memory_np = ltm.retrieve(query_key_np, config.EPSILON)
                    stm_memory_np = stm.retrieve(query_key_np, config.EPSILON)
                    mtm_memory_np = mtm.retrieve(query_key_np, config.EPSILON)
                except Exception as e:
                    print(f"[WARNING] Memory retrieval failed: {e}")
                    ltm_memory_np, stm_memory_np, mtm_memory_np = None, None, None
                if ltm_memory_np is not None:
                    ltm_memory = jnp.array(ltm_memory_np, dtype=jnp.float32)
                if stm_memory_np is not None:
                    stm_memory = jnp.array(stm_memory_np, dtype=jnp.float32)
                if mtm_memory_np is not None:
                    mtm_memory = jnp.array(mtm_memory_np, dtype=jnp.float32)

            # Ensure input_ids is int32 before applying with deep copy
            if input_ids.dtype != jnp.int32:
                print(f"[WARNING] Forcing input_ids to int32 before model.apply: original dtype {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")
                input_ids = jnp.asarray(input_ids, dtype=jnp.int32, copy=True)  # Force deep copy to prevent implicit coercion
            print(f"[DEBUG] Step {step} input_ids before model.apply shape: {input_ids.shape}, dtype: {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")  # Use flatten() instead of tolist()
            # Predict next token (using pre-proj output, d_model)
            output, new_state = model.apply(params, current_state, rng, input_ids,  # Correct order: params, state, rng, inputs
                                          retrieved_memory_ltm=ltm_memory,
                                          retrieved_memory_stm=stm_memory,
                                          retrieved_memory_mtm=mtm_memory)
            print(f"[DEBUG] Model output shape: {output.shape}, dtype: {output.dtype}")  # Should be (1, 64, 384), float32
            print(f"[DEBUG] New state after model: {type(new_state)}, content: {new_state}")
            # Since we’re stopping before proj, output is (batch_size, seq_len, d_model)
            # Manually project to vocab_size for token prediction
            logits = hk.Linear(config.vocab_size)(output)[:, -1, :]  # Project to vocab_size and take last token

            # Apply temperature and top-k sampling
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
                logits = jnp.where(jnp.isin(jnp.arange(config.vocab_size), top_k_indices), top_k_logits, -1e9)
            probs = jax.nn.softmax(logits, axis=-1)
            next_token_id = jax.random.categorical(rng, probs, shape=(1,))[0]

            # Append to sequence and ensure int32 for input_ids with deep copy
            generated_ids.append(int(next_token_id))
            input_ids = jnp.array([generated_ids[-config.max_seq_length:]], dtype=jnp.int32, copy=True)  # Explicitly force int32 with deep copy
            print(f"[DEBUG] Updated input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, values: {input_ids.flatten()[:10]}...")  # Use flatten() instead of tolist()

            current_state = new_state  # Update state for the next iteration

            # Optional thought analysis (for debugging)
            if analyze_thoughts and thought_log is not None:
                print(f"[DEBUG] Step {step}: Analyzing thoughts for similar task...")
                # Example: Compare current embeddings/similarities to thought_log
                if thought_log:
                    print(f"Thought log sample: {thought_log[0]['similarity_score'][:5]}")

            # Stop if EOS token (adjust '1' to your vocab’s EOS token)
            if next_token_id == 1:  # Verify this is your EOS token in DataProcessor
                break

        generated_text = processor.decode_tokens(generated_ids)
        print(f"Generated Token IDs: {generated_ids}")
        print(f"Generated Text: {generated_text}")
        return generated_text, current_state  # Return final state for potential reuse
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return None, None

if __name__ == "__main__":
    prompt = "The future of AI is"
    generated_text, final_state = generate_text(prompt, max_length=50, temperature=1.0, top_k=5, use_memory=True, analyze_thoughts=False)
    print("\nGenerated Output:\n", generated_text)