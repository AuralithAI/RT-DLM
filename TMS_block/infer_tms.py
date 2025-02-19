import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_config import TrainConfig
from model_tms import TMSModel
from data_utils import DataProcessor

# Load config
config = TrainConfig()

# Load vocabulary & processor
processor = DataProcessor(vocab_size=config.vocab_size)

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

# Load trained parameters
params_path = "TMS_block/tms_params.pkl"
with open(params_path, "rb") as f:
    params = pickle.load(f)
print(f"[INFO] Loaded trained parameters from {params_path}")

# Function to generate text
def generate_text(prompt, max_length=50):
    tokens = processor.tokenize(prompt)
    print(f"Tokenized Prompt: {tokens}")
    
    tokens = processor.pad_sequence(tokens, config.max_seq_length)
    print(f"Tokenized Prompt PAD Seq: {tokens}")
    
    inputs = jnp.array([tokens], dtype=jnp.int32)
    rng = jax.random.PRNGKey(42)

    logits = model.apply(params, rng, inputs, return_attention=False)
    generated_ids = jnp.argmax(logits, axis=-1)[0].tolist()

    print(f"Generated Token IDs: {generated_ids}")  # Debugging step

    # Convert to text
    generated_text = processor.decode_tokens(generated_ids)
    print("\nDecoded Tokens:", generated_text)  # Print decoded tokens before joining

    return generated_text

# Run inference
if __name__ == "__main__":
    prompt = "The future of AI is"
    generated_text = generate_text(prompt)
    print("\nGenerated Output:\n", generated_text)