import jax
import jax.numpy as jnp
import haiku as hk
import json
from model_tms import TMSModel
from train_config import TrainConfig
from data_utils import DataProcessor

# Load config & vocab
config = TrainConfig()
processor = DataProcessor(vocab_size=config.vocab_size)
processor.load_vocab()

# Load trained parameters
params_path = "TMS_block/tms_params.pkl"
with open(params_path, "rb") as f:
    params = json.load(f)
print(f"[INFO] Loaded trained parameters from {params_path}")

# Initialize model
def forward_fn(inputs):
    model = TMSModel(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        moe_experts=config.moe_experts,
        moe_top_k=config.moe_top_k
    )
    return model(inputs)

model = hk.transform(forward_fn)

# Prediction function
def predict(text):
    tokens = processor.convert_text_to_tokens(text)
    inputs = jnp.array([processor.pad_sequence(tokens, config.max_seq_length)], dtype=jnp.int32)
    logits = model.apply(params, inputs)
    predictions = jnp.argmax(logits, axis=-1)  # Get highest probability words
    return processor.decode(predictions[0])  # Convert back to text

# Example Test
print("Generated Output:", predict("This is a test sentence."))
