from fastapi import FastAPI
from pydantic import BaseModel
import jax.numpy as jnp
import haiku as hk
import jax
import os
import pickle
from data_processing.data_utils import DataProcessor
from text_summarization.text_summary_module import TextSummarizationModel
from datasets import load_dataset

app = FastAPI()
processor = DataProcessor()
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "..", "text_summarization", "trained_model.pkl")

processor = DataProcessor()
data = load_dataset("cnn_dailymail", "3.0.0", split="train[:5%]")
text_samples = [sample["article"] for sample in data][:10000]
processor.build_vocab(text_samples)

class TextInput(BaseModel):
    text: str

global_rng = jax.random.PRNGKey(42)

with open(MODEL_PATH, "rb") as f:
    params, state = pickle.load(f)
print("Loaded trained model parameters from 'trained_model.pkl'")

def forward_fn(inputs):
    model = TextSummarizationModel(
        vocab_size=4000, d_model=64, num_heads=2, num_layers=2, max_seq_length=64
    )
    return model(inputs)

model = hk.transform_with_state(forward_fn)

@jax.jit 
def fast_infer(params, state, rng, inputs):
    return model.apply(params, state, rng, inputs)

dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
fast_infer(params, state, jax.random.PRNGKey(42), dummy_input)
print("Model Compilation Done!")

@app.post("/summarize")
def summarize(input_text: TextInput):
    global global_rng  

    print(f"[DEBUG] Received Input: {input_text.text}") 

    tokens = processor.convert_text_to_tokens(input_text.text)
    padded_tokens = processor.pad_sequence(tokens, 64)
    inputs = jnp.array([padded_tokens])

    print(f"[DEBUG] Final JAX Input Shape: {inputs.shape}")

    global_rng, subkey = jax.random.split(global_rng)  
    summary_logits, _ = fast_infer(params, state, subkey, inputs) 

    print(f"[DEBUG] Model Output Logits Shape: {summary_logits.shape}")

    summary_tokens = jnp.argmax(summary_logits, axis=-1)[0]
    print(f"[DEBUG] Summary Token IDs: {summary_tokens}")

    summary_words = [word for idx, word in enumerate(processor.vocab.keys()) if idx in summary_tokens]
    print(f"[DEBUG] Summary Words: {summary_words}")

    summary = " ".join(summary_words)
    print(f"[DEBUG] Final Summary: {summary}")

    return {"summary": summary}
