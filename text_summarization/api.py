from fastapi import FastAPI
from pydantic import BaseModel
import jax.numpy as jnp
import jax
import haiku as hk
from text_summarization.text_summary_module import TextSummarizationModel
from data_utils import DataProcessor
import nltk
from nltk.corpus import words

nltk.download("words")

app = FastAPI()
processor = DataProcessor()

vocab_list = sorted(words.words())
vocab_size = min(len(vocab_list), 4000)
limited_vocab = vocab_list[:vocab_size]
processor.build_vocab(limited_vocab)

class TextInput(BaseModel):
    text: str

global_rng = jax.random.PRNGKey(42)  

def forward_fn(inputs):
    model = TextSummarizationModel(
        vocab_size=vocab_size, d_model=64, num_heads=2, num_layers=2, max_seq_length=64
    )
    return model(inputs)

model = hk.transform_with_state(forward_fn)

dummy_inputs = jnp.ones((1, 64), dtype=jnp.int32)
params, state = model.init(global_rng, dummy_inputs)  

@app.post("/summarize")
def summarize(input_text: TextInput):
    global global_rng 
    print(f"[DEBUG] Received Input: {input_text.text}") 
    tokens = processor.convert_text_to_tokens(input_text.text)
    padded_tokens = processor.pad_sequence(tokens, 64)
    inputs = jnp.array([padded_tokens])
    print(f"[DEBUG] Final JAX Input Shape: {inputs.shape}")
    global_rng, subkey = jax.random.split(global_rng)  
    summary_logits, _ = model.apply(params, state, subkey, inputs)  
    print(f"[DEBUG] Model Output Logits Shape: {summary_logits.shape}")
    summary_tokens = jnp.argmax(summary_logits, axis=-1)[0]
    print(f"[DEBUG] Summary Token IDs: {summary_tokens}")
    summary_words = [word for idx, word in enumerate(processor.vocab.keys()) if idx in summary_tokens]
    print(f"[DEBUG] Summary Words: {summary_words}")
    summary = " ".join(summary_words)
    print(f"[DEBUG] Final Summary: {summary}")
    return {"summary": summary}
