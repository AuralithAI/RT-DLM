####################################################################################
#                   Test Embedding Model - Using Example
####################################################################################
import jax
import haiku as hk
import jax.numpy as jnp
from model import EmbeddingLayer

def test_embedding():
    vocab_size = 32000
    d_model = 512
    max_seq_length = 128

    embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length)
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]]) 
    seq_length = inputs.shape[1]

    model = hk.transform(lambda x: embedding_layer(x, seq_length))
    params = model.init(jax.random.PRNGKey(42), inputs)
    outputs = model.apply(params, None, inputs)

    print("Embedding Output Shape:", outputs.shape)

test_embedding()