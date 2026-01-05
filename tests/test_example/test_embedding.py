####################################################################################
#                   Test Embedding Model - Using Example
####################################################################################
import os
import sys
import jax
import haiku as hk
import jax.numpy as jnp

# Add tests folder to path (one level up from test_example)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_model import EmbeddingLayer

def test_embedding():
    vocab_size = 32000
    d_model = 512
    max_seq_length = 128

    # Define a forward function
    """
    This function initializes the EmbeddingLayer and applies it to the input token_ids.
    """
    def forward_fn(token_ids):
        embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length)
        return embedding_layer(token_ids, seq_length=token_ids.shape[1])

    model = hk.transform(forward_fn)   # So here we wrap the feedforward function in a Haiku transform.

    # Example inputs
    inputs = jnp.array([[1, 2, 3], [4, 5, 6]])  
    rng = jax.random.PRNGKey(42) 

    # Initialize the model
    params = model.init(rng, inputs)

    # Apply the model to inputs
    outputs = model.apply(params, rng, inputs)
    print("Embedding Output Shape:", outputs.shape)

    # This should print:  (2, 3, 512)
    # This ensures that models remain stateless, a key feature for functional programming in JAX.


if __name__ == "__main__":
    test_embedding()