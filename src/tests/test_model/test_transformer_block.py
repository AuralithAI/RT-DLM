####################################################################################
#                 Test transformer block - Using Example
####################################################################################
import os
import sys
import jax
import haiku as hk
import jax.numpy as jnp

# Add tests folder to path (one level up from test_example)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_model import TransformerBlock

def test_transformer_block():
    d_model = 512
    num_heads = 8
    batch_size = 2
    sequence_length = 128

    def forward_fn(x):
        transformer_block = TransformerBlock(d_model, num_heads)
        return transformer_block(x)

    model = hk.transform(forward_fn)
    inputs = jnp.ones((batch_size, sequence_length, d_model))
    rng = jax.random.PRNGKey(42)

    params = model.init(rng, inputs)
    outputs = model.apply(params, rng, inputs)

    print("Transformer Block Output Shape:", outputs.shape)
    # This should print:  (2, 128, 512)
    # This ensures the TransformerBlock is correctly processing the input tensor without altering its structure,

test_transformer_block()

