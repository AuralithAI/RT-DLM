####################################################################################
#                 Test Attention Multi_Headed - Using Example
####################################################################################
import os
import sys
import jax
import haiku as hk
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SelfAttention

def test_attention():
    """
    This function initializes the SelfAttention and applies it to the input tensor.
    All parameters are assumed paramters to satisfy the outcome.
    """
    d_model = 512
    num_heads = 8
    batch_size = 2
    sequence_length = 128

    def forward_fn(x):
        attention_layer = SelfAttention(d_model, num_heads)
        return attention_layer(x)

    # Transform the forward function with Haiku
    model = hk.transform(forward_fn)

    inputs = jnp.ones((batch_size, sequence_length, d_model))
    rng = jax.random.PRNGKey(42)

    # Initialize the model
    params = model.init(rng, inputs)

    # Apply the model to inputs
    outputs = model.apply(params, rng, inputs)

    print("Attention Output Shape:", outputs.shape)

if __name__ == "__main__":
    test_attention()