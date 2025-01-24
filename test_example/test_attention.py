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

    attention_layer = SelfAttention(d_model, num_heads)
    inputs = jnp.ones((batch_size, sequence_length, d_model))  

    model = hk.transform(lambda x: attention_layer(x))
    params = model.init(jax.random.PRNGKey(42), inputs)
    outputs = model.apply(params, None, inputs)

    print("Attention Output Shape:", outputs.shape)

test_attention()