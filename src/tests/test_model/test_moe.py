####################################################################################
#                 Test Mixture of Experts - Using Example
####################################################################################
import os
import sys
import jax
import haiku as hk
import jax.numpy as jnp

# Add tests folder to path (one level up from test_example)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_model import MixtureOfExperts

def test_moe():
    d_model = 512
    num_experts = 16
    top_k = 4
    batch_size = 2
    sequence_length = 128

    def forward_fn(x):
        mixture_of_experts = MixtureOfExperts(d_model, num_experts, top_k)
        return mixture_of_experts(x)

    model = hk.transform(forward_fn)
    inputs = jnp.ones((batch_size, sequence_length, d_model))
    rng = jax.random.PRNGKey(42)  

    params = model.init(rng, inputs)
    outputs = model.apply(params, rng, inputs)

    print("MoE Output Shape:", outputs.shape)
    # This should print:  (2, 128, 512)

if __name__ == "__main__":
    test_moe()
