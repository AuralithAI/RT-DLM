import haiku as hk
import jax
import jax.numpy as jnp
from model import TransformerBlock, SelfAttention, EmbeddingLayer, MixtureOfExperts, to_device
from train_config import TrainConfig

class RTDLMModel(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_length)
        self.transformer_blocks = [TransformerBlock(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        self.moe_layer = MixtureOfExperts(config.d_model, config.moe_experts, config.moe_top_k, dropout_rate=0.1, temperature=1.0)
        self.final_layer = hk.Linear(config.vocab_size, w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))

    def __call__(self, inputs, rng):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        if inputs.shape[1] != TrainConfig().max_seq_length:
            pad_width = TrainConfig().max_seq_length - inputs.shape[1]
            inputs = jnp.pad(inputs, ((0, 0), (0, pad_width)), constant_values=0)

        x = self.embed(inputs, seq_length=inputs.shape[1])
        rng, *subkeys = jax.random.split(rng, num=len(self.transformer_blocks) + 2)

        for block, subkey in zip(self.transformer_blocks, subkeys[:-1]):
            x = block(x)

        # Apply Mixture of Experts (MoE)
        x, load_balancing_loss = self.moe_layer(x, rng=subkeys[-1], is_training=True)

        logits = self.final_layer(x)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)

        return to_device(logits), load_balancing_loss

def forward_fn(inputs, rng):
    if rng.shape != (2,):  
        rng = jax.random.PRNGKey(42)
    subkeys = jax.random.split(rng, num=2)
    model = RTDLMModel(TrainConfig())
    return model(inputs, rng=subkeys[0])

model = hk.without_apply_rng(hk.transform_with_state(forward_fn))