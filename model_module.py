import haiku as hk
import jax
from model import TransformerBlock, SelfAttention, EmbeddingLayer, MixtureOfExperts, to_device
from train_config import TrainConfig

class RTDLMModel(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_length)
        self.transformer_blocks = [TransformerBlock(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        self.moe_layer = MixtureOfExperts(config.d_model, config.moe_experts, config.moe_top_k, dropout_rate=0.1)
        self.final_layer = hk.Linear(config.vocab_size, w_init=hk.initializers.TruncatedNormal(0.02))

    def __call__(self, inputs, rng):
        x = self.embed(inputs, seq_length=inputs.shape[1])
        rng, *subkeys = jax.random.split(rng, num=4)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.moe_layer(x, rng=subkeys[0], is_training=True)
        x = self.final_layer(x)

        return to_device(x)

def forward_fn(inputs, rng):
    model = RTDLMModel(TrainConfig())
    rng, subkey = jax.random.split(rng)
    return model(inputs, rng=subkey)

model = hk.without_apply_rng(hk.transform_with_state(forward_fn))