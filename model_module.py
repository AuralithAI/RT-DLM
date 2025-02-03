from model import TransformerBlock, SelfAttention, EmbeddingLayer, MixtureOfExperts, to_device
from train_config import TrainConfig
import haiku as hk

class RTDLMModel(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_length)
        self.transformer_blocks = [TransformerBlock(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        self.moe_layer = MixtureOfExperts(config.d_model, config.moe_experts, config.moe_top_k, dropout_rate=0.1)
        self.final_layer = hk.Linear(config.vocab_size, w_init=hk.initializers.TruncatedNormal(0.02))  
        
    def __call__(self, inputs):
        x = self.embed(inputs, seq_length=inputs.shape[1])

        for block in self.transformer_blocks:
            x = block(x)

        x = self.moe_layer(x, hk.next_rng_key(), is_training=True)  
        x = self.final_layer(x) 

        return to_device(x)


def forward_fn(inputs):
    config = TrainConfig()
    model = RTDLMModel(config)
    return model(inputs)