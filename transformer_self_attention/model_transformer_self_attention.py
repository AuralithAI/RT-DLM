import haiku as hk
from transformer_self_attention.model import TransformerSelfAttentionModel

class ModelTransformerSelfAttention(hk.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.model = TransformerSelfAttentionModel(d_model, num_heads, num_layers, vocab_size, max_seq_length)

    def __call__(self, inputs, rng=None, return_attention=False):
        return self.model(inputs, rng, return_attention)