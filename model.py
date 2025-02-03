###########################################################################################################
#                RT-DLM Model With Transformer Block, Embedding Layer and Mixture of Experts
###########################################################################################################
import haiku as hk
import jax.numpy as jnp
import jax
import os
from config import RTDLMConfig
from jax.lib import xla_bridge

"""
    GPU or CPU device selection. (Based on CUDA availability.)
"""
if jax.devices("gpu"):
    print(f"Using GPU: {jax.devices('gpu')[0]}")
    device = jax.devices("gpu")[0]
else:
    print("No GPU found. Falling back to CPU.")
    device = jax.devices("cpu")[0]

def to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float16), device)

"""
    EmbeddingLayer class is used to create embeddings for token and positional embeddings.
"""
class EmbeddingLayer(hk.Module):
    """
       Constructor for EmbeddingLayer   
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def __call__(self, token_ids: jnp.ndarray, seq_length: int):
        """
        Combine token and positional embeddings.
        Parameters:
            token_ids: jnp.ndarray - Token IDs
            seq_length: int - Sequence length
        Returns:
            jnp.ndarray: Combined embeddings
        """
        token_embeddings = hk.get_parameter("token_embedding", 
                                            shape=[self.vocab_size, self.d_model], 
                                            init=hk.initializers.RandomNormal())

        position_embeddings = hk.get_parameter("position_embedding", 
                                               shape=[self.max_seq_length, self.d_model], 
                                               init=hk.initializers.RandomNormal())

        position_ids = jnp.arange(seq_length)[None, :]
        token_embeds = jnp.take(token_embeddings, token_ids, axis=0)
        position_embeds = jnp.take(position_embeddings, position_ids, axis=0)
        print(f"[EmbeddingLayer] position_ids.shape: {position_ids.shape}")
        print(f"[EmbeddingLayer] token_embeds.shape: {token_embeds.shape}, position_embeds.shape: {position_embeds.shape}")
 
        return token_embeds + position_embeds
    

"""
    SelfAttention class is used to create self-attention mechanism using hk.multihead_attention.
"""
class SelfAttention(hk.Module):
    """
       Constructor for SelfAttention 
       By-Default:
            MultiHeadAttention is used with key_size = d_model // num_heads
            But it needs a w_init parameter to be passed.
            So we are passing it as hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"):
                * scale = 1.0 : scale factor of weights.
                * mode = "fan_avg" : Mode of computation. (Balances initialization for both forward and backward passes)
                * distribution = "uniform" : Distribution of weights is uniform
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = hk.MultiHeadAttention(
            num_heads=num_heads, 
            key_size=d_model // num_heads,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )

    def __call__(self, x: jnp.ndarray):
        """
        Apply self-attention mechanism.
        Parameters:
            x: jnp.ndarray - Input tensor
        Returns:
            jnp.ndarray: Output tensor
            Here, assumption is that query, key and value [Q,K,V] are same. (May change later..)
        """
        print(f"[SelfAttention] Input shape: {x.shape}")
        output = self.attention(x, x, x)
        print(f"[SelfAttention] Output shape: {output.shape}")
        return output
    

"""
    Create a Transformer model using EmbeddingLayer and SelfAttention.
"""
class TransformerBlock(hk.Module):
    """
       Constructor for TransformerBlock:
        Here we are using a 2 Layer MLP (Multi-Layer Preceptron) with 4*d_model and d_model units. (May change later)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.layer_norm_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.feedforward = hk.nets.MLP([d_model * 4, d_model])  
        self.layer_norm_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass for Transformer Block.
        """
        print(f"[TransformerBlock] Input shape: {x.shape}")

        attention_output = self.attention(x)
        print(f"[TransformerBlock] Attention output shape: {attention_output.shape}")

        x = self.addAndNormalize(x, attention_output, self.layer_norm_1)
        print(f"[TransformerBlock] After LayerNorm 1: {x.shape}")

        feedforward_output = self.feedforward(x)
        print(f"[TransformerBlock] Feedforward output shape: {feedforward_output.shape}")

        x = self.addAndNormalize(x, feedforward_output, self.layer_norm_2)
        print(f"[TransformerBlock] After LayerNorm 2: {x.shape}")

        return to_device(x)
    
    def addAndNormalize(self, x: jnp.ndarray, output: jnp.ndarray, layer_norm: hk.LayerNorm) -> jnp.ndarray:
        """
        Add and normalize the output using the provided LayerNorm.
        Parameters:
            x: jnp.ndarray - Input tensor
            output: jnp.ndarray - Output tensor to add
            layer_norm: hk.LayerNorm - LayerNorm module to apply normalization
        Returns:
            jnp.ndarray: Output tensor after addition and normalization
        """
        return layer_norm(x + output)
    

"""
    MixtureOfExperts class is used to create a Mixture of Experts mechanism.
    This mechanism selects top-k experts and combines their outputs. (Good for scalability)
"""
class MixtureOfExperts(hk.Module):
    """
    Advanced Mixture of Experts (MoE):
    - Uses dynamic gating to determine expert allocation.
    - Supports top-k expert selection.
    - Incorporates dropout for regularization.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, dropout_rate: float = 0.1):
        super().__init__()
        self.experts = [hk.nets.MLP([d_model * 4, d_model]) for _ in range(num_experts)]
        self.gating = hk.Linear(num_experts)
        self.top_k = top_k
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool = True):
        gate_scores = jax.nn.softmax(self.gating(x), axis=-1)
        if is_training:
            gate_scores = hk.dropout(hk.next_rng_key(), self.dropout_rate, gate_scores)

        print(f"[DEBUG] gate_scores sharding type: {type(gate_scores.sharding)}")
        print(f"[DEBUG] gate_scores device: {gate_scores.devices() if hasattr(gate_scores, 'devices') else 'N/A'}")


        top_k_scores, top_k_indices = jax.lax.top_k(gate_scores, self.top_k)
        print(f"[MixtureOfExperts] Top-k scores shape: {top_k_scores.shape}, Top-k indices shape: {top_k_indices.shape}")

        outputs = hk.vmap(self._process_batch, in_axes=(0, 0, 0), split_rng=False)(x, top_k_scores, top_k_indices)

        print(f"[MixtureOfExperts] Output shape: {outputs.shape}")
        return to_device(outputs)

    def _process_batch(self, x_batch, scores, indices):
        """
        Process a batch of inputs by computing outputs for top-k experts.
        Args:
            x_batch (jnp.ndarray): Input batch of shape [seq_length, d_model].
            scores (jnp.ndarray): Gating scores of shape [seq_length, top_k].
            indices (jnp.ndarray): Top-k expert indices of shape [seq_length, top_k].
        Returns:
            jnp.ndarray: Combined outputs of shape [seq_length, d_model].
        """

        def compute_expert_output(index, x_slice):
            """
            Compute the output of a single expert.
            Args:
                index: Index of the expert to invoke.
                x_slice: Input slice for the expert.
            Returns:
                Output of the expert computation.
            """
            return hk.switch(index, [lambda x_slice=x_slice: expert(x_slice) for expert in self.experts])

        def process_single_position(x_slice, scores_pos, indices_pos):
            """
            Compute outputs for the top-k experts at a single position.
            Args:
                x_slice: A single input slice of shape [d_model].
                scores_pos: Gating scores for top-k experts of shape [top_k].
                indices_pos: Top-k expert indices of shape [top_k].
            """
            x_repeated = jnp.repeat(x_slice[None, :], self.top_k, axis=0)  
            expert_outputs = jax.vmap(compute_expert_output, in_axes=(0, 0))(indices_pos, x_repeated)  
            return jnp.sum(expert_outputs * scores_pos[:, None], axis=0)

        combined_outputs = hk.vmap(process_single_position, in_axes=(0, 0, 0), split_rng=False)(
            x_batch, scores, indices
        )

        print(f"[MixtureOfExperts] Combined outputs shape: {combined_outputs.shape}")
        return to_device(combined_outputs)
