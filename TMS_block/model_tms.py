import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_attention.model_module_self_attention import SelfAttentionModel
from transformer_block.model_transformer_module import TransformerModel
from moe_block.sparse_moe import SparseMoE
from memory_bank import MemoryBank, ShortTermMemory, MidTermMemory

class TMSModel(hk.Module):
    """
    Transformer + MoE + Self-Attention (TMS) Model
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, 
                moe_experts: int, moe_top_k: int, memory_size: int, retrieval_k: int, 
                ltm_weight: float, stm_weight: float, mtm_weight: float, 
                audio_sample_rate: int = 16000, image_size: int = 64, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.audio_sample_rate = audio_sample_rate
        self.image_size = image_size
        self.full_audio_length = audio_sample_rate * max_seq_length
        self.full_image_size = image_size * image_size * 64
        self.full_video_size = max_seq_length * image_size * image_size * 32

        # Modality Encoders (Text, Audio, Image, Video)
        self.text_encoder = hk.Embed(vocab_size, d_model, name="text_embedding")
        self.audio_encoder = hk.Sequential([
                                hk.Conv1D(output_channels=128, kernel_shape=10, stride=5, padding="VALID"),
                                jax.nn.relu,
                                hk.Conv1D(output_channels=d_model, kernel_shape=5, stride=2),
                                jax.nn.relu,
                                hk.Linear(d_model)
                            ], name="audio_encoder")
        self.image_encoder = hk.Sequential([
                                hk.Conv2D(output_channels=32, kernel_shape=3, stride=2, padding="VALID"),
                                jax.nn.relu,
                                hk.Conv2D(output_channels=64, kernel_shape=3, stride=2),
                                jax.nn.relu,
                                hk.Flatten(),
                                hk.Linear(d_model)
                            ], name="image_encoder")
        self.video_encoder = hk.Sequential([
                                hk.Conv3D(output_channels=32, kernel_shape=(3, 3, 3), stride=(1, 2, 2), padding="VALID"),
                                jax.nn.relu,
                                hk.Flatten(),
                                hk.Linear(d_model)
                            ], name="video_encoder")
        self.position_enc = hk.Embed(max_seq_length, d_model)

        # Model Components (Self-Attention, Transformer, MoE)
        self.self_attention = SelfAttentionModel(d_model, num_heads, vocab_size, max_seq_length)
        self.transformer = TransformerModel(d_model, num_heads, num_layers, vocab_size, max_seq_length)
        self.moe = SparseMoE(d_model, moe_experts, moe_top_k, expert_capacity=3)

        # Memory Banks (Long-Term, Short-Term, Mid-Term)
        self.ltm = MemoryBank(memory_size, d_model, retrieval_k)
        self.stm = ShortTermMemory(max_seq_length, d_model)
        self.mtm = MidTermMemory(max_seq_length * 10, d_model, retention_steps=100)

        # Memory Projections
        self.memory_projection_ltm = hk.Sequential([
                                        hk.Linear(d_model * 2),
                                        jax.nn.relu,
                                        hk.Linear(d_model)
                                    ], name="ltm_projection")
        self.memory_projection_stm = hk.Sequential([
                                        hk.Linear(d_model * 2),
                                        jax.nn.relu,
                                        hk.Linear(d_model)
                                    ], name="stm_projection")
        self.memory_projection_mtm = hk.Sequential([
                                        hk.Linear(d_model * 2),
                                        jax.nn.relu,
                                        hk.Linear(d_model)
                                    ], name="mtm_projection")
        
        # Decoders (Text, Audio, Image, Video)
        self.decoders = {
            "text": hk.Linear(vocab_size, name="text_decoder"),
            "audio": hk.Sequential([
                        hk.Linear(d_model, name="audio_linear_1"),
                        jax.nn.relu,
                        hk.Linear(self.full_audio_length, name="audio_linear_2"),
                        lambda x: x.reshape(-1, max_seq_length * audio_sample_rate)
                    ], name="audio_decoder"),
            "image": hk.Sequential([
                        hk.Linear(self.full_image_size, name="image_linear_1"),
                        jax.nn.relu,
                        lambda x: x.reshape(-1, image_size, image_size, 64),
                        hk.Conv2DTranspose(output_channels=32, kernel_shape=3, stride=2, name="image_conv2d_transpose_1"),
                        jax.nn.relu,
                        hk.Conv2DTranspose(output_channels=3, kernel_shape=3, stride=2, name="image_conv2d_transpose_2"),
                    ], name="image_decoder"),
            "video": hk.Sequential([
                        hk.Linear(self.full_video_size, name="video_linear_1"),
                        jax.nn.relu,
                        lambda x: x.reshape(-1, max_seq_length, image_size, image_size, 32),
                        hk.Conv3DTranspose(output_channels=16, kernel_shape=(3, 3, 3), stride=(1, 2, 2), name="video_conv3d_transpose_1"),
                        jax.nn.relu,
                        hk.Conv3DTranspose(output_channels=3, kernel_shape=(3, 3, 3), stride=(1, 2, 2), name="video_conv3d_transpose_2"),
                    ], name="video_decoder")
        }

        # Norm and Projection Layers
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ltm_weight = ltm_weight  
        self.stm_weight = stm_weight
        self.mtm_weight = mtm_weight

        # Multimodal Fusion Attention
        self.fusion_attention = hk.MultiHeadAttention(
                                    num_heads=num_heads,
                                    key_size=d_model // num_heads,
                                    model_size=d_model,
                                    w_init=hk.initializers.VarianceScaling(1.0),
                                    name="fusion_attention"
                                )

    def encode_modality(self, input_data, modality_type):
        """Encode input based on modality type."""
        if modality_type == "text":
            input_data = jnp.asarray(input_data, dtype=jnp.int32)
            return self.text_encoder(input_data)
        elif modality_type == "audio":
            input_data = jnp.asarray(input_data, dtype=jnp.float32)
            if input_data.ndim == 2:
                input_data = input_data[:, :, None]
            audio_emb = self.audio_encoder(input_data)
            return audio_emb[:, :self.max_seq_length, :]
        elif modality_type == "image":
            input_data = jnp.asarray(input_data, dtype=jnp.float32)
            image_emb = self.image_encoder(input_data)
            return image_emb[:, None, :]
        elif modality_type == "video":
            input_data = jnp.asarray(input_data, dtype=jnp.float32)
            video_emb = self.video_encoder(input_data)
            return video_emb[:, None, :]
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
        
    def decode_output(self, x, output_modality):
        """
        Decode model output based on modality type.
        """
        decoder = self.decoders.get(output_modality)
        if decoder is None:
            raise ValueError(f"No decoder for output modality: {output_modality}")
        # Adjust input shape if needed
        if x.shape[-1] != self.d_model:
            x = hk.Linear(self.d_model, name=f"{output_modality}_input_proj")(x)
        return decoder(x)

    def __call__(self, inputs, modality_types, output_modality="text", rng=None, return_attention=False, 
                retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, 
                spike_threshold=0.1, epsilon=1e-8):
        """
        Forward pass for TMS model.
        """
        modality_embeddings = []
        for inp, m_type in zip(inputs, modality_types):
            emb = self.encode_modality(inp, m_type)
            if emb.shape[1] > self.max_seq_length:
                emb = emb[:, :self.max_seq_length, :]
            elif emb.shape[1] < self.max_seq_length:
                padding = jnp.zeros((emb.shape[0], self.max_seq_length - emb.shape[1], self.d_model))
                emb = jnp.concatenate([emb, padding], axis=1)
            modality_embeddings.append(emb)

        # Multimodal Fusion with Cross-Attention
        if len(modality_embeddings) > 1:
            all_embeddings = jnp.concatenate(modality_embeddings, axis=1)
            query = jnp.mean(all_embeddings, axis=1, keepdims=True)
            fused_x = self.fusion_attention(query=query, key=all_embeddings, value=all_embeddings)
            fused_x = jnp.repeat(fused_x, self.max_seq_length, axis=1)
        else:
            fused_x = modality_embeddings[0]

        # Add positional encoding
        pos_enc = self.position_enc(jnp.arange(self.max_seq_length))
        x = fused_x + pos_enc[None, :, :]

        # ** dummy_memory is used to initialize the memory projection layers [Not to be used anywhere else] ** 
        batch_size = modality_embeddings[0].shape[0]
        dummy_memory = jnp.zeros((batch_size, self.max_seq_length, self.d_model), dtype=jnp.float32)
        dummy_memory = dummy_memory[:, :1, :]
    
        # Handle LTM
        if retrieved_memory_ltm is not None:
            ltm_emb = self.memory_projection_ltm(retrieved_memory_ltm[:, None, :])
            ltm_emb = jnp.repeat(ltm_emb, self.max_seq_length, axis=1)
            x += self.ltm_weight * ltm_emb
        else:
            _ = self.memory_projection_ltm(dummy_memory)

        # Handle STM
        if retrieved_memory_stm is not None:
            stm_emb = self.memory_projection_stm(retrieved_memory_stm[:, None, :])
            stm_emb = jnp.repeat(stm_emb, self.max_seq_length, axis=1)
            x += self.stm_weight * stm_emb 
        else:
            _ = self.memory_projection_stm(dummy_memory)

        # Handle MTM
        if retrieved_memory_mtm is not None:
            mtm_emb = self.memory_projection_mtm(retrieved_memory_mtm[:, None, :])
            mtm_emb = jnp.repeat(mtm_emb, self.max_seq_length, axis=1)
            x += self.mtm_weight * mtm_emb
        else:
            _ = self.memory_projection_mtm(dummy_memory)

        attn_weights_self = None
        if "text" in modality_types and modality_types[0] == "text":
            x_sa, attn_weights_self = self.self_attention(inputs[0], return_attention=True, 
                                                        spike_threshold=spike_threshold, epsilon=epsilon,
                                                        output_logits=False)  
        x = x + x_sa
        x, attn_weights_transformer = self.transformer(x, rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, top_k_expert_indices, aux_loss = self.moe(x, spike_threshold=spike_threshold, epsilon=epsilon)
        x = self.norm(x)

        logits = self.decode_output(x, output_modality)

        if return_attention:
            return logits, (attn_weights_self, attn_weights_transformer), top_k_expert_indices, aux_loss
        return logits