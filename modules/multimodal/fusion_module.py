import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple, List

class CrossModalAttention(hk.Module):
    """Cross-modal attention for text-image-audio fusion"""
    
    def __init__(self, d_model: int, num_heads: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = hk.Linear(d_model, name="query")
        self.key_proj = hk.Linear(d_model, name="key") 
        self.value_proj = hk.Linear(d_model, name="value")
        self.output_proj = hk.Linear(d_model, name="output")
        
    def __call__(self, query_modal, key_modal, value_modal, mask=None):
        batch_size, seq_len = query_modal.shape[:2]
        
        # Project to query, key, value
        Q = self.query_proj(query_modal)
        K = self.key_proj(key_modal)
        V = self.value_proj(value_modal)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
            
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attended_values = jnp.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = self.output_proj(attended_values)
        
        return output, attention_weights

class MultiModalFusionLayer(hk.Module):
    """Advanced multi-modal fusion with cross-attention and adaptive gating"""
    
    def __init__(self, d_model: int, num_heads: int, modalities: List[str], name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.modalities = modalities
        
        # Cross-modal attention layers
        self.cross_attentions = {}
        for source in modalities:
            for target in modalities:
                if source != target:
                    self.cross_attentions[f"{source}_to_{target}"] = CrossModalAttention(
                        d_model, num_heads, name=f"cross_attn_{source}_to_{target}"
                    )
        
        # Adaptive gating for each modality
        self.modality_gates = {}
        for modality in modalities:
            self.modality_gates[modality] = hk.Sequential([
                hk.Linear(d_model),
                jax.nn.silu,
                hk.Linear(1),
                jax.nn.sigmoid
            ], name=f"gate_{modality}")
        
        # Final fusion layer
        self.fusion_ffn = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="fusion_ffn")
        
    def __call__(self, modal_inputs: Dict[str, jnp.ndarray], modal_masks: Optional[Dict[str, Optional[jnp.ndarray]]] = None):
        """
        Fuse multiple modalities with cross-attention
        
        Args:
            modal_inputs: Dict of {modality_name: features}
            modal_masks: Optional masks for each modality
        """
        if modal_masks is None:
            modal_masks = {mod: None for mod in modal_inputs.keys()}
            
        enhanced_features = {}
        cross_attention_maps = {}
        
        # Apply cross-modal attention
        for source_mod, source_features in modal_inputs.items():
            enhanced_source = source_features
            
            for target_mod, target_features in modal_inputs.items():
                if source_mod != target_mod:
                    cross_attn_key = f"{source_mod}_to_{target_mod}"
                    if cross_attn_key in self.cross_attentions:
                        cross_attended, attn_weights = self.cross_attentions[cross_attn_key](
                            source_features, target_features, target_features,
                            mask=modal_masks.get(target_mod)
                        )
                        enhanced_source = enhanced_source + cross_attended
                        cross_attention_maps[cross_attn_key] = attn_weights
            
            enhanced_features[source_mod] = enhanced_source
        
        # Apply adaptive gating
        gated_features = {}
        for modality, features in enhanced_features.items():
            gate_score = self.modality_gates[modality](features.mean(axis=1, keepdims=True))
            gated_features[modality] = features * gate_score
        
        # Final fusion
        if len(gated_features) > 1:
            # Concatenate all modalities
            all_features = jnp.concatenate(list(gated_features.values()), axis=-1)
            fused_output = self.fusion_ffn(all_features)
        else:
            fused_output = list(gated_features.values())[0]
            
        return fused_output, cross_attention_maps, gated_features

class VisionEncoder(hk.Module):
    """Vision Transformer + CNN hybrid for image processing"""
    
    def __init__(self, d_model: int, patch_size: int = 16, num_layers: int = 6, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.cnn_backbone = hk.Sequential([
            hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(128, kernel_shape=3, stride=2, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(256, kernel_shape=3, stride=2, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(d_model, kernel_shape=3, stride=1, padding="SAME")
        ], name="cnn_backbone")
        
        # Patch embedding for ViT
        self.patch_embed = hk.Linear(d_model, name="patch_embed")
        self.pos_embed = hk.get_parameter("pos_embed", [1, 196, d_model], 
                                         init=hk.initializers.TruncatedNormal(0.02))
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                hk.MultiHeadAttention(
                    num_heads=8, key_size=d_model//8, 
                    name=f"vit_layer_{i}"
                )
            )
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
    def __call__(self, images):
        batch_size = images.shape[0]
        
        # CNN features
        cnn_features = self.cnn_backbone(images)
        h, w = cnn_features.shape[1:3]
        
        # Convert to patches for ViT
        patches = cnn_features.reshape(batch_size, h * w, self.d_model)
        
        # Add positional encoding
        if patches.shape[1] <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :patches.shape[1], :]
        else:
            # Interpolate if needed
            pos_embed = jax.image.resize(
                self.pos_embed, 
                (1, patches.shape[1], self.d_model), 
                method="linear"
            )
        
        x = patches + pos_embed
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = self.norm(x + layer(x, x, x))
            
        return x

class AudioEncoder(hk.Module):
    """Audio encoder using spectrograms and temporal attention"""
    
    def __init__(self, d_model: int, num_freq_bins: int = 128, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_freq_bins = num_freq_bins
        
        # Spectrogram processing
        self.conv_layers = hk.Sequential([
            hk.Conv1D(64, kernel_shape=3, stride=2, padding="SAME"),
            jax.nn.relu,
            hk.Conv1D(128, kernel_shape=3, stride=2, padding="SAME"),
            jax.nn.relu,
            hk.Conv1D(d_model, kernel_shape=3, stride=1, padding="SAME")
        ], name="audio_conv")
        
        # Temporal attention
        self.temporal_attention = hk.MultiHeadAttention(
            num_heads=8, key_size=d_model//8, name="temporal_attn"
        )
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
    def __call__(self, audio_features):
        """
        Args:
            audio_features: [batch, time_steps, freq_bins] spectrograms
        """
        # Apply convolutions
        x = self.conv_layers(audio_features)
        
        # Apply temporal attention
        x = self.norm(x + self.temporal_attention(x, x, x))
        
        return x

class VideoEncoder(hk.Module):
    """Video encoder with frame-wise and temporal processing"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Frame encoder (reuse vision encoder)
        self.frame_encoder = VisionEncoder(d_model, name="frame_encoder")
        
        # Temporal modeling
        self.temporal_layers = []
        for i in range(3):
            self.temporal_layers.append(
                hk.MultiHeadAttention(
                    num_heads=8, key_size=d_model//8,
                    name=f"temporal_layer_{i}"
                )
            )
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
    def __call__(self, video_frames):
        """
        Args:
            video_frames: [batch, num_frames, height, width, channels]
        """
        _, num_frames = video_frames.shape[:2]
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame_feat = self.frame_encoder(video_frames[:, i])  # [batch, patches, d_model]
            frame_features.append(frame_feat.mean(axis=1))  # Pool patches
        
        # Stack temporal features
        temporal_features = jnp.stack(frame_features, axis=1)  # [batch, num_frames, d_model]
        
        # Apply temporal attention
        for layer in self.temporal_layers:
            temporal_features = self.norm(temporal_features + layer(temporal_features, temporal_features, temporal_features))
        
        return temporal_features

class MultiModalRTDLM(hk.Module):
    """Complete Multi-Modal RT-DLM with advanced fusion"""
    
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        
        # Encoders for each modality
        self.text_encoder = None  # Will use existing TMS model
        self.vision_encoder = VisionEncoder(config.d_model)
        self.audio_encoder = AudioEncoder(config.d_model)
        self.video_encoder = VideoEncoder(config.d_model)
        
        # Multi-modal fusion
        self.fusion_layer = MultiModalFusionLayer(
            config.d_model, 
            config.num_heads,
            modalities=["text", "vision", "audio", "video"]
        )
        
        # Final projection layers
        self.output_projection = hk.Linear(config.vocab_size, name="output_proj")
        self.modality_classifier = hk.Sequential([
            hk.Linear(config.d_model),
            jax.nn.silu,
            hk.Linear(4),  # 4 modalities
            jax.nn.softmax
        ], name="modality_classifier")
        
    def __call__(self, inputs: Dict[str, jnp.ndarray], text_features=None):
        """
        Args:
            inputs: Dict containing:
                - 'images': [batch, height, width, channels] (optional)
                - 'audio': [batch, time_steps, freq_bins] (optional)  
                - 'video': [batch, num_frames, height, width, channels] (optional)
            text_features: [batch, seq_len, d_model] from TMS model
        """
        modal_features = {}
        
        # Process text (provided externally from TMS)
        if text_features is not None:
            modal_features["text"] = text_features
            
        # Process images
        if "images" in inputs and inputs["images"] is not None:
            modal_features["vision"] = self.vision_encoder(inputs["images"])
            
        # Process audio
        if "audio" in inputs and inputs["audio"] is not None:
            modal_features["audio"] = self.audio_encoder(inputs["audio"])
            
        # Process video
        if "video" in inputs and inputs["video"] is not None:
            modal_features["video"] = self.video_encoder(inputs["video"])
        
        # Multi-modal fusion
        if len(modal_features) > 1:
            fused_output, cross_attention_maps, gated_features = self.fusion_layer(modal_features)
            
            # Classify dominant modality
            modality_probs = self.modality_classifier(fused_output.mean(axis=1))
            
            return {
                "fused_features": fused_output,
                "cross_attention_maps": cross_attention_maps,
                "modality_weights": gated_features,
                "modality_classification": modality_probs,
                "logits": self.output_projection(fused_output)
            }
        else:
            # Single modality
            single_features = list(modal_features.values())[0]
            return {
                "fused_features": single_features,
                "logits": self.output_projection(single_features)
            }

