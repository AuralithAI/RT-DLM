import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, Optional, List


def sinusoidal_timestamp_embedding(timestamps: jnp.ndarray, dim: int, max_period: float = 10000.0) -> jnp.ndarray:
    """Continuous-time sinusoidal embedding. timestamps [...,] -> [..., dim]."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / max(half, 1))
    args = timestamps[..., None] * freqs
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.concatenate([emb, jnp.zeros((*emb.shape[:-1], dim - emb.shape[-1]))], axis=-1)
    return emb


def modality_synchronization_loss(
    feats_a: jnp.ndarray,
    feats_b: jnp.ndarray,
    timestamps_a: jnp.ndarray,
    timestamps_b: jnp.ndarray,
    tolerance: float = 0.05,
) -> jnp.ndarray:
    """Penalize cosine dissimilarity between cross-modal tokens at near-equal timestamps."""
    a = feats_a / (jnp.linalg.norm(feats_a, axis=-1, keepdims=True) + 1e-6)
    b = feats_b / (jnp.linalg.norm(feats_b, axis=-1, keepdims=True) + 1e-6)
    sim = jnp.einsum("bnd,bmd->bnm", a, b)
    dt = jnp.abs(timestamps_a[:, :, None] - timestamps_b[:, None, :])
    mask = (dt <= tolerance).astype(jnp.float32)
    weight = mask.sum() + 1e-6
    return ((1.0 - sim) * mask).sum() / weight


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

    def __call__(
        self,
        query_modal,
        key_modal,
        value_modal,
        mask=None,
        query_timestamps: Optional[jnp.ndarray] = None,
        key_timestamps: Optional[jnp.ndarray] = None,
        temporal_bias_scale: float = 1.0,
    ):
        """Cross-modal attention with optional temporal proximity bias."""
        batch_size, seq_len = query_modal.shape[:2]

        Q = self.query_proj(query_modal)
        K = self.key_proj(key_modal)
        V = self.value_proj(value_modal)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)

        if query_timestamps is not None and key_timestamps is not None:
            dt = jnp.abs(query_timestamps[:, :, None] - key_timestamps[:, None, :])
            bias = -temporal_bias_scale * dt
            scores = scores + bias[:, None, :, :]

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
            self.modality_gates[modality] = hk.Sequential(
                [hk.Linear(d_model), jax.nn.silu, hk.Linear(1), jax.nn.sigmoid],
                name=f"gate_{modality}",
            )

        self.fusion_ffn = hk.Sequential(
            [
                hk.Linear(d_model * max(len(modalities), 2)),
                jax.nn.silu,
                hk.Linear(d_model),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            ],
            name="fusion_ffn",
        )

    def __call__(
        self,
        modal_inputs: Dict[str, jnp.ndarray],
        modal_masks: Optional[Dict[str, Optional[jnp.ndarray]]] = None,
    ):
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
                            source_features,
                            target_features,
                            target_features,
                            mask=modal_masks.get(target_mod),
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


class MultiResolutionPatchEmbed(hk.Module):
    """Multi-resolution patch embedding for vision transformers.

    Instead of a single fixed patch size, uses multiple convolutional kernels
    with different patch sizes.  Each kernel produces a different spatial
    resolution.  The outputs are concatenated along the sequence dimension and
    a learned resolution embedding is added so the transformer can distinguish
    the resolution of each token.

    Supported patch sizes are given at init (e.g. [8, 16, 32]).  For a 224×224
    image these yield 784, 196, and 49 tokens respectively.

    Positional embeddings are *interpolated* at runtime from a canonical
    resolution so that the module works with arbitrary input sizes.
    """

    def __init__(
        self,
        d_model: int,
        patch_sizes: Optional[List[int]] = None,
        canonical_img_size: int = 224,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_sizes = patch_sizes or [8, 16, 32]
        self.canonical_img_size = canonical_img_size

        # One Conv2D per patch size (acts as the patch projection)
        self.patch_convs = {}
        for ps in self.patch_sizes:
            self.patch_convs[ps] = hk.Conv2D(
                d_model,
                kernel_shape=ps,
                stride=ps,
                padding="VALID",
                name=f"patch_conv_{ps}",
            )

        # Linear projection to merge if needed
        self.merge_proj = hk.Linear(d_model, name="multi_res_merge")

    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """Embed images using multiple patch resolutions.

        Args:
            images: [batch, H, W, C]

        Returns:
            tokens: [batch, total_num_patches, d_model]
        """
        batch_size = images.shape[0]
        all_tokens = []

        for i, ps in enumerate(self.patch_sizes):
            # Conv2D patch projection  → [batch, h//ps, w//ps, d_model]
            patch_features = self.patch_convs[ps](images)
            ph, pw = patch_features.shape[1], patch_features.shape[2]
            num_patches = ph * pw

            # Flatten spatial  → [batch, num_patches, d_model]
            tokens = patch_features.reshape(batch_size, num_patches, self.d_model)

            # Canonical positional embedding (one per patch size)
            canonical_num = (self.canonical_img_size // ps) ** 2
            pos_embed = hk.get_parameter(
                f"pos_embed_ps{ps}",
                [1, canonical_num, self.d_model],
                init=hk.initializers.TruncatedNormal(0.02),
            )

            # Interpolate if actual num_patches differs from canonical
            if num_patches != canonical_num:
                pos_embed = jax.image.resize(
                    pos_embed,
                    (1, num_patches, self.d_model),
                    method="linear",
                )

            tokens = tokens + pos_embed

            # Resolution embedding — learned scalar per resolution
            res_embed = hk.get_parameter(
                f"res_embed_ps{ps}",
                [1, 1, self.d_model],
                init=hk.initializers.TruncatedNormal(0.02),
            )
            tokens = tokens + res_embed

            all_tokens.append(tokens)

        # Concatenate tokens from all resolutions  → [batch, Σ(num_patches_i), d_model]
        combined = jnp.concatenate(all_tokens, axis=1)

        # Linear merge to re-project
        combined = self.merge_proj(combined)

        return combined


class VisionEncoder(hk.Module):
    """Vision Transformer + CNN hybrid for image processing"""

    def __init__(
        self,
        d_model: int,
        patch_size: int = 16,
        num_layers: int = 6,
        use_multi_resolution: bool = False,
        multi_res_patch_sizes: Optional[List[int]] = None,
        in_channels: int = 3,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.use_multi_resolution = use_multi_resolution
        self.in_channels = in_channels

        # CNN feature extractor
        self.cnn_backbone = hk.Sequential(
            [
                hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME"),
                jax.nn.relu,
                hk.Conv2D(128, kernel_shape=3, stride=2, padding="SAME"),
                jax.nn.relu,
                hk.Conv2D(256, kernel_shape=3, stride=2, padding="SAME"),
                jax.nn.relu,
                hk.Conv2D(d_model, kernel_shape=3, stride=1, padding="SAME"),
            ],
            name="cnn_backbone",
        )

        # Multi-resolution or standard patch embedding
        if use_multi_resolution:
            self.multi_res_embed = MultiResolutionPatchEmbed(
                d_model=d_model,
                patch_sizes=multi_res_patch_sizes,
                name="multi_res_patch_embed",
            )

        # Standard patch embedding (used when not multi-resolution)
        self.patch_embed = hk.Linear(d_model, name="patch_embed")
        self.pos_embed = hk.get_parameter("pos_embed", [1, 196, d_model], init=hk.initializers.TruncatedNormal(0.02))

        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                hk.MultiHeadAttention(
                    num_heads=8,
                    key_size=d_model // 8,
                    w_init=hk.initializers.VarianceScaling(1.0),
                    name=f"vit_layer_{i}",
                )
            )

        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, images):
        batch_size = images.shape[0]
        if images.shape[-1] != self.in_channels:
            if images.shape[-1] < self.in_channels:
                pad = self.in_channels - images.shape[-1]
                images = jnp.concatenate([images, jnp.zeros((*images.shape[:-1], pad), dtype=images.dtype)], axis=-1)
            else:
                images = images[..., : self.in_channels]

        # Multi-resolution path: bypass CNN, use multi-res patch embedding directly
        if self.use_multi_resolution:
            x = self.multi_res_embed(images)
            # Apply transformer layers on multi-res tokens
            for layer in self.transformer_layers:
                x = self.norm(x + layer(x, x, x))
            return x

        # Standard path: CNN features → ViT
        cnn_features = self.cnn_backbone(images)
        h, w = cnn_features.shape[1:3]

        # Convert to patches for ViT
        patches = cnn_features.reshape(batch_size, h * w, self.d_model)

        # Add positional encoding
        if patches.shape[1] <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, : patches.shape[1], :]
        else:
            # Interpolate if needed
            pos_embed = jax.image.resize(self.pos_embed, (1, patches.shape[1], self.d_model), method="linear")

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
        self.conv_layers = hk.Sequential(
            [
                hk.Conv1D(64, kernel_shape=3, stride=2, padding="SAME"),
                jax.nn.relu,
                hk.Conv1D(128, kernel_shape=3, stride=2, padding="SAME"),
                jax.nn.relu,
                hk.Conv1D(d_model, kernel_shape=3, stride=1, padding="SAME"),
            ],
            name="audio_conv",
        )

        # Temporal attention
        self.temporal_attention = hk.MultiHeadAttention(
            num_heads=8,
            key_size=d_model // 8,
            w_init=hk.initializers.VarianceScaling(1.0),
            name="temporal_attn",
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


def apply_3d_rope(x: jnp.ndarray, axis_lens: tuple, base: float = 10000.0) -> jnp.ndarray:
    """3D RoPE applied as a pre-attention rotation. axis_lens=(T,H,W). Splits d_model into 3 axes."""
    *_, seq, d = x.shape
    T, H, W = axis_lens
    assert seq == T * H * W, f"seq {seq} != T*H*W={T*H*W}"
    third = d // 6
    if third == 0:
        return x

    def rope_axis(positions: jnp.ndarray, dims: int) -> jnp.ndarray:
        freqs = base ** (-jnp.arange(0, dims, dtype=jnp.float32) / dims)
        ang = positions[:, None] * freqs[None, :]
        return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)

    t_idx = jnp.repeat(jnp.arange(T), H * W)
    h_idx = jnp.tile(jnp.repeat(jnp.arange(H), W), T)
    w_idx = jnp.tile(jnp.arange(W), T * H)

    rot_t = rope_axis(t_idx, third)
    rot_h = rope_axis(h_idx, third)
    rot_w = rope_axis(w_idx, third)
    rot = jnp.concatenate([rot_t, rot_h, rot_w], axis=-1)
    pad = d - rot.shape[-1]
    if pad > 0:
        rot = jnp.concatenate([rot, jnp.ones((seq, pad))], axis=-1)
    while rot.ndim < x.ndim:
        rot = rot[None, ...]
    return x * rot


class VideoEncoder(hk.Module):
    """Spatiotemporal video encoder: per-frame patches + motion + 3D RoPE."""

    def __init__(
        self,
        d_model: int,
        patch_size: int = 16,
        max_frames: int = 16,
        motion_window: int = 4,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.motion_window = motion_window

        self.frame_encoder = VisionEncoder(d_model, patch_size=patch_size, num_layers=2, name="video_frame_encoder")
        self.motion_proj = hk.Linear(d_model // 4, name="motion_proj")
        self.fuse_proj = hk.Linear(d_model, name="motion_fuse_proj")

        self.temporal_layers = []
        for i in range(3):
            self.temporal_layers.append(
                hk.MultiHeadAttention(
                    num_heads=8,
                    key_size=d_model // 8,
                    w_init=hk.initializers.VarianceScaling(1.0),
                    name=f"st_layer_{i}",
                )
            )
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def _motion_features(self, frames: jnp.ndarray, target_patches: int) -> jnp.ndarray:
        """frames [B,F,H,W,C] -> motion tokens [B,(F-1),target_patches,d_model/4]."""
        diffs = frames[:, 1:] - frames[:, :-1]
        b, fm1, h, w, c = diffs.shape
        flat = diffs.reshape(b * fm1, h, w, c)
        x = hk.Conv2D(32, kernel_shape=3, stride=2, padding="SAME", name="motion_c1")(flat)
        x = jax.nn.silu(x)
        x = hk.Conv2D(64, kernel_shape=3, stride=2, padding="SAME", name="motion_c2")(x)
        x = jax.nn.silu(x)
        side = max(int(target_patches**0.5), 1)
        x = jax.image.resize(x, (b * fm1, side, side, 64), method="linear")
        x = self.motion_proj(x)
        x = x.reshape(b, fm1, side * side, -1)
        if x.shape[2] != target_patches:
            x = jax.image.resize(x, (b, fm1, target_patches, x.shape[-1]), method="linear")
        return x

    def _build_st_mask(self, frames: int, patches: int, max_back: int) -> jnp.ndarray:
        """Sparse temporal causal mask: full intra-frame, sparse inter-frame within max_back."""
        seq = frames * patches
        f_idx = jnp.repeat(jnp.arange(frames), patches)
        diff = f_idx[None, :] - f_idx[:, None]
        intra = diff == 0
        within = (diff > 0) & (diff <= max_back)
        return (intra | within).astype(jnp.float32).reshape(1, 1, seq, seq)

    def __call__(self, video_frames: jnp.ndarray) -> jnp.ndarray:
        """video_frames [B,F,H,W,C] -> [B, F*patches, d_model]."""
        b, fr, h, w, c = video_frames.shape
        flat_frames = video_frames.reshape(b * fr, h, w, c)
        frame_tokens = self.frame_encoder(flat_frames)
        patches = frame_tokens.shape[1]
        appearance = frame_tokens.reshape(b, fr, patches, self.d_model)

        motion = self._motion_features(video_frames, target_patches=patches)
        motion_padded = jnp.concatenate([jnp.zeros((b, 1, patches, motion.shape[-1])), motion], axis=1)

        appearance_motion = jnp.concatenate([appearance, motion_padded], axis=-1)
        fused = self.fuse_proj(appearance_motion)

        ph = pw = int(patches**0.5)
        if ph * pw != patches:
            ph, pw = 1, patches
        tokens = fused.reshape(b, fr * patches, self.d_model)
        tokens = apply_3d_rope(tokens, (fr, ph, pw))

        mask = self._build_st_mask(fr, patches, self.motion_window)
        for layer in self.temporal_layers:
            attn = layer(tokens, tokens, tokens, mask=mask)
            tokens = self.norm(tokens + attn)
        return tokens


class MultiModalRTDLM(hk.Module):
    """Complete Multi-Modal RT-DLM with advanced fusion"""

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config

        self.text_encoder = None
        self.vision_encoder = VisionEncoder(
            config.d_model,
            patch_size=getattr(config, "vision_patch_size", 16),
            num_layers=getattr(config, "vision_layers", 6),
            use_multi_resolution=getattr(config, "enable_multi_res_vision", False),
            multi_res_patch_sizes=getattr(config, "vision_patch_sizes", None),
            in_channels=getattr(config, "vision_in_channels", 3),
        )
        self.audio_encoder = AudioEncoder(config.d_model)
        self.video_encoder = VideoEncoder(
            config.d_model,
            patch_size=getattr(config, "video_patch_size", 16),
            max_frames=getattr(config, "video_frames", 16),
            motion_window=getattr(config, "video_motion_window", 4),
        )

        self.enable_document = getattr(config, "enable_document_modality", False)
        self.enable_pointcloud = getattr(config, "enable_pointcloud_modality", False)
        self.enable_biosignal = getattr(config, "enable_biosignal_modality", False)
        self.enable_tactile = getattr(config, "enable_tactile_modality", False)
        self.enable_action = getattr(config, "enable_action_modality", False)

        if self.enable_document:
            from src.modules.multimodal.document_encoder import DocumentEncoder

            self.document_encoder = DocumentEncoder(config.d_model, name="document_encoder")
        if self.enable_pointcloud:
            from src.modules.multimodal.point_cloud_encoder import PointCloudEncoder

            self.pointcloud_encoder = PointCloudEncoder(config.d_model, name="pointcloud_encoder")
        if self.enable_biosignal:
            from src.modules.multimodal.biosignal_encoder import BiosignalEncoder

            self.biosignal_encoder = BiosignalEncoder(config.d_model, name="biosignal_encoder")
        if self.enable_tactile:
            from src.modules.multimodal.tactile_encoder import TactileEncoder

            self.tactile_encoder = TactileEncoder(config.d_model, name="tactile_encoder")
        if self.enable_action:
            from src.modules.multimodal.action_encoder import ActionEncoder

            self.action_encoder = ActionEncoder(
                config.d_model,
                num_axes=getattr(config, "action_num_axes", 15),
                num_bins=getattr(config, "action_num_bins", 256),
                name="action_encoder",
            )

        modalities = ["text", "vision", "audio", "video"]
        if self.enable_document:
            modalities.append("document")
        if self.enable_pointcloud:
            modalities.append("pointcloud")
        if self.enable_biosignal:
            modalities.append("biosignal")
        if self.enable_tactile:
            modalities.append("tactile")
        if self.enable_action:
            modalities.append("action")
        self.active_modalities = modalities

        self.fusion_layer = MultiModalFusionLayer(config.d_model, config.num_heads, modalities=modalities)

        self.output_projection = hk.Linear(config.vocab_size, name="output_proj")
        self.modality_classifier = hk.Sequential(
            [hk.Linear(config.d_model), jax.nn.silu, hk.Linear(len(modalities)), jax.nn.softmax],
            name="modality_classifier",
        )

    def __call__(self, inputs: Dict[str, jnp.ndarray], text_features=None):
        modal_features: Dict[str, jnp.ndarray] = {}

        if text_features is not None:
            modal_features["text"] = text_features
        if "images" in inputs and inputs["images"] is not None:
            modal_features["vision"] = self.vision_encoder(inputs["images"])
        if "audio" in inputs and inputs["audio"] is not None:
            modal_features["audio"] = self.audio_encoder(inputs["audio"])
        if "video" in inputs and inputs["video"] is not None:
            modal_features["video"] = self.video_encoder(inputs["video"])
        if self.enable_document and "document" in inputs and inputs["document"] is not None:
            modal_features["document"] = self.document_encoder(inputs["document"])["tokens"]
        if self.enable_pointcloud and "pointcloud" in inputs and inputs["pointcloud"] is not None:
            pc_out = self.pointcloud_encoder(inputs["pointcloud"])
            modal_features["pointcloud"] = pc_out["local"]
        if self.enable_biosignal and "biosignal" in inputs and inputs["biosignal"] is not None:
            modal_features["biosignal"] = self.biosignal_encoder(inputs["biosignal"])["features"]
        if self.enable_tactile and "tactile" in inputs and inputs["tactile"] is not None:
            tac = self.tactile_encoder(inputs["tactile"])
            modal_features["tactile"] = tac["temporal"]
        if self.enable_action and "action" in inputs and inputs["action"] is not None:
            modal_features["action"] = self.action_encoder(inputs["action"])["features"]

        if len(modal_features) > 1:
            fused_output, cross_attention_maps, gated_features = self.fusion_layer(modal_features)
            modality_probs = self.modality_classifier(fused_output.mean(axis=1))
            return {
                "fused_features": fused_output,
                "cross_attention_maps": cross_attention_maps,
                "modality_weights": gated_features,
                "modality_classification": modality_probs,
                "logits": self.output_projection(fused_output),
                "modal_features": modal_features,
            }
        elif modal_features:
            single_features = next(iter(modal_features.values()))
            return {
                "fused_features": single_features,
                "logits": self.output_projection(single_features),
                "modal_features": modal_features,
            }
        else:
            raise ValueError("No active modality inputs")
