import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class HybridVideoEncoder(hk.Module):
    """Hybrid video encoder combining multiple computer vision approaches"""
    
    def __init__(self, d_model: int, frame_height: int = 224, frame_width: int = 224, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Frame-level processing (combines CNN + ViT approaches)
        self.frame_encoder = HybridFrameEncoder(d_model)
        
        # Temporal modeling
        self.temporal_encoder = TemporalEncoder(d_model)
        
        # Object tracking module
        self.object_tracker = ObjectTrackingModule(d_model)
        
        # Action recognition module
        self.action_recognizer = ActionRecognitionModule(d_model)
        
        # Scene understanding module
        self.scene_analyzer = SceneUnderstandingModule(d_model)
        
        # Motion analysis
        self.motion_analyzer = MotionAnalysisModule(d_model)
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(d_model)
        
        # Task router for different video understanding tasks
        self.task_router = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.softmax
        ], name="task_router")
    
    def __call__(self, video_input: jnp.ndarray, task_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process video through hybrid architecture
        
        Args:
            video_input: Video frames [batch, num_frames, height, width, channels]
            task_hint: Optional hint about the task ('tracking', 'action', 'scene')
        """
        
        # Frame-level encoding
        frame_features = self.frame_encoder(video_input)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(frame_features)
        
        # Object tracking
        tracking_output = self.object_tracker(frame_features, temporal_features)
        
        # Action recognition
        action_output = self.action_recognizer(temporal_features)
        
        # Scene understanding
        scene_output = self.scene_analyzer(frame_features, temporal_features)
        
        # Motion analysis
        motion_output = self.motion_analyzer(frame_features)
        
        # Multi-scale feature fusion
        fused_features = self.feature_fusion([
            frame_features, temporal_features, 
            tracking_output['features'], action_output['features'],
            scene_output['features'], motion_output['features']
        ])
        
        # Task-specific routing
        task_weights = self.task_router(fused_features.mean(axis=1))
        
        # Combine outputs based on task
        if task_hint == 'tracking':
            primary_output = tracking_output['features']
        elif task_hint == 'action':
            primary_output = action_output['features']
        elif task_hint == 'scene':
            primary_output = scene_output['features']
        else:
            # Automatic weighting
            primary_output = (
                task_weights[:, 0:1, None] * tracking_output['features'] +
                task_weights[:, 1:2, None] * action_output['features'] +
                task_weights[:, 2:3, None] * scene_output['features'] +
                task_weights[:, 3:4, None] * motion_output['features']
            )
        
        return {
            'primary_features': primary_output,
            'frame_features': frame_features,
            'temporal_features': temporal_features,
            'tracking_analysis': tracking_output,
            'action_analysis': action_output,
            'scene_analysis': scene_output,
            'motion_analysis': motion_output,
            'task_weights': task_weights,
            'fused_features': fused_features
        }


class HybridFrameEncoder(hk.Module):
    """Hybrid frame encoder combining CNN and ViT approaches"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # CNN backbone (ResNet-like)
        self.cnn_backbone = CNNBackbone(d_model)
        
        # Vision Transformer branch
        self.vit_branch = VisionTransformerBranch(d_model)
        
        # Feature fusion
        self.fusion_layer = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="feature_fusion")
    
    def __call__(self, video_frames: jnp.ndarray) -> jnp.ndarray:
        """Encode video frames"""
        batch_size, num_frames, height, width, channels = video_frames.shape
        
        # Reshape for frame-by-frame processing
        frames_reshaped = video_frames.reshape(-1, height, width, channels)
        
        # CNN features
        cnn_features = self.cnn_backbone(frames_reshaped)
        
        # ViT features
        vit_features = self.vit_branch(frames_reshaped)
        
        # Fuse CNN and ViT features
        combined_features = jnp.concatenate([cnn_features, vit_features], axis=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # Reshape back to video format
        output = fused_features.reshape(batch_size, num_frames, -1)
        
        return output


class CNNBackbone(hk.Module):
    """CNN backbone for frame processing (ResNet-inspired)"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Convolutional layers
        self.conv_layers = []
        channels = [64, 128, 256, 512]
        
        for i, out_channels in enumerate(channels):
            conv_block = ResNetBlock(out_channels, name=f"resnet_block_{i}")
            self.conv_layers.append(conv_block)
        
        # Global average pooling and projection
        self.final_projection = hk.Linear(d_model, name="final_projection")
    
    def __call__(self, frames: jnp.ndarray) -> jnp.ndarray:
        """Extract CNN features from frames"""
        x = frames
        
        # Apply conv blocks
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Global average pooling
        pooled = jnp.mean(x, axis=(1, 2))  # [batch, channels]
        
        # Project to d_model
        output = self.final_projection(pooled)
        
        return output


class ResNetBlock(hk.Module):
    """ResNet-style convolutional block"""
    
    def __init__(self, out_channels: int, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ResNet block"""
        # First conv
        conv1 = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=3,
            stride=1,
            padding='SAME',
            name="conv1"
        )
        
        # Second conv
        conv2 = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=3,
            stride=1,
            padding='SAME',
            name="conv2"
        )
        
        # Downsample for spatial reduction
        downsample = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=1,
            stride=2,
            padding='SAME',
            name="downsample"
        )
        
        # Apply operations
        residual = downsample(x)
        
        out = conv1(x)
        out = jax.nn.relu(out)
        out = conv2(out)
        
        # Downsample output to match residual
        out = hk.avg_pool(out, window_shape=2, strides=2, padding='SAME')
        
        # Residual connection
        if out.shape == residual.shape:
            output = residual + out
        else:
            # Adjust dimensions if needed
            if residual.shape[-1] != out.shape[-1]:
                residual_proj = hk.Conv2D(
                    output_channels=self.out_channels,
                    kernel_shape=1,
                    name="residual_proj"
                )
                residual = residual_proj(residual)
            output = residual + out
        
        return jax.nn.relu(output)


class VisionTransformerBranch(hk.Module):
    """Vision Transformer branch for frame processing"""
    
    def __init__(self, d_model: int, patch_size: int = 16, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embedding = hk.Conv2D(
            output_channels=d_model,
            kernel_shape=patch_size,
            stride=patch_size,
            padding='VALID',
            name="patch_embedding"
        )
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(4):  # 4 transformer layers
            layer = hk.MultiHeadAttention(
                num_heads=8, 
                key_size=d_model//8, 
                w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)
        
        # Layer norms
        self.layer_norms = []
        for i in range(4):
            norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"layer_norm_{i}")
            self.layer_norms.append(norm)
    
    def __call__(self, frames: jnp.ndarray) -> jnp.ndarray:
        """Extract ViT features from frames"""
        batch_size, _, _, _ = frames.shape
        
        # Create patches
        patches = self.patch_embedding(frames)  # [batch, h_patches, w_patches, d_model]
        
        # Flatten patches
        patches_flat = patches.reshape(batch_size, -1, self.d_model)
        
        # Add positional encoding
        num_patches = patches_flat.shape[1]
        pos_encoding = hk.get_parameter(
            "pos_encoding", 
            shape=[1, num_patches, self.d_model], 
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        
        x = patches_flat + pos_encoding
        
        # Apply transformer layers
        for transformer_layer, layer_norm in zip(self.transformer_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = transformer_layer(x, x, x)
            x = residual + x
        
        # Global average pooling
        output = jnp.mean(x, axis=1)  # [batch, d_model]
        
        return output


class TemporalEncoder(hk.Module):
    """Temporal encoder for modeling frame sequences"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Temporal attention
        self.temporal_attention = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="temporal_attention"
        )
        
        # Temporal convolution (3D-like)
        self.temporal_conv = hk.Conv1D(
            output_channels=d_model,
            kernel_shape=3,
            padding='SAME',
            name="temporal_conv"
        )
        
        # Frame difference modeling
        self.difference_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="difference_encoder")
    
    def __call__(self, frame_features: jnp.ndarray) -> jnp.ndarray:
        """Encode temporal relationships between frames"""
        # Apply temporal attention
        attended_features = self.temporal_attention(
            frame_features, frame_features, frame_features
        )
        
        # Apply temporal convolution
        conv_features = self.temporal_conv(attended_features)
        
        # Frame difference modeling
        frame_diffs = jnp.diff(frame_features, axis=1, prepend=frame_features[:, :1])
        diff_features = self.difference_encoder(frame_diffs)
        
        # Combine all temporal features
        temporal_output = attended_features + conv_features + diff_features
        
        return temporal_output


class ObjectTrackingModule(hk.Module):
    """Object tracking and detection module"""
    
    def __init__(self, d_model: int, max_objects: int = 10, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_objects = max_objects
        
        # Object detector
        self.object_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(max_objects),
            jax.nn.sigmoid
        ], name="object_detector")
        
        # Object feature extractor
        self.object_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="object_encoder")
        
        # Tracking state
        self.track_attention = hk.MultiHeadAttention(
            num_heads=4,
            key_size=d_model//4,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="track_attention"
        )
    
    def __call__(self, frame_features: jnp.ndarray, temporal_features: jnp.ndarray) -> Dict[str, Any]:
        """Track objects across frames"""
        # Detect objects in each frame
        object_scores = self.object_detector(frame_features)
        
        # Extract object features
        object_features = self.object_encoder(frame_features)
        
        # Track objects across time using attention
        tracked_features = self.track_attention(
            temporal_features, object_features, object_features
        )
        
        return {
            'features': tracked_features,
            'object_scores': object_scores,
            'object_features': object_features,
            'tracking_confidence': jnp.mean(object_scores)
        }


class ActionRecognitionModule(hk.Module):
    """Action recognition module"""
    
    def __init__(self, d_model: int, num_actions: int = 20, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_actions = num_actions
        
        # Action classifier
        self.action_classifier = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(num_actions),
            jax.nn.softmax
        ], name="action_classifier")
        
        # Temporal pooling for action recognition
        self.temporal_pooler = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="temporal_pooler")
        
        # Motion emphasis
        self.motion_enhancer = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="motion_enhancer")
    
    def __call__(self, temporal_features: jnp.ndarray) -> Dict[str, Any]:
        """Recognize actions in video"""
        # Pool temporal features
        pooled_features = self.temporal_pooler(temporal_features.mean(axis=1))
        
        # Enhance motion-related features
        motion_enhanced = self.motion_enhancer(temporal_features)
        
        # Classify actions
        action_probs = self.action_classifier(pooled_features)
        
        return {
            'features': motion_enhanced,
            'action_probabilities': action_probs,
            'pooled_features': pooled_features,
            'action_confidence': jnp.max(action_probs, axis=-1)
        }


class SceneUnderstandingModule(hk.Module):
    """Scene understanding and spatial relationship module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Scene classifier
        self.scene_classifier = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(10),  # Scene types
            jax.nn.softmax
        ], name="scene_classifier")
        
        # Spatial relationship analyzer
        self.spatial_analyzer = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="spatial_analyzer"
        )
        
        # Depth estimator (monocular)
        self.depth_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="depth_estimator")
    
    def __call__(self, frame_features: jnp.ndarray, temporal_features: jnp.ndarray) -> Dict[str, Any]:
        """Understand scene structure and relationships"""
        # Classify scene type
        scene_probs = self.scene_classifier(frame_features.mean(axis=1))
        
        # Analyze spatial relationships
        spatial_features = self.spatial_analyzer(
            frame_features, temporal_features, temporal_features
        )
        
        # Estimate depth
        depth_maps = self.depth_estimator(frame_features)
        
        return {
            'features': spatial_features,
            'scene_probabilities': scene_probs,
            'depth_estimates': depth_maps,
            'scene_confidence': jnp.max(scene_probs, axis=-1)
        }


class MotionAnalysisModule(hk.Module):
    """Motion analysis and optical flow module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Motion detector
        self.motion_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(2),  # x, y motion vectors
            jax.nn.tanh
        ], name="motion_detector")
        
        # Motion magnitude estimator
        self.motion_magnitude = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="motion_magnitude")
    
    def __call__(self, frame_features: jnp.ndarray) -> Dict[str, Any]:
        """Analyze motion in video"""
        # Detect motion vectors
        motion_vectors = self.motion_detector(frame_features)
        
        # Estimate motion magnitude
        motion_mag = self.motion_magnitude(frame_features)
        
        return {
            'features': frame_features,
            'motion_vectors': motion_vectors,
            'motion_magnitude': motion_mag,
            'motion_confidence': jnp.mean(motion_mag)
        }


class MultiScaleFeatureFusion(hk.Module):
    """Multi-scale feature fusion for video understanding"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Attention-based fusion
        self.fusion_attention = hk.MultiHeadAttention(
            num_heads=8,
            key_size=d_model//8,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="fusion_attention"
        )
        
        # Feature weighting - outputs num_features weights
        self.feature_weights_proj = hk.Linear(d_model, name="feature_weights_proj")
        
        # Final projection
        self.final_projection = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="final_projection")
    
    def __call__(self, feature_list: List[jnp.ndarray]) -> jnp.ndarray:
        """Fuse multiple feature representations"""
        # Stack all features
        stacked_features = jnp.stack(feature_list, axis=2)  # [batch, time, num_features, d_model]
        
        # Flatten for attention
        batch_size, time_steps, num_features, d_model = stacked_features.shape
        features_flat = stacked_features.reshape(batch_size, time_steps * num_features, d_model)
        
        # Apply fusion attention
        fused = self.fusion_attention(features_flat, features_flat, features_flat)
        
        # Reshape and aggregate
        fused_reshaped = fused.reshape(batch_size, time_steps, num_features, d_model)
        
        # Weight different feature types
        pooled_features = fused_reshaped.mean(axis=1)
        feature_scores = self.feature_weights_proj(pooled_features).mean(axis=-1)
        feature_weights = jax.nn.softmax(feature_scores, axis=-1)
        
        weighted_features = jnp.sum(
            fused_reshaped * feature_weights[:, None, :, None], axis=2
        )
        
        output = self.final_projection(weighted_features)
        
        return output

