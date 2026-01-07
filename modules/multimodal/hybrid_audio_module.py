import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import librosa
import scipy.signal
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

class HybridAudioEncoder(hk.Module):
    """Hybrid audio encoder that combines multiple ML approaches"""
    
    def __init__(self, d_model: int, sample_rate: int = 16000, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.sample_rate = sample_rate
        
        # Traditional signal processing backbone
        self.signal_processor = SignalProcessingBackbone(d_model)
        
        # CNN for local patterns (like ConvNeXt)
        self.cnn_encoder = CNNAudioEncoder(d_model)
        
        # RNN for temporal modeling (like LSTM)
        self.rnn_encoder = RNNAudioEncoder(d_model)
        
        # Transformer for long-range dependencies
        self.transformer_encoder = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="audio_transformer"
        )
        
        # Speech recognition module
        self.speech_module = SpeechRecognitionModule(d_model)
        
        # Music analysis module
        self.music_module = MusicAnalysisModule(d_model)
        
        # Emotion detection module
        self.emotion_module = AudioEmotionModule(d_model)
        
        # Feature fusion layer
        self.feature_fusion = hk.Sequential([
            hk.Linear(d_model * 4),  # CNN + RNN + Transformer + Signal
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="feature_fusion")
        
        # Task-specific heads
        self.task_router = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.softmax
        ], name="task_router")
    
    def __call__(self, audio_input: jnp.ndarray, task_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio through hybrid architecture
        
        Args:
            audio_input: Audio waveform [batch, time] or spectrogram [batch, time, freq]
            task_hint: Optional hint about the audio type ('speech', 'music', 'environment')
        """
        
        # Ensure we have proper input format
        if len(audio_input.shape) == 2:
            # Convert waveform to spectrogram
            audio_features = self._waveform_to_features(audio_input)
        else:
            audio_features = audio_input
        
        # Traditional signal processing features
        signal_features = self.signal_processor(audio_features)
        
        # CNN features for local patterns
        cnn_features = self.cnn_encoder(audio_features)
        
        # RNN features for temporal modeling
        rnn_features = self.rnn_encoder(audio_features)
        
        # Transformer features for long-range dependencies
        transformer_features = self.transformer_encoder(
            audio_features, audio_features, audio_features
        )
        
        # Fuse all features
        all_features = jnp.concatenate([
            signal_features, cnn_features, rnn_features, transformer_features
        ], axis=-1)
        
        fused_features = self.feature_fusion(all_features)
        
        # Task-specific processing
        task_weights = self.task_router(fused_features.mean(axis=1))
        
        # Speech processing
        speech_output = self.speech_module(fused_features)
        
        # Music processing
        music_output = self.music_module(fused_features)
        
        # Emotion processing
        emotion_output = self.emotion_module(fused_features)
        
        # Weighted combination based on task
        if task_hint == 'speech':
            primary_output = speech_output['features']
        elif task_hint == 'music':
            primary_output = music_output['features']
        else:
            # Automatic weighting
            primary_output = (
                task_weights[:, 0:1, None] * speech_output['features'] +
                task_weights[:, 1:2, None] * music_output['features'] +
                task_weights[:, 2:3, None] * emotion_output['features']
            )
        
        return {
            'primary_features': primary_output,
            'speech_analysis': speech_output,
            'music_analysis': music_output,
            'emotion_analysis': emotion_output,
            'task_weights': task_weights,
            'raw_features': {
                'signal': signal_features,
                'cnn': cnn_features,
                'rnn': rnn_features,
                'transformer': transformer_features
            }
        }
    
    def _waveform_to_features(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Convert waveform to spectral features"""
        # Simple spectrogram computation (in practice, use proper STFT)
        # This is a placeholder - in real implementation, use librosa or JAX-based STFT
        batch_size, time_steps = waveform.shape
        
        # Create mock spectrogram features
        freq_bins = 128
        time_frames = time_steps // 256  # Typical hop length
        
        # Simulate spectral features
        features = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, time_frames, freq_bins)
        ) * 0.1
        
        return features


class SignalProcessingBackbone(hk.Module):
    """Traditional signal processing features as backbone"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        self.feature_extractor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="signal_features")
    
    def __call__(self, spectrogram: jnp.ndarray) -> jnp.ndarray:
        """Extract traditional signal processing features"""
        # Extract spectral features
        spectral_features = self._extract_spectral_features(spectrogram)
        
        # Encode features
        encoded = self.feature_extractor(spectral_features)
        
        return encoded
    
    def _extract_spectral_features(self, spectrogram: jnp.ndarray) -> jnp.ndarray:
        """Extract spectral features like MFCC, spectral centroid, etc."""
        batch_size, time_frames, freq_bins = spectrogram.shape
        
        # Spectral centroid (frequency-weighted average)
        freq_axis = jnp.arange(freq_bins)
        spectral_centroid = jnp.sum(spectrogram * freq_axis, axis=-1, keepdims=True) / (
            jnp.sum(spectrogram, axis=-1, keepdims=True) + 1e-8
        )
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = jnp.cumsum(spectrogram, axis=-1)
        total_energy = jnp.sum(spectrogram, axis=-1, keepdims=True)
        rolloff_threshold = 0.85 * total_energy
        rolloff_indices = jnp.argmax(cumulative_energy >= rolloff_threshold, axis=-1, keepdims=True)
        
        # Zero crossing rate (approximated from spectral features)
        zcr = jnp.mean(jnp.abs(jnp.diff(spectrogram, axis=-1)), axis=-1, keepdims=True)
        
        # RMS energy
        rms_energy = jnp.sqrt(jnp.mean(spectrogram**2, axis=-1, keepdims=True))
        
        # Combine features
        features = jnp.concatenate([
            spectral_centroid, rolloff_indices.astype(jnp.float32), zcr, rms_energy
        ], axis=-1)
        
        # Pad to d_model if needed
        if features.shape[-1] < self.d_model:
            padding = jnp.zeros((batch_size, time_frames, self.d_model - features.shape[-1]))
            features = jnp.concatenate([features, padding], axis=-1)
        elif features.shape[-1] > self.d_model:
            features = features[..., :self.d_model]
        
        return features


class CNNAudioEncoder(hk.Module):
    """CNN encoder for local audio patterns (ConvNeXt-inspired)"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Depthwise convolutions (like ConvNeXt)
        self.conv_blocks = []
        channels = [64, 128, 256, d_model]
        
        for i, out_channels in enumerate(channels):
            block = ConvNeXtBlock(out_channels, name=f"conv_block_{i}")
            self.conv_blocks.append(block)
    
    def __call__(self, spectrogram: jnp.ndarray) -> jnp.ndarray:
        """Extract CNN features from spectrogram"""
        x = spectrogram
        
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        return x


class ConvNeXtBlock(hk.Module):
    """ConvNeXt-inspired block for audio processing"""
    
    def __init__(self, out_channels: int, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, _, freq_bins = x.shape
        
        # Depthwise convolution (simulated)
        depthwise_conv = hk.Conv1D(
            output_channels=freq_bins,
            kernel_shape=7,
            padding='SAME',
            feature_group_count=freq_bins,
            name="depthwise_conv"
        )
        
        # Pointwise convolution
        pointwise_conv = hk.Conv1D(
            output_channels=self.out_channels,
            kernel_shape=1,
            name="pointwise_conv"
        )
        
        # Layer normalization
        layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm")
        
        # Feedforward network
        ffn = hk.Sequential([
            hk.Linear(self.out_channels * 4),
            jax.nn.gelu,
            hk.Linear(self.out_channels)
        ], name="ffn")
        
        # Apply operations
        residual = x
        
        # Reshape for conv1d (time dimension)
        x_reshaped = x.transpose(0, 2, 1)  # [batch, freq, time]
        x_conv = depthwise_conv(x_reshaped)
        x_conv = pointwise_conv(x_conv)
        x_conv = x_conv.transpose(0, 2, 1)  # Back to [batch, time, channels]
        
        # Ensure correct output channels
        if x_conv.shape[-1] != self.out_channels:
            projection = hk.Linear(self.out_channels, name="projection")
            x_conv = projection(x_conv)
        
        x_norm = layer_norm(x_conv)
        x_ffn = ffn(x_norm)
        
        # Residual connection (if dimensions match)
        if residual.shape[-1] == x_ffn.shape[-1]:
            output = residual + x_ffn
        else:
            # Project residual to match dimensions
            residual_proj = hk.Linear(self.out_channels, name="residual_proj")
            output = residual_proj(residual) + x_ffn
        
        return output


class RNNAudioEncoder(hk.Module):
    """RNN encoder for temporal audio modeling (simplified)"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Simple RNN-like temporal modeling using convolutions
        self.temporal_conv = hk.Conv1D(
            output_channels=d_model,
            kernel_shape=5,
            padding='SAME',
            name="temporal_conv"
        )
        
        # Attention for temporal modeling
        self.temporal_attention = hk.MultiHeadAttention(
            num_heads=4, 
            key_size=d_model//4, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="temporal_attention"
        )
        
        # Output projection
        self.output_proj = hk.Linear(d_model, name="output_proj")
    
    def __call__(self, spectrogram: jnp.ndarray) -> jnp.ndarray:
        """Extract temporal features from spectrogram"""
        # Apply temporal convolution
        x_transposed = spectrogram.transpose(0, 2, 1)  # [batch, freq, time]
        conv_output = self.temporal_conv(x_transposed)
        conv_output = conv_output.transpose(0, 2, 1)  # [batch, time, channels]
        
        # Apply temporal attention
        attended_output = self.temporal_attention(
            conv_output, conv_output, conv_output
        )
        
        # Project to final dimensions
        output = self.output_proj(attended_output)
        
        return output


class SpeechRecognitionModule(hk.Module):
    """Speech recognition and analysis module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Phoneme detector
        self.phoneme_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(128),  # Phoneme classes
            jax.nn.softmax
        ], name="phoneme_detector")
        
        # Speech activity detector
        self.speech_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="speech_detector")
        
        # Speaker embedding
        self.speaker_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(256)  # Speaker embedding
        ], name="speaker_encoder")
    
    def __call__(self, audio_features: jnp.ndarray) -> Dict[str, Any]:
        """Analyze speech content"""
        # Detect phonemes
        phoneme_probs = self.phoneme_detector(audio_features)
        
        # Detect speech activity
        speech_activity = self.speech_detector(audio_features)
        
        # Extract speaker embedding
        speaker_embedding = self.speaker_encoder(audio_features.mean(axis=1))
        
        return {
            'features': audio_features,
            'phoneme_probabilities': phoneme_probs,
            'speech_activity': speech_activity,
            'speaker_embedding': speaker_embedding,
            'speech_confidence': jnp.mean(speech_activity)
        }


class MusicAnalysisModule(hk.Module):
    """Music analysis and understanding module"""
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Chord recognition
        self.chord_detector = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(24),  # 12 major + 12 minor chords
            jax.nn.softmax
        ], name="chord_detector")
        
        # Tempo estimation
        self.tempo_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            lambda x: jax.nn.sigmoid(x) * 200 + 60  # 60-260 BPM
        ], name="tempo_estimator")
        
        # Genre classifier
        self.genre_classifier = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(10),  # Music genres
            jax.nn.softmax
        ], name="genre_classifier")
        
        # Beat tracker
        self.beat_tracker = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="beat_tracker")
    
    def __call__(self, audio_features: jnp.ndarray) -> Dict[str, Any]:
        """Analyze musical content"""
        # Detect chords
        chord_probs = self.chord_detector(audio_features)
        
        # Estimate tempo
        tempo = self.tempo_estimator(audio_features.mean(axis=1))
        
        # Classify genre
        genre_probs = self.genre_classifier(audio_features.mean(axis=1))
        
        # Track beats
        beat_strength = self.beat_tracker(audio_features)
        
        return {
            'features': audio_features,
            'chord_probabilities': chord_probs,
            'estimated_tempo': tempo,
            'genre_probabilities': genre_probs,
            'beat_strength': beat_strength,
            'musical_confidence': jnp.mean(chord_probs.max(axis=-1))
        }


class AudioEmotionModule(hk.Module):
    """Audio emotion recognition module with 14-emotion taxonomy"""
    
    # 14-emotion labels matching SocialEmotionalIntelligence
    EMOTION_LABELS = [
        "joy", "sadness", "anger", "fear", "disgust", 
        "surprise", "neutral", "anticipation", "trust",
        "love", "guilt", "pride", "confusion", "curiosity"
    ]
    NUM_EMOTIONS = 14
    
    def __init__(self, d_model: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        
        # Emotion classifier with 14 emotions
        self.emotion_classifier = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(self.NUM_EMOTIONS),  # 14 emotions matching SocialEmotionalIntelligence
            jax.nn.softmax
        ], name="emotion_classifier")
        
        # Valence predictor (positive/negative)
        self.valence_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.tanh  # -1 to 1
        ], name="valence_predictor")
        
        # Arousal predictor (calm/excited)
        self.arousal_predictor = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid  # 0 to 1
        ], name="arousal_predictor")
    
    def __call__(self, audio_features: jnp.ndarray) -> Dict[str, Any]:
        """Analyze emotional content with 14-emotion taxonomy"""
        # Classify emotions (14 classes)
        emotion_probs = self.emotion_classifier(audio_features.mean(axis=1))
        
        # Predict valence and arousal
        valence = self.valence_predictor(audio_features.mean(axis=1))
        arousal = self.arousal_predictor(audio_features.mean(axis=1))
        
        # Get dominant emotion index
        dominant_emotion_idx = jnp.argmax(emotion_probs, axis=-1)
        
        return {
            'features': audio_features,
            'emotion_probabilities': emotion_probs,
            'emotion_labels': self.EMOTION_LABELS,
            'num_emotions': self.NUM_EMOTIONS,
            'dominant_emotion_idx': dominant_emotion_idx,
            'valence': valence,
            'arousal': arousal,
            'emotion_confidence': jnp.max(emotion_probs, axis=-1)
        }

