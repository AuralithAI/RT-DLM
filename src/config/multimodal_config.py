"""Multimodal sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MultimodalConfig:
    """Multi-modal processing settings."""

    multimodal_enabled: bool = True
    vision_patch_size: int = 16
    vision_layers: int = 6
    audio_freq_bins: int = 128
    video_frames: int = 16
    enable_multi_res_vision: bool = False
    vision_patch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    vision_in_channels: int = 3
    video_patch_size: int = 16
    video_motion_window: int = 4
    enable_document_modality: bool = False
    enable_pointcloud_modality: bool = False
    enable_biosignal_modality: bool = False
    enable_tactile_modality: bool = False
    enable_action_modality: bool = False
    action_num_axes: int = 15
    action_num_bins: int = 256
    enable_image_vq: bool = False
    image_vq_codes: int = 8192
    image_vq_code_dim: int = 256
    image_vq_downsample: int = 16
    enable_spectrogram_decoder: bool = False
    spectrogram_n_mels: int = 128
    streaming_video_max_frames: int = 32
    streaming_compressed_size: int = 64

    def validate(self) -> None:
        if self.multimodal_enabled:
            if self.vision_patch_size <= 0:
                raise ValueError("vision_patch_size must be positive")
            if self.audio_freq_bins <= 0:
                raise ValueError("audio_freq_bins must be positive")
            if self.video_frames <= 0:
                raise ValueError("video_frames must be positive")
        if self.enable_multi_res_vision:
            if not self.vision_patch_sizes:
                raise ValueError("vision_patch_sizes must be non-empty")
            if any(ps <= 0 for ps in self.vision_patch_sizes):
                raise ValueError("all vision_patch_sizes must be positive")
