#!/usr/bin/env python3
"""
Test Configuration for RT-DLM
Device-specific settings and test parameters.
"""

import jax
import os
from typing import Dict, Any

class TestConfig:
    """Configuration class for tests based on available hardware"""
    
    def __init__(self):
        self.device_type = self._detect_device()
        self.config = self._get_device_config()
    
    def _detect_device(self) -> str:
        """Detect available compute device"""
        try:
            if jax.device_count('tpu') > 0:
                return 'tpu'
            elif jax.device_count('gpu') > 0:
                return 'gpu'
            else:
                return 'cpu'
        except Exception:
            return 'cpu'
    
    def _get_device_config(self) -> Dict[str, Any]:
        """Get configuration based on device capabilities"""
        
        if self.device_type == 'tpu':
            return {
                'd_model': 1024,
                'num_heads': 16,
                'num_layers': 12,
                'vocab_size': 50000,
                'max_seq_length': 4096,
                'batch_size': 8,
                'audio_frames': 256,
                'video_frames': 32,
                'video_size': 224,
                'enable_multimodal': True,
                'enable_consciousness': True,
                'enable_hybrid': True
            }
        elif self.device_type == 'gpu':
            return {
                'd_model': 512,
                'num_heads': 8,
                'num_layers': 6,
                'vocab_size': 10000,
                'max_seq_length': 1024,
                'batch_size': 4,
                'audio_frames': 128,
                'video_frames': 16,
                'video_size': 128,
                'enable_multimodal': True,
                'enable_consciousness': True,
                'enable_hybrid': True
            }
        else:  # CPU
            return {
                'd_model': 256,
                'num_heads': 4,
                'num_layers': 3,
                'vocab_size': 5000,
                'max_seq_length': 512,
                'batch_size': 1,
                'audio_frames': 64,
                'video_frames': 8,
                'video_size': 64,
                'enable_multimodal': False,  # Disable for CPU
                'enable_consciousness': False,
                'enable_hybrid': False
            }
    
    def setup_jax_config(self):
        """Setup JAX configuration for the device"""
        if self.device_type == 'cpu':
            # CPU optimizations
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            jax.config.update('jax_platform_name', 'cpu')
        elif self.device_type == 'gpu':
            # GPU optimizations
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
            jax.config.update('jax_enable_x64', False)
        # TPU uses default settings
    
    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            'd_model': self.config['d_model'],
            'num_heads': self.config['num_heads'],
            'num_layers': self.config['num_layers'],
            'vocab_size': self.config['vocab_size'],
            'max_seq_length': self.config['max_seq_length'],
            'multimodal_enabled': self.config['enable_multimodal'],
            'consciousness_simulation': self.config['enable_consciousness'],
            'scientific_reasoning': True,
            'creative_generation': True
        }
    
    def print_info(self):
        """Print configuration information"""
        print(f"Device Type: {self.device_type.upper()}")
        print(f"JAX Devices: {jax.devices()}")
        print("Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

# Global test configuration instance
test_config = TestConfig()

