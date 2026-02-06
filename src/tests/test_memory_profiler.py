"""
Tests for Memory Profiler Utilities.

Tests memory tracking, preset estimation, and batch memory calculations.
"""

import pytest
import jax.numpy as jnp

from src.core.memory_profiler import (
    MemorySnapshot,
    MemoryProfile,
    MemoryProfiler,
    estimate_memory_for_preset,
    estimate_batch_memory,
    get_all_preset_memory_requirements,
    print_memory_requirements_table,
    _recommend_gpu,
    _recommend_batch_size,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            step=100,
            phase="forward",
            bytes_in_use=1024 * 1024 * 1024,  # 1 GB
            peak_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
            device_id=0,
        )
        
        assert snapshot.step == 100
        assert snapshot.phase == "forward"
        assert snapshot.bytes_in_use == 1024 * 1024 * 1024
        assert snapshot.peak_bytes == 2 * 1024 * 1024 * 1024
        assert snapshot.device_id == 0
    
    def test_snapshot_default_device_id(self):
        """Test default device ID is 0."""
        snapshot = MemorySnapshot(
            timestamp=0.0,
            step=0,
            phase="idle",
            bytes_in_use=0,
            peak_bytes=0,
        )
        assert snapshot.device_id == 0


class TestMemoryProfile:
    """Tests for MemoryProfile dataclass."""
    
    def test_empty_profile(self):
        """Test empty profile properties."""
        profile = MemoryProfile()
        
        assert profile.peak_memory_gb == pytest.approx(0.0)
        assert profile.average_memory_gb == pytest.approx(0.0)
        assert len(profile.snapshots) == 0
    
    def test_profile_with_snapshots(self):
        """Test profile with multiple snapshots."""
        profile = MemoryProfile()
        
        # Add snapshots with varying memory usage
        profile.snapshots = [
            MemorySnapshot(0, 0, "forward", 1 * 1024**3, 2 * 1024**3),
            MemorySnapshot(1, 1, "backward", 2 * 1024**3, 3 * 1024**3),
            MemorySnapshot(2, 2, "optimizer", 1 * 1024**3, 3 * 1024**3),
        ]
        
        assert profile.peak_memory_gb == pytest.approx(3.0)  # Max peak
        assert profile.average_memory_gb == pytest.approx(4/3, rel=0.01)  # (1+2+1)/3
    
    def test_profile_summary(self):
        """Test profile summary generation."""
        profile = MemoryProfile()
        profile.model_params_bytes = 100 * 1024**2  # 100 MB
        profile.optimizer_state_bytes = 200 * 1024**2  # 200 MB
        profile.snapshots = [
            MemorySnapshot(0, 0, "forward", 1 * 1024**3, 2 * 1024**3),
        ]
        
        summary = profile.summary()
        
        assert "peak_memory_gb" in summary
        assert "average_memory_gb" in summary
        assert "num_snapshots" in summary
        assert summary["num_snapshots"] == 1


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = MemoryProfiler(enabled=True, log_every_n_steps=10)
        
        assert profiler.enabled
        assert profiler.log_every_n_steps == 10
    
    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        profiler = MemoryProfiler(enabled=False)
        
        # Snapshot should return None when disabled
        result = profiler.snapshot(step=0, phase="forward")
        assert result is None
    
    def test_profiler_snapshot(self):
        """Test taking a memory snapshot."""
        profiler = MemoryProfiler(enabled=True)
        
        # Snapshot may or may not return data depending on device
        result = profiler.snapshot(step=0, phase="forward")
        # Result can be None if device doesn't support memory stats
        # Just ensure it doesn't raise
        assert result is None or isinstance(result, MemorySnapshot)
    
    def test_profiler_set_model_size(self):
        """Test recording model size."""
        profiler = MemoryProfiler(enabled=True)
        
        # Create mock params
        params = {"w": jnp.ones((100, 100)), "b": jnp.zeros((100,))}
        
        # Should not raise
        profiler.set_model_size(params)
        assert profiler.profile.model_params_bytes > 0
    
    def test_profiler_set_optimizer_size(self):
        """Test recording optimizer state size."""
        profiler = MemoryProfiler(enabled=True)
        
        # Create mock optimizer state
        opt_state = {"momentum": jnp.zeros((100, 100)), "variance": jnp.zeros((100, 100))}
        
        profiler.set_optimizer_size(opt_state)
        assert profiler.profile.optimizer_state_bytes > 0
    
    def test_profiler_summary(self):
        """Test profiler summary."""
        profiler = MemoryProfiler()
        summary = profiler.summary()
        
        assert isinstance(summary, dict)
        assert "peak_memory_gb" in summary
        assert "average_memory_gb" in summary
        assert "num_snapshots" in summary
    
    def test_profiler_reset(self):
        """Test profiler reset."""
        profiler = MemoryProfiler()
        
        # Set some values
        profiler.profile.model_params_bytes = 1000
        
        # Reset
        profiler.profile = MemoryProfile()
        
        assert profiler.profile.model_params_bytes == 0
    
    def test_profiler_disabled_set_model_size(self):
        """Test set_model_size when disabled does nothing."""
        profiler = MemoryProfiler(enabled=False)
        params = {"w": jnp.ones((100, 100))}
        
        profiler.set_model_size(params)
        # Should not crash, and should not set size
        assert profiler.profile.model_params_bytes == 0


class TestEstimateMemoryForPreset:
    """Tests for estimate_memory_for_preset function."""
    
    @pytest.mark.parametrize("preset", ["tiny", "small", "base", "large", "xlarge", "xxlarge"])
    def test_valid_presets(self, preset):
        """Test all valid presets return proper estimates."""
        result = estimate_memory_for_preset(preset)
        
        assert "preset" in result
        assert result["preset"] == preset
        assert "parameters_millions" in result
        assert "d_model" in result
        assert "layers" in result
        assert "heads" in result
        assert "memory" in result
        assert "recommended_gpu" in result
        assert "recommended_batch_size" in result
    
    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            estimate_memory_for_preset("invalid_preset")
    
    def test_memory_fields(self):
        """Test memory fields are present and positive."""
        result = estimate_memory_for_preset("base")
        memory = result["memory"]
        
        assert "params_fp32_gb" in memory
        assert "params_fp16_gb" in memory
        assert "training_fp32_gb" in memory
        assert "training_fp16_gb" in memory
        assert "training_fp16_checkpointed_gb" in memory
        
        # All memory values should be positive
        assert all(v > 0 for v in memory.values())
    
    def test_memory_ordering(self):
        """Test FP16 with checkpointing uses less memory than FP32."""
        result = estimate_memory_for_preset("large")
        memory = result["memory"]
        
        assert memory["training_fp16_checkpointed_gb"] < memory["training_fp32_gb"]
        assert memory["training_fp16_gb"] < memory["training_fp32_gb"]
    
    def test_preset_size_ordering(self):
        """Test larger presets require more memory."""
        tiny = estimate_memory_for_preset("tiny")
        small = estimate_memory_for_preset("small")
        base = estimate_memory_for_preset("base")
        large = estimate_memory_for_preset("large")
        
        assert tiny["memory"]["training_fp16_gb"] < small["memory"]["training_fp16_gb"]
        assert small["memory"]["training_fp16_gb"] < base["memory"]["training_fp16_gb"]
        assert base["memory"]["training_fp16_gb"] < large["memory"]["training_fp16_gb"]


class TestEstimateBatchMemory:
    """Tests for estimate_batch_memory function."""
    
    def test_basic_estimation(self):
        """Test basic batch memory estimation."""
        result = estimate_batch_memory(
            num_params=125_000_000,  # 125M params
            batch_size=16,
            seq_length=512,
            d_model=768,
            num_layers=12,
            dtype_bytes=2,
            gradient_checkpointing=True,
        )
        
        assert "params_gb" in result
        assert "optimizer_gb" in result
        assert "gradients_gb" in result
        assert "activations_gb" in result
        assert "total_gb" in result
    
    def test_checkpointing_reduces_activations(self):
        """Test gradient checkpointing reduces activation memory."""
        base_args = {
            "num_params": 125_000_000,
            "batch_size": 16,
            "seq_length": 512,
            "d_model": 768,
            "num_layers": 12,
            "dtype_bytes": 2,
        }
        
        with_checkpointing = estimate_batch_memory(**base_args, gradient_checkpointing=True)
        without_checkpointing = estimate_batch_memory(**base_args, gradient_checkpointing=False)
        
        assert with_checkpointing["activations_gb"] < without_checkpointing["activations_gb"]
    
    def test_larger_batch_more_memory(self):
        """Test larger batch sizes require more memory."""
        base_args = {
            "num_params": 125_000_000,
            "seq_length": 512,
            "d_model": 768,
            "num_layers": 12,
            "dtype_bytes": 2,
            "gradient_checkpointing": True,
        }
        
        small_batch = estimate_batch_memory(**base_args, batch_size=8)
        large_batch = estimate_batch_memory(**base_args, batch_size=32)
        
        assert small_batch["activations_gb"] < large_batch["activations_gb"]
    
    def test_fp32_uses_more_memory(self):
        """Test FP32 uses more memory than FP16."""
        base_args = {
            "num_params": 125_000_000,
            "batch_size": 16,
            "seq_length": 512,
            "d_model": 768,
            "num_layers": 12,
            "gradient_checkpointing": True,
        }
        
        fp16 = estimate_batch_memory(**base_args, dtype_bytes=2)
        fp32 = estimate_batch_memory(**base_args, dtype_bytes=4)
        
        assert fp16["params_gb"] < fp32["params_gb"]


class TestGetAllPresetMemoryRequirements:
    """Tests for get_all_preset_memory_requirements function."""
    
    def test_returns_all_presets(self):
        """Test function returns all presets."""
        result = get_all_preset_memory_requirements()
        
        expected_presets = ["tiny", "small", "base", "large", "xlarge", "xxlarge"]
        assert all(preset in result for preset in expected_presets)
    
    def test_each_preset_has_required_fields(self):
        """Test each preset has required fields."""
        result = get_all_preset_memory_requirements()
        
        for preset, data in result.items():
            assert "preset" in data
            assert "memory" in data
            assert "recommended_gpu" in data


class TestPrintMemoryRequirementsTable:
    """Tests for print_memory_requirements_table function."""
    
    def test_returns_string(self):
        """Test function returns formatted string."""
        result = print_memory_requirements_table()
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_contains_preset_names(self):
        """Test output contains all preset names."""
        result = print_memory_requirements_table()
        
        assert "tiny" in result
        assert "small" in result
        assert "base" in result
        assert "large" in result
    
    def test_contains_headers(self):
        """Test output contains headers."""
        result = print_memory_requirements_table()
        
        assert "Preset" in result
        assert "Params" in result
        assert "RT-DLM Memory Requirements" in result


class TestRecommendGPU:
    """Tests for _recommend_gpu helper function."""
    
    def test_small_memory_recommendation(self):
        """Test recommendation for small memory requirement."""
        result = _recommend_gpu(6.0)
        assert "8GB" in result or "3070" in result or "4070" in result
    
    def test_medium_memory_recommendation(self):
        """Test recommendation for medium memory requirement."""
        result = _recommend_gpu(20.0)
        assert "24GB" in result or "4090" in result or "A10" in result
    
    def test_large_memory_recommendation(self):
        """Test recommendation for large memory requirement."""
        result = _recommend_gpu(60.0)
        assert "80GB" in result or "A100" in result or "H100" in result
    
    def test_very_large_memory_recommendation(self):
        """Test recommendation for very large memory requirement."""
        result = _recommend_gpu(150.0)
        assert "Multi-GPU" in result


class TestRecommendBatchSize:
    """Tests for _recommend_batch_size helper function."""
    
    def test_returns_recommendations_for_gpus(self):
        """Test returns batch size recommendations for different GPUs."""
        result = _recommend_batch_size("base", 5.0)
        
        assert isinstance(result, dict)
        assert "RTX_4090_24GB" in result
        assert "A100_40GB" in result
        assert "A100_80GB" in result
    
    def test_larger_gpu_larger_batch(self):
        """Test larger GPUs get larger batch recommendations."""
        result = _recommend_batch_size("base", 5.0)
        
        # A100 80GB should support larger batches than RTX 4090
        assert result["A100_80GB"] >= result["RTX_4090_24GB"]
    
    def test_smaller_preset_larger_batch(self):
        """Test smaller presets get larger batch recommendations."""
        tiny_result = _recommend_batch_size("tiny", 1.0)
        large_result = _recommend_batch_size("large", 20.0)
        
        # Tiny model should support larger batches
        assert tiny_result["A100_40GB"] >= large_result["A100_40GB"]
