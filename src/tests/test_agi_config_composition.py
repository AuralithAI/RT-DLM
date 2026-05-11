"""Unit tests for AGIConfig composition into sub-configs."""

import pytest

from src.config.agi_config import AGIConfig, MODEL_PRESETS
from src.config.architecture_config import ArchitectureConfig
from src.config.multimodal_config import MultimodalConfig
from src.config.parallelism_config import ParallelismConfig
from src.config.precision_config import PrecisionConfig
from src.config.safety_config import SafetyConfig
from src.config.training_config import TrainingConfig


class TestSubConfigPresence:
    def test_all_subconfigs_attached(self):
        cfg = AGIConfig()
        assert isinstance(cfg.architecture, ArchitectureConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.precision, PrecisionConfig)
        assert isinstance(cfg.parallelism, ParallelismConfig)
        assert isinstance(cfg.multimodal, MultimodalConfig)
        assert isinstance(cfg.safety, SafetyConfig)


class TestFlatAndComposedAgree:
    def test_architecture_mirrors_flat_fields(self):
        cfg = AGIConfig(d_model=512, num_heads=8, num_layers=10)
        assert cfg.architecture.d_model == cfg.d_model == 512
        assert cfg.architecture.num_heads == cfg.num_heads == 8
        assert cfg.architecture.num_layers == cfg.num_layers == 10

    def test_training_mirrors_flat_fields(self):
        cfg = AGIConfig(batch_size=4, learning_rate=3e-4, clip_norm=1.0)
        assert cfg.training.batch_size == 4
        assert cfg.training.learning_rate == pytest.approx(3e-4)
        assert cfg.training.clip_norm == pytest.approx(1.0)

    def test_z_loss_weights_flow_through(self):
        cfg = AGIConfig(moe_z_loss_weight=5e-4, moe_router_z_loss_weight=2e-3)
        assert cfg.training.moe_z_loss_weight == pytest.approx(5e-4)
        assert cfg.training.moe_router_z_loss_weight == pytest.approx(2e-3)

    def test_base_d_model_in_architecture(self):
        cfg = AGIConfig(base_d_model=128)
        assert cfg.architecture.base_d_model == 128


class TestPresetRoundtrip:
    @pytest.mark.parametrize("name", list(MODEL_PRESETS.keys()))
    def test_preset_validates(self, name):
        cfg = AGIConfig.from_preset(name)
        assert cfg.architecture.d_model > 0
        assert cfg.architecture.base_d_model == 256

    def test_to_dict_excludes_subconfigs(self):
        cfg = AGIConfig()
        d = cfg.to_dict()
        for key in ("architecture", "training", "precision", "parallelism", "multimodal", "safety"):
            assert key not in d


class TestValidationOnUpdate:
    def test_update_refreshes_subconfigs(self):
        cfg = AGIConfig()
        cfg.update(d_model=1024, num_heads=16)
        assert cfg.architecture.d_model == 1024
        assert cfg.architecture.num_heads == 16

    def test_invalid_update_raises(self):
        cfg = AGIConfig()
        with pytest.raises((AssertionError, ValueError)):
            cfg.update(d_model=-1)
