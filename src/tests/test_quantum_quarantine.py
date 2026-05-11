"""Unit tests for quantum subsystem quarantine."""

import importlib
import os
import pytest

from src.config.agi_config import AGIConfig, MODEL_PRESETS


class TestQuantumDefaultsDisabled:
    def test_default_quantum_layers_zero(self):
        cfg = AGIConfig()
        assert cfg.quantum_layers == 0
        assert cfg.quantum_qubits == 0

    def test_all_presets_have_zero_quantum(self):
        for name in MODEL_PRESETS:
            cfg = AGIConfig.from_preset(name)
            assert cfg.quantum_layers == 0, f"preset {name} leaks quantum_layers"

    def test_config_allows_enabling_quantum(self):
        cfg = AGIConfig(quantum_layers=2, quantum_qubits=4)
        assert cfg.quantum_layers == 2
        assert cfg.quantum_qubits == 4

    def test_quantum_layers_without_qubits_rejected(self):
        with pytest.raises(AssertionError):
            AGIConfig(quantum_layers=2, quantum_qubits=0)


class TestQuantumPackageGate:
    def test_old_path_removed(self):
        with pytest.raises(ImportError):
            importlib.import_module("src.core.quantum")

    def test_new_path_requires_env(self):
        prev = os.environ.pop("AGI_ENABLE_QUANTUM", None)
        try:
            for mod in [
                "experimental.quantum",
                "experimental.quantum.quantum_agi_core",
                "experimental.quantum.quantum_readiness",
            ]:
                import sys
                sys.modules.pop(mod, None)
            with pytest.raises(ImportError):
                importlib.import_module("experimental.quantum")
        finally:
            if prev is not None:
                os.environ["AGI_ENABLE_QUANTUM"] = prev


class TestQuantumPresetParams:
    def test_size_estimate_does_not_use_quantum_when_disabled(self):
        cfg = AGIConfig(quantum_layers=0, quantum_qubits=0)
        size_disabled = cfg.get_model_size_estimate()
        cfg2 = AGIConfig(quantum_layers=4, quantum_qubits=8)
        size_enabled = cfg2.get_model_size_estimate()
        assert size_enabled > size_disabled
