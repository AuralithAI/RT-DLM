"""
RT-DLM Core Package

This package contains all model-specific components:
- core/: Core model components, training utilities, quantum modules
- config/: Model and training configurations
- modules/: Feature modules (multimodal, retrieval, hybrid architecture)
- tests/: Test suite

Main entry points:
- rtdlm.py: Model definitions and factory functions
- train.py: Training script
"""

from src.rtdlm import (
    create_rtdlm_agi,
    create_agi_optimizer,
    compute_agi_loss,
    RTDLMAGISystem,
    ConsciousnessSimulator,
    ScientificDiscoveryEngine,
    CreativeGenerationEngine,
    SocialEmotionalIntelligence,
)

from src.config.agi_config import AGIConfig

__all__ = [
    # Model factory functions
    "create_rtdlm_agi",
    "create_agi_optimizer", 
    "compute_agi_loss",
    # Model classes
    "RTDLMAGISystem",
    "ConsciousnessSimulator",
    "ScientificDiscoveryEngine",
    "CreativeGenerationEngine",
    "SocialEmotionalIntelligence",
    # Config
    "AGIConfig",
]
