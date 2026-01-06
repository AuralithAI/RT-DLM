"""
Advanced Learning Module for RT-DLM AGI

Implements continual learning algorithms to enable adaptive learning
for long-term human-centric tasks without catastrophic forgetting.

Key Components:
- ContinualLearner: EWC-based continual learning
- ProgressiveNeuralNetworks: Progressive network expansion
- MemoryAwareRegularization: Memory-aware synaptic intelligence
"""

from .advanced_algorithms import (
    ContinualLearner,
    ElasticWeightConsolidation,
    SynapticIntelligence,
    ProgressiveNeuralNetwork,
    compute_fisher_information,
    compute_ewc_loss,
    compute_si_loss,
)

__all__ = [
    "ContinualLearner",
    "ElasticWeightConsolidation",
    "SynapticIntelligence",
    "ProgressiveNeuralNetwork",
    "compute_fisher_information",
    "compute_ewc_loss",
    "compute_si_loss",
]
