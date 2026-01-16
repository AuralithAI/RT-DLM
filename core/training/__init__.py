"""
RT-DLM Training Package

Training utilities and evaluation metrics:
- Evaluation metrics (perplexity, accuracy, gradient health)
- Structured logging
- Validation runner
- Training utilities (mixed precision, gradient checkpointing)
- Scalable training (distributed, model parallel)

Usage:
    from core.training import (
        TrainingEvaluator,
        EvaluationMetrics,
        MetricLogger,
        ValidationRunner,
    )
"""

from core.training.evaluation import (
    # Data structures
    BatchMetrics,
    GradientMetrics,
    TrainingStepMetrics,
    ValidationMetrics,
    
    # Core components
    EvaluationMetrics,
    GradientMonitor,
    MetricLogger,
    ValidationRunner,
    
    # High-level integration
    TrainingEvaluator,
)

__all__ = [
    # Metric data structures
    'BatchMetrics',
    'GradientMetrics',
    'TrainingStepMetrics',
    'ValidationMetrics',
    
    # Core evaluation components
    'EvaluationMetrics',
    'GradientMonitor',
    'MetricLogger',
    'ValidationRunner',
    
    # High-level integration
    'TrainingEvaluator',
]
