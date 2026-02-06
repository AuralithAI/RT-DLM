"""
RT-DLM Evaluation Module (Backward Compatibility)

This module re-exports from the new locations:
- Decorators: core.utils.decorators
- Evaluation: core.training.evaluation

For new code, import directly from:
    from src.core.utils import dev_utility, is_dev_utility
    from src.core.training import EvaluationMetrics, TrainingEvaluator
"""

# Re-export decorators from src.core.utils
from src.core.utils.decorators import (
    dev_utility,
    is_dev_utility,
    get_dev_utility_reason,
)

# Re-export evaluation from src.core.training
from src.core.training.evaluation import (
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
    # Decorator (from utils)
    'dev_utility',
    'is_dev_utility',
    'get_dev_utility_reason',
    
    # Data structures
    'BatchMetrics',
    'GradientMetrics',
    'TrainingStepMetrics',
    'ValidationMetrics',
    
    # Core components
    'EvaluationMetrics',
    'GradientMonitor',
    'MetricLogger',
    'ValidationRunner',
    
    # High-level integration
    'TrainingEvaluator',
]
