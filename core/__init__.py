# Core module
from core.training_utils import (
    MixedPrecisionPolicy,
    GradientCheckpointingConfig,
    DistributedTrainingConfig,
    TrainingOptimizations,
    create_mixed_precision_policy,
    create_checkpointed_layer,
    create_pmap_train_step,
    GradientAccumulator,
)
# Scalable training (production-ready unified approach)
from core.scalable_training import (
    ScalableMesh,
    create_scalable_mesh,
    setup_scalable_training,
    replicate_for_data_parallel,
    unreplicate_params,
    estimate_model_memory,
    recommend_parallelism,
    get_param_sharding_spec,
    create_sharded_params,
    create_scalable_train_step,
)
# Legacy model parallel (kept for backwards compatibility)
from core.model_parallel import (
    DeviceMesh,
    TensorParallelLinear,
    TensorParallelAttention,
    TensorParallelMLP,
    PipelineStage,
    PipelineParallelModel,
    ModelParallelTransformer,
    shard_params_for_tensor_parallel,
    create_sharded_train_step,
    create_model_parallel_system,
    create_model_parallel_transformer,
)
# Import ModelParallelConfig from config folder
from config.model_parallel_config import ModelParallelConfig

# Evaluation metrics (from core.training - recommended import path)
# Backward-compatible re-exports via core.evaluation
from core.training import (
    # Metric data structures
    BatchMetrics,
    GradientMetrics,
    TrainingStepMetrics,
    ValidationMetrics,
    # Core evaluation components
    EvaluationMetrics,
    GradientMonitor,
    MetricLogger,
    ValidationRunner,
    # High-level integration
    TrainingEvaluator,
)

# Decorators (from core.utils - recommended import path)
# Backward-compatible re-exports
from core.utils import (
    dev_utility,
    is_dev_utility,
    get_dev_utility_reason,
)

__all__ = [
    # Training utilities
    'MixedPrecisionPolicy',
    'GradientCheckpointingConfig', 
    'DistributedTrainingConfig',
    'TrainingOptimizations',
    'create_mixed_precision_policy',
    'create_checkpointed_layer',
    'create_pmap_train_step',
    'GradientAccumulator',
    # Scalable training (recommended)
    'ScalableMesh',
    'create_scalable_mesh',
    'setup_scalable_training',
    'replicate_for_data_parallel',
    'unreplicate_params',
    'estimate_model_memory',
    'recommend_parallelism',
    'get_param_sharding_spec',
    'create_sharded_params',
    'create_scalable_train_step',
    # Legacy model parallelism (backwards compatibility)
    'ModelParallelConfig',
    'DeviceMesh',
    'TensorParallelLinear',
    'TensorParallelAttention',
    'TensorParallelMLP',
    'PipelineStage',
    'PipelineParallelModel',
    'ModelParallelTransformer',
    'shard_params_for_tensor_parallel',
    'create_sharded_train_step',
    'create_model_parallel_system',
    'create_model_parallel_transformer',
    # Evaluation metrics
    'dev_utility',
    'is_dev_utility',
    'get_dev_utility_reason',
    'BatchMetrics',
    'GradientMetrics',
    'TrainingStepMetrics',
    'ValidationMetrics',
    'EvaluationMetrics',
    'GradientMonitor',
    'MetricLogger',
    'ValidationRunner',
    'TrainingEvaluator',
]
