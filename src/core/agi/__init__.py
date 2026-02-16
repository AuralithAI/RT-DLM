"""
Core AGI System Module

This module provides the central abstraction for AGI system orchestration,
including component unification, stage tracking, ethical alignment, and
dynamic compute allocation via the ComputeController.

Key Components:
- AGISystemAbstraction: Central hub for AGI component integration
- ComponentFusion: Fuses outputs from multiple AGI components
- StageTracker: Tracks AGI progression stages (0-6)
- EthicalAlignmentModule: Evaluates ethical alignment of decisions
- AGIStage: Enum for AGI progression stages
- StageThresholds: Configurable thresholds for stage transitions
- AGIMetrics: Dataclass for tracking AGI performance metrics

Compute Controller (Dynamic Orchestration):
- ComputeController: Learned controller for dynamic compute allocation
- ComputePlan: Per-request execution plan with adaptive steps
- ComputeState: State tracking for controller decisions
- ComputeAction: Action selected by controller
- ModuleContract: Standardized interface for callable modules
- ModuleOutput: Standardized output from modules
- ModuleRegistry: Registry of available modules
- ModuleType: Enum of module types

Training:
- ControllerLossComputer: Multi-objective loss computation (efficiency, calibration, etc.)
- ControllerRewardShaper: RL reward shaping for dense training signals

AGI Integration:
- ControllerIntegrationMixin: Mixin to wire controller into AGI system
- ControlledAGIForward: Controller-driven forward pass
- create_controlled_agi_fn: Factory for transformed controlled forward

Module Adapters:
- ModuleAdapter: Base class for adapters
- MemoryRetrievalAdapter: Adapter for memory retrieval
- GraphReasoningAdapter: Adapter for graph reasoning
- SymbolicReasoningAdapter: Adapter for symbolic reasoning
- ProbabilisticAdapter: Adapter for probabilistic inference
- MoERoutingAdapter: Adapter for MoE routing
- ConsciousnessAdapter: Adapter for consciousness simulation
- OutputGenerationAdapter: Adapter for output generation

Utility Functions:
- create_agi_system_fn: Factory for transformed AGISystemAbstraction
- create_compute_controller_fn: Factory for transformed ComputeController
- create_controlled_agi_fn: Factory for controller-driven AGI forward
- create_module_executors: Creates executor functions for all module types
"""

from .agi_system import (
    # Main abstraction
    AGISystemAbstraction,
    
    # Core components
    ComponentFusion,
    StageTracker,
    EthicalAlignmentModule,
    
    # Data structures
    AGIStage,
    StageThresholds,
    AGIMetrics,
    
    # Factory function
    create_agi_system_fn,
)

from .compute_controller import (
    # Controller
    ComputeController,
    ComputePlan,
    
    # State and actions
    ComputeState,
    ComputeAction,
    
    # Module contracts
    ModuleContract,
    ModuleOutput,
    ModuleRegistry,
    ModuleType,
    
    # Factory function
    create_compute_controller_fn,
    
    ControllerLossComputer,
    ControllerRewardShaper,
    
    ControllerIntegrationMixin,
    ControlledAGIForward,
    create_controlled_agi_fn,
)

from .module_adapters import (
    # Base adapter
    ModuleAdapter,
    
    # Specific adapters
    MemoryRetrievalAdapter,
    GraphReasoningAdapter,
    SymbolicReasoningAdapter,
    ProbabilisticAdapter,
    MoERoutingAdapter,
    ConsciousnessAdapter,
    OutputGenerationAdapter,
    
    # Factory function
    create_module_executors,
)

__all__ = [
    # Main abstraction
    "AGISystemAbstraction",
    
    # Core components
    "ComponentFusion",
    "StageTracker",
    "EthicalAlignmentModule",
    
    # Data structures
    "AGIStage",
    "StageThresholds",
    "AGIMetrics",
    
    # Factory function
    "create_agi_system_fn",
    
    # Compute Controller
    "ComputeController",
    "ComputePlan",
    "ComputeState",
    "ComputeAction",
    "ModuleContract",
    "ModuleOutput",
    "ModuleRegistry",
    "ModuleType",
    "create_compute_controller_fn",
    
    "ControllerLossComputer",
    "ControllerRewardShaper",

    "ControllerIntegrationMixin",
    "ControlledAGIForward",
    "create_controlled_agi_fn",
    
    # Module Adapters
    "ModuleAdapter",
    "MemoryRetrievalAdapter",
    "GraphReasoningAdapter",
    "SymbolicReasoningAdapter",
    "ProbabilisticAdapter",
    "MoERoutingAdapter",
    "ConsciousnessAdapter",
    "OutputGenerationAdapter",
    "create_module_executors",
]
