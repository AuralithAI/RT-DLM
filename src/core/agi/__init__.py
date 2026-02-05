"""
Core AGI System Module

This module provides the central abstraction for AGI system orchestration,
including component unification, stage tracking, and ethical alignment.

Key Components:
- AGISystemAbstraction: Central hub for AGI component integration
- ComponentFusion: Fuses outputs from multiple AGI components
- StageTracker: Tracks AGI progression stages (0-6)
- EthicalAlignmentModule: Evaluates ethical alignment of decisions
- AGIStage: Enum for AGI progression stages
- StageThresholds: Configurable thresholds for stage transitions
- AGIMetrics: Dataclass for tracking AGI performance metrics

Utility Functions:
- create_agi_system_fn: Factory for transformed AGISystemAbstraction
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
]
