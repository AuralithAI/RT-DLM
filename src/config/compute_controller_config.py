"""
Compute Controller Configuration

Configuration settings for the ComputeController and dynamic compute allocation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ComputeStrategy(Enum):
    """Strategies for compute allocation."""
    ADAPTIVE = "adaptive"      # Learn optimal allocation
    GREEDY = "greedy"          # Always use minimum compute
    THOROUGH = "thorough"      # Always use maximum compute
    BALANCED = "balanced"      # Fixed moderate allocation


@dataclass
class ModuleCostConfig:
    """Configuration for module compute costs."""
    # Base costs (normalized 0-1)
    memory_retrieval: float = 0.05
    graph_reasoning: float = 0.15
    symbolic_reasoning: float = 0.10
    probabilistic: float = 0.08
    quantum_simulation: float = 0.20
    moe_routing: float = 0.12
    scientific_discovery: float = 0.18
    creative_generation: float = 0.15
    consciousness: float = 0.10
    multimodal_fusion: float = 0.12
    attention_refinement: float = 0.08
    output_generation: float = 0.05
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "memory_retrieval": self.memory_retrieval,
            "graph_reasoning": self.graph_reasoning,
            "symbolic_reasoning": self.symbolic_reasoning,
            "probabilistic": self.probabilistic,
            "quantum_simulation": self.quantum_simulation,
            "moe_routing": self.moe_routing,
            "scientific_discovery": self.scientific_discovery,
            "creative_generation": self.creative_generation,
            "consciousness": self.consciousness,
            "multimodal_fusion": self.multimodal_fusion,
            "attention_refinement": self.attention_refinement,
            "output_generation": self.output_generation,
        }


@dataclass
class ComputeControllerConfig:
    """
    Configuration for the ComputeController.
    
    Attributes:
        enabled: Whether to use the compute controller
        max_steps: Maximum compute steps per request
        initial_budget: Initial compute budget [0, 1]
        min_budget: Minimum budget to continue (below this, halt)
        halt_threshold: Confidence threshold for early halting
        temperature: Temperature for module selection softmax
        strategy: Compute allocation strategy
        
        # Training parameters
        lambda_compute: Weight for compute cost in loss
        mu_error: Weight for error penalty in loss
        ponder_lambda: Weight for ponder cost regularization
        
        # Module settings
        module_costs: Custom module cost configuration
        enabled_modules: List of enabled module types (None = all)
        disabled_modules: List of explicitly disabled modules
        
        # Exploration
        exploration_rate: Epsilon for exploration during training
        exploration_decay: Decay rate for exploration
    """
    enabled: bool = True
    max_steps: int = 10
    initial_budget: float = 1.0
    min_budget: float = 0.05
    halt_threshold: float = 0.8
    temperature: float = 1.0
    strategy: ComputeStrategy = ComputeStrategy.ADAPTIVE
    
    # Training parameters
    lambda_compute: float = 0.01
    mu_error: float = 0.1
    ponder_lambda: float = 0.01
    
    # Module settings
    module_costs: ModuleCostConfig = field(default_factory=ModuleCostConfig)
    enabled_modules: Optional[List[str]] = None
    disabled_modules: List[str] = field(default_factory=list)
    
    # Exploration
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    
    def get_effective_budget(self, difficulty_estimate: float = 0.5) -> float:
        """
        Get effective budget based on difficulty estimate.
        
        Args:
            difficulty_estimate: Estimated difficulty [0, 1]
            
        Returns:
            Adjusted budget
        """
        if self.strategy == ComputeStrategy.GREEDY:
            return 0.3  # Minimal budget
        elif self.strategy == ComputeStrategy.THOROUGH:
            return 1.0  # Full budget
        elif self.strategy == ComputeStrategy.BALANCED:
            return 0.6  # Moderate budget
        else:  # ADAPTIVE
            # Scale budget with difficulty
            return self.initial_budget * (0.5 + 0.5 * difficulty_estimate)
    
    def get_effective_max_steps(self, difficulty_estimate: float = 0.5) -> int:
        """
        Get effective max steps based on difficulty.
        
        Args:
            difficulty_estimate: Estimated difficulty [0, 1]
            
        Returns:
            Adjusted max steps
        """
        if self.strategy == ComputeStrategy.GREEDY:
            return max(3, self.max_steps // 3)
        elif self.strategy == ComputeStrategy.THOROUGH:
            return self.max_steps
        elif self.strategy == ComputeStrategy.BALANCED:
            return max(5, self.max_steps // 2)
        else:  # ADAPTIVE
            min_steps = 3
            return int(min_steps + (self.max_steps - min_steps) * difficulty_estimate)


@dataclass
class TaskTypeConfig:
    """Configuration for task-specific compute allocation."""
    
    # Default module priorities per task type
    # Higher weight = more likely to be selected
    task_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "classification": {
            "memory_retrieval": 0.8,
            "graph_reasoning": 0.5,
            "symbolic_reasoning": 1.2,
            "probabilistic": 1.0,
            "moe_routing": 1.3,
        },
        "generation": {
            "memory_retrieval": 1.0,
            "graph_reasoning": 0.8,
            "creative_generation": 1.5,
            "consciousness": 1.2,
            "moe_routing": 1.0,
        },
        "reasoning": {
            "memory_retrieval": 1.2,
            "graph_reasoning": 1.5,
            "symbolic_reasoning": 1.5,
            "probabilistic": 1.2,
            "scientific_discovery": 1.3,
        },
        "multimodal": {
            "memory_retrieval": 0.8,
            "multimodal_fusion": 1.5,
            "graph_reasoning": 1.0,
            "consciousness": 1.0,
        },
        "conversation": {
            "memory_retrieval": 1.3,
            "consciousness": 1.2,
            "creative_generation": 1.0,
            "probabilistic": 0.8,
        },
    })
    
    def get_weights(self, task_type: str) -> Dict[str, float]:
        """Get module weights for a task type."""
        return self.task_weights.get(task_type, {})


# Default configurations for different use cases
DEFAULT_CONFIG = ComputeControllerConfig()

FAST_CONFIG = ComputeControllerConfig(
    max_steps=5,
    initial_budget=0.5,
    halt_threshold=0.7,
    strategy=ComputeStrategy.GREEDY,
    lambda_compute=0.05,
)

THOROUGH_CONFIG = ComputeControllerConfig(
    max_steps=15,
    initial_budget=1.0,
    halt_threshold=0.9,
    strategy=ComputeStrategy.THOROUGH,
    lambda_compute=0.005,
)

BALANCED_CONFIG = ComputeControllerConfig(
    max_steps=8,
    initial_budget=0.7,
    halt_threshold=0.8,
    strategy=ComputeStrategy.BALANCED,
    lambda_compute=0.02,
)
