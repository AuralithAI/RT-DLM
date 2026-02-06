"""
Compute Controller for Dynamic Module Orchestration

This module implements a learned controller that explicitly allocates compute
across modules, memory, and tools under a budget. Instead of running all modules
every forward pass, the controller decides:
- What to do next (retrieve memory, route to experts, run graph reasoner, etc.)
- How much compute to spend (budget, depth, number of steps)
- When to stop (early exit when confident, more compute when uncertain)

This transforms RT-DLM from a static neural network into a cognitive engine.

Architecture:
    State = hidden representation + memory summary + uncertainty signals
    Action = choose module(s) to run next
    Update = run chosen module(s) → update state + memory
    Halt = decide to stop and emit output

Training Objective:
    Total = task_loss + λ * compute_cost + μ * error_penalty

Key Components:
    - ModuleContract: Standardized interface for all callable modules
    - ModuleRegistry: Registry of available modules with their contracts
    - ComputeState: Current state including hidden, memory, uncertainty
    - ComputeAction: Action to take (module selection, parameters)
    - ComputeController: Learned controller for dynamic compute allocation
    - ComputePlan: Per-request execution plan with K steps

References:
    - Adaptive Computation Time (Graves, 2016)
    - PonderNet (Banino et al., 2021)
    - Universal Transformers (Dehghani et al., 2018)
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ModuleType(Enum):
    """Types of modules that can be invoked by the controller."""
    MEMORY_RETRIEVAL = auto()      # LTM/STM/MTM retrieval
    GRAPH_REASONING = auto()        # Multi-hop graph reasoning
    SYMBOLIC_REASONING = auto()     # Rule-based, logic reasoning
    PROBABILISTIC = auto()          # Bayesian inference, uncertainty
    QUANTUM_SIMULATION = auto()     # Quantum-enhanced processing
    MOE_ROUTING = auto()            # Mixture of Experts
    SCIENTIFIC_DISCOVERY = auto()   # Hypothesis generation
    CREATIVE_GENERATION = auto()    # Creative content generation
    CONSCIOUSNESS = auto()          # Self-awareness, introspection
    MULTIMODAL_FUSION = auto()      # Cross-modal attention
    ATTENTION_REFINEMENT = auto()   # Additional attention layers
    OUTPUT_GENERATION = auto()      # Final output (always last)


@dataclass
class ModuleContract:
    """
    Standardized contract for modules callable by the controller.
    
    Each module must report:
    - what it changed (delta to representation)
    - confidence / uncertainty
    - cost (FLOPs/latency proxy)
    - evidence (for symbolic/probabilistic/graph branches)
    
    This makes the controller stable and debuggable.
    """
    module_type: ModuleType
    name: str
    base_cost: float  # Normalized cost [0, 1] relative to full forward pass
    
    # Callable signatures
    # forward_fn: (state, params, rng) -> ModuleOutput
    forward_fn: Optional[Callable] = None
    
    # Whether this module can be skipped
    skippable: bool = True
    
    # Dependencies (must run after these modules)
    dependencies: List[ModuleType] = field(default_factory=list)
    
    # Maximum times this module can be called per request
    max_calls: int = 3
    
    # Description for debugging
    description: str = ""


@dataclass
class ModuleOutput:
    """
    Standardized output from a module.
    
    Every module returns this contract to enable the controller
    to make informed decisions about next steps.
    """
    # The updated hidden representation
    hidden_delta: jnp.ndarray  # [batch, d_model] - change to apply
    
    # Confidence in this module's output [0, 1]
    confidence: jnp.ndarray  # [batch, 1]
    
    # Uncertainty estimate [0, 1] - high means controller should explore more
    uncertainty: jnp.ndarray  # [batch, 1]
    
    # Actual compute cost (can differ from base_cost based on input)
    actual_cost: float
    
    # Evidence/explanation for this module's contribution
    evidence: Optional[Dict[str, Any]] = None
    
    # Whether the module suggests halting (output is ready)
    suggests_halt: bool = False
    
    # Optional: specific recommendations for next module
    recommended_next: Optional[List[ModuleType]] = None


class ComputeState(NamedTuple):
    """
    Current state of the compute plan execution.
    
    Tracks everything needed for the controller to make decisions.
    """
    # Current hidden representation [batch, seq, d_model]
    hidden: jnp.ndarray
    
    # Pooled representation for decision making [batch, d_model]
    hidden_pooled: jnp.ndarray
    
    # Memory summary [batch, d_model]
    memory_summary: jnp.ndarray
    
    # Cumulative uncertainty [batch, 1]
    uncertainty: jnp.ndarray
    
    # Cumulative confidence [batch, 1]
    confidence: jnp.ndarray
    
    # Compute budget remaining [0, 1]
    budget_remaining: float
    
    # Step counter
    step: int
    
    # History of modules called
    modules_called: List[ModuleType]
    
    # History of module outputs (for analysis)
    module_outputs: List[ModuleOutput]


@dataclass 
class ComputeAction:
    """
    Action selected by the controller.
    """
    # Which module(s) to run next
    modules: List[ModuleType]
    
    # Probability of halting after this action
    halt_probability: float
    
    # Optional parameters for modules
    module_params: Dict[ModuleType, Dict[str, Any]] = field(default_factory=dict)
    
    # Allocated budget for this action
    budget_allocation: float = 0.1


class ModuleRegistry:
    """
    Registry of all available modules with their contracts.
    
    Allows the controller to query available actions and their costs.
    """
    
    def __init__(self):
        self._modules: Dict[ModuleType, ModuleContract] = {}
        self._initialize_default_contracts()
    
    def _initialize_default_contracts(self):
        """Initialize default module contracts with estimated costs."""
        defaults = [
            ModuleContract(
                module_type=ModuleType.MEMORY_RETRIEVAL,
                name="memory_retrieval",
                base_cost=0.05,
                skippable=True,
                description="Retrieve from LTM/STM/MTM memory stores"
            ),
            ModuleContract(
                module_type=ModuleType.GRAPH_REASONING,
                name="graph_reasoning",
                base_cost=0.15,
                skippable=True,
                dependencies=[ModuleType.MEMORY_RETRIEVAL],
                description="Multi-hop reasoning over knowledge graphs"
            ),
            ModuleContract(
                module_type=ModuleType.SYMBOLIC_REASONING,
                name="symbolic_reasoning",
                base_cost=0.10,
                skippable=True,
                description="Rule-based and logic reasoning"
            ),
            ModuleContract(
                module_type=ModuleType.PROBABILISTIC,
                name="probabilistic",
                base_cost=0.08,
                skippable=True,
                description="Bayesian inference and uncertainty quantification"
            ),
            ModuleContract(
                module_type=ModuleType.QUANTUM_SIMULATION,
                name="quantum_simulation",
                base_cost=0.20,
                skippable=True,
                max_calls=1,
                description="Quantum-enhanced optimization (expensive)"
            ),
            ModuleContract(
                module_type=ModuleType.MOE_ROUTING,
                name="moe_routing",
                base_cost=0.12,
                skippable=True,
                description="Route to specialized experts"
            ),
            ModuleContract(
                module_type=ModuleType.SCIENTIFIC_DISCOVERY,
                name="scientific_discovery",
                base_cost=0.18,
                skippable=True,
                dependencies=[ModuleType.GRAPH_REASONING],
                description="Hypothesis generation and causal reasoning"
            ),
            ModuleContract(
                module_type=ModuleType.CREATIVE_GENERATION,
                name="creative_generation",
                base_cost=0.15,
                skippable=True,
                description="Novel content generation"
            ),
            ModuleContract(
                module_type=ModuleType.CONSCIOUSNESS,
                name="consciousness",
                base_cost=0.10,
                skippable=True,
                description="Self-awareness and introspection"
            ),
            ModuleContract(
                module_type=ModuleType.MULTIMODAL_FUSION,
                name="multimodal_fusion",
                base_cost=0.12,
                skippable=True,
                description="Cross-modal attention and fusion"
            ),
            ModuleContract(
                module_type=ModuleType.ATTENTION_REFINEMENT,
                name="attention_refinement",
                base_cost=0.08,
                skippable=True,
                max_calls=5,
                description="Additional self-attention refinement"
            ),
            ModuleContract(
                module_type=ModuleType.OUTPUT_GENERATION,
                name="output_generation",
                base_cost=0.05,
                skippable=False,  # Must always run
                description="Generate final output"
            ),
        ]
        
        for contract in defaults:
            self._modules[contract.module_type] = contract
    
    def register(self, contract: ModuleContract):
        """Register a module contract."""
        self._modules[contract.module_type] = contract
    
    def get(self, module_type: ModuleType) -> Optional[ModuleContract]:
        """Get a module contract by type."""
        return self._modules.get(module_type)
    
    def get_all(self) -> Dict[ModuleType, ModuleContract]:
        """Get all registered modules."""
        return self._modules.copy()
    
    def get_available(
        self, 
        state: ComputeState,
        budget: float
    ) -> List[ModuleContract]:
        """
        Get modules available given current state and budget.
        
        Filters out:
        - Modules that exceed remaining budget
        - Modules with unmet dependencies
        - Modules that have exceeded max_calls
        """
        available = []
        call_counts = {}
        for m in state.modules_called:
            call_counts[m] = call_counts.get(m, 0) + 1
        
        for module_type, contract in self._modules.items():
            # Check budget
            if contract.base_cost > budget:
                continue
            
            # Check max calls
            if call_counts.get(module_type, 0) >= contract.max_calls:
                continue
            
            # Check dependencies
            deps_met = all(
                dep in state.modules_called 
                for dep in contract.dependencies
            )
            if not deps_met:
                continue
            
            available.append(contract)
        
        return available
    
    def get_total_cost(self, modules: List[ModuleType]) -> float:
        """Calculate total cost of running given modules."""
        return sum(
            self._modules[m].base_cost 
            for m in modules 
            if m in self._modules
        )


class ComputeController(hk.Module):
    """
    Learned controller for dynamic compute allocation.
    
    The controller observes the current state (hidden representation,
    memory summary, uncertainty) and decides:
    1. Which module(s) to run next
    2. How much budget to allocate
    3. Whether to halt and output
    
    Training:
        The controller is trained with:
        - Task loss (cross-entropy, etc.)
        - Compute cost penalty (encourages efficiency)
        - Error penalty (discourages premature halting)
    
    Architecture:
        - State encoder: Encodes current state into decision features
        - Module selector: Outputs probability over modules
        - Halt predictor: Outputs probability of halting
        - Budget allocator: Decides budget for next step
    """
    
    def __init__(
        self,
        d_model: int,
        num_modules: int = len(ModuleType),
        max_steps: int = 10,
        min_budget: float = 0.05,
        halt_threshold: float = 0.8,
        temperature: float = 1.0,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_modules = num_modules
        self.max_steps = max_steps
        self.min_budget = min_budget
        self.halt_threshold = halt_threshold
        self.temperature = temperature
        
    def _encode_state(self, state: ComputeState) -> jnp.ndarray:
        """Encode compute state into decision features."""
        # State encoder network
        state_encoder = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="state_encoder")
        
        # Combine state components
        combined = jnp.concatenate([
            state.hidden_pooled,
            state.memory_summary,
            state.uncertainty.repeat(self.d_model // 2, axis=-1),
            state.confidence.repeat(self.d_model // 2, axis=-1),
        ], axis=-1)
        
        # Project to d_model
        projection = hk.Linear(self.d_model, name="state_projection")
        projected = projection(combined)
        
        # Encode
        encoded = state_encoder(projected)
        
        # Add step and budget information
        step_embedding = hk.Embed(
            vocab_size=self.max_steps + 1,
            embed_dim=self.d_model,
            name="step_embedding"
        )
        step_features = step_embedding(jnp.array([min(state.step, self.max_steps)]))
        step_features = jnp.broadcast_to(step_features, encoded.shape)
        
        budget_encoder = hk.Linear(self.d_model, name="budget_encoder")
        budget_features = budget_encoder(
            jnp.full((encoded.shape[0], 1), state.budget_remaining)
        )
        
        # Combine all features
        final_features = encoded + step_features * 0.1 + budget_features * 0.1
        
        return final_features
    
    def _encode_module_history(self, state: ComputeState) -> jnp.ndarray:
        """Encode history of modules called."""
        history_encoder = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model)
        ], name="history_encoder")
        
        # Create one-hot encoding of modules called
        history_onehot = jnp.zeros(self.num_modules)
        for m in state.modules_called:
            history_onehot = history_onehot.at[m.value - 1].set(1.0)
        
        # Expand to batch dimension
        batch_size = state.hidden_pooled.shape[0]
        history_onehot = jnp.broadcast_to(
            history_onehot[None, :], 
            (batch_size, self.num_modules)
        )
        
        # Project and encode
        projection = hk.Linear(self.d_model, name="history_projection")
        projected = projection(history_onehot)
        
        return history_encoder(projected)
    
    def __call__(
        self,
        state: ComputeState,
        available_modules: List[ModuleContract],
        is_training: bool = True
    ) -> Tuple[ComputeAction, Dict[str, Any]]:
        """
        Decide next action given current state.
        
        Args:
            state: Current compute state
            available_modules: Modules available given budget/dependencies
            is_training: Whether in training mode (affects exploration)
            
        Returns:
            action: The selected action
            info: Dictionary with decision details for analysis/training
        """
        # Encode state
        state_features = self._encode_state(state)
        history_features = self._encode_module_history(state)
        
        # Combine features
        decision_features = state_features + history_features * 0.5
        
        # Module selection network
        module_selector = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model // 2),
            jax.nn.silu,
            hk.Linear(self.num_modules)
        ], name="module_selector")
        
        module_logits = module_selector(decision_features)
        
        # Mask unavailable modules
        available_mask = jnp.zeros(self.num_modules)
        for contract in available_modules:
            available_mask = available_mask.at[contract.module_type.value - 1].set(1.0)
        
        # Apply mask (set unavailable to -inf)
        masked_logits = jnp.where(
            available_mask[None, :] > 0,
            module_logits,
            jnp.full_like(module_logits, -1e9)
        )
        
        # Temperature-scaled softmax
        module_probs = jax.nn.softmax(masked_logits / self.temperature, axis=-1)
        
        # Halt prediction network
        halt_predictor = hk.Sequential([
            hk.Linear(self.d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="halt_predictor")
        
        # Halt probability increases with confidence, decreases with uncertainty
        halt_input = jnp.concatenate([
            decision_features,
            state.confidence,
            state.uncertainty,
            jnp.full((decision_features.shape[0], 1), state.budget_remaining)
        ], axis=-1)
        halt_projection = hk.Linear(self.d_model, name="halt_input_proj")
        halt_features = halt_projection(halt_input)
        halt_prob = halt_predictor(halt_features).squeeze(-1)
        
        # Budget allocation network
        budget_allocator = hk.Sequential([
            hk.Linear(self.d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="budget_allocator")
        
        # Allocate fraction of remaining budget
        budget_fraction = budget_allocator(decision_features).squeeze(-1)
        budget_allocation = budget_fraction * state.budget_remaining
        budget_allocation = jnp.maximum(budget_allocation, self.min_budget)
        
        # Select modules (can select multiple if budget allows)
        # For now, select top-k based on probability
        if is_training:
            # During training, sample from distribution
            rng = hk.next_rng_key()
            selected_idx = jax.random.categorical(rng, masked_logits[0])
            selected_modules = [ModuleType(int(selected_idx) + 1)]
        else:
            # During inference, take argmax
            selected_idx = jnp.argmax(module_probs[0])
            selected_modules = [ModuleType(int(selected_idx) + 1)]
        
        # Check if we should halt
        should_halt = (
            float(halt_prob[0]) > self.halt_threshold or
            state.step >= self.max_steps - 1 or
            state.budget_remaining < self.min_budget
        )
        
        # If halting, ensure OUTPUT_GENERATION is selected
        if should_halt and ModuleType.OUTPUT_GENERATION not in selected_modules:
            selected_modules = [ModuleType.OUTPUT_GENERATION]
        
        action = ComputeAction(
            modules=selected_modules,
            halt_probability=float(halt_prob[0]),
            budget_allocation=float(budget_allocation[0])
        )
        
        info = {
            "module_probs": module_probs,
            "module_logits": module_logits,
            "halt_prob": halt_prob,
            "budget_fraction": budget_fraction,
            "budget_allocation": budget_allocation,
            "decision_features": decision_features,
            "available_mask": available_mask,
            "should_halt": should_halt
        }
        
        return action, info
    
    def compute_loss(
        self,
        task_loss: jnp.ndarray,
        compute_cost: float,
        module_probs: jnp.ndarray,
        halt_probs: List[jnp.ndarray],
        actual_steps: int,
        confidence_at_halt: jnp.ndarray,
        lambda_compute: float = 0.01,
        mu_error: float = 0.1
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute controller loss for training.
        
        Total objective = task_loss + λ * compute_cost + μ * error_penalty
        
        Args:
            task_loss: Primary task loss (e.g., cross-entropy)
            compute_cost: Normalized compute cost [0, 1]
            module_probs: Probability distributions over modules
            halt_probs: Halt probabilities at each step
            actual_steps: Number of steps actually taken
            confidence_at_halt: Model confidence when halting
            lambda_compute: Weight for compute cost penalty
            mu_error: Weight for error penalty
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        # Compute cost penalty - encourages efficiency
        compute_penalty = lambda_compute * compute_cost
        
        # Error penalty - discourages halting when uncertain
        # If we halt with low confidence, penalize
        error_penalty = mu_error * (1.0 - confidence_at_halt) * jnp.array(actual_steps < self.max_steps, dtype=jnp.float32)
        
        # Ponder cost (from PonderNet) - regularizes thinking time
        # Encourages geometric distribution of halt times
        if halt_probs:
            halt_probs_stack = jnp.stack(halt_probs)
            # KL divergence from geometric prior
            prior_lambda = 0.5  # Prior probability of halting
            prior_probs = prior_lambda * (1 - prior_lambda) ** jnp.arange(len(halt_probs))
            prior_probs = prior_probs / prior_probs.sum()
            
            # Actual halt distribution
            actual_halt_dist = halt_probs_stack.mean(axis=-1)
            actual_halt_dist = actual_halt_dist / (actual_halt_dist.sum() + 1e-8)
            
            ponder_cost = jnp.sum(
                actual_halt_dist * jnp.log(actual_halt_dist / (prior_probs + 1e-8) + 1e-8)
            )
        else:
            ponder_cost = jnp.array(0.0)
        
        # Total loss
        total_loss = task_loss + compute_penalty + error_penalty.mean() + 0.01 * ponder_cost
        
        loss_components = {
            "task_loss": task_loss,
            "compute_penalty": compute_penalty,
            "error_penalty": error_penalty.mean(),
            "ponder_cost": ponder_cost,
            "total_loss": total_loss
        }
        
        return total_loss, loss_components


class ComputePlan(hk.Module):
    """
    Executes a compute plan for a single request/batch.
    
    Implements the main loop:
    1. Initialize state from input
    2. Controller selects action
    3. Execute selected modules
    4. Update state
    5. Check halt condition
    6. Repeat or output
    """
    
    def __init__(
        self,
        d_model: int,
        max_steps: int = 10,
        initial_budget: float = 1.0,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        
    def initialize_state(
        self,
        hidden: jnp.ndarray,
        memory_summary: Optional[jnp.ndarray] = None
    ) -> ComputeState:
        """Initialize compute state from input hidden representation."""
        batch_size = hidden.shape[0]
        
        # Pool hidden if 3D
        if hidden.ndim == 3:
            hidden_pooled = hidden.mean(axis=1)
        else:
            hidden_pooled = hidden
        
        # Default memory summary if not provided
        if memory_summary is None:
            memory_summary = jnp.zeros((batch_size, self.d_model))
        
        return ComputeState(
            hidden=hidden,
            hidden_pooled=hidden_pooled,
            memory_summary=memory_summary,
            uncertainty=jnp.ones((batch_size, 1)) * 0.5,  # Start with medium uncertainty
            confidence=jnp.ones((batch_size, 1)) * 0.5,  # Start with medium confidence
            budget_remaining=self.initial_budget,
            step=0,
            modules_called=[],
            module_outputs=[]
        )
    
    def update_state(
        self,
        state: ComputeState,
        module_output: ModuleOutput,
        module_type: ModuleType,
        cost: float
    ) -> ComputeState:
        """Update state after running a module."""
        # Apply hidden delta with gating
        gate = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.sigmoid
        ], name=f"update_gate_{state.step}")
        
        hidden_pooled = state.hidden_pooled
        delta = module_output.hidden_delta
        
        # Ensure shapes match
        if delta.shape != hidden_pooled.shape:
            if delta.ndim < hidden_pooled.ndim:
                delta = jnp.broadcast_to(delta[:, None, :], hidden_pooled.shape)
            else:
                delta = delta.mean(axis=1) if delta.ndim > 2 else delta
        
        gate_value = gate(hidden_pooled)
        new_hidden_pooled = hidden_pooled + gate_value * delta
        
        # Update hidden (3D) if needed
        if state.hidden.ndim == 3:
            new_hidden = state.hidden + gate_value[:, None, :] * module_output.hidden_delta[:, None, :]
        else:
            new_hidden = new_hidden_pooled
        
        # Update uncertainty and confidence with EMA
        alpha = 0.3
        new_uncertainty = (1 - alpha) * state.uncertainty + alpha * module_output.uncertainty
        new_confidence = (1 - alpha) * state.confidence + alpha * module_output.confidence
        
        # Update modules called list
        new_modules_called = state.modules_called + [module_type]
        new_module_outputs = state.module_outputs + [module_output]
        
        return ComputeState(
            hidden=new_hidden,
            hidden_pooled=new_hidden_pooled,
            memory_summary=state.memory_summary,  # Updated by memory module if called
            uncertainty=new_uncertainty,
            confidence=new_confidence,
            budget_remaining=state.budget_remaining - cost,
            step=state.step + 1,
            modules_called=new_modules_called,
            module_outputs=new_module_outputs
        )
    
    def __call__(
        self,
        hidden: jnp.ndarray,
        controller: ComputeController,
        registry: ModuleRegistry,
        module_executors: Dict[ModuleType, Callable],
        memory_summary: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[ComputeState, Dict[str, Any]]:
        """
        Execute the compute plan.
        
        Args:
            hidden: Initial hidden representation [batch, seq, d_model]
            controller: The compute controller
            registry: Module registry
            module_executors: Dict mapping ModuleType to callable functions
            memory_summary: Optional initial memory summary
            is_training: Whether in training mode
            
        Returns:
            final_state: Final compute state
            execution_info: Dictionary with execution trace
        """
        # Initialize state
        state = self.initialize_state(hidden, memory_summary)
        
        # Execution trace
        execution_trace = {
            "steps": [],
            "total_cost": 0.0,
            "modules_executed": [],
            "halt_probs": [],
            "confidences": [],
            "uncertainties": []
        }
        
        # Main loop
        for step in range(self.max_steps):
            # Get available modules
            available = registry.get_available(state, state.budget_remaining)
            
            if not available:
                logger.warning(f"No modules available at step {step}")
                break
            
            # Controller decides action
            action, decision_info = controller(state, available, is_training)
            
            execution_trace["halt_probs"].append(decision_info["halt_prob"])
            execution_trace["confidences"].append(float(state.confidence.mean()))
            execution_trace["uncertainties"].append(float(state.uncertainty.mean()))
            
            # Execute selected modules
            for module_type in action.modules:
                if module_type in module_executors:
                    # Execute module
                    executor = module_executors[module_type]
                    module_output = executor(state, is_training)
                    
                    # Get actual cost
                    contract = registry.get(module_type)
                    actual_cost = contract.base_cost if contract else action.budget_allocation
                    
                    # Update state
                    state = self.update_state(state, module_output, module_type, actual_cost)
                    
                    # Track execution
                    execution_trace["total_cost"] += actual_cost
                    execution_trace["modules_executed"].append(module_type.name)
                    execution_trace["steps"].append({
                        "step": step,
                        "module": module_type.name,
                        "cost": actual_cost,
                        "confidence": float(module_output.confidence.mean()),
                        "uncertainty": float(module_output.uncertainty.mean()),
                        "suggests_halt": module_output.suggests_halt
                    })
                else:
                    logger.warning(f"No executor for module {module_type}")
            
            # Check halt condition
            if decision_info["should_halt"]:
                logger.debug(f"Halting at step {step} with confidence {state.confidence.mean():.3f}")
                break
        
        execution_trace["final_step"] = state.step
        execution_trace["final_confidence"] = float(state.confidence.mean())
        execution_trace["final_uncertainty"] = float(state.uncertainty.mean())
        
        return state, execution_trace


def create_compute_controller_fn(
    d_model: int,
    max_steps: int = 10,
    initial_budget: float = 1.0,
    halt_threshold: float = 0.8
):
    """
    Create a transformed compute controller function.
    
    Returns Haiku transformed pair (init, apply) for the controller system.
    """
    def _forward(
        hidden: jnp.ndarray,
        module_executors: Dict[ModuleType, Callable],
        memory_summary: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ):
        controller = ComputeController(
            d_model=d_model,
            max_steps=max_steps,
            halt_threshold=halt_threshold
        )
        
        plan = ComputePlan(
            d_model=d_model,
            max_steps=max_steps,
            initial_budget=initial_budget
        )
        
        registry = ModuleRegistry()
        
        return plan(
            hidden=hidden,
            controller=controller,
            registry=registry,
            module_executors=module_executors,
            memory_summary=memory_summary,
            is_training=is_training
        )
    
    return hk.transform(_forward)
