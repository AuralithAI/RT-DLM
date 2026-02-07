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


# =============================================================================
# Training Losses
# =============================================================================

class ControllerLossComputer:
    """
    Comprehensive loss computation for the Compute Controller.
    
    Implements multiple auxiliary losses to train the controller:
    1. Task loss (cross-entropy, etc.) - primary objective
    2. Compute efficiency loss - penalize unnecessary computation
    3. Module utilization loss - encourage diverse module usage
    4. Confidence calibration loss - align confidence with accuracy
    5. Budget adherence loss - penalize budget violations
    6. Ponder cost (PonderNet) - regularize thinking time
    """
    
    def __init__(
        self,
        lambda_compute: float = 0.01,
        lambda_utilization: float = 0.005,
        lambda_calibration: float = 0.1,
        lambda_budget: float = 0.05,
        lambda_ponder: float = 0.01,
        target_utilization: float = 0.3,  # Target: use 30% of available modules
        prior_halt_prob: float = 0.2,  # Geometric prior for halting
    ):
        self.lambda_compute = lambda_compute
        self.lambda_utilization = lambda_utilization
        self.lambda_calibration = lambda_calibration
        self.lambda_budget = lambda_budget
        self.lambda_ponder = lambda_ponder
        self.target_utilization = target_utilization
        self.prior_halt_prob = prior_halt_prob
    
    def compute_efficiency_loss(
        self,
        total_cost: float,
        task_difficulty: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute efficiency loss - penalize unnecessary computation.
        
        If task_difficulty is provided, scale penalty by difficulty
        (easy tasks should use less compute).
        """
        base_penalty = jnp.array(total_cost)
        
        if task_difficulty is not None:
            # Scale by inverse difficulty - easy tasks get higher penalty for compute
            difficulty_scale = 1.0 / (task_difficulty + 0.1)
            return self.lambda_compute * base_penalty * difficulty_scale.mean()
        
        return self.lambda_compute * base_penalty
    
    def compute_utilization_loss(
        self,
        modules_called: List[ModuleType],
        total_available: int = len(ModuleType)
    ) -> jnp.ndarray:
        """
        Module utilization loss - encourage diverse but not excessive usage.
        
        Penalizes both under-utilization (missing important modules) and
        over-utilization (calling too many modules unnecessarily).
        """
        unique_modules = len(set(modules_called))
        utilization = unique_modules / total_available
        
        # Deviation from target utilization
        deviation = jnp.abs(utilization - self.target_utilization)
        
        return self.lambda_utilization * deviation
    
    def compute_calibration_loss(
        self,
        predicted_confidence: jnp.ndarray,
        actual_accuracy: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Confidence calibration loss - align confidence with accuracy.
        
        Uses Expected Calibration Error (ECE) style loss.
        """
        # Simple MSE between confidence and accuracy
        calibration_error = jnp.mean((predicted_confidence - actual_accuracy) ** 2)
        
        return self.lambda_calibration * calibration_error
    
    def compute_budget_loss(
        self,
        budget_remaining: float,
        initial_budget: float = 1.0
    ) -> jnp.ndarray:
        """
        Budget adherence loss - penalize budget violations.
        
        Penalizes:
        - Negative budget (overspending)
        - Too much remaining budget (underspending when task incomplete)
        """
        if budget_remaining < 0:
            # Heavy penalty for overspending
            return jnp.array(self.lambda_budget * jnp.abs(budget_remaining) * 10.0)
        
        # Small penalty for excessive remaining budget
        remaining_ratio = budget_remaining / initial_budget
        if remaining_ratio > 0.5:  # More than 50% unused
            return jnp.array(self.lambda_budget * (remaining_ratio - 0.5))
        
        return jnp.array(0.0)
    
    def compute_ponder_loss(
        self,
        halt_probs: List[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Ponder cost (from PonderNet) - regularize thinking time.
        
        Encourages geometric distribution of halt times, preventing
        the model from always taking max steps or always halting early.
        """
        if not halt_probs:
            return jnp.array(0.0)
        
        # Stack halt probabilities
        halt_probs_stack = jnp.stack([h.mean() for h in halt_probs])
        num_steps = len(halt_probs)
        
        # Geometric prior: P(halt at step t) = p * (1-p)^t
        p = self.prior_halt_prob
        prior = jnp.array([p * ((1 - p) ** t) for t in range(num_steps)])
        prior = prior / prior.sum()  # Normalize
        
        # Compute actual halt distribution
        # p(halt at t) = halt_prob[t] * prod(1 - halt_prob[0:t])
        not_halted = jnp.cumprod(1 - jnp.concatenate([jnp.array([0.0]), halt_probs_stack[:-1]]))
        actual_dist = halt_probs_stack * not_halted
        actual_dist = actual_dist / (actual_dist.sum() + 1e-8)
        
        # KL divergence from prior
        kl = jnp.sum(actual_dist * jnp.log((actual_dist + 1e-8) / (prior + 1e-8)))
        
        return self.lambda_ponder * kl
    
    def compute_total_loss(
        self,
        task_loss: jnp.ndarray,
        execution_trace: Dict[str, Any],
        predicted_confidence: jnp.ndarray,
        actual_accuracy: jnp.ndarray,
        task_difficulty: Optional[jnp.ndarray] = None,
        initial_budget: float = 1.0
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute total controller loss with all components.
        
        Args:
            task_loss: Primary task loss
            execution_trace: Trace from ComputePlan execution
            predicted_confidence: Model's confidence at halt
            actual_accuracy: Actual accuracy (0 or 1 per sample)
            task_difficulty: Optional difficulty estimate per sample
            initial_budget: Initial compute budget
            
        Returns:
            total_loss: Combined loss
            loss_components: Dict of individual losses
        """
        # Extract from execution trace
        total_cost = execution_trace.get("total_cost", 0.0)
        modules_called = [ModuleType[m] for m in execution_trace.get("modules_executed", [])]
        halt_probs = execution_trace.get("halt_probs", [])
        budget_remaining = initial_budget - total_cost
        
        # Compute individual losses
        efficiency_loss = self.compute_efficiency_loss(total_cost, task_difficulty)
        utilization_loss = self.compute_utilization_loss(modules_called)
        calibration_loss = self.compute_calibration_loss(predicted_confidence, actual_accuracy)
        budget_loss = self.compute_budget_loss(budget_remaining, initial_budget)
        ponder_loss = self.compute_ponder_loss(halt_probs)
        
        # Total loss
        total_loss = (
            task_loss +
            efficiency_loss +
            utilization_loss +
            calibration_loss +
            budget_loss +
            ponder_loss
        )
        
        loss_components = {
            "task_loss": task_loss,
            "efficiency_loss": efficiency_loss,
            "utilization_loss": utilization_loss,
            "calibration_loss": calibration_loss,
            "budget_loss": budget_loss,
            "ponder_loss": ponder_loss,
            "total_loss": total_loss
        }
        
        return total_loss, loss_components


class ControllerRewardShaper:
    """
    Reward shaping for RL-based controller training.
    
    Provides dense rewards during execution to guide the controller,
    rather than only sparse rewards at the end.
    """
    
    def __init__(
        self,
        reward_correct: float = 1.0,
        reward_efficiency: float = 0.1,
        penalty_wrong: float = -0.5,
        penalty_wasteful: float = -0.05,
        gamma: float = 0.99
    ):
        self.reward_correct = reward_correct
        self.reward_efficiency = reward_efficiency
        self.penalty_wrong = penalty_wrong
        self.penalty_wasteful = penalty_wasteful
        self.gamma = gamma
    
    def compute_step_reward(
        self,
        state: ComputeState,
        module_output: ModuleOutput,
        module_cost: float
    ) -> float:
        """Compute reward for a single step."""
        reward = 0.0
        
        # Reward confidence increase
        if hasattr(state, 'confidence'):
            confidence_delta = float(module_output.confidence.mean() - state.confidence.mean())
            reward += 0.1 * confidence_delta
        
        # Penalize high uncertainty
        uncertainty = float(module_output.uncertainty.mean())
        if uncertainty > 0.7:
            reward += self.penalty_wasteful
        
        # Efficiency bonus for low-cost modules
        if module_cost < 0.1:
            reward += self.reward_efficiency * (0.1 - module_cost)
        
        return reward
    
    def compute_final_reward(
        self,
        is_correct: bool,
        total_cost: float,
        num_steps: int,
        max_steps: int
    ) -> float:
        """Compute final reward after execution."""
        reward = 0.0
        
        # Task correctness
        if is_correct:
            reward += self.reward_correct
            # Bonus for efficiency (correct with less compute)
            efficiency_bonus = (1.0 - total_cost) * self.reward_efficiency
            reward += efficiency_bonus
        else:
            reward += self.penalty_wrong
        
        # Penalty for using max steps (inefficient)
        if num_steps >= max_steps:
            reward += self.penalty_wasteful * 2
        
        return reward
    
    def compute_returns(
        self,
        step_rewards: List[float],
        final_reward: float
    ) -> List[float]:
        """Compute discounted returns for each step."""
        rewards = step_rewards + [final_reward]
        returns = []
        G = 0.0
        
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        return returns[:-1]  # Exclude the final step return


# =============================================================================
# AGI System Integration
# =============================================================================

class ControllerIntegrationMixin:
    """
    Mixin class to integrate ComputeController with RTDLMAGISystem.
    
    Provides methods to:
    - Create module executors from AGI system components
    - Wire controller into forward pass
    - Handle controller state persistence
    """
    
    @staticmethod
    def create_module_executors_from_agi(
        agi_system: Any,
        d_model: int
    ) -> Dict[ModuleType, Callable]:
        """
        Create module executor functions from an AGI system.
        
        Maps ModuleType to actual AGI system method calls.
        """
        executors = {}
        
        # Helper to create standard output
        def make_output(
            hidden_delta: jnp.ndarray,
            confidence: float = 0.5,
            uncertainty: float = 0.5,
            cost: float = 0.1,
            suggests_halt: bool = False
        ) -> ModuleOutput:
            batch_size = hidden_delta.shape[0]
            return ModuleOutput(
                hidden_delta=hidden_delta,
                confidence=jnp.full((batch_size, 1), confidence),
                uncertainty=jnp.full((batch_size, 1), uncertainty),
                actual_cost=cost,
                suggests_halt=suggests_halt
            )
        
        # Memory Retrieval
        def memory_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'memory_bank') and agi_system.memory_bank is not None:
                # Query memory with current hidden state
                query = state.hidden_pooled
                # Simplified memory retrieval
                retrieved = query * 0.1  # Placeholder - actual implementation in adapters
                return make_output(retrieved, confidence=0.6, cost=0.05)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.MEMORY_RETRIEVAL] = memory_executor
        
        # Graph Reasoning
        def graph_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'graph_reasoner') and agi_system.graph_reasoner is not None:
                delta = state.hidden_pooled * 0.15  # Placeholder
                return make_output(delta, confidence=0.55, cost=0.15)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.GRAPH_REASONING] = graph_executor
        
        # Symbolic Reasoning
        def symbolic_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'hybrid_architecture') and agi_system.hybrid_architecture is not None:
                delta = state.hidden_pooled * 0.1  # Placeholder
                return make_output(delta, confidence=0.7, cost=0.10)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.SYMBOLIC_REASONING] = symbolic_executor
        
        # Probabilistic Inference
        def probabilistic_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'hybrid_architecture') and agi_system.hybrid_architecture is not None:
                delta = state.hidden_pooled * 0.08  # Placeholder
                uncertainty = 0.4  # Probabilistic module reduces uncertainty
                return make_output(delta, confidence=0.6, uncertainty=uncertainty, cost=0.08)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.PROBABILISTIC] = probabilistic_executor
        
        # Quantum Simulation
        def quantum_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'quantum_core') and agi_system.quantum_core is not None:
                delta = state.hidden_pooled * 0.2  # Placeholder - expensive
                return make_output(delta, confidence=0.5, cost=0.20)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.QUANTUM_SIMULATION] = quantum_executor
        
        # MoE Routing
        def moe_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'model') and agi_system.model is not None:
                delta = state.hidden_pooled * 0.12  # Placeholder
                return make_output(delta, confidence=0.65, cost=0.12)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.MOE_ROUTING] = moe_executor
        
        # Scientific Discovery
        def scientific_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'scientific_engine') and agi_system.scientific_engine is not None:
                delta = state.hidden_pooled * 0.18  # Placeholder
                return make_output(delta, confidence=0.55, cost=0.18)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.SCIENTIFIC_DISCOVERY] = scientific_executor
        
        # Creative Generation
        def creative_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'creative_engine') and agi_system.creative_engine is not None:
                delta = state.hidden_pooled * 0.15  # Placeholder
                return make_output(delta, confidence=0.5, cost=0.15)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.CREATIVE_GENERATION] = creative_executor
        
        # Consciousness
        def consciousness_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'consciousness') and agi_system.consciousness is not None:
                delta = state.hidden_pooled * 0.1  # Placeholder
                # Consciousness provides introspection - high confidence
                return make_output(delta, confidence=0.75, cost=0.10)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.CONSCIOUSNESS] = consciousness_executor
        
        # Multimodal Fusion
        def multimodal_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            if hasattr(agi_system, 'multimodal_fusion') and agi_system.multimodal_fusion is not None:
                delta = state.hidden_pooled * 0.12  # Placeholder
                return make_output(delta, confidence=0.6, cost=0.12)
            return make_output(jnp.zeros_like(state.hidden_pooled), cost=0.01)
        
        executors[ModuleType.MULTIMODAL_FUSION] = multimodal_executor
        
        # Attention Refinement
        def attention_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            delta = state.hidden_pooled * 0.08  # Simple refinement
            return make_output(delta, confidence=0.6, cost=0.08)
        
        executors[ModuleType.ATTENTION_REFINEMENT] = attention_executor
        
        # Output Generation (always succeeds)
        def output_executor(state: ComputeState, is_training: bool) -> ModuleOutput:
            # Final output - high confidence, suggests halt
            delta = jnp.zeros_like(state.hidden_pooled)
            return make_output(delta, confidence=0.9, cost=0.05, suggests_halt=True)
        
        executors[ModuleType.OUTPUT_GENERATION] = output_executor
        
        return executors


class ControlledAGIForward(hk.Module):
    """
    Controller-driven forward pass for AGI system.
    
    Replaces static module execution with dynamic, controller-driven
    execution that adapts to input complexity and available budget.
    """
    
    def __init__(
        self,
        d_model: int,
        max_steps: int = 10,
        initial_budget: float = 1.0,
        halt_threshold: float = 0.8,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.halt_threshold = halt_threshold
    
    def __call__(
        self,
        hidden: jnp.ndarray,
        module_executors: Dict[ModuleType, Callable],
        memory_summary: Optional[jnp.ndarray] = None,
        is_training: bool = True,
        return_trace: bool = False
    ) -> Tuple[jnp.ndarray, Optional[Dict[str, Any]]]:
        """
        Execute controller-driven forward pass.
        
        Args:
            hidden: Input hidden representation [batch, seq, d_model]
            module_executors: Dict of module type -> executor function
            memory_summary: Optional memory context
            is_training: Whether in training mode
            return_trace: Whether to return execution trace
            
        Returns:
            output: Final hidden representation
            trace: Execution trace (if return_trace=True)
        """
        # Create controller and plan
        controller = ComputeController(
            d_model=self.d_model,
            max_steps=self.max_steps,
            halt_threshold=self.halt_threshold
        )
        
        plan = ComputePlan(
            d_model=self.d_model,
            max_steps=self.max_steps,
            initial_budget=self.initial_budget
        )
        
        registry = ModuleRegistry()
        
        # Execute plan
        final_state, execution_trace = plan(
            hidden=hidden,
            controller=controller,
            registry=registry,
            module_executors=module_executors,
            memory_summary=memory_summary,
            is_training=is_training
        )
        
        # Return final hidden representation
        output = final_state.hidden_pooled
        
        if return_trace:
            return output, execution_trace
        return output, None


def create_controlled_agi_fn(
    d_model: int,
    max_steps: int = 10,
    initial_budget: float = 1.0,
    halt_threshold: float = 0.8
):
    """
    Create a transformed controlled AGI forward function.
    
    Returns Haiku transformed pair for controller-driven AGI forward pass.
    """
    def _forward(
        hidden: jnp.ndarray,
        module_executors: Dict[ModuleType, Callable],
        memory_summary: Optional[jnp.ndarray] = None,
        is_training: bool = True,
        return_trace: bool = False
    ):
        forward_module = ControlledAGIForward(
            d_model=d_model,
            max_steps=max_steps,
            initial_budget=initial_budget,
            halt_threshold=halt_threshold
        )
        
        return forward_module(
            hidden=hidden,
            module_executors=module_executors,
            memory_summary=memory_summary,
            is_training=is_training,
            return_trace=return_trace
        )
    
    return hk.transform(_forward)
