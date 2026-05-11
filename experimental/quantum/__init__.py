"""Opt-in quantum subsystem: CLASSICAL SIMULATION only (no real quantum hardware). Gated by AGI_ENABLE_QUANTUM=1."""

import os

if os.environ.get("AGI_ENABLE_QUANTUM") != "1":
    raise ImportError(
        "experimental.quantum is opt-in. Set AGI_ENABLE_QUANTUM=1 to enable."
    )

from experimental.quantum.quantum_agi_core import QuantumAGICore
from experimental.quantum.quantum_readiness import (
    QubitAssistedOptimization,
    SelfEvolvingArchitecture,
    AutonomousScientificDiscovery,
    AutonomousMultiAgentSystem,
    VariationalQuantumCircuit,
    QuantumSimulator,
)
from experimental.quantum.extended_quantum_sim import (
    ExtendedQuantumSimulator,
    ExtendedQuantumConfig,
    SparseStateVector,
    ChunkedQuantumSimulator,
    create_extended_quantum_simulator,
)
from experimental.quantum.tensor_network import (
    MatrixProductState,
    TreeTensorNetwork,
    TensorNetworkQuantumSimulator,
    create_tensor_network_simulator,
    estimate_memory_usage,
)
from src.config.tensor_network_config import TensorNetworkConfig


def estimate_quantum_overhead(
    num_qubits: int = 16,
    num_layers: int = 4,
    d_model: int = 384,
    use_tensor_network: bool = False,
    bond_dimension: int = 64,
) -> dict:
    """Estimate memory and compute overhead of quantum simulation."""
    if use_tensor_network:
        state_memory_bytes = num_qubits * bond_dimension**2 * 16
        memory_formula = f"O(n × χ²) = {num_qubits} × {bond_dimension}² × 16 bytes"
    else:
        state_memory_bytes = (2**num_qubits) * 16
        memory_formula = f"O(2^n) = 2^{num_qubits} × 16 bytes"
    gate_params = num_layers * num_qubits * 3
    projection_params = 2 * d_model * (2 ** min(6, num_qubits))
    total_params = gate_params + projection_params
    return {
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "use_tensor_network": use_tensor_network,
        "state_memory_bytes": state_memory_bytes,
        "state_memory_mb": state_memory_bytes / (1024**2),
        "memory_formula": memory_formula,
        "gate_parameters": gate_params,
        "projection_parameters": projection_params,
        "total_trainable_params": total_params,
        "recommended_max_qubits_full_state": 24,
        "recommended_min_qubits_tensor_network": 20,
        "note": "Classical simulation - set quantum_layers=0 to disable",
    }


__all__ = [
    "QuantumAGICore",
    "QubitAssistedOptimization",
    "SelfEvolvingArchitecture",
    "AutonomousScientificDiscovery",
    "AutonomousMultiAgentSystem",
    "VariationalQuantumCircuit",
    "QuantumSimulator",
    "ExtendedQuantumSimulator",
    "ExtendedQuantumConfig",
    "SparseStateVector",
    "ChunkedQuantumSimulator",
    "create_extended_quantum_simulator",
    "TensorNetworkConfig",
    "MatrixProductState",
    "TreeTensorNetwork",
    "TensorNetworkQuantumSimulator",
    "create_tensor_network_simulator",
    "estimate_memory_usage",
    "estimate_quantum_overhead",
]
