# Core quantum module
from core.quantum.quantum_agi_core import QuantumAGICore
from core.quantum.quantum_readiness import (
    QubitAssistedOptimization,
    SelfEvolvingArchitecture,
    AutonomousScientificDiscovery,
    AutonomousMultiAgentSystem,
    VariationalQuantumCircuit,
    QuantumSimulator
)
from core.quantum.extended_quantum_sim import (
    ExtendedQuantumSimulator,
    ExtendedQuantumConfig,
    SparseStateVector,
    ChunkedQuantumSimulator,
    create_extended_quantum_simulator
)
from core.quantum.tensor_network import (
    TensorNetworkConfig,
    MatrixProductState,
    TreeTensorNetwork,
    TensorNetworkQuantumSimulator,
    create_tensor_network_simulator,
    estimate_memory_usage
)

__all__ = [
    'QuantumAGICore',
    'QubitAssistedOptimization',
    'SelfEvolvingArchitecture',
    'AutonomousScientificDiscovery',
    'AutonomousMultiAgentSystem',
    'VariationalQuantumCircuit',
    'QuantumSimulator',
    # Extended quantum simulation (64+ qubits)
    'ExtendedQuantumSimulator',
    'ExtendedQuantumConfig',
    'SparseStateVector',
    'ChunkedQuantumSimulator',
    'create_extended_quantum_simulator',
    # Tensor network approximations
    'TensorNetworkConfig',
    'MatrixProductState',
    'TreeTensorNetwork',
    'TensorNetworkQuantumSimulator',
    'create_tensor_network_simulator',
    'estimate_memory_usage',
]
