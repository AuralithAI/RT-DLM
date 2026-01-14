"""
Tensor Network Configuration for RT-DLM

Configuration for tensor network-based optimizations:
- Quantum simulation approximation (100+ qubits)
- Memory-efficient large tensor operations
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TensorNetworkConfig:
    """
    Configuration for tensor network operations.
    
    Tensor networks provide efficient representations of high-dimensional
    tensors, enabling:
    - Quantum simulation with 100+ qubits (vs ~30 with full state vector)
    - Memory-efficient large model operations
    - Approximate computations with controlled error
    """
    
    # Network structure
    num_qubits: int = 16  # Number of qubits (for quantum simulation)
    bond_dimension: int = 64  # Maximum bond dimension (controls accuracy vs memory)
    network_type: Literal["mps", "ttn", "mera"] = "mps"  # Network topology
    
    # Accuracy settings
    truncation_threshold: float = 1e-10  # SVD truncation threshold
    max_iterations: int = 100  # Max iterations for optimization
    convergence_threshold: float = 1e-8  # Convergence criterion
    
    # Memory optimization
    use_sparse: bool = True  # Use sparse representations where possible
    chunk_size: int = 1024  # Chunk size for batched operations
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_qubits > 0, "num_qubits must be positive"
        assert self.bond_dimension > 0, "bond_dimension must be positive"
        assert self.network_type in ["mps", "ttn", "mera"], \
            "network_type must be 'mps', 'ttn', or 'mera'"
        assert self.truncation_threshold > 0, "truncation_threshold must be positive"
    
    @classmethod
    def from_agi_config(cls, agi_config) -> "TensorNetworkConfig":
        """Create tensor network config from AGI config"""
        return cls(
            num_qubits=agi_config.quantum_max_qubits,
            bond_dimension=agi_config.quantum_qubits * 4,  # Scale with model qubits
            network_type="mps",
            use_sparse=agi_config.quantum_sparse_mode,
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "num_qubits": self.num_qubits,
            "bond_dimension": self.bond_dimension,
            "network_type": self.network_type,
            "truncation_threshold": self.truncation_threshold,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "use_sparse": self.use_sparse,
            "chunk_size": self.chunk_size,
        }
    
    def estimate_memory_gb(self) -> float:
        """Estimate memory usage in GB"""
        if self.network_type == "mps":
            # MPS: O(n * d * chi^2) where n=qubits, d=2 (physical), chi=bond
            memory_bytes = self.num_qubits * 2 * (self.bond_dimension ** 2) * 16  # complex128
        elif self.network_type == "ttn":
            # TTN: O(n * d * chi^3) roughly
            memory_bytes = self.num_qubits * 2 * (self.bond_dimension ** 3) * 16
        else:  # mera
            memory_bytes = self.num_qubits * 4 * (self.bond_dimension ** 4) * 16
        return memory_bytes / (1024 ** 3)
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 50)
        print("Tensor Network Configuration")
        print("=" * 50)
        print(f"  Network type: {self.network_type.upper()}")
        print(f"  Qubits: {self.num_qubits}")
        print(f"  Bond dimension: {self.bond_dimension}")
        print(f"  Estimated memory: {self.estimate_memory_gb():.3f} GB")
        print(f"  Truncation threshold: {self.truncation_threshold}")
        print(f"  Sparse mode: {self.use_sparse}")
        print("=" * 50)
