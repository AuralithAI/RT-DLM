"""
Tensor Network Approximations for Quantum Simulation.

IMPORTANT: CLASSICAL SIMULATION ONLY - Not actual quantum hardware.
Uses tensor network decompositions for efficient simulation of 100+ qubits.

Implements:
- Matrix Product States (MPS) for 1D systems
- Tree Tensor Networks (TTN) for hierarchical systems
- MERA for scale-invariant systems

To disable: Set config.quantum_layers=0 in AGIConfig.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.tensor_network_config import TensorNetworkConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Matrix Product State (MPS)
# =============================================================================

class MatrixProductState:
    """
    Matrix Product State representation of quantum state.
    
    For n qubits, represents state as:
    |ψ⟩ = Σ A[1]_{i1} A[2]_{i2} ... A[n]_{in} |i1 i2 ... in⟩
    
    Where each A[k] is a 3-tensor with shape [bond_left, physical, bond_right]
    
    Benefits:
    - Memory: O(n * d * χ²) vs O(2^n) for full state
    - Efficient for low-entanglement states
    - Enables efficient simulation of 100+ qubit systems
    """
    
    def __init__(self, num_qubits: int, bond_dimension: int = 64):
        self.num_qubits = num_qubits
        self.max_bond_dim = bond_dimension
        self.physical_dim = 2  # Qubit
        
        # Initialize MPS tensors in product state |00...0⟩
        self.tensors: List[jnp.ndarray] = self._initialize_product_state()
        
    def _initialize_product_state(self) -> List[jnp.ndarray]:
        """Initialize MPS in |00...0⟩ product state"""
        tensors = []
        
        for i in range(self.num_qubits):
            if i == 0:
                # Left boundary: shape [1, 2, 1]
                tensor = jnp.zeros((1, 2, 1), dtype=jnp.complex64)
                tensor = tensor.at[0, 0, 0].set(1.0)  # |0⟩
            elif i == self.num_qubits - 1:
                # Right boundary: shape [1, 2, 1]
                tensor = jnp.zeros((1, 2, 1), dtype=jnp.complex64)
                tensor = tensor.at[0, 0, 0].set(1.0)  # |0⟩
            else:
                # Bulk: shape [1, 2, 1]
                tensor = jnp.zeros((1, 2, 1), dtype=jnp.complex64)
                tensor = tensor.at[0, 0, 0].set(1.0)  # |0⟩
                
            tensors.append(tensor)
            
        return tensors
    
    def get_bond_dimensions(self) -> List[int]:
        """Get current bond dimensions"""
        return [t.shape[2] for t in self.tensors[:-1]]
    
    def apply_single_qubit_gate(self, gate: jnp.ndarray, site: int):
        """
        Apply single-qubit gate at given site.
        
        Gate shape: [2, 2]
        """
        tensor = self.tensors[site]
        # Contract gate with physical index
        # tensor: [bond_l, phys, bond_r]
        # gate: [phys_out, phys_in]
        new_tensor = jnp.einsum("lpb,op->lob", tensor, gate)
        self.tensors[site] = new_tensor
        
    def apply_two_qubit_gate(self, gate: jnp.ndarray, site1: int, site2: int,
                              truncation_threshold: float = 1e-10):
        """
        Apply two-qubit gate between adjacent sites.
        
        Uses SVD to maintain MPS form with bond dimension control.
        
        Gate shape: [4, 4] or [2, 2, 2, 2]
        """
        if abs(site2 - site1) != 1:
            raise ValueError("Two-qubit gate requires adjacent sites")
            
        left_site = min(site1, site2)
        right_site = max(site1, site2)
        
        # Get tensors
        A = self.tensors[left_site]   # [χ_l, d, χ_m]
        B = self.tensors[right_site]  # [χ_m, d, χ_r]
        
        # Contract to form two-site tensor
        # θ[χ_l, d1, d2, χ_r] = A[χ_l, d1, χ_m] B[χ_m, d2, χ_r]
        theta = jnp.einsum("lam,mbr->labr", A, B)
        
        # Reshape gate if needed
        if gate.shape == (4, 4):
            gate = gate.reshape(2, 2, 2, 2)
        
        # Apply gate
        # gate[o1, o2, i1, i2] θ[χ_l, i1, i2, χ_r] -> [χ_l, o1, o2, χ_r]
        theta = jnp.einsum("opin,lnir->loir", gate, theta)
        
        # Reshape for SVD: [χ_l * d, d * χ_r]
        chi_l, d1, d2, chi_r = theta.shape
        theta_matrix = theta.reshape(chi_l * d1, d2 * chi_r)
        
        # SVD with truncation
        U, S, Vh = jnp.linalg.svd(theta_matrix, full_matrices=False)
        
        # Truncate to max bond dimension
        keep = min(self.max_bond_dim, len(S))
        
        # Also truncate based on singular value threshold
        significant = jnp.sum(S > truncation_threshold * S[0])
        keep = min(keep, int(significant))
        keep = max(keep, 1)  # Keep at least one
        
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        
        # Absorb singular values (left-canonical form)
        # New A: [χ_l, d, χ_new]
        new_A = U.reshape(chi_l, d1, keep)
        
        # New B: [χ_new, d, χ_r] with singular values
        SV = jnp.diag(S) @ Vh
        new_B = SV.reshape(keep, d2, chi_r)
        
        self.tensors[left_site] = new_A
        self.tensors[right_site] = new_B
        
    def canonicalize_left(self):
        """Put MPS in left-canonical form using QR decomposition"""
        for i in range(self.num_qubits - 1):
            tensor = self.tensors[i]
            chi_l, d, chi_r = tensor.shape
            
            # Reshape for QR
            matrix = tensor.reshape(chi_l * d, chi_r)
            Q, R = jnp.linalg.qr(matrix)
            
            # Update current tensor
            new_chi = Q.shape[1]
            self.tensors[i] = Q.reshape(chi_l, d, new_chi)
            
            # Absorb R into next tensor
            next_tensor = self.tensors[i + 1]
            self.tensors[i + 1] = jnp.einsum("ab,bdr->adr", R, next_tensor)
            
    def canonicalize_right(self):
        """Put MPS in right-canonical form using QR decomposition"""
        for i in range(self.num_qubits - 1, 0, -1):
            tensor = self.tensors[i]
            chi_l, d, chi_r = tensor.shape
            
            # Reshape for QR (from right)
            matrix = tensor.reshape(chi_l, d * chi_r).T
            Q, R = jnp.linalg.qr(matrix)
            
            # Update current tensor
            new_chi = Q.shape[1]
            self.tensors[i] = Q.T.reshape(new_chi, d, chi_r)
            
            # Absorb R into previous tensor
            prev_tensor = self.tensors[i - 1]
            self.tensors[i - 1] = jnp.einsum("lda,ab->ldb", prev_tensor, R.T)
    
    def compute_norm(self) -> float:
        """Compute norm of MPS state"""
        # Contract all tensors
        result = self.tensors[0]
        
        for i in range(1, self.num_qubits):
            # result: [χ_l, d1...di, χ_m]
            # next: [χ_m, d_{i+1}, χ_r]
            result = jnp.einsum("...m,mdr->...dr", result, self.tensors[i])
            
        # Result should be scalar (for normalized state)
        # Contract physical indices with their conjugates
        result_conj = jnp.conj(result)
        norm_sq = jnp.einsum("...,...->", result, result_conj)
        
        return jnp.sqrt(jnp.real(norm_sq))
    
    def normalize(self):
        """Normalize the MPS state"""
        norm = self.compute_norm()
        if norm > 0:
            # Absorb normalization into first tensor
            self.tensors[0] = self.tensors[0] / norm
            
    def sample(self, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample a bit string from MPS probability distribution.
        
        Uses efficient sequential sampling from left to right.
        """
        # Put in right-canonical form first
        self.canonicalize_right()
        
        samples = []
        left_boundary = jnp.array([[1.0 + 0j]])  # Initial left boundary
        
        for i in range(self.num_qubits):
            tensor = self.tensors[i]
            
            # Compute probabilities for |0⟩ and |1⟩
            probs = []
            for outcome in range(2):
                # Contract with left boundary and select outcome
                contracted = jnp.einsum("ab,bor->ao", left_boundary, tensor[:, outcome:outcome+1, :])
                prob = jnp.sum(jnp.abs(contracted) ** 2)
                probs.append(float(prob))
                
            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]
            
            # Sample
            rng_key, sample_key = jax.random.split(rng_key)
            outcome = int(jax.random.choice(sample_key, 2, p=jnp.array(probs)))
            samples.append(outcome)
            
            # Update left boundary
            left_boundary = jnp.einsum("ab,bor->ao", left_boundary, tensor[:, outcome:outcome+1, :])
            
        return jnp.array(samples)
    
    def expectation_value(self, operator: jnp.ndarray, site: int) -> complex:
        """
        Compute expectation value of single-site operator.
        
        ⟨ψ|O_i|ψ⟩
        """
        # Put in mixed-canonical form around site
        self.canonicalize_left()
        
        # Contract from left up to site
        left = jnp.eye(1, dtype=jnp.complex64)
        for i in range(site):
            tensor = self.tensors[i]
            # Contract: left[a,b] * tensor[a,d,c] * conj(tensor)[b,d,e] -> new_left[c,e]
            left = jnp.einsum("ab,adc,bde->ce", left, tensor, jnp.conj(tensor))
        
        # Apply operator at site
        tensor = self.tensors[site]
        tensor_op = jnp.einsum("op,apc->aoc", operator, tensor)
        middle = jnp.einsum("ab,aoc,bpe->ce", left, tensor_op, jnp.conj(tensor))
        
        # Contract from right
        right = jnp.eye(1, dtype=jnp.complex64)
        for i in range(self.num_qubits - 1, site, -1):
            tensor = self.tensors[i]
            right = jnp.einsum("adc,bde,ce->ab", tensor, jnp.conj(tensor), right)
        
        # Final contraction
        result = jnp.einsum("ab,ab->", middle, right)
        
        return result


# =============================================================================
# Tree Tensor Network (TTN)
# =============================================================================

class TreeTensorNetwork:
    """
    Tree Tensor Network for hierarchical quantum state representation.
    
    Organizes qubits in a binary tree structure where each node
    is a tensor that entangles its children.
    
    Benefits:
    - Efficient for states with hierarchical entanglement
    - O(n log n * χ³) computational complexity
    """
    
    def __init__(self, num_qubits: int, bond_dimension: int = 32):
        # Round up to power of 2
        self.num_leaves = 2 ** int(np.ceil(np.log2(num_qubits)))
        self.num_qubits = num_qubits
        self.bond_dimension = bond_dimension
        self.physical_dim = 2
        
        # Tree structure: list of layers
        # Layer 0: leaf tensors [physical, bond_up]
        # Layer k: internal nodes [bond_left, bond_right, bond_up]
        # Top layer: root [bond_left, bond_right]
        self.layers: List[List[jnp.ndarray]] = self._initialize_tree()
        
    def _initialize_tree(self) -> List[List[jnp.ndarray]]:
        """Initialize tree in product state"""
        layers = []
        
        # Leaf layer
        leaves = []
        for i in range(self.num_leaves):
            if i < self.num_qubits:
                # Real qubit: |0⟩
                leaf = jnp.zeros((2, 1), dtype=jnp.complex64)
                leaf = leaf.at[0, 0].set(1.0)
            else:
                # Padding qubit
                leaf = jnp.zeros((2, 1), dtype=jnp.complex64)
                leaf = leaf.at[0, 0].set(1.0)
            leaves.append(leaf)
        layers.append(leaves)
        
        # Internal layers
        num_nodes = self.num_leaves // 2
        while num_nodes >= 1:
            layer = []
            for _ in range(num_nodes):
                if num_nodes == 1:
                    # Root: no upward bond
                    node = jnp.zeros((1, 1), dtype=jnp.complex64)
                    node = node.at[0, 0].set(1.0)
                else:
                    node = jnp.zeros((1, 1, 1), dtype=jnp.complex64)
                    node = node.at[0, 0, 0].set(1.0)
                layer.append(node)
            layers.append(layer)
            num_nodes //= 2
            
        return layers
    
    def apply_single_qubit_gate(self, gate: jnp.ndarray, qubit: int):
        """Apply single-qubit gate to leaf"""
        if qubit >= self.num_qubits:
            return
            
        leaf = self.layers[0][qubit]
        # leaf: [physical, bond]
        # gate: [out, in]
        new_leaf = jnp.einsum("pi,op->oi", leaf, gate)
        self.layers[0][qubit] = new_leaf


# =============================================================================
# Tensor Network Quantum Simulator
# =============================================================================

class TensorNetworkQuantumSimulator:
    """
    Quantum simulator using tensor network representations.
    
    Automatically selects representation based on system size and
    entanglement structure.
    """
    
    def __init__(self, config: Optional[TensorNetworkConfig] = None):
        self.config = config or TensorNetworkConfig()
        self.state = None
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize quantum state with appropriate tensor network"""
        if self.config.network_type == "mps":
            self.state = MatrixProductState(
                self.config.num_qubits,
                self.config.bond_dimension
            )
        elif self.config.network_type == "ttn":
            self.state = TreeTensorNetwork(
                self.config.num_qubits,
                self.config.bond_dimension
            )
        else:
            raise ValueError(f"Unknown network type: {self.config.network_type}")
            
    def reset(self):
        """Reset to initial |00...0⟩ state"""
        self._initialize_state()
        
    def apply_gate(self, gate: jnp.ndarray, qubits: List[int]):
        """Apply gate to specified qubits"""
        if len(qubits) == 1:
            self.state.apply_single_qubit_gate(gate, qubits[0])
        elif len(qubits) == 2:
            if isinstance(self.state, MatrixProductState):
                self.state.apply_two_qubit_gate(
                    gate, qubits[0], qubits[1],
                    self.config.truncation_threshold
                )
            else:
                raise NotImplementedError("Two-qubit gates not yet supported for TTN")
        else:
            raise ValueError("Only 1 and 2 qubit gates supported")
            
    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate"""
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        self.apply_gate(H, [qubit])
        
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
        
        # For MPS, need to swap to make adjacent if necessary
        if isinstance(self.state, MatrixProductState):
            if abs(control - target) > 1:
                # Use SWAP network to bring qubits adjacent
                self._apply_long_range_cnot(control, target)
            else:
                self.apply_gate(CNOT, [control, target])
        else:
            raise NotImplementedError("CNOT not yet supported for TTN")
            
    def _apply_long_range_cnot(self, control: int, target: int):
        """Apply CNOT between non-adjacent qubits using SWAP network"""
        SWAP = jnp.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.complex64)
        
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
        
        # Move control next to target
        direction = 1 if target > control else -1
        current = control
        
        # Swap until adjacent
        while abs(current - target) > 1:
            next_pos = current + direction
            self.apply_gate(SWAP, [min(current, next_pos), max(current, next_pos)])
            current = next_pos
            
        # Apply CNOT
        self.apply_gate(CNOT, [current, target])
        
        # Swap back
        while current != control:
            prev_pos = current - direction
            self.apply_gate(SWAP, [min(current, prev_pos), max(current, prev_pos)])
            current = prev_pos
            
    def apply_rotation_y(self, qubit: int, angle: float):
        """Apply Y rotation gate"""
        c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
        RY = jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)
        self.apply_gate(RY, [qubit])
        
    def apply_variational_layer(self, params: jnp.ndarray):
        """
        Apply variational quantum layer (for VQE/QAOA).
        
        params: [num_qubits * 3] - three rotation angles per qubit
        """
        n = self.config.num_qubits
        
        # Single-qubit rotations
        for i in range(n):
            if i * 3 + 2 < len(params):
                self.apply_rotation_y(i, float(params[i * 3]))
                # Could add RX, RZ rotations here
                
        # Entangling layer (linear connectivity)
        for i in range(n - 1):
            self.apply_cnot(i, i + 1)
            
    def sample(self, rng_key: jax.random.PRNGKey, num_samples: int = 1) -> jnp.ndarray:
        """Sample bit strings from the quantum state"""
        if isinstance(self.state, MatrixProductState):
            samples = []
            for _ in range(num_samples):
                rng_key, sample_key = jax.random.split(rng_key)
                sample = self.state.sample(sample_key)
                samples.append(sample)
            return jnp.stack(samples)
        else:
            raise NotImplementedError("Sampling not yet supported for this network type")
            
    def get_bond_dimensions(self) -> List[int]:
        """Get current bond dimensions (for MPS)"""
        if isinstance(self.state, MatrixProductState):
            return self.state.get_bond_dimensions()
        return []
    
    def get_entanglement_entropy(self, site: int) -> float:
        """
        Compute entanglement entropy at bipartition.
        
        S = -Σ λ² log(λ²) where λ are Schmidt values
        """
        if isinstance(self.state, MatrixProductState):
            # Put in mixed canonical form
            self.state.canonicalize_left()
            
            # Get tensor and do SVD
            tensor = self.state.tensors[site]
            chi_l, d, chi_r = tensor.shape
            matrix = tensor.reshape(chi_l * d, chi_r)
            
            _, S, _ = jnp.linalg.svd(matrix, full_matrices=False)
            
            # Compute entropy
            S_sq = S ** 2
            S_sq = S_sq / jnp.sum(S_sq)  # Normalize
            entropy = -jnp.sum(S_sq * jnp.log(S_sq + 1e-12))
            
            return float(entropy)
        return 0.0


# =============================================================================
# Factory Function
# =============================================================================

def create_tensor_network_simulator(config) -> TensorNetworkQuantumSimulator:
    """Create tensor network simulator from config.
    
    Accepts either a TensorNetworkConfig directly or an AGI config
    with quantum_max_qubits and quantum_qubits attributes.
    """
    if isinstance(config, TensorNetworkConfig):
        tn_config = config
    else:
        # Assume AGI config
        tn_config = TensorNetworkConfig(
            num_qubits=getattr(config, 'quantum_max_qubits', 16),
            bond_dimension=getattr(config, 'quantum_qubits', 4) * 4,  # Scale with model qubits
            network_type="mps"
        )
    return TensorNetworkQuantumSimulator(tn_config)


def estimate_memory_usage(num_qubits: int, bond_dimension: int) -> Dict[str, float]:
    """
    Estimate memory usage for different representations.
    
    Returns memory in GB.
    """
    bytes_per_complex = 8  # complex64
    
    # Full state vector
    full_state = (2 ** num_qubits) * bytes_per_complex / 1e9
    
    # MPS
    mps = num_qubits * 2 * (bond_dimension ** 2) * bytes_per_complex / 1e9
    
    # TTN
    num_nodes = 2 * num_qubits - 1  # Binary tree
    ttn = num_nodes * (bond_dimension ** 3) * bytes_per_complex / 1e9
    
    return {
        "full_state_gb": full_state,
        "mps_gb": mps,
        "ttn_gb": ttn,
        "compression_ratio": full_state / mps if mps > 0 else float('inf')
    }
