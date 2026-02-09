"""
Tests for Tensor Network Config Module

Tests for tensor network configuration dataclass.
"""

import unittest


class TestTensorNetworkConfig(unittest.TestCase):
    """Test TensorNetworkConfig dataclass."""
    
    def test_default_config(self):
        """Test default TensorNetworkConfig values."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig()
        
        self.assertEqual(config.num_qubits, 16)
        self.assertEqual(config.bond_dimension, 64)
        self.assertEqual(config.network_type, "mps")
        self.assertEqual(config.truncation_threshold, 1e-10)
    
    def test_custom_config(self):
        """Test custom TensorNetworkConfig values."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig(
            num_qubits=32,
            bond_dimension=128,
            network_type="ttn",
            truncation_threshold=1e-8
        )
        
        self.assertEqual(config.num_qubits, 32)
        self.assertEqual(config.bond_dimension, 128)
        self.assertEqual(config.network_type, "ttn")
    
    def test_mera_network_type(self):
        """Test MERA network type."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig(network_type="mera")
        
        self.assertEqual(config.network_type, "mera")
    
    def test_post_init_validation_positive_qubits(self):
        """Test validation requires positive num_qubits."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        with self.assertRaises(AssertionError):
            TensorNetworkConfig(num_qubits=0)
    
    def test_post_init_validation_positive_bond_dimension(self):
        """Test validation requires positive bond_dimension."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        with self.assertRaises(AssertionError):
            TensorNetworkConfig(bond_dimension=0)
    
    def test_post_init_validation_network_type(self):
        """Test validation for valid network_type."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        # Valid types should work
        for network_type in ["mps", "ttn", "mera"]:
            config = TensorNetworkConfig(network_type=network_type)
            self.assertEqual(config.network_type, network_type)
    
    def test_post_init_validation_truncation_threshold(self):
        """Test validation requires positive truncation_threshold."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        with self.assertRaises(AssertionError):
            TensorNetworkConfig(truncation_threshold=0)


class TestMemoryOptimizationSettings(unittest.TestCase):
    """Test memory optimization settings."""
    
    def test_sparse_default(self):
        """Test use_sparse default value."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig()
        
        self.assertTrue(config.use_sparse)
    
    def test_chunk_size_default(self):
        """Test chunk_size default value."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig()
        
        self.assertEqual(config.chunk_size, 1024)
    
    def test_custom_memory_settings(self):
        """Test custom memory optimization settings."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig(
            use_sparse=False,
            chunk_size=512
        )
        
        self.assertFalse(config.use_sparse)
        self.assertEqual(config.chunk_size, 512)


class TestOptimizationSettings(unittest.TestCase):
    """Test optimization settings."""
    
    def test_max_iterations_default(self):
        """Test max_iterations default value."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig()
        
        self.assertEqual(config.max_iterations, 100)
    
    def test_convergence_threshold_default(self):
        """Test convergence_threshold default value."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig()
        
        self.assertEqual(config.convergence_threshold, 1e-8)
    
    def test_custom_optimization_settings(self):
        """Test custom optimization settings."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        config = TensorNetworkConfig(
            max_iterations=200,
            convergence_threshold=1e-6
        )
        
        self.assertEqual(config.max_iterations, 200)
        self.assertEqual(config.convergence_threshold, 1e-6)


class TestFromAGIConfig(unittest.TestCase):
    """Test from_agi_config factory method."""
    
    def test_from_agi_config_exists(self):
        """Test from_agi_config method exists."""
        from src.config.tensor_network_config import TensorNetworkConfig
        
        self.assertTrue(hasattr(TensorNetworkConfig, 'from_agi_config'))


class TestConfigSerialization(unittest.TestCase):
    """Test config serialization capabilities."""
    
    def test_config_as_dict(self):
        """Test config can be converted to dict."""
        from src.config.tensor_network_config import TensorNetworkConfig
        from dataclasses import asdict
        
        config = TensorNetworkConfig()
        config_dict = asdict(config)
        
        self.assertIn("num_qubits", config_dict)
        self.assertIn("bond_dimension", config_dict)
        self.assertIn("network_type", config_dict)


if __name__ == "__main__":
    unittest.main()
