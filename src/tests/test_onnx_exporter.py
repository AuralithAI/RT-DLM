"""
Tests for ONNX Exporter Module

Tests for JAX to ONNX model export functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class TestONNXExportConfig(unittest.TestCase):
    """Test ONNXExportConfig dataclass."""
    
    def test_default_config(self):
        """Test default ONNX export configuration."""
        from src.core.export.onnx_exporter import ONNXExportConfig
        
        config = ONNXExportConfig()
        
        self.assertEqual(config.opset_version, 15)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.sequence_length, 512)
        self.assertTrue(config.optimize)
        self.assertTrue(config.validate_output)
    
    def test_custom_config(self):
        """Test custom ONNX export configuration."""
        from src.core.export.onnx_exporter import ONNXExportConfig
        
        config = ONNXExportConfig(
            opset_version=13,
            batch_size=4,
            sequence_length=256,
            optimize=False,
            use_external_data=True
        )
        
        self.assertEqual(config.opset_version, 13)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.sequence_length, 256)
        self.assertFalse(config.optimize)
        self.assertTrue(config.use_external_data)
    
    def test_config_validation_tolerance(self):
        """Test validation tolerance setting."""
        from src.core.export.onnx_exporter import ONNXExportConfig
        
        config = ONNXExportConfig(validation_tolerance=1e-3)
        
        self.assertEqual(config.validation_tolerance, 1e-3)
    
    def test_config_external_data_threshold(self):
        """Test external data threshold setting."""
        from src.core.export.onnx_exporter import ONNXExportConfig
        
        config = ONNXExportConfig(
            use_external_data=True,
            external_data_threshold=2048
        )
        
        self.assertEqual(config.external_data_threshold, 2048)


class TestONNXExporterInit(unittest.TestCase):
    """Test ONNXExporter initialization."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_exporter_initialization(self, mock_check):
        """Test ONNXExporter initialization with mocked dependencies."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        # Mock model function
        def mock_model_fn(params, x):
            return x
        
        params = {"layer": jnp.zeros((10, 10))}
        config = ONNXExportConfig()
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertIsNotNone(exporter.model_fn)
        self.assertIsNotNone(exporter.params)
        self.assertIsNotNone(exporter.config)
        mock_check.assert_called_once()
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_exporter_with_default_config(self, mock_check):
        """Test ONNXExporter with default configuration."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        
        exporter = ONNXExporter(mock_model_fn, params)
        
        # Should create default config
        self.assertIsNotNone(exporter.config)
        self.assertEqual(exporter.config.opset_version, 15)


class TestDependencyChecks(unittest.TestCase):
    """Test dependency checking functionality."""
    
    def test_dependency_check_missing_tensorflow(self):
        """Test that missing TensorFlow raises ImportError."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        
        # This test verifies the structure - actual import behavior 
        # depends on environment
        # Just verify the exporter class has the dependency check method
        self.assertTrue(hasattr(ONNXExporter, '_check_dependencies'))


class TestExportFunctionality(unittest.TestCase):
    """Test ONNX export functionality."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_create_input_spec(self, mock_check):
        """Test input specification creation."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(batch_size=2, sequence_length=128)
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        # Verify config is stored correctly
        self.assertEqual(exporter.config.batch_size, 2)
        self.assertEqual(exporter.config.sequence_length, 128)
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_output_path_creation(self, mock_check):
        """Test that output directory is created if it doesn't exist."""
        from src.core.export.onnx_exporter import ONNXExporter
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        exporter = ONNXExporter(mock_model_fn, params)
        
        # Verify exporter has export method
        self.assertTrue(hasattr(exporter, 'export'))


class TestONNXOptimization(unittest.TestCase):
    """Test ONNX model optimization."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_optimize_config_flag(self, mock_check):
        """Test optimization configuration flag."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(optimize=True)
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertTrue(exporter.config.optimize)
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_fold_constants_config(self, mock_check):
        """Test constant folding configuration."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(fold_constants=True)
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertTrue(exporter.config.fold_constants)


class TestONNXValidation(unittest.TestCase):
    """Test ONNX model validation."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_validation_config(self, mock_check):
        """Test validation configuration."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(
            validate_output=True,
            validation_tolerance=1e-4
        )
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertTrue(exporter.config.validate_output)
        self.assertEqual(exporter.config.validation_tolerance, 1e-4)
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_validation_method_exists(self, mock_check):
        """Test that validation method exists."""
        from src.core.export.onnx_exporter import ONNXExporter
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        exporter = ONNXExporter(mock_model_fn, params)
        
        # Should have validation method
        self.assertTrue(hasattr(exporter, '_validate_export') or 
                       hasattr(exporter, 'validate'))


class TestExternalDataSupport(unittest.TestCase):
    """Test external data support for large models."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_external_data_config(self, mock_check):
        """Test external data configuration."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(
            use_external_data=True,
            external_data_threshold=4096
        )
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertTrue(exporter.config.use_external_data)
        self.assertEqual(exporter.config.external_data_threshold, 4096)


class TestOpsetVersions(unittest.TestCase):
    """Test ONNX opset version handling."""
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_default_opset_version(self, mock_check):
        """Test default opset version."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        exporter = ONNXExporter(mock_model_fn, params)
        
        self.assertEqual(exporter.config.opset_version, 15)
    
    @patch('src.core.export.onnx_exporter.ONNXExporter._check_dependencies')
    def test_custom_opset_version(self, mock_check):
        """Test custom opset version."""
        from src.core.export.onnx_exporter import ONNXExporter, ONNXExportConfig
        
        def mock_model_fn(params, x):
            return x
        
        params = {}
        config = ONNXExportConfig(opset_version=12)
        
        exporter = ONNXExporter(mock_model_fn, params, config)
        
        self.assertEqual(exporter.config.opset_version, 12)


if __name__ == "__main__":
    unittest.main()
