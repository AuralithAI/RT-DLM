"""
Pytest configuration and shared fixtures for RT-DLM tests
"""

import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


@pytest.fixture
def rng():
    """Provide a JAX random key"""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_input(rng):
    """Provide sample input tensor"""
    return jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))


@pytest.fixture
def random_input(rng):
    """Provide random input tensor"""
    return jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))


@pytest.fixture
def d_model():
    """Model dimension"""
    return D_MODEL


@pytest.fixture
def batch_size():
    """Batch size"""
    return BATCH_SIZE


@pytest.fixture
def seq_len():
    """Sequence length"""
    return SEQ_LEN
