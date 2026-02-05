"""Pytest configuration for distributed tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "multi_gpu: mark test as requiring multiple GPUs"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    import jax
    
    num_devices = jax.device_count()
    
    for item in items:
        if "multi_gpu" in item.keywords and num_devices < 2:
            item.add_marker(pytest.mark.skip(
                reason=f"Test requires 2+ GPUs, found {num_devices}"
            ))
