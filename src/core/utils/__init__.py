"""
RT-DLM Core Utilities

Cross-cutting utilities for the core package:
- Decorators for code annotation (@dev_utility, @deprecated, @experimental)
- Helper functions

Usage:
    from src.core.utils import dev_utility, deprecated, experimental
    
    @dev_utility("Testing only")
    def test_helper():
        ...
"""

from src.core.utils.decorators import (
    # Dev utility
    dev_utility,
    is_dev_utility,
    get_dev_utility_reason,
    
    # Deprecation
    deprecated,
    is_deprecated,
    
    # Experimental
    experimental,
    is_experimental,
    
    # Internal
    internal,
    is_internal,
    
    # Dependencies
    requires,
)

__all__ = [
    # Dev utility
    'dev_utility',
    'is_dev_utility',
    'get_dev_utility_reason',
    
    # Deprecation
    'deprecated',
    'is_deprecated',
    
    # Experimental
    'experimental',
    'is_experimental',
    
    # Internal
    'internal',
    'is_internal',
    
    # Dependencies
    'requires',
]
