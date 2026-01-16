"""
RT-DLM Utility Decorators

Cross-cutting decorators for code annotation and behavior modification.
These are shared utilities used across the entire codebase.

Usage:
    from core.utils import dev_utility, deprecated, experimental
    
    @dev_utility("Testing only - not for production inference")
    def sample_tokens(...):
        ...
    
    @deprecated("Use new_function() instead")
    def old_function(...):
        ...
    
    @experimental("API may change")
    class NewFeature:
        ...
"""

import warnings
import functools
import logging
from typing import Optional, Callable, Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Development Utility Decorator
# =============================================================================

def dev_utility(reason: str = "Development/testing utility"):
    """
    Mark code as a development/internal utility.
    
    These components are NOT part of the production training pipeline
    but are useful for debugging, testing, or development.
    
    Usage:
        @dev_utility("Used for testing generation quality")
        def sample_tokens(...):
            ...
        
        @dev_utility("Internal debugging tool")
        class DebugVisualizer:
            ...
    
    Args:
        reason: Explanation of why this is a dev utility
        
    Returns:
        Decorated function/class with metadata attached
    """
    def decorator(obj):
        obj._is_dev_utility = True
        obj._dev_utility_reason = reason
        
        if callable(obj) and not isinstance(obj, type):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                return obj(*args, **kwargs)
            wrapper._is_dev_utility = True
            wrapper._dev_utility_reason = reason
            wrapper.__doc__ = f"[DEV UTILITY: {reason}]\n\n{obj.__doc__ or ''}"
            return wrapper
        else:
            # For classes
            original_doc = obj.__doc__ or ""
            obj.__doc__ = f"[DEV UTILITY: {reason}]\n\n{original_doc}"
            return obj
    return decorator


def is_dev_utility(obj) -> bool:
    """Check if an object is marked as a dev utility."""
    return getattr(obj, '_is_dev_utility', False)


def get_dev_utility_reason(obj) -> Optional[str]:
    """Get the reason why something is marked as dev utility."""
    return getattr(obj, '_dev_utility_reason', None)


# =============================================================================
# Deprecation Decorator
# =============================================================================

def deprecated(reason: str = "This is deprecated", 
               version: Optional[str] = None,
               replacement: Optional[str] = None):
    """
    Mark code as deprecated with optional replacement guidance.
    
    Emits a DeprecationWarning when the decorated item is used.
    
    Usage:
        @deprecated("Use new_api() instead", version="2.0", replacement="new_api")
        def old_api(...):
            ...
    
    Args:
        reason: Why it's deprecated
        version: Version when it will be removed
        replacement: Name of replacement function/class
    """
    def decorator(obj):
        obj._is_deprecated = True
        obj._deprecation_reason = reason
        obj._deprecation_version = version
        obj._deprecation_replacement = replacement
        
        msg_parts = [f"{obj.__name__} is deprecated: {reason}"]
        if replacement:
            msg_parts.append(f"Use {replacement} instead.")
        if version:
            msg_parts.append(f"Will be removed in version {version}.")
        full_msg = " ".join(msg_parts)
        
        if callable(obj) and not isinstance(obj, type):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(full_msg, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)
            wrapper._is_deprecated = True
            wrapper._deprecation_reason = reason
            wrapper.__doc__ = f"[DEPRECATED: {reason}]\n\n{obj.__doc__ or ''}"
            return wrapper
        else:
            # For classes, warn on instantiation
            original_init = obj.__init__
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(full_msg, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)
            obj.__init__ = new_init
            obj.__doc__ = f"[DEPRECATED: {reason}]\n\n{obj.__doc__ or ''}"
            return obj
    return decorator


def is_deprecated(obj) -> bool:
    """Check if an object is marked as deprecated."""
    return getattr(obj, '_is_deprecated', False)


# =============================================================================
# Experimental Decorator
# =============================================================================

def experimental(reason: str = "Experimental feature - API may change"):
    """
    Mark code as experimental/unstable.
    
    Logs a warning on first use to indicate the API may change.
    
    Usage:
        @experimental("New quantum backend - interface not finalized")
        def quantum_forward(...):
            ...
    
    Args:
        reason: Description of experimental status
    """
    _warned = set()
    
    def decorator(obj):
        obj._is_experimental = True
        obj._experimental_reason = reason
        
        if callable(obj) and not isinstance(obj, type):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                if obj.__name__ not in _warned:
                    logger.warning(f"[EXPERIMENTAL] {obj.__name__}: {reason}")
                    _warned.add(obj.__name__)
                return obj(*args, **kwargs)
            wrapper._is_experimental = True
            wrapper._experimental_reason = reason
            wrapper.__doc__ = f"[EXPERIMENTAL: {reason}]\n\n{obj.__doc__ or ''}"
            return wrapper
        else:
            obj.__doc__ = f"[EXPERIMENTAL: {reason}]\n\n{obj.__doc__ or ''}"
            return obj
    return decorator


def is_experimental(obj) -> bool:
    """Check if an object is marked as experimental."""
    return getattr(obj, '_is_experimental', False)


# =============================================================================
# Internal Use Only Decorator
# =============================================================================

def internal(reason: str = "Internal implementation detail"):
    """
    Mark code as internal implementation detail.
    
    These are not part of the public API and should not be used directly.
    No runtime warning, just documentation.
    
    Usage:
        @internal("Low-level memory management")
        def _allocate_buffer(...):
            ...
    """
    def decorator(obj):
        obj._is_internal = True
        obj._internal_reason = reason
        
        if callable(obj) and not isinstance(obj, type):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                return obj(*args, **kwargs)
            wrapper._is_internal = True
            wrapper._internal_reason = reason
            wrapper.__doc__ = f"[INTERNAL: {reason}]\n\n{obj.__doc__ or ''}"
            return wrapper
        else:
            obj.__doc__ = f"[INTERNAL: {reason}]\n\n{obj.__doc__ or ''}"
            return obj
    return decorator


def is_internal(obj) -> bool:
    """Check if an object is marked as internal."""
    return getattr(obj, '_is_internal', False)


# =============================================================================
# Requires Decorator (Dependency Check)
# =============================================================================

def requires(*dependencies: str):
    """
    Decorator to specify runtime dependencies.
    
    Raises ImportError with helpful message if dependencies missing.
    
    Usage:
        @requires("torch", "transformers")
        def pytorch_compatibility_layer(...):
            ...
    """
    def decorator(obj):
        obj._requires = dependencies
        
        def check_dependencies():
            missing = []
            for dep in dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    missing.append(dep)
            if missing:
                raise ImportError(
                    f"{obj.__name__} requires: {', '.join(missing)}. "
                    f"Install with: pip install {' '.join(missing)}"
                )
        
        if callable(obj) and not isinstance(obj, type):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                check_dependencies()
                return obj(*args, **kwargs)
            wrapper._requires = dependencies
            return wrapper
        else:
            original_init = obj.__init__
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                check_dependencies()
                original_init(self, *args, **kwargs)
            obj.__init__ = new_init
            return obj
    return decorator


# =============================================================================
# Exports
# =============================================================================

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
