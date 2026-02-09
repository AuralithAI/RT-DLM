"""
Checkpoint Manager with SafeTensors Support

Provides secure model weight serialization using SafeTensors format,
replacing insecure pickle-based checkpointing.

SafeTensors advantages:
- No arbitrary code execution (secure by design)
- Memory-mapped loading (fast and memory efficient)
- Cross-framework compatible (JAX, PyTorch, TensorFlow)
- Simple and reliable

Usage:
    from src.core.checkpoint_manager import CheckpointManager
    
    # Save checkpoint
    manager = CheckpointManager("checkpoints")
    manager.save_checkpoint(params, opt_state, epoch, metrics, config)
    
    # Load checkpoint
    checkpoint = manager.load_checkpoint("checkpoints/rtdlm_agi_epoch_10.safetensors")
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# SafeTensors import with fallback
try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as save_safetensors
    from safetensors.numpy import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning(
        "SafeTensors not installed. Install with: pip install safetensors>=0.4.0"
    )

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside checkpoint tensors."""
    epoch: int
    step_count: int
    timestamp: str
    model_name: str = "rtdlm_agi"
    framework: str = "jax"
    version: str = "2.0"
    training_losses: List[float] = None
    validation_losses: List[float] = None
    metrics: Dict[str, Any] = None
    config: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "epoch": self.epoch,
            "step_count": self.step_count,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "framework": self.framework,
            "version": self.version,
            "training_losses": self.training_losses or [],
            "validation_losses": self.validation_losses or [],
            "metrics": self.metrics or {},
            "config": self.config or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            step_count=data.get("step_count", 0),
            timestamp=data.get("timestamp", ""),
            model_name=data.get("model_name", "rtdlm_agi"),
            framework=data.get("framework", "jax"),
            version=data.get("version", "2.0"),
            training_losses=data.get("training_losses"),
            validation_losses=data.get("validation_losses"),
            metrics=data.get("metrics"),
            config=data.get("config")
        )


def flatten_params(params: Dict, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    Flatten nested JAX parameter dictionary to flat dict with dot-separated keys.
    
    Args:
        params: Nested parameter dictionary from Haiku
        prefix: Key prefix for recursion
        
    Returns:
        Flat dictionary mapping "layer.sublayer.weight" -> numpy array
    """
    flat = {}
    
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            flat.update(flatten_params(value, full_key))
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            # Convert JAX arrays to numpy
            flat[full_key] = np.array(value)
        elif hasattr(value, '__array__'):
            # Handle other array-like objects
            flat[full_key] = np.array(value)
        else:
            # Skip non-array values (they go in metadata)
            logger.debug(f"Skipping non-array parameter: {full_key}")
    
    return flat


def unflatten_params(flat_params: Dict[str, np.ndarray]) -> Dict:
    """
    Unflatten dot-separated keys back to nested dictionary.
    
    Args:
        flat_params: Flat dictionary with dot-separated keys
        
    Returns:
        Nested dictionary structure matching Haiku params
    """
    nested: Dict[str, Any] = {}
    
    for full_key, value in flat_params.items():
        keys = full_key.split(".")
        current = nested
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert back to JAX array
        current[keys[-1]] = jnp.array(value)
    
    return nested


def flatten_opt_state(opt_state) -> Dict[str, np.ndarray]:
    """
    Flatten optimizer state to flat dictionary.
    
    Handles optax optimizer states which can be complex nested structures.
    """
    flat = {}
    
    def _flatten(obj, prefix="opt"):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            flat[prefix] = np.array(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}.{i}")
        elif hasattr(obj, '_fields'):  # NamedTuple
            for field in obj._fields:
                _flatten(getattr(obj, field), f"{prefix}.{field}")
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):
                    _flatten(v, f"{prefix}.{k}")
    
    _flatten(opt_state)
    return flat


def unflatten_opt_state(flat_state: Dict[str, np.ndarray], reference_state):
    """
    Unflatten optimizer state using reference structure.
    
    Args:
        flat_state: Flattened state dictionary
        reference_state: Original optimizer state for structure reference
        
    Returns:
        Reconstructed optimizer state
    """
    # This is complex because optax states are NamedTuples
    # For now, we'll use JAX tree utilities
    
    flat_values = {}
    for key, value in flat_state.items():
        if key.startswith("opt."):
            flat_values[key] = jnp.array(value)
    
    # Use JAX's tree utilities to reconstruct
    leaves, treedef = jax.tree_util.tree_flatten(reference_state)
    
    # Match flattened values to tree structure
    # This is a simplified approach - full reconstruction needs structure info
    try:
        new_leaves = []
        for i, leaf in enumerate(leaves):
            key = f"opt.{i}"
            if key in flat_values:
                new_leaves.append(flat_values[key])
            else:
                new_leaves.append(jnp.array(leaf))
        
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
    except Exception as e:
        logger.warning(f"Could not fully reconstruct opt_state: {e}")
        return None


class CheckpointManager:
    """
    Manages model checkpoints using SafeTensors format.
    
    Provides secure, efficient saving and loading of model weights,
    optimizer state, and training metadata.
    """
    
    def __init__(
        self, 
        checkpoint_dir: str = "checkpoints",
        model_name: str = "rtdlm_agi",
        keep_last_n: int = 5
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            model_name: Base name for checkpoint files
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "SafeTensors is required for secure checkpointing. "
                "Install with: pip install safetensors>=0.4.0"
            )
    
    def save_checkpoint(
        self,
        params: Dict,
        opt_state: Any,
        epoch: int,
        step_count: int = 0,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None,
        training_losses: Optional[List[float]] = None,
        validation_losses: Optional[List[float]] = None,
        extra_tensors: Optional[Dict[str, np.ndarray]] = None
    ) -> str:
        """
        Save model checkpoint using SafeTensors.
        
        Args:
            params: Model parameters (nested dict from Haiku)
            opt_state: Optimizer state from optax
            epoch: Current training epoch
            step_count: Current training step
            metrics: Training metrics dict
            config: Model configuration dict
            training_losses: List of training losses
            validation_losses: List of validation losses
            extra_tensors: Additional tensors to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()
        
        # Flatten parameters for SafeTensors
        flat_params = flatten_params(params, prefix="params")
        
        # Flatten optimizer state
        flat_opt = flatten_opt_state(opt_state)
        
        # Combine all tensors
        all_tensors = {}
        all_tensors.update(flat_params)
        all_tensors.update(flat_opt)
        
        # Add extra tensors if provided
        if extra_tensors:
            for key, tensor in extra_tensors.items():
                all_tensors[f"extra.{key}"] = np.array(tensor)
        
        # Create metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            step_count=step_count,
            timestamp=timestamp,
            model_name=self.model_name,
            training_losses=training_losses,
            validation_losses=validation_losses,
            metrics=metrics,
            config=config
        )
        
        # Save tensors using SafeTensors
        checkpoint_name = f"{self.model_name}_epoch_{epoch}.safetensors"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # SafeTensors saves tensors, metadata goes in separate JSON
        save_safetensors(all_tensors, str(checkpoint_path))
        
        # Save metadata as JSON (human-readable, no security risk)
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        epoch: Optional[int] = None,
        load_opt_state: bool = True,
        reference_opt_state: Any = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint from SafeTensors.
        
        Args:
            checkpoint_path: Path to checkpoint file (or None to load latest)
            epoch: Specific epoch to load (if checkpoint_path not provided)
            load_opt_state: Whether to load optimizer state
            reference_opt_state: Reference optimizer state for reconstruction
            
        Returns:
            Dictionary with params, opt_state, metadata
        """
        # Determine checkpoint path
        checkpoint_file: Path
        if checkpoint_path is None:
            if epoch is not None:
                checkpoint_file = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.safetensors"
            else:
                latest = self._get_latest_checkpoint()
                if latest is None:
                    raise FileNotFoundError("No checkpoint found")
                checkpoint_file = latest
        else:
            checkpoint_file = Path(checkpoint_path)
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        print(f"[INFO] Loading checkpoint: {checkpoint_file}")
        
        # Load tensors using SafeTensors
        flat_tensors = load_safetensors(str(checkpoint_file))
        
        # Load metadata from JSON
        metadata_path = checkpoint_file.with_suffix(".json")
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
        
        # Separate params and opt_state
        param_tensors = {k: v for k, v in flat_tensors.items() if k.startswith("params.")}
        opt_tensors = {k: v for k, v in flat_tensors.items() if k.startswith("opt.")}
        extra_tensors = {k[6:]: v for k, v in flat_tensors.items() if k.startswith("extra.")}
        
        # Unflatten parameters
        # Remove "params." prefix
        param_tensors_clean = {k[7:]: v for k, v in param_tensors.items()}
        params = unflatten_params(param_tensors_clean)
        
        # Unflatten optimizer state if requested
        opt_state = None
        if load_opt_state and opt_tensors and reference_opt_state is not None:
            opt_state = unflatten_opt_state(opt_tensors, reference_opt_state)
        
        result = {
            "params": params,
            "opt_state": opt_state,
            "metadata": metadata,
            "epoch": metadata.epoch if metadata else 0,
            "step_count": metadata.step_count if metadata else 0,
            "config": metadata.config if metadata else None,
            "training_losses": metadata.training_losses if metadata else [],
            "validation_losses": metadata.validation_losses if metadata else [],
            "metrics": metadata.metrics if metadata else {},
            "extra_tensors": extra_tensors
        }
        
        print(f"[INFO] Loaded checkpoint from epoch {result['epoch']}")
        
        return result
    
    def _get_latest_checkpoint(self) -> Path:
        """Get the most recent checkpoint file."""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.safetensors"))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        # Sort by epoch number
        def get_epoch(path):
            try:
                return int(path.stem.split("_epoch_")[1])
            except (IndexError, ValueError):
                return 0
        
        checkpoints.sort(key=get_epoch, reverse=True)
        return checkpoints[0]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.safetensors"))
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by epoch number
        def get_epoch(path):
            try:
                return int(path.stem.split("_epoch_")[1])
            except (IndexError, ValueError):
                return 0
        
        checkpoints.sort(key=get_epoch, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.keep_last_n:]:
            try:
                checkpoint.unlink()
                # Also remove metadata JSON
                metadata_path = checkpoint.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {checkpoint}: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints: List[Dict[str, Any]] = []
        
        for path in self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.safetensors"):
            info: Dict[str, Any] = {
                "path": str(path),
                "filename": path.name,
                "size_mb": path.stat().st_size / (1024 * 1024)
            }
            
            # Try to load metadata
            metadata_path = path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    info["metadata"] = json.load(f)
            
            checkpoints.append(info)
        
        # Sort by epoch
        def get_epoch(x: Dict[str, Any]) -> int:
            metadata = x.get("metadata", {})
            if isinstance(metadata, dict):
                return int(metadata.get("epoch", 0))
            return 0
        
        checkpoints.sort(key=get_epoch, reverse=True)
        
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a specific checkpoint without loading tensors.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata and info
        """
        path = Path(checkpoint_path)
        
        info: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else 0
        }
        
        if path.exists():
            # Get tensor names without loading values
            with safe_open(str(path), framework="numpy") as f:
                tensor_names = list(f.keys())
                info["tensor_names"] = tensor_names
                info["num_tensors"] = len(tensor_names)
            
            # Load metadata
            metadata_path = path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    info["metadata"] = json.load(f)
        
        return info


# Convenience functions for direct use
def save_model_weights(
    params: Dict,
    path: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    Simple function to save just model weights (no optimizer state).
    
    Args:
        params: Model parameters
        path: Output path (will add .safetensors extension)
        metadata: Optional metadata dict
        
    Returns:
        Path to saved file
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors required: pip install safetensors>=0.4.0")
    
    file_path = Path(path)
    if not file_path.suffix:
        file_path = file_path.with_suffix(".safetensors")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten and save
    flat_params = flatten_params(params)
    save_safetensors(flat_params, str(file_path))
    
    # Save metadata if provided
    if metadata:
        metadata_path = file_path.with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model weights saved: {file_path}")
    return str(file_path)


def load_model_weights(path: str) -> Tuple[Dict, Optional[Dict]]:
    """
    Simple function to load just model weights.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Tuple of (params dict, metadata dict or None)
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors required: pip install safetensors>=0.4.0")
    
    file_path = Path(path)
    
    # Load tensors
    flat_params = load_safetensors(str(file_path))
    params = unflatten_params(flat_params)
    
    # Load metadata if exists
    metadata = None
    metadata_path = file_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return params, metadata


# For backwards compatibility with pickle checkpoints
def convert_pickle_to_safetensors(
    pickle_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Convert a pickle checkpoint to SafeTensors format.
    
    Args:
        pickle_path: Path to pickle checkpoint
        output_path: Output path (default: same name with .safetensors)
        
    Returns:
        Path to converted checkpoint
    """
    import pickle
    
    pickle_path_obj = Path(pickle_path)
    
    if output_path is None:
        output_path_obj = pickle_path_obj.with_suffix(".safetensors")
    else:
        output_path_obj = Path(output_path)
    
    # Load pickle checkpoint (with warning about security)
    logger.warning(f"Loading pickle file (potentially unsafe): {pickle_path_obj}")
    with open(pickle_path_obj, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract components
    params = checkpoint.get("params", {})
    opt_state = checkpoint.get("opt_state")
    
    # Create manager and save
    manager = CheckpointManager(
        checkpoint_dir=str(output_path_obj.parent),
        model_name=output_path_obj.stem.split("_epoch_")[0] if "_epoch_" in output_path_obj.stem else "model"
    )
    
    saved_path = manager.save_checkpoint(
        params=params,
        opt_state=opt_state,
        epoch=checkpoint.get("epoch", 0),
        step_count=checkpoint.get("step_count", 0),
        metrics=checkpoint.get("metrics"),
        config=checkpoint.get("config"),
        training_losses=checkpoint.get("training_losses"),
        validation_losses=checkpoint.get("validation_losses")
    )
    
    logger.info(f"Converted {pickle_path_obj} -> {saved_path}")
    return saved_path
