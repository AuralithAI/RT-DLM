"""Precision sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass

_VALID_DTYPES = ("float32", "bfloat16", "float16")


@dataclass
class PrecisionConfig:
    """Mixed precision and dtype policy."""

    mixed_precision: bool = False
    precision_dtype: str = "float32"
    compute_dtype: str = "float32"
    gradient_checkpointing: bool = False
    checkpoint_every_n_layers: int = 2

    def validate(self) -> None:
        if self.precision_dtype not in _VALID_DTYPES:
            raise ValueError(f"precision_dtype must be one of {_VALID_DTYPES}")
        if self.compute_dtype not in _VALID_DTYPES:
            raise ValueError(f"compute_dtype must be one of {_VALID_DTYPES}")
        if self.gradient_checkpointing and self.checkpoint_every_n_layers <= 0:
            raise ValueError("checkpoint_every_n_layers must be positive")
