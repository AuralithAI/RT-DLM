"""Architecture sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ArchitectureConfig:
    """Model architecture hyperparameters."""

    d_model: int = 384
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    max_seq_length: int = 4096
    base_d_model: int = 256
    moe_experts: int = 8
    moe_top_k: int = 2
    attention_type: str = "standard"
    num_kv_heads: Optional[int] = None
    position_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    sliding_window_size: int = 512
    use_flash_attention: bool = False

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.base_d_model <= 0:
            raise ValueError("base_d_model must be positive")
        if self.position_encoding not in {"rope", "learned", "alibi", "none"}:
            raise ValueError(f"unknown position_encoding: {self.position_encoding}")
