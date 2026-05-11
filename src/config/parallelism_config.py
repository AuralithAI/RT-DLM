"""Parallelism sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass


@dataclass
class ParallelismConfig:
    """Distributed/parallel execution settings."""

    distributed_training: bool = False
    num_devices: int = 1
    data_parallel: bool = True
    model_parallel: bool = False
    gradient_accumulation_steps: int = 1

    def validate(self) -> None:
        if self.num_devices < 1:
            raise ValueError("num_devices must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
