"""Safety/alignment sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """Safety, ethics and alignment settings."""

    ethics_enabled: bool = True
    ethics_weight: float = 0.1
    bias_detection_enabled: bool = True
    fairness_constraints: bool = True
    alignment_training: bool = True
    value_learning: bool = True
    interpretability: bool = True
    safety_constraints: bool = True

    def validate(self) -> None:
        if self.ethics_weight < 0:
            raise ValueError("ethics_weight must be non-negative")
