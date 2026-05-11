"""Training sub-configuration (composed into AGIConfig)."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 5000
    decay_steps: int = 200000
    init_lr: float = 2e-6
    end_lr: float = 2e-6
    weight_decay: float = 1e-3
    clip_norm: float = 0.5
    label_smoothing: float = 0.1
    eval_interval: int = 25
    moe_z_loss_weight: float = 1e-4
    moe_router_z_loss_weight: float = 1e-3
    use_mup: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if self.moe_z_loss_weight < 0:
            raise ValueError("moe_z_loss_weight must be non-negative")
        if self.moe_router_z_loss_weight < 0:
            raise ValueError("moe_router_z_loss_weight must be non-negative")
