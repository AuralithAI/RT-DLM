"""Goodput and per-expert utilization metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class GoodputTracker:
    """Rolling goodput estimator: useful_step_time / wall_step_time."""

    window: int = 100
    _useful: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _wall: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        self._useful = deque(maxlen=self.window)
        self._wall = deque(maxlen=self.window)

    def record(self, useful_seconds: float, wall_seconds: float) -> None:
        if wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")
        if useful_seconds < 0 or useful_seconds > wall_seconds:
            raise ValueError("useful_seconds must be in [0, wall_seconds]")
        self._useful.append(float(useful_seconds))
        self._wall.append(float(wall_seconds))

    def goodput(self) -> float:
        """Fraction of wall time spent doing useful work."""
        if not self._wall:
            return 0.0
        return float(sum(self._useful) / sum(self._wall))

    def reset(self) -> None:
        self._useful.clear()
        self._wall.clear()

    def summary(self) -> Dict[str, float]:
        return {
            "goodput": self.goodput(),
            "samples": float(len(self._wall)),
            "wall_seconds_total": float(sum(self._wall)),
            "useful_seconds_total": float(sum(self._useful)),
        }


@dataclass
class ExpertUtilizationTracker:
    """Per-expert utilization stats across MoE forward passes."""

    num_experts: int
    _counts: np.ndarray = field(init=False)
    _total_tokens: int = 0

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self._counts = np.zeros(self.num_experts, dtype=np.int64)

    def update(self, top_k_indices: np.ndarray) -> None:
        """Update from int array of expert indices selected per token."""
        flat = np.asarray(top_k_indices).reshape(-1)
        if flat.size == 0:
            return
        self._counts += np.bincount(flat, minlength=self.num_experts)[: self.num_experts]
        self._total_tokens += int(flat.size)

    def utilization(self) -> np.ndarray:
        """Per-expert selection probability over all tokens seen."""
        if self._total_tokens == 0:
            return np.zeros(self.num_experts, dtype=np.float32)
        return (self._counts.astype(np.float32) / float(self._total_tokens))

    def entropy(self) -> float:
        """Shannon entropy of expert utilization (nats); higher = more balanced."""
        p = self.utilization()
        nz = p[p > 0]
        if nz.size == 0:
            return 0.0
        return float(-(nz * np.log(nz)).sum())

    def imbalance(self) -> float:
        """Max - min utilization. 0 = perfectly balanced."""
        p = self.utilization()
        return float(p.max() - p.min())

    def reset(self) -> None:
        self._counts[:] = 0
        self._total_tokens = 0

    def summary(self) -> Dict[str, float]:
        p = self.utilization()
        return {
            "expert_entropy": self.entropy(),
            "expert_imbalance": self.imbalance(),
            "expert_max_util": float(p.max()) if p.size else 0.0,
            "expert_min_util": float(p.min()) if p.size else 0.0,
            "total_tokens": float(self._total_tokens),
        }
