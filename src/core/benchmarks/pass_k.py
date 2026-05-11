"""pass@k / maj@k aggregation utilities for generation benchmarks."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence


def pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021 (HumanEval)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_correct < 0 or num_correct > num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    if num_samples - num_correct < k:
        return 1.0
    prob = 1.0
    for i in range(num_samples - num_correct + 1, num_samples + 1):
        prob *= 1.0 - k / i
    return 1.0 - prob


def maj_at_k(predictions: Sequence, gold) -> float:
    """maj@k: 1 if majority-voted prediction equals gold else 0."""
    if not predictions:
        raise ValueError("predictions must be non-empty")
    counts = Counter(predictions)
    top, _ = counts.most_common(1)[0]
    return 1.0 if top == gold else 0.0


def best_of_n(
    candidates: Sequence,
    score_fn: Callable,
    gold,
    is_correct: Optional[Callable] = None,
) -> float:
    """Return 1.0 if the highest-scoring candidate matches gold."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    scored = sorted(enumerate(candidates), key=lambda kv: score_fn(kv[1]), reverse=True)
    best = scored[0][1]
    if is_correct is not None:
        return 1.0 if is_correct(best, gold) else 0.0
    return 1.0 if best == gold else 0.0


def aggregate_pass_at_k(per_problem_results: Iterable[tuple[int, int]], ks: Sequence[int]) -> dict[str, float]:
    """Aggregate pass@k across problems given (num_correct, num_samples) per problem."""
    per_problem = list(per_problem_results)
    if not per_problem:
        raise ValueError("per_problem_results must be non-empty")
    out: dict[str, float] = {}
    for k in ks:
        scores = [pass_at_k(c, n, k) for c, n in per_problem]
        out[f"pass@{k}"] = float(sum(scores) / len(scores))
    return out
