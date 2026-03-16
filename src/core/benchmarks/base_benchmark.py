"""
Abstract Benchmark Interface and Shared Data Structures

Provides the base class for all benchmark evaluators and common
result/sample containers used across benchmarks.

Design:
    - BenchmarkBase: abstract class with load_data(), evaluate(), format_result()
    - BenchmarkSample: uniform container for a single evaluation sample
    - BenchmarkResult: aggregated results with per-category breakdown
    - BenchmarkSuite: orchestrates multiple benchmarks in one pass

All benchmarks follow the same lifecycle:
    1. load_data()   → fetch/parse dataset into BenchmarkSample list
    2. evaluate()    → run model on all samples, return BenchmarkResult
    3. format_result()→ human-readable summary string
"""

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BenchmarkSample:
    """Single evaluation sample in uniform format.

    All benchmarks convert their raw data into this container so the
    evaluation loop, scoring, and logging are benchmark-agnostic.

    Attributes:
        sample_id: Unique identifier (dataset row id or index)
        prompt: Full prompt text sent to the model
        choices: Answer choices for multiple-choice (None for open-ended)
        correct_answer: Ground-truth answer (index for MC, string for open-ended)
        category: Optional sub-category for breakdown (e.g., GPQA domain)
        metadata: Extra benchmark-specific data (difficulty, source, etc.)
    """

    sample_id: str
    prompt: str
    choices: Optional[List[str]] = None
    correct_answer: Union[str, int, None] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark evaluation result.

    Attributes:
        benchmark_name: Short identifier (e.g., "gpqa_diamond")
        accuracy: Overall accuracy (correct / total)
        num_correct: Number of correct predictions
        num_total: Total number of samples evaluated
        category_scores: Per-category accuracy breakdown
        total_time_sec: Wall-clock time for the entire benchmark
        samples_per_sec: Throughput
        predictions: List of (sample_id, predicted, correct, is_correct) tuples
        config: Evaluation config snapshot
    """

    benchmark_name: str
    accuracy: float
    num_correct: int
    num_total: int
    category_scores: Dict[str, float] = field(default_factory=dict)
    total_time_sec: float = 0.0
    samples_per_sec: float = 0.0
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "benchmark": self.benchmark_name,
            "accuracy": round(self.accuracy, 4),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "total_time_sec": round(self.total_time_sec, 2),
            "samples_per_sec": round(self.samples_per_sec, 2),
            "config": self.config,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark result to {path}")


# =============================================================================
# Abstract Benchmark Base
# =============================================================================


class BenchmarkBase(abc.ABC):
    """Abstract base class for benchmark evaluators.

    Subclasses must implement:
        - name (property): short benchmark identifier
        - load_data(): returns list of BenchmarkSample
        - score_sample(): compare prediction against ground truth

    The ``evaluate()`` method is provided and calls the model on each
    sample, then aggregates scores.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Local directory for cached data (None → download)
            split: Dataset split to evaluate on
            max_samples: Cap number of samples (useful for CI)
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self._samples: Optional[List[BenchmarkSample]] = None

    # ---- Abstract interface ------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short benchmark name (e.g., 'gpqa_diamond')."""
        ...

    @abc.abstractmethod
    def load_data(self) -> List[BenchmarkSample]:
        """Load and parse benchmark data into BenchmarkSample list.

        Implementations should handle:
        - Downloading from HuggingFace datasets or other sources
        - Caching to self.data_dir
        - Applying self.max_samples cap
        - Setting self.seed for reproducibility
        """
        ...

    def score_sample(
        self,
        sample: BenchmarkSample,
        prediction: Any,
    ) -> bool:
        """Score a single prediction against ground truth.

        Default implementation handles both MC (int index) and
        open-ended (string match). Subclasses can override for
        custom scoring (e.g., exact numeric match for AIME).

        Args:
            sample: The benchmark sample
            prediction: Model's prediction (int index or string)

        Returns:
            True if prediction matches ground truth
        """
        if sample.choices is not None:
            # Multiple-choice: compare index
            if isinstance(prediction, str):
                # Map letter to index: A→0, B→1, C→2, D→3
                letter_map = {c: i for i, c in enumerate("ABCDEFGHIJ")}
                prediction = letter_map.get(prediction.upper().strip(), -1)
            return prediction == sample.correct_answer
        else:
            # Open-ended: exact string match (normalized)
            pred_str = str(prediction).strip().lower()
            correct_str = str(sample.correct_answer).strip().lower()
            return pred_str == correct_str

    # ---- Provided evaluation loop ------------------------------------------

    @property
    def samples(self) -> List[BenchmarkSample]:
        """Lazy-loaded samples."""
        if self._samples is None:
            self._samples = self.load_data()
        return self._samples

    def evaluate(
        self,
        model_fn: Callable[..., Any],
        params: Any = None,
        state: Any = None,
        rng: Optional[jnp.ndarray] = None,
        batch_size: int = 1,
        think_budget: Optional[int] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run model on all samples and aggregate results.

        Args:
            model_fn: Callable that takes (prompt_str, **kwargs) and
                       returns a prediction (int for MC, str for open-ended).
                       Signature: model_fn(prompt, params, state, rng, **kwargs)
            params: Model parameters (Haiku params dict)
            state: Model state (Haiku state dict, may be None)
            rng: JAX PRNGKey
            batch_size: Number of samples to process at once
            think_budget: Optional think-budget override
            verbose: Print per-sample results

        Returns:
            BenchmarkResult with accuracy and breakdown
        """
        samples = self.samples
        if not samples:
            logger.warning(f"{self.name}: No samples loaded, returning empty result")
            return BenchmarkResult(
                benchmark_name=self.name,
                accuracy=0.0,
                num_correct=0,
                num_total=0,
            )

        rng = rng if rng is not None else jax.random.PRNGKey(self.seed)

        correct = 0
        total = 0
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        predictions_log: List[Dict[str, Any]] = []

        start_time = time.time()

        for sample in samples:
            rng, sub_rng = jax.random.split(rng)

            # Build kwargs for model_fn
            kwargs: Dict[str, Any] = {}
            if params is not None:
                kwargs["params"] = params
            if state is not None:
                kwargs["state"] = state
            if think_budget is not None:
                kwargs["think_budget"] = think_budget

            try:
                prediction = model_fn(
                    sample.prompt,
                    rng=sub_rng,
                    choices=sample.choices,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.sample_id}: {e}")
                prediction = None

            is_correct = self.score_sample(sample, prediction)
            if is_correct:
                correct += 1

            total += 1

            # Category tracking
            cat = sample.category or "default"
            cat_correct[cat] = cat_correct.get(cat, 0) + (1 if is_correct else 0)
            cat_total[cat] = cat_total.get(cat, 0) + 1

            predictions_log.append(
                {
                    "sample_id": sample.sample_id,
                    "predicted": str(prediction),
                    "correct_answer": str(sample.correct_answer),
                    "is_correct": is_correct,
                    "category": cat,
                }
            )

            if verbose:
                status = "✓" if is_correct else "✗"
                logger.info(
                    f"  [{status}] {sample.sample_id}: "
                    f"pred={prediction}, truth={sample.correct_answer}"
                )

        elapsed = time.time() - start_time

        # Per-category accuracy
        category_scores = {
            cat: cat_correct[cat] / cat_total[cat] for cat in cat_total if cat_total[cat] > 0
        }

        result = BenchmarkResult(
            benchmark_name=self.name,
            accuracy=correct / total if total > 0 else 0.0,
            num_correct=correct,
            num_total=total,
            category_scores=category_scores,
            total_time_sec=elapsed,
            samples_per_sec=total / elapsed if elapsed > 0 else 0.0,
            predictions=predictions_log,
            config={
                "split": self.split,
                "max_samples": self.max_samples,
                "seed": self.seed,
                "think_budget": think_budget,
                "batch_size": batch_size,
            },
        )

        logger.info(
            f"{self.name}: {result.accuracy:.2%} "
            f"({result.num_correct}/{result.num_total}) "
            f"in {result.total_time_sec:.1f}s"
        )

        return result

    def format_result(self, result: BenchmarkResult) -> str:
        """Format result as human-readable string."""
        lines = [
            f"\n{'='*60}",
            f"  {result.benchmark_name.upper()}",
            f"{'='*60}",
            f"  Accuracy: {result.accuracy:.2%} ({result.num_correct}/{result.num_total})",
            f"  Time:     {result.total_time_sec:.1f}s ({result.samples_per_sec:.1f} samples/s)",
        ]
        if result.category_scores:
            lines.append("  Categories:")
            for cat, score in sorted(result.category_scores.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat}: {score:.2%}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# =============================================================================
# Benchmark Suite (multi-benchmark orchestrator)
# =============================================================================


class BenchmarkSuite:
    """Orchestrate evaluation across multiple benchmarks.

    Usage:
        suite = BenchmarkSuite()
        suite.add(GPQABenchmark(max_samples=100))
        suite.add(AIMEBenchmark(max_samples=50))
        results = suite.run(model_fn, params, state, rng)
    """

    def __init__(self):
        self.benchmarks: List[BenchmarkBase] = []
        self._results: Dict[str, BenchmarkResult] = {}

    def add(self, benchmark: BenchmarkBase) -> "BenchmarkSuite":
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
        return self

    def run(
        self,
        model_fn: Callable[..., Any],
        params: Any = None,
        state: Any = None,
        rng: Optional[jnp.ndarray] = None,
        think_budget: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return results.

        Args:
            model_fn: Callable for inference (see BenchmarkBase.evaluate)
            params: Model parameters
            state: Model state
            rng: JAX PRNGKey
            think_budget: Optional think-budget override
            verbose: Print per-sample results

        Returns:
            Dict mapping benchmark name → BenchmarkResult
        """
        rng = rng if rng is not None else jax.random.PRNGKey(42)
        results: Dict[str, BenchmarkResult] = {}

        for bench in self.benchmarks:
            rng, bench_rng = jax.random.split(rng)
            logger.info(f"Running benchmark: {bench.name}")
            try:
                result = bench.evaluate(
                    model_fn=model_fn,
                    params=params,
                    state=state,
                    rng=bench_rng,
                    think_budget=think_budget,
                    verbose=verbose,
                )
                results[bench.name] = result
            except Exception as e:
                logger.error(f"Benchmark {bench.name} failed: {e}")
                results[bench.name] = BenchmarkResult(
                    benchmark_name=bench.name,
                    accuracy=0.0,
                    num_correct=0,
                    num_total=0,
                )

        self._results = results
        return results

    def summary_table(self) -> str:
        """Generate a summary table of all results."""
        if not self._results:
            return "No results available. Run benchmarks first."

        lines = [
            "\n" + "=" * 70,
            "  BENCHMARK EVALUATION SUMMARY",
            "=" * 70,
            f"  {'Benchmark':<25s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s} {'Time':>8s}",
            "-" * 70,
        ]
        for name, result in self._results.items():
            lines.append(
                f"  {name:<25s} {result.accuracy:>9.2%} "
                f"{result.num_correct:>10d} {result.num_total:>8d} "
                f"{result.total_time_sec:>7.1f}s"
            )
        lines.append("=" * 70 + "\n")
        return "\n".join(lines)

    def save_all(self, output_dir: Union[str, Path]) -> None:
        """Save all results as JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, result in self._results.items():
            result.save_json(output_dir / f"{name}_result.json")
        logger.info(f"All results saved to {output_dir}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all results to a single dict."""
        return {name: result.to_dict() for name, result in self._results.items()}
