"""
Tests for Benchmark Harness + MLflow Integration

Covers:
    - BenchmarkBase / BenchmarkSample / BenchmarkResult / BenchmarkSuite
    - GPQABenchmark (synthetic fallback)
    - AIMEBenchmark (synthetic fallback + custom scoring)
    - SWEBenchBenchmark (synthetic fallback + lightweight scoring)
    - LiveCodeBenchmark (synthetic fallback + code structure scoring)
    - MLflowTracker (with/without MLflow installed)
    - AGIConfig benchmark/MLflow flags
    - run_eval CLI parser
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import jax
import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.agi_config import AGIConfig
from core.benchmarks.base_benchmark import (
    BenchmarkBase,
    BenchmarkResult,
    BenchmarkSample,
    BenchmarkSuite,
)
from core.benchmarks.gpqa_benchmark import GPQABenchmark
from core.benchmarks.aime_benchmark import AIMEBenchmark
from core.benchmarks.swe_bench import SWEBenchBenchmark
from core.benchmarks.livecode_bench import LiveCodeBenchmark
from core.benchmarks.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def dummy_model_fn():
    """Model fn that returns random MC answer or string."""

    def model_fn(prompt, rng, choices=None, **kwargs):
        if choices is not None:
            # Return random choice index
            return int(jax.random.randint(rng, (), 0, len(choices)))
        else:
            # Return random integer string
            return str(int(jax.random.randint(rng, (), 0, 1000)))

    return model_fn


@pytest.fixture
def correct_model_fn():
    """Model fn that always returns the correct answer (for testing scoring)."""

    def model_fn(prompt, rng, choices=None, **kwargs):
        # Extract correct answer from prompt metadata
        # For MC: always return 0 (tests set correct=0)
        if choices is not None:
            return 0
        return "42"

    return model_fn


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# =============================================================================
# BenchmarkSample Tests
# =============================================================================


class TestBenchmarkSample:
    """Tests for BenchmarkSample dataclass."""

    def test_create_mc_sample(self):
        s = BenchmarkSample(
            sample_id="q1",
            prompt="What is 2+2?",
            choices=["3", "4", "5", "6"],
            correct_answer=1,
            category="arithmetic",
        )
        assert s.sample_id == "q1"
        assert len(s.choices) == 4
        assert s.correct_answer == 1
        assert s.category == "arithmetic"

    def test_create_open_ended_sample(self):
        s = BenchmarkSample(
            sample_id="q2",
            prompt="What is the answer to life?",
            correct_answer="42",
        )
        assert s.choices is None
        assert s.correct_answer == "42"

    def test_metadata_defaults(self):
        s = BenchmarkSample(sample_id="q3", prompt="test")
        assert s.metadata == {}
        assert s.category is None

    def test_metadata_custom(self):
        s = BenchmarkSample(
            sample_id="q4",
            prompt="test",
            metadata={"difficulty": "hard", "source": "gpqa"},
        )
        assert s.metadata["difficulty"] == "hard"


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self):
        r = BenchmarkResult(
            benchmark_name="test_bench",
            accuracy=0.75,
            num_correct=15,
            num_total=20,
        )
        assert r.accuracy == 0.75
        assert r.num_correct == 15
        assert r.num_total == 20

    def test_to_dict(self):
        r = BenchmarkResult(
            benchmark_name="test_bench",
            accuracy=0.7523,
            num_correct=15,
            num_total=20,
            category_scores={"physics": 0.8, "chemistry": 0.7},
            total_time_sec=12.345,
            samples_per_sec=1.62,
        )
        d = r.to_dict()
        assert d["benchmark"] == "test_bench"
        assert d["accuracy"] == 0.7523
        assert d["category_scores"]["physics"] == 0.8
        assert d["total_time_sec"] == 12.35  # Rounded

    def test_save_json(self, tmp_dir):
        r = BenchmarkResult(
            benchmark_name="test_bench",
            accuracy=0.85,
            num_correct=17,
            num_total=20,
        )
        path = Path(tmp_dir) / "result.json"
        r.save_json(path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["benchmark"] == "test_bench"
        assert data["accuracy"] == 0.85

    def test_save_json_creates_dirs(self, tmp_dir):
        r = BenchmarkResult(
            benchmark_name="x",
            accuracy=0.5,
            num_correct=5,
            num_total=10,
        )
        path = Path(tmp_dir) / "nested" / "dir" / "result.json"
        r.save_json(path)
        assert path.exists()

    def test_default_fields(self):
        r = BenchmarkResult(
            benchmark_name="x",
            accuracy=0.0,
            num_correct=0,
            num_total=0,
        )
        assert r.category_scores == {}
        assert r.predictions == []
        assert r.config == {}
        assert r.total_time_sec == 0.0


# =============================================================================
# BenchmarkBase Tests
# =============================================================================


class _DummyBenchmark(BenchmarkBase):
    """Concrete subclass for testing abstract base."""

    @property
    def name(self):
        return "dummy"

    def load_data(self):
        samples = [
            BenchmarkSample(
                sample_id=f"d_{i}",
                prompt=f"Question {i}",
                choices=["A", "B", "C", "D"],
                correct_answer=i % 4,
                category="cat_a" if i < 5 else "cat_b",
            )
            for i in range(10)
        ]
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples


class TestBenchmarkBase:
    """Tests for abstract BenchmarkBase interface."""

    def test_name(self):
        b = _DummyBenchmark()
        assert b.name == "dummy"

    def test_load_data(self):
        b = _DummyBenchmark()
        samples = b.load_data()
        assert len(samples) == 10
        assert all(isinstance(s, BenchmarkSample) for s in samples)

    def test_max_samples(self):
        b = _DummyBenchmark(max_samples=3)
        samples = b.load_data()
        assert len(samples) == 3

    def test_lazy_samples(self):
        b = _DummyBenchmark(max_samples=5)
        assert b._samples is None
        _ = b.samples
        assert b._samples is not None
        assert len(b.samples) == 5

    def test_evaluate(self, dummy_model_fn, rng):
        b = _DummyBenchmark(max_samples=5)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "dummy"
        assert result.num_total == 5
        assert 0 <= result.accuracy <= 1.0
        assert result.total_time_sec > 0
        assert result.samples_per_sec > 0

    def test_evaluate_empty(self, dummy_model_fn, rng):
        b = _DummyBenchmark(max_samples=0)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert result.num_total == 0
        assert result.accuracy == 0.0

    def test_evaluate_with_categories(self, dummy_model_fn, rng):
        b = _DummyBenchmark(max_samples=10)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert "cat_a" in result.category_scores
        assert "cat_b" in result.category_scores

    def test_format_result(self):
        b = _DummyBenchmark()
        result = BenchmarkResult(
            benchmark_name="dummy",
            accuracy=0.8,
            num_correct=8,
            num_total=10,
            category_scores={"cat_a": 0.9, "cat_b": 0.7},
            total_time_sec=5.0,
            samples_per_sec=2.0,
        )
        text = b.format_result(result)
        assert "DUMMY" in text
        assert "80.00%" in text
        assert "cat_a" in text

    def test_score_sample_mc(self):
        b = _DummyBenchmark()
        sample = BenchmarkSample(
            sample_id="t1",
            prompt="test",
            choices=["A", "B", "C", "D"],
            correct_answer=2,
        )
        assert b.score_sample(sample, 2) is True
        assert b.score_sample(sample, 0) is False
        assert b.score_sample(sample, "C") is True  # Letter mapping
        assert b.score_sample(sample, "A") is False

    def test_score_sample_open_ended(self):
        b = _DummyBenchmark()
        sample = BenchmarkSample(
            sample_id="t2",
            prompt="test",
            correct_answer="42",
        )
        assert b.score_sample(sample, "42") is True
        assert b.score_sample(sample, " 42 ") is True
        assert b.score_sample(sample, "43") is False

    def test_evaluate_error_handling(self, rng):
        def failing_model(prompt, rng, choices=None, **kwargs):
            raise RuntimeError("Model error")

        b = _DummyBenchmark(max_samples=3)
        result = b.evaluate(failing_model, rng=rng)
        assert result.num_total == 3
        assert result.num_correct == 0

    def test_predictions_logged(self, dummy_model_fn, rng):
        b = _DummyBenchmark(max_samples=3)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert len(result.predictions) == 3
        for pred in result.predictions:
            assert "sample_id" in pred
            assert "predicted" in pred
            assert "is_correct" in pred


# =============================================================================
# BenchmarkSuite Tests
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite orchestrator."""

    def test_add_benchmarks(self):
        suite = BenchmarkSuite()
        b1 = _DummyBenchmark(max_samples=5)
        b2 = _DummyBenchmark(max_samples=3)
        suite.add(b1).add(b2)
        assert len(suite.benchmarks) == 2

    def test_run_all(self, dummy_model_fn, rng):
        suite = BenchmarkSuite()
        suite.add(_DummyBenchmark(max_samples=5))
        results = suite.run(dummy_model_fn, rng=rng)
        assert "dummy" in results
        assert results["dummy"].num_total == 5

    def test_summary_table(self, dummy_model_fn, rng):
        suite = BenchmarkSuite()
        suite.add(_DummyBenchmark(max_samples=5))
        suite.run(dummy_model_fn, rng=rng)
        table = suite.summary_table()
        assert "dummy" in table
        assert "BENCHMARK EVALUATION SUMMARY" in table

    def test_summary_table_no_results(self):
        suite = BenchmarkSuite()
        table = suite.summary_table()
        assert "No results" in table

    def test_save_all(self, dummy_model_fn, rng, tmp_dir):
        suite = BenchmarkSuite()
        suite.add(_DummyBenchmark(max_samples=5))
        suite.run(dummy_model_fn, rng=rng)
        suite.save_all(tmp_dir)
        files = list(Path(tmp_dir).glob("*.json"))
        assert len(files) == 1

    def test_to_dict(self, dummy_model_fn, rng):
        suite = BenchmarkSuite()
        suite.add(_DummyBenchmark(max_samples=5))
        suite.run(dummy_model_fn, rng=rng)
        d = suite.to_dict()
        assert "dummy" in d
        assert "accuracy" in d["dummy"]

    def test_failed_benchmark(self, rng):
        """Suite should handle benchmark failures gracefully."""

        def bad_model(prompt, rng, choices=None, **kwargs):
            raise RuntimeError("boom")

        suite = BenchmarkSuite()
        suite.add(_DummyBenchmark(max_samples=3))
        results = suite.run(bad_model, rng=rng)
        # Should still get a result (with 0 accuracy since all predictions fail)
        assert "dummy" in results


# =============================================================================
# GPQA Benchmark Tests
# =============================================================================


class TestGPQABenchmark:
    """Tests for GPQA Diamond benchmark."""

    def test_name(self):
        b = GPQABenchmark()
        assert b.name == "gpqa_diamond"

    def test_synthetic_fallback(self):
        b = GPQABenchmark(max_samples=10)
        samples = b.load_data()
        assert len(samples) > 0
        assert all(isinstance(s, BenchmarkSample) for s in samples)

    def test_synthetic_has_choices(self):
        b = GPQABenchmark(max_samples=5)
        samples = b.load_data()
        for s in samples:
            assert s.choices is not None
            assert len(s.choices) == 4
            assert isinstance(s.correct_answer, int)
            assert 0 <= s.correct_answer <= 3

    def test_synthetic_has_prompt_format(self):
        b = GPQABenchmark(max_samples=3)
        samples = b.load_data()
        for s in samples:
            assert "Answer the following question" in s.prompt
            assert "(A)" in s.prompt
            assert "(B)" in s.prompt
            assert "Answer:" in s.prompt

    def test_synthetic_has_categories(self):
        b = GPQABenchmark(max_samples=10)
        samples = b.load_data()
        categories = {s.category for s in samples}
        assert len(categories) > 1  # Multiple domains

    def test_evaluate_runs(self, dummy_model_fn, rng):
        b = GPQABenchmark(max_samples=5)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert result.benchmark_name == "gpqa_diamond"
        assert result.num_total == 5

    def test_no_shuffle(self):
        b = GPQABenchmark(max_samples=5, shuffle_choices=False)
        samples = b.load_data()
        # Without shuffle, correct answer should be at index 0
        for s in samples:
            assert s.correct_answer == 0

    def test_seed_reproducibility(self):
        b1 = GPQABenchmark(max_samples=5, seed=123)
        b2 = GPQABenchmark(max_samples=5, seed=123)
        s1 = b1.load_data()
        s2 = b2.load_data()
        for a, b_sample in zip(s1, s2):
            assert a.prompt == b_sample.prompt
            assert a.correct_answer == b_sample.correct_answer


# =============================================================================
# AIME Benchmark Tests
# =============================================================================


class TestAIMEBenchmark:
    """Tests for AIME benchmark."""

    def test_name_default(self):
        b = AIMEBenchmark()
        assert b.name == "aime"

    def test_name_with_year(self):
        b = AIMEBenchmark(year=2024)
        assert b.name == "aime_2024"

    def test_synthetic_fallback(self):
        b = AIMEBenchmark(max_samples=10)
        samples = b.load_data()
        assert len(samples) > 0
        for s in samples:
            assert s.choices is None  # Open-ended
            assert s.correct_answer is not None

    def test_synthetic_prompt_format(self):
        b = AIMEBenchmark(max_samples=3)
        samples = b.load_data()
        for s in samples:
            assert "AIME" in s.prompt
            assert "integer" in s.prompt.lower()

    def test_score_exact_match(self):
        b = AIMEBenchmark()
        sample = BenchmarkSample(
            sample_id="a1",
            prompt="test",
            correct_answer="42",
        )
        assert b.score_sample(sample, "42") is True
        assert b.score_sample(sample, "42.0") is True  # Float → int
        assert b.score_sample(sample, " 42 ") is True
        assert b.score_sample(sample, "43") is False

    def test_score_digit_extraction(self):
        b = AIMEBenchmark()
        sample = BenchmarkSample(
            sample_id="a2",
            prompt="test",
            correct_answer="100",
        )
        assert b.score_sample(sample, "The answer is 100.") is True

    def test_score_invalid_prediction(self):
        b = AIMEBenchmark()
        sample = BenchmarkSample(
            sample_id="a3",
            prompt="test",
            correct_answer="42",
        )
        assert b.score_sample(sample, "no digits here!") is False
        assert b.score_sample(sample, None) is False

    def test_evaluate_runs(self, dummy_model_fn, rng):
        b = AIMEBenchmark(max_samples=5)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert result.benchmark_name == "aime"
        assert result.num_total == 5


# =============================================================================
# SWE-Bench Tests
# =============================================================================


class TestSWEBenchBenchmark:
    """Tests for SWE-Bench Verified benchmark."""

    def test_name(self):
        b = SWEBenchBenchmark()
        assert b.name == "swe_bench_verified"

    def test_synthetic_fallback(self):
        b = SWEBenchBenchmark(max_samples=5)
        samples = b.load_data()
        assert len(samples) > 0
        for s in samples:
            assert s.choices is None
            assert "patch" in s.prompt.lower() or "issue" in s.prompt.lower()

    def test_synthetic_has_metadata(self):
        b = SWEBenchBenchmark(max_samples=3)
        samples = b.load_data()
        for s in samples:
            assert "gold_patch" in s.metadata
            assert "target_files" in s.metadata
            assert "repo" in s.metadata

    def test_lightweight_score_patch(self):
        b = SWEBenchBenchmark(lightweight=True)
        sample = BenchmarkSample(
            sample_id="swe1",
            prompt="Fix the bug",
            correct_answer="patch",
            metadata={
                "gold_patch": (
                    "diff --git a/foo.py b/foo.py\n"
                    "--- a/foo.py\n"
                    "+++ b/foo.py\n"
                    "@@ -10,3 +10,5 @@\n"
                    "+    return fixed_value\n"
                ),
                "target_files": ["foo.py"],
            },
        )

        # Good prediction with patch format
        good_pred = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -10,3 +10,5 @@\n"
            "+    return fixed_value\n"
        )
        assert b.score_sample(sample, good_pred) is True

    def test_lightweight_score_no_patch(self):
        b = SWEBenchBenchmark(lightweight=True)
        sample = BenchmarkSample(
            sample_id="swe2",
            prompt="Fix the bug",
            metadata={
                "gold_patch": "diff --git a/foo.py\n+fix",
                "target_files": ["foo.py"],
            },
        )
        assert b.score_sample(sample, "Just some text without a patch") is False

    def test_score_none_prediction(self):
        b = SWEBenchBenchmark()
        sample = BenchmarkSample(sample_id="swe3", prompt="test")
        assert b.score_sample(sample, None) is False
        assert b.score_sample(sample, "") is False

    def test_extract_files_from_patch(self):
        patch = "diff --git a/src/foo.py b/src/foo.py\n" "--- a/src/foo.py\n" "+++ b/src/foo.py\n"
        files = SWEBenchBenchmark._extract_files_from_patch(patch)
        assert "src/foo.py" in files

    def test_evaluate_runs(self, dummy_model_fn, rng):
        b = SWEBenchBenchmark(max_samples=3)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert result.benchmark_name == "swe_bench_verified"
        assert result.num_total == 3


# =============================================================================
# LiveCodeBench Tests
# =============================================================================


class TestLiveCodeBenchmark:
    """Tests for LiveCodeBench benchmark."""

    def test_name(self):
        b = LiveCodeBenchmark()
        assert b.name == "livecode_bench"

    def test_synthetic_fallback(self):
        b = LiveCodeBenchmark(max_samples=6)
        samples = b.load_data()
        assert len(samples) > 0
        for s in samples:
            assert s.choices is None
            assert "problem" in s.prompt.lower() or "solve" in s.prompt.lower()

    def test_synthetic_has_difficulty(self):
        b = LiveCodeBenchmark(max_samples=6)
        samples = b.load_data()
        difficulties = {s.category for s in samples}
        assert len(difficulties) >= 1

    def test_lightweight_score_code(self):
        b = LiveCodeBenchmark(lightweight=True)
        sample = BenchmarkSample(
            sample_id="lc1",
            prompt="Solve the problem",
            correct_answer="n = int(input())\nprint(n)",
            metadata={
                "test_inputs": ["5"],
                "test_outputs": ["5"],
            },
        )

        # Good prediction with code structure
        good_code = "n = int(input())\n" "result = n * 2\n" "print(result)\n"
        assert b.score_sample(sample, good_code) is True

    def test_lightweight_score_no_code(self):
        b = LiveCodeBenchmark(lightweight=True)
        sample = BenchmarkSample(
            sample_id="lc2",
            prompt="test",
            metadata={"test_inputs": [], "test_outputs": []},
        )
        assert b.score_sample(sample, "This is just text") is False

    def test_score_none_prediction(self):
        b = LiveCodeBenchmark()
        sample = BenchmarkSample(sample_id="lc3", prompt="test")
        assert b.score_sample(sample, None) is False

    def test_evaluate_runs(self, dummy_model_fn, rng):
        b = LiveCodeBenchmark(max_samples=3)
        result = b.evaluate(dummy_model_fn, rng=rng)
        assert result.benchmark_name == "livecode_bench"
        assert result.num_total == 3


# =============================================================================
# MLflowTracker Tests
# =============================================================================


class TestMLflowTracker:
    """Tests for MLflow tracker wrapper."""

    def test_init_disabled(self):
        tracker = MLflowTracker(enabled=False)
        assert tracker.enabled is False
        assert tracker._mlflow is None

    def test_init_no_mlflow(self):
        """Test graceful fallback when MLflow is not installed."""
        with mock.patch("core.benchmarks.mlflow_tracker._is_mlflow_available", return_value=False):
            tracker = MLflowTracker(enabled=True)
            assert tracker.enabled is False

    def test_context_manager_disabled(self):
        tracker = MLflowTracker(enabled=False)
        with tracker.start_run(run_name="test"):
            tracker.log_param("key", "value")
            tracker.log_metric("loss", 0.5, step=0)
        assert tracker.run_id is None

    def test_log_params_disabled(self):
        tracker = MLflowTracker(enabled=False)
        # Should not raise
        tracker.log_params({"lr": 0.001, "batch_size": 32})

    def test_log_metrics_disabled(self):
        tracker = MLflowTracker(enabled=False)
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)

    def test_log_benchmark_result_disabled(self):
        tracker = MLflowTracker(enabled=False)
        result = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.85,
            num_correct=17,
            num_total=20,
        )
        # Should not raise
        tracker.log_benchmark_result(result)

    def test_log_model_config_disabled(self):
        tracker = MLflowTracker(enabled=False)
        config = AGIConfig()
        tracker.log_model_config(config)

    def test_flatten_dict(self):
        d = {
            "a": 1,
            "b": {"c": 2, "d": {"e": 3}},
            "f": [1, 2, 3],
        }
        flat = MLflowTracker._flatten_dict(d)
        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d.e"] == 3
        assert flat["f"] == "[1, 2, 3]"

    def test_flatten_dict_empty(self):
        assert MLflowTracker._flatten_dict({}) == {}

    def test_is_active(self):
        tracker = MLflowTracker(enabled=False)
        assert tracker.is_active is False

    def test_log_artifact_nonexistent(self, tmp_dir):
        tracker = MLflowTracker(enabled=False)
        # Should not raise, just warn
        tracker.log_artifact(Path(tmp_dir) / "nonexistent.txt")

    def test_metric_buffer_flush(self):
        tracker = MLflowTracker(enabled=False)
        for i in range(200):
            tracker.log_metric("loss", float(i) * 0.01, step=i)
        # Buffer should have been flushed at least once
        # After flush, buffer should be empty or small
        assert len(tracker._metric_buffer) < tracker._buffer_size

    def test_nan_metric_skipped(self):
        tracker = MLflowTracker(enabled=False)
        tracker.log_metric("loss", float("nan"), step=0)
        tracker.log_metric("loss", float("inf"), step=1)
        # NaN/Inf metrics should be skipped
        assert len(tracker._metric_buffer) == 0

    def test_numpy_metric_conversion(self):
        tracker = MLflowTracker(enabled=False)
        tracker.log_metric("loss", np.float32(0.5), step=0)
        assert len(tracker._metric_buffer) == 1
        assert isinstance(tracker._metric_buffer[0]["value"], float)


# =============================================================================
# AGIConfig Benchmark/MLflow Flag Tests
# =============================================================================


class TestAGIConfigBenchmarkFlags:
    """Tests for new AGIConfig benchmark and MLflow flags."""

    def test_default_mlflow_disabled(self):
        config = AGIConfig()
        assert config.mlflow_enabled is False
        assert config.mlflow_tracking_uri is None
        assert config.mlflow_experiment_name == "rtdlm_training"

    def test_mlflow_enabled(self):
        config = AGIConfig(
            mlflow_enabled=True,
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="test_exp",
            mlflow_log_interval=5,
        )
        assert config.mlflow_enabled is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_experiment_name == "test_exp"
        assert config.mlflow_log_interval == 5

    def test_default_benchmark_disabled(self):
        config = AGIConfig()
        assert config.benchmark_enabled is False
        assert config.benchmark_names == ["gpqa"]

    def test_benchmark_enabled(self):
        config = AGIConfig(
            benchmark_enabled=True,
            benchmark_names=["gpqa", "aime", "livecode"],
            benchmark_max_samples=50,
            benchmark_think_budget="high",
            benchmark_eval_interval=2,
        )
        assert config.benchmark_enabled is True
        assert len(config.benchmark_names) == 3
        assert config.benchmark_max_samples == 50
        assert config.benchmark_think_budget == "high"

    def test_benchmark_in_to_dict(self):
        config = AGIConfig(
            benchmark_enabled=True,
            benchmark_names=["gpqa", "aime"],
        )
        d = config.to_dict()
        assert d["benchmark_enabled"] is True
        assert d["benchmark_names"] == ["gpqa", "aime"]
        assert "mlflow_enabled" in d

    def test_mlflow_invalid_log_interval(self):
        with pytest.raises(AssertionError, match="mlflow_log_interval"):
            AGIConfig(mlflow_enabled=True, mlflow_log_interval=0)

    def test_mlflow_empty_experiment_name(self):
        with pytest.raises(AssertionError, match="mlflow_experiment_name"):
            AGIConfig(mlflow_enabled=True, mlflow_experiment_name="")

    def test_benchmark_invalid_name(self):
        with pytest.raises(AssertionError, match="Unknown benchmark"):
            AGIConfig(benchmark_enabled=True, benchmark_names=["invalid_bench"])

    def test_benchmark_invalid_eval_interval(self):
        with pytest.raises(AssertionError, match="benchmark_eval_interval"):
            AGIConfig(benchmark_enabled=True, benchmark_eval_interval=0)

    def test_benchmark_invalid_think_budget(self):
        with pytest.raises(AssertionError, match="benchmark_think_budget"):
            AGIConfig(benchmark_enabled=True, benchmark_think_budget="extreme")


# =============================================================================
# run_eval CLI Parser Tests
# =============================================================================


class TestRunEvalCLI:
    """Tests for the CLI parser in run_eval."""

    def test_build_parser(self):
        from core.benchmarks.run_eval import build_parser

        parser = build_parser()
        args = parser.parse_args(["--benchmarks", "gpqa", "aime"])
        assert args.benchmarks == ["gpqa", "aime"]

    def test_default_args(self):
        from core.benchmarks.run_eval import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.benchmarks == ["gpqa"]
        assert args.checkpoint is None
        assert args.max_samples is None
        assert args.seed == 42
        assert args.output_dir == "eval_results"

    def test_think_budget_arg(self):
        from core.benchmarks.run_eval import build_parser

        parser = build_parser()
        args = parser.parse_args(["--think-budget", "high"])
        assert args.think_budget == "high"

    def test_mlflow_args(self):
        from core.benchmarks.run_eval import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--mlflow-uri",
                "http://localhost:5000",
                "--mlflow-experiment",
                "my_exp",
            ]
        )
        assert args.mlflow_uri == "http://localhost:5000"
        assert args.mlflow_experiment == "my_exp"

    def test_all_benchmarks(self):
        from core.benchmarks.run_eval import build_parser

        parser = build_parser()
        args = parser.parse_args(["--benchmarks", "all"])
        assert args.benchmarks == ["all"]

    def test_think_budget_presets(self):
        from core.benchmarks.run_eval import THINK_BUDGET_PRESETS

        assert "low" in THINK_BUDGET_PRESETS
        assert "medium" in THINK_BUDGET_PRESETS
        assert "high" in THINK_BUDGET_PRESETS
        assert "max" in THINK_BUDGET_PRESETS
        assert THINK_BUDGET_PRESETS["low"] < THINK_BUDGET_PRESETS["max"]

    def test_benchmark_registry(self):
        from core.benchmarks.run_eval import BENCHMARK_REGISTRY

        assert "gpqa" in BENCHMARK_REGISTRY
        assert "aime" in BENCHMARK_REGISTRY
        assert "swe" in BENCHMARK_REGISTRY
        assert "livecode" in BENCHMARK_REGISTRY


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarkIntegration:
    """End-to-end integration tests."""

    def test_full_suite_synthetic(self, dummy_model_fn, rng, tmp_dir):
        """Run all benchmarks in synthetic mode and save results."""
        suite = BenchmarkSuite()
        suite.add(GPQABenchmark(max_samples=3))
        suite.add(AIMEBenchmark(max_samples=3))
        suite.add(SWEBenchBenchmark(max_samples=2))
        suite.add(LiveCodeBenchmark(max_samples=2))

        results = suite.run(dummy_model_fn, rng=rng)
        assert len(results) == 4
        assert all(r.num_total > 0 for r in results.values())

        suite.save_all(tmp_dir)
        json_files = list(Path(tmp_dir).glob("*.json"))
        assert len(json_files) == 4

    def test_suite_with_mlflow_disabled(self, dummy_model_fn, rng):
        """MLflow tracker should work transparently when disabled."""
        tracker = MLflowTracker(enabled=False)
        suite = BenchmarkSuite()
        suite.add(GPQABenchmark(max_samples=3))
        results = suite.run(dummy_model_fn, rng=rng)

        with tracker.start_run(run_name="test_eval"):
            for name, result in results.items():
                tracker.log_benchmark_result(result)

    def test_result_json_roundtrip(self, tmp_dir):
        """BenchmarkResult should survive JSON serialization."""
        original = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.75,
            num_correct=15,
            num_total=20,
            category_scores={"a": 0.8, "b": 0.7},
            total_time_sec=10.5,
            samples_per_sec=1.9,
            config={"seed": 42, "max_samples": 20},
        )
        path = Path(tmp_dir) / "test.json"
        original.save_json(path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == 0.75
        assert loaded["num_correct"] == 15
        assert loaded["category_scores"]["a"] == 0.8

    def test_benchmark_with_think_budget(self, rng):
        """Verify think_budget is passed through."""
        received_budgets = []

        def model_fn(prompt, rng, choices=None, think_budget=None, **kwargs):
            received_budgets.append(think_budget)
            return 0 if choices else "42"

        b = _DummyBenchmark(max_samples=3)
        b.evaluate(model_fn, rng=rng, think_budget=1024)
        assert all(tb == 1024 for tb in received_budgets)

    def test_config_benchmark_flags_propagate(self):
        """Verify config flags create correct benchmark setup."""
        config = AGIConfig(
            benchmark_enabled=True,
            benchmark_names=["gpqa", "aime"],
            benchmark_max_samples=10,
            benchmark_think_budget="high",
        )
        assert config.benchmark_enabled
        assert config.benchmark_names == ["gpqa", "aime"]
        assert config.benchmark_max_samples == 10
