"""
MLflow Experiment Tracker

Wraps MLflow for experiment tracking, metric logging, artifact storage,
and model versioning. Integrates with the RT-DLM training loop and
benchmark evaluation pipeline.

Design:
    - Drop-in tracker: start_run → log_params/metrics → end_run
    - Auto-creates experiment if it doesn't exist
    - Supports nested runs (parent train run → child benchmark runs)
    - Graceful degradation if MLflow server is unavailable
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _is_mlflow_available() -> bool:
    """Check if MLflow is installed."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


class MLflowTracker:
    """MLflow experiment tracker for RT-DLM.

    Provides a unified interface for logging training metrics,
    benchmark results, model artifacts, and hyperparameters to MLflow.

    Falls back to local-only logging if MLflow is not available.

    Usage:
        tracker = MLflowTracker(
            experiment_name="rtdlm_training",
            tracking_uri="http://localhost:5000",
        )
        with tracker.start_run(run_name="train_v1"):
            tracker.log_params(config.to_dict())
            for step in range(num_steps):
                tracker.log_metric("loss", loss_val, step=step)
            tracker.log_benchmark_result(benchmark_result)
            tracker.log_artifact("checkpoints/best.safetensors")
    """

    def __init__(
        self,
        experiment_name: str = "rtdlm_default",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI (None → local ./mlruns)
            enabled: Whether MLflow tracking is active
            tags: Global tags to apply to all runs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled and _is_mlflow_available()
        self.tags = tags or {}
        self._run = None
        self._mlflow = None
        self._metric_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Batch log every N metrics

        if self.enabled:
            self._setup_mlflow()
        elif enabled and not _is_mlflow_available():
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow>=2.10.0. "
                "Falling back to local logging only."
            )

    def _setup_mlflow(self) -> None:
        """Initialize MLflow client and experiment."""
        try:
            import mlflow
            self._mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)

            logger.info(
                f"MLflow tracker initialized: experiment='{self.experiment_name}', "
                f"uri='{self.tracking_uri or 'local'}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for an MLflow run.

        Args:
            run_name: Name for this run
            nested: Whether this is a nested (child) run
            tags: Run-specific tags (merged with global tags)

        Yields:
            self (for chaining)
        """
        merged_tags = {**self.tags, **(tags or {})}

        if self.enabled and self._mlflow is not None:
            try:
                self._run = self._mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    tags=merged_tags,
                )
                logger.info(f"MLflow run started: {run_name}")
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
                self._run = None
        else:
            logger.info(f"MLflow disabled, skipping run: {run_name}")
            self._run = None

        try:
            yield self
        finally:
            self._flush_metrics()
            if self._run is not None:
                try:
                    self._mlflow.end_run()
                    logger.info(f"MLflow run ended: {run_name}")
                except Exception as e:
                    logger.error(f"Failed to end MLflow run: {e}")
            self._run = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        if self.enabled and self._mlflow is not None and self._run is not None:
            try:
                self._mlflow.log_param(key, value)
            except Exception as e:
                logger.debug(f"Failed to log param {key}: {e}")
        logger.debug(f"Param: {key}={value}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters (flattened dict).

        Nested dicts are flattened with '.' separator.
        MLflow has a 500-param limit per run.
        """
        flat = self._flatten_dict(params)
        # MLflow limits param values to 500 chars
        truncated = {
            k: str(v)[:500] for k, v in flat.items()
        }

        if self.enabled and self._mlflow is not None and self._run is not None:
            try:
                # Log in batches (MLflow limit: 100 per call)
                items = list(truncated.items())
                for i in range(0, len(items), 100):
                    batch = dict(items[i:i + 100])
                    self._mlflow.log_params(batch)
            except Exception as e:
                logger.debug(f"Failed to log params batch: {e}")

        logger.debug(f"Params logged: {len(truncated)} entries")

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single metric value.

        Buffers metrics and flushes in batches for efficiency.
        """
        # Validate value
        if isinstance(value, (np.floating, np.integer)):
            value = float(value)
        if not isinstance(value, (int, float)):
            return

        if np.isnan(value) or np.isinf(value):
            logger.debug(f"Skipping NaN/Inf metric: {key}={value}")
            return

        self._metric_buffer.append({
            "key": key,
            "value": value,
            "step": step,
            "timestamp": int(time.time() * 1000),
        })

        if len(self._metric_buffer) >= self._buffer_size:
            self._flush_metrics()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics at the same step."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def _flush_metrics(self) -> None:
        """Flush buffered metrics to MLflow."""
        if not self._metric_buffer:
            return

        if self.enabled and self._mlflow is not None and self._run is not None:
            try:
                client = self._mlflow.tracking.MlflowClient()
                run_id = self._run.info.run_id

                # Use batch logging
                from mlflow.entities import Metric

                mlflow_metrics = [
                    Metric(
                        key=m["key"],
                        value=m["value"],
                        timestamp=m["timestamp"],
                        step=m["step"] or 0,
                    )
                    for m in self._metric_buffer
                ]

                # Batch log (max 1000 per call)
                for i in range(0, len(mlflow_metrics), 1000):
                    batch = mlflow_metrics[i:i + 1000]
                    client.log_batch(run_id, metrics=batch)

            except Exception as e:
                logger.debug(f"Failed to flush {len(self._metric_buffer)} metrics: {e}")

        self._metric_buffer.clear()

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a file or directory as an artifact.

        Args:
            local_path: Path to file/directory to log
            artifact_path: Optional subdirectory in artifact store
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning(f"Artifact not found: {local_path}")
            return

        if self.enabled and self._mlflow is not None and self._run is not None:
            try:
                if local_path.is_dir():
                    self._mlflow.log_artifacts(
                        str(local_path), artifact_path
                    )
                else:
                    self._mlflow.log_artifact(
                        str(local_path), artifact_path
                    )
                logger.info(f"Logged artifact: {local_path}")
            except Exception as e:
                logger.debug(f"Failed to log artifact {local_path}: {e}")

    def log_benchmark_result(
        self,
        result: Any,
        prefix: str = "benchmark",
    ) -> None:
        """Log a BenchmarkResult to MLflow.

        Logs accuracy, category scores, timing, and config as metrics
        and parameters.

        Args:
            result: BenchmarkResult instance
            prefix: Metric key prefix
        """
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            result_dict = {"value": str(result)}
            return

        bench_name = result_dict.get("benchmark", "unknown")
        metric_prefix = f"{prefix}/{bench_name}"

        # Log primary metrics
        self.log_metric(f"{metric_prefix}/accuracy", result_dict.get("accuracy", 0.0))
        self.log_metric(f"{metric_prefix}/num_correct", result_dict.get("num_correct", 0))
        self.log_metric(f"{metric_prefix}/num_total", result_dict.get("num_total", 0))
        self.log_metric(f"{metric_prefix}/time_sec", result_dict.get("total_time_sec", 0.0))
        self.log_metric(f"{metric_prefix}/samples_per_sec", result_dict.get("samples_per_sec", 0.0))

        # Log per-category scores
        for cat, score in result_dict.get("category_scores", {}).items():
            safe_cat = cat.replace(" ", "_").replace("/", "_")
            self.log_metric(f"{metric_prefix}/category/{safe_cat}", score)

        self._flush_metrics()
        logger.info(
            f"Logged benchmark result: {bench_name} "
            f"(accuracy={result_dict.get('accuracy', 0.0):.2%})"
        )

    def log_model_config(self, config: Any) -> None:
        """Log AGIConfig as MLflow parameters.

        Args:
            config: AGIConfig instance (or any object with to_dict())
        """
        if hasattr(config, "to_dict"):
            params = config.to_dict()
        elif isinstance(config, dict):
            params = config
        else:
            params = {"config": str(config)}

        self.log_params(params)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        if self.enabled and self._mlflow is not None and self._run is not None:
            try:
                self._mlflow.set_tag(key, value)
            except Exception as e:
                logger.debug(f"Failed to set tag {key}: {e}")

    @property
    def run_id(self) -> Optional[str]:
        """Current run ID, if any."""
        if self._run is not None:
            return self._run.info.run_id
        return None

    @property
    def is_active(self) -> bool:
        """Whether a run is currently active."""
        return self._run is not None

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten nested dict with separator."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    MLflowTracker._flatten_dict(v, new_key, sep).items()
                )
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
