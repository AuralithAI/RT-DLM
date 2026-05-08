"""
RT-DLM Benchmark Harness

Production-grade evaluation on frontier model benchmarks:
- GPQA Diamond (graduate-level QA)
- AIME 2024/2025 (competition math)
- SWE-Bench Verified (software engineering)
- LiveCodeBench (competitive programming)

Usage:
    python -m src.core.benchmarks.run_eval \
        --benchmarks gpqa aime \
        --checkpoint checkpoints/rtdlm_agi_epoch_5.safetensors
"""

from core.benchmarks.base_benchmark import (
    BenchmarkBase,
    BenchmarkSample,
    BenchmarkResult,
    BenchmarkSuite,
)
from core.benchmarks.gpqa_benchmark import GPQABenchmark
from core.benchmarks.aime_benchmark import AIMEBenchmark
from core.benchmarks.swe_bench import SWEBenchBenchmark
from core.benchmarks.livecode_bench import LiveCodeBenchmark
from core.benchmarks.mlflow_tracker import MLflowTracker

__all__ = [
    "BenchmarkBase",
    "BenchmarkSample",
    "BenchmarkResult",
    "BenchmarkSuite",
    "GPQABenchmark",
    "AIMEBenchmark",
    "SWEBenchBenchmark",
    "LiveCodeBenchmark",
    "MLflowTracker",
]
