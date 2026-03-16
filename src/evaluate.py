#!/usr/bin/env python3
"""
RT-DLM Benchmark Evaluation Runner
====================================

Root-level entry point for running benchmark evaluations against
the RT-DLM AGI model.  Works just like train.py — run from the
src/ directory:

    python evaluate.py --benchmarks gpqa aime --max-samples 50
    python evaluate.py --benchmarks all --checkpoint ckpts/best
    python evaluate.py --benchmarks gpqa --think-budget high --use-huggingface

Supported benchmarks:
  gpqa       — GPQA Diamond (graduate-level science / MC)
  aime       — AIME math competition (integer answers)
  swe        — SWE-Bench Verified (patch generation)
  livecode   — LiveCodeBench (competitive programming)
  all        — Run every registered benchmark

By default, curated built-in problems are used (no network access).
Pass --use-huggingface to download the real datasets from HuggingFace
Hub (requires the `datasets` package).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── path setup (same pattern as train.py) ──────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# We also need `src/` itself on the path so that
# `from config.agi_config import AGIConfig` etc. work inside the
# benchmark modules that use un-prefixed imports.
SRC_DIR = Path(__file__).parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ── lazy imports (after path is set up) ────────────────────────
from core.benchmarks.run_eval import (       # noqa: E402
    BENCHMARK_REGISTRY,
    THINK_BUDGET_PRESETS,
    build_model_fn,
    build_parser as _build_inner_parser,
    run_evaluation,
    print_rich_summary,
)
from config.agi_config import AGIConfig      # noqa: E402


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Mirrors the arg surface of ``core.benchmarks.run_eval.build_parser``
    but adds a few root-level conveniences (e.g. ``--config-*`` overrides,
    ``--epochs-between-evals``).
    """
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="RT-DLM AGI Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Quick smoke-test with built-in curated problems (no downloads)
  python evaluate.py --benchmarks gpqa --max-samples 10

  # Run all benchmarks against a trained checkpoint
  python evaluate.py --benchmarks all --checkpoint checkpoints/best

  # Use real HuggingFace datasets with high thinking budget
  python evaluate.py --benchmarks gpqa aime --use-huggingface --think-budget high

  # Log results to MLflow
  python evaluate.py --benchmarks gpqa --mlflow-uri http://localhost:5000 \\
                     --mlflow-experiment rtdlm_eval
        """,
    )

    # ── benchmark selection ────────────────────────────────────
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gpqa"],
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        help="Benchmarks to run.  Use 'all' to run every registered benchmark.  "
             "(default: gpqa)",
    )

    # ── checkpoint & model ─────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint directory.  "
             "If omitted, a randomly-initialised model is used.",
    )

    # ── sampling / budget ──────────────────────────────────────
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of samples per benchmark (default: all).",
    )
    parser.add_argument(
        "--think-budget",
        type=str,
        default=None,
        help="Thinking-token budget.  "
             "Presets: low (256), medium (1024), high (4096), max (8192), "
             "or pass an integer.",
    )

    # ── data source ────────────────────────────────────────────
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Download benchmark datasets from HuggingFace Hub.  "
             "Requires the 'datasets' package.  "
             "Default: use built-in curated problems (no network).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local cache directory for benchmark data.",
    )

    # ── output ─────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save per-benchmark JSON results (default: eval_results).",
    )

    # ── MLflow ─────────────────────────────────────────────────
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI (e.g. http://localhost:5000).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name for this evaluation run.",
    )

    # ── misc ───────────────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample predictions.",
    )

    return parser


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — parse args, run benchmarks, report results."""
    parser = build_parser()
    args = parser.parse_args()

    # ── banner ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  RT-DLM AGI — Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Benchmarks     : {', '.join(args.benchmarks)}")
    logger.info(f"  Checkpoint     : {args.checkpoint or 'random init'}")
    logger.info(f"  Think budget   : {args.think_budget or 'default'}")
    logger.info(f"  Max samples    : {args.max_samples or 'all'}")
    logger.info(f"  HuggingFace    : {'yes' if args.use_huggingface else 'no (built-in data)'}")
    logger.info(f"  Output dir     : {args.output_dir}")
    logger.info(f"  Seed           : {args.seed}")
    logger.info("=" * 60)

    # ── delegate to the evaluation pipeline ────────────────────
    results = run_evaluation(args)

    if not results:
        logger.error("No results produced!  Check benchmark names and data.")
        sys.exit(1)

    # ── summary ────────────────────────────────────────────────
    total_correct = sum(
        r.num_correct for r in results.values() if hasattr(r, "num_correct")
    )
    total_samples = sum(
        r.num_total for r in results.values() if hasattr(r, "num_total")
    )
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    logger.info("-" * 60)
    logger.info(f"  Overall: {total_correct}/{total_samples} = {overall_acc:.2%}")
    logger.info("-" * 60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
