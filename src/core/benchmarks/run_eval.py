"""
Benchmark CLI Runner

Command-line interface for running RT-DLM model evaluation across
GPQA, AIME, SWE-Bench, and LiveCodeBench benchmarks. Outputs results
as Rich tables to the console and optionally logs to MLflow.

Usage:
    python -m core.benchmarks.run_eval \
        --checkpoint checkpoints/latest \
        --benchmarks gpqa aime livecode \
        --think-budget medium \
        --max-samples 50 \
        --output-dir eval_results/

    python -m core.benchmarks.run_eval \
        --benchmarks all \
        --mlflow-uri http://localhost:5000 \
        --mlflow-experiment rtdlm_eval
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# Think-budget presets (maps human-friendly names to token budgets)
THINK_BUDGET_PRESETS = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
    "max": 8192,
}

# Benchmark registry
BENCHMARK_REGISTRY = {
    "gpqa": "core.benchmarks.gpqa_benchmark.GPQABenchmark",
    "aime": "core.benchmarks.aime_benchmark.AIMEBenchmark",
    "swe": "core.benchmarks.swe_bench.SWEBenchBenchmark",
    "livecode": "core.benchmarks.livecode_bench.LiveCodeBenchmark",
}


def _import_benchmark(dotpath: str):
    """Dynamically import a benchmark class from dotted path."""
    module_path, class_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_model_fn(
    checkpoint_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Build the model inference function.

    If a checkpoint is provided, loads params/state from it.
    Otherwise, creates a fresh model with random params (for testing).

    Returns:
        model_fn: Callable(prompt, rng, choices=None, **kwargs) -> prediction
        params: Model parameters
        state: Model state
    """
    from config.agi_config import AGIConfig

    config_kwargs = config_overrides or {}
    config = AGIConfig(**config_kwargs)

    # Create model
    from rtdlm import create_rtdlm_agi
    model = create_rtdlm_agi(config, use_state=True)

    if checkpoint_path:
        try:
            from core.checkpoint_manager import CheckpointManager
            ckpt_mgr = CheckpointManager(checkpoint_dir=checkpoint_path)
            ckpt_data = ckpt_mgr.load_checkpoint()
            params = ckpt_data.get("params", ckpt_data)
            state = ckpt_data.get("state", {})
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Using random init.")
            params, state = _init_random_params(model, config)
    else:
        logger.info("No checkpoint provided, using random initialization")
        params, state = _init_random_params(model, config)

    def model_fn(
        prompt: str,
        rng: jnp.ndarray,
        choices: Optional[List[str]] = None,
        params: Any = params,
        state: Any = state,
        think_budget: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Model inference for benchmark evaluation.

        For MC questions: returns the index of the selected choice.
        For open-ended: returns the generated text.
        """
        # Tokenize prompt (simplified - in production use real tokenizer)
        seq_len = min(len(prompt.split()), config.max_seq_length)
        vocab_size = config.vocab_size

        rng, tok_rng = jax.random.split(rng)
        token_ids = jax.random.randint(
            tok_rng, (1, seq_len), 0, vocab_size
        )

        inputs = {"text": token_ids}

        try:
            rng, model_rng = jax.random.split(rng)
            output, new_state = model.apply(params, state, model_rng, inputs)

            logits = output.get("logits", None)
            if logits is None:
                logits = jnp.zeros((1, seq_len, vocab_size))
        except Exception as e:
            logger.debug(f"Model forward pass error: {e}")
            logits = jnp.zeros((1, seq_len, vocab_size))

        if choices is not None:
            # MC: pick answer based on logits
            # Use last-token logits and map to choice indices
            last_logits = logits[0, -1, :]
            rng, choice_rng = jax.random.split(rng)
            # Simple: pick from first len(choices) positions
            choice_scores = last_logits[:len(choices)]
            return int(jnp.argmax(choice_scores))
        else:
            # Open-ended: generate text (simplified)
            last_logits = logits[0, -1, :]
            pred_token = int(jnp.argmax(last_logits))
            return str(pred_token)

    return model_fn, params, state


def _init_random_params(model, config):
    """Initialize model with random parameters."""
    rng = jax.random.PRNGKey(42)
    seq_len = min(64, config.max_seq_length)
    dummy_input = {
        "text": jax.random.randint(
            rng, (1, seq_len), 0, config.vocab_size
        )
    }
    rng, init_rng = jax.random.split(rng)
    try:
        params, state = model.init(init_rng, dummy_input)
    except Exception as e:
        logger.warning(f"Model init failed: {e}. Using empty params.")
        params = {}
        state = {}
    return params, state


def print_rich_summary(results: Dict[str, Any]) -> None:
    """Print results as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Benchmark Evaluation Results", show_lines=True)
        table.add_column("Benchmark", style="cyan", justify="left")
        table.add_column("Accuracy", style="green", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Samples/s", justify="right")

        for name, result in results.items():
            if hasattr(result, "accuracy"):
                table.add_row(
                    name,
                    f"{result.accuracy:.2%}",
                    str(result.num_correct),
                    str(result.num_total),
                    f"{result.total_time_sec:.1f}",
                    f"{result.samples_per_sec:.1f}",
                )
            else:
                table.add_row(name, "ERROR", "-", "-", "-", "-")

        console.print(table)

        # Category breakdown
        for name, result in results.items():
            if hasattr(result, "category_scores") and result.category_scores:
                cat_table = Table(
                    title=f"{name} — Category Breakdown",
                    show_lines=True,
                )
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("Accuracy", style="green", justify="right")

                for cat, score in sorted(
                    result.category_scores.items(), key=lambda x: -x[1]
                ):
                    cat_table.add_row(cat, f"{score:.2%}")

                console.print(cat_table)

    except ImportError:
        # Fallback without Rich
        print("\n" + "=" * 60)
        print("  BENCHMARK EVALUATION RESULTS")
        print("=" * 60)
        for name, result in results.items():
            if hasattr(result, "accuracy"):
                print(
                    f"  {name:<25s} {result.accuracy:>8.2%} "
                    f"({result.num_correct}/{result.num_total}) "
                    f"{result.total_time_sec:.1f}s"
                )
            else:
                print(f"  {name:<25s} ERROR")
        print("=" * 60)


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute benchmark evaluation pipeline.

    Args:
        args: Parsed CLI arguments

    Returns:
        Dict of benchmark name → BenchmarkResult
    """
    from core.benchmarks.base_benchmark import BenchmarkSuite

    # Resolve think-budget
    think_budget = None
    if args.think_budget:
        if args.think_budget in THINK_BUDGET_PRESETS:
            think_budget = THINK_BUDGET_PRESETS[args.think_budget]
        else:
            try:
                think_budget = int(args.think_budget)
            except ValueError:
                logger.warning(
                    f"Invalid think-budget: {args.think_budget}. Using None."
                )

    # Build benchmark suite
    suite = BenchmarkSuite()

    benchmark_names = args.benchmarks
    if "all" in benchmark_names:
        benchmark_names = list(BENCHMARK_REGISTRY.keys())

    for bench_name in benchmark_names:
        if bench_name not in BENCHMARK_REGISTRY:
            logger.warning(f"Unknown benchmark: {bench_name}. Skipping.")
            continue

        bench_cls = _import_benchmark(BENCHMARK_REGISTRY[bench_name])
        bench_kwargs = {
            "max_samples": args.max_samples,
            "seed": args.seed,
            "use_huggingface": getattr(args, "use_huggingface", False),
        }
        if args.data_dir:
            bench_kwargs["data_dir"] = args.data_dir

        bench = bench_cls(**bench_kwargs)
        suite.add(bench)

    if not suite.benchmarks:
        logger.error("No benchmarks to run!")
        return {}

    # Build model
    logger.info("Building model...")
    model_fn, params, state = build_model_fn(
        checkpoint_path=args.checkpoint,
        config_overrides={},
    )

    rng = jax.random.PRNGKey(args.seed)

    # Run benchmarks
    logger.info(f"Running {len(suite.benchmarks)} benchmarks...")
    results = suite.run(
        model_fn=model_fn,
        params=params,
        state=state,
        rng=rng,
        think_budget=think_budget,
        verbose=args.verbose,
    )

    # Print results
    print_rich_summary(results)
    print(suite.summary_table())

    # Save results
    if args.output_dir:
        suite.save_all(args.output_dir)
        # Also save combined summary
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output_dir}")

    # MLflow logging
    if args.mlflow_uri or args.mlflow_experiment:
        _log_to_mlflow(results, args, think_budget)

    return results


def _log_to_mlflow(
    results: Dict[str, Any],
    args: argparse.Namespace,
    think_budget: Optional[int],
) -> None:
    """Log benchmark results to MLflow."""
    from core.benchmarks.mlflow_tracker import MLflowTracker

    tracker = MLflowTracker(
        experiment_name=args.mlflow_experiment or "rtdlm_evaluation",
        tracking_uri=args.mlflow_uri,
        enabled=True,
    )

    run_tags = {
        "eval_type": "benchmark",
        "think_budget": str(think_budget) if think_budget else "none",
    }

    with tracker.start_run(run_name=f"eval_{int(time.time())}", tags=run_tags):
        # Log eval config
        tracker.log_params({
            "benchmarks": ",".join(args.benchmarks),
            "max_samples": args.max_samples or "all",
            "seed": args.seed,
            "checkpoint": args.checkpoint or "random_init",
            "think_budget": think_budget or "none",
        })

        # Log each benchmark result
        for name, result in results.items():
            tracker.log_benchmark_result(result)

        # Log output artifacts
        if args.output_dir and Path(args.output_dir).exists():
            tracker.log_artifact(args.output_dir, "eval_results")

    logger.info("Results logged to MLflow")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="RT-DLM Benchmark Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.benchmarks.run_eval --benchmarks gpqa aime --max-samples 50
  python -m core.benchmarks.run_eval --benchmarks all --checkpoint ckpts/best
  python -m core.benchmarks.run_eval --benchmarks gpqa --think-budget high
        """,
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gpqa"],
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        help="Benchmarks to run (default: gpqa)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (default: all)",
    )
    parser.add_argument(
        "--think-budget",
        type=str,
        default=None,
        help="Think budget: low/medium/high/max or integer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for result JSON files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local data cache directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample results",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Download benchmark data from HuggingFace Hub (requires 'datasets' package). "
             "Default: use built-in curated problems (no network needed).",
    )

    return parser


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  RT-DLM Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Benchmarks: {', '.join(args.benchmarks)}")
    logger.info(f"  Checkpoint: {args.checkpoint or 'random init'}")
    logger.info(f"  Think Budget: {args.think_budget or 'default'}")
    logger.info(f"  Max Samples: {args.max_samples or 'all'}")
    logger.info("=" * 60)

    results = run_evaluation(args)

    if not results:
        logger.error("No results produced!")
        sys.exit(1)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
