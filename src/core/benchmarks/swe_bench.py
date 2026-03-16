"""
SWE-Bench Benchmark Evaluator

Evaluates the model on SWE-Bench Verified — a curated subset of real-world
GitHub issues that test code generation and debugging capabilities.

Dataset: princeton-nlp/SWE-bench_Verified (HuggingFace Hub)
Metric: pass@1 — whether the model's generated patch resolves the issue.

Note: Full SWE-Bench evaluation requires execution-based testing. This
module provides the prompt construction and basic output parsing. For
full execution-based eval, use the official SWE-bench harness.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from core.benchmarks.base_benchmark import BenchmarkBase, BenchmarkSample

logger = logging.getLogger(__name__)


class SWEBenchBenchmark(BenchmarkBase):
    """SWE-Bench Verified benchmark evaluator.

    Loads SWE-Bench Verified from HuggingFace Hub. Each sample is a
    real GitHub issue + repository context. The model must generate
    a patch that fixes the issue.

    Scoring: For lightweight eval, we check whether the model's output
    contains key elements from the gold patch. For full eval, the
    official SWE-bench harness should be used.

    Usage:
        bench = SWEBenchBenchmark(max_samples=20)
        result = bench.evaluate(model_fn, params, state, rng)
    """

    DATASET_ID = "princeton-nlp/SWE-bench_Verified"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        seed: int = 42,
        lightweight: bool = True,
        use_huggingface: bool = False,
    ):
        """
        Args:
            data_dir: Local cache directory
            split: Dataset split
            max_samples: Cap number of samples
            seed: Random seed
            lightweight: If True, use heuristic scoring instead of
                        execution-based evaluation
            use_huggingface: If True, attempt to download from HuggingFace Hub.
                            Default is False — uses built-in curated issues.
        """
        super().__init__(data_dir, split, max_samples, seed)
        self.lightweight = lightweight
        self.use_huggingface = use_huggingface

    @property
    def name(self) -> str:
        return "swe_bench_verified"

    def score_sample(
        self,
        sample: BenchmarkSample,
        prediction: Any,
    ) -> bool:
        """SWE-Bench scoring.

        Lightweight mode: checks for key patch elements.
        Full mode: would require execution (not implemented here).
        """
        if prediction is None:
            return False

        pred_str = str(prediction).strip()
        if not pred_str:
            return False

        if self.lightweight:
            return self._lightweight_score(sample, pred_str)
        else:
            # Full execution-based scoring requires the SWE-bench harness
            logger.warning(
                "Full SWE-bench execution scoring not implemented. "
                "Use the official SWE-bench harness for execution-based eval."
            )
            return self._lightweight_score(sample, pred_str)

    def _lightweight_score(
        self, sample: BenchmarkSample, prediction: str
    ) -> bool:
        """Heuristic scoring for SWE-bench.

        Checks whether the prediction:
        1. Contains a diff/patch format
        2. References the correct file(s)
        3. Contains key tokens from the gold patch
        """
        gold_patch = sample.metadata.get("gold_patch", "")
        if not gold_patch:
            return False

        # Check basic patch structure
        has_diff = any(
            marker in prediction
            for marker in ["diff --git", "---", "+++", "@@", "+", "-"]
        )

        # Check file references
        target_files = sample.metadata.get("target_files", [])
        files_referenced = sum(
            1 for f in target_files
            if f in prediction
        )

        # Check for key tokens from gold patch (non-trivial lines)
        gold_lines = [
            line.strip()
            for line in gold_patch.split("\n")
            if line.strip()
            and not line.startswith("---")
            and not line.startswith("+++")
            and not line.startswith("@@")
            and not line.startswith("diff")
            and len(line.strip()) > 5
        ]
        if gold_lines:
            matched_lines = sum(
                1 for line in gold_lines[:10]  # Check first 10 significant lines
                if line.lstrip("+-").strip() in prediction
            )
            token_overlap = matched_lines / min(len(gold_lines), 10)
        else:
            token_overlap = 0.0

        # Heuristic: pass if has patch format AND (file refs OR token overlap)
        return has_diff and (files_referenced > 0 or token_overlap > 0.3)

    def load_data(self) -> List[BenchmarkSample]:
        """Load SWE-Bench Verified problems.

        Default: uses built-in curated GitHub issues (no network needed).
        If use_huggingface=True, attempts to download from HuggingFace Hub.
        """
        if not self.use_huggingface:
            return self._synthetic_fallback()

        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.DATASET_ID,
                split=self.split,
                cache_dir=self.data_dir,
                trust_remote_code=True,
            )
            logger.info(f"Loaded SWE-bench Verified: {len(ds)} samples")
        except Exception as e:
            logger.warning(
                f"Could not load SWE-bench from HuggingFace: {e}. "
                f"Using synthetic fallback data."
            )
            return self._synthetic_fallback()

        samples = self._parse_dataset(ds)
        logger.info(f"Prepared {len(samples)} SWE-bench samples")
        self._samples = samples
        return samples

    def _parse_dataset(self, ds) -> List[BenchmarkSample]:
        """Parse HuggingFace dataset into BenchmarkSamples."""
        samples: List[BenchmarkSample] = []

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            instance_id = row.get("instance_id", f"swe_{idx}")
            problem_statement = row.get("problem_statement", "")
            gold_patch = row.get("patch", "")
            repo = row.get("repo", "")
            base_commit = row.get("base_commit", "")

            # Extract target files from patch
            target_files = self._extract_files_from_patch(gold_patch)

            # Build prompt
            prompt = (
                f"Fix the following GitHub issue by generating a unified diff patch.\n\n"
                f"Repository: {repo}\n"
                f"Base commit: {base_commit}\n\n"
                f"Issue:\n{problem_statement}\n\n"
                f"Generate a patch (unified diff format) that resolves this issue:"
            )

            # Category is the repo
            category = repo.split("/")[-1] if "/" in repo else repo

            samples.append(BenchmarkSample(
                sample_id=instance_id,
                prompt=prompt,
                choices=None,
                correct_answer=gold_patch,
                category=category,
                metadata={
                    "repo": repo,
                    "base_commit": base_commit,
                    "gold_patch": gold_patch,
                    "target_files": target_files,
                    "instance_id": instance_id,
                },
            ))

        return samples

    @staticmethod
    def _extract_files_from_patch(patch: str) -> List[str]:
        """Extract modified file paths from a unified diff patch."""
        files = []
        for line in patch.split("\n"):
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    # "diff --git a/file.py b/file.py"
                    file_path = parts[2].lstrip("a/")
                    files.append(file_path)
            elif line.startswith("--- a/"):
                files.append(line[6:])
            elif line.startswith("+++ b/"):
                files.append(line[6:])
        return list(set(files))

    def _synthetic_fallback(self) -> List[BenchmarkSample]:
        """Generate synthetic SWE-bench-like samples for testing/CI."""
        rng = random.Random(self.seed)

        issues = [
            {
                "repo": "django/django",
                "issue": (
                    "QuerySet.filter() raises TypeError when using "
                    "__in lookup with empty list on PostgreSQL backend."
                ),
                "patch": (
                    "diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py\n"
                    "--- a/django/db/models/sql/query.py\n"
                    "+++ b/django/db/models/sql/query.py\n"
                    "@@ -1234,6 +1234,8 @@\n"
                    "     def build_filter(self, filter_expr):\n"
                    "+        if isinstance(value, (list, tuple)) and len(value) == 0:\n"
                    "+            return self.where_class()\n"
                ),
                "files": ["django/db/models/sql/query.py"],
            },
            {
                "repo": "scikit-learn/scikit-learn",
                "issue": (
                    "KMeans.fit() produces different results with n_init='auto' "
                    "versus n_init=10 when random_state is set."
                ),
                "patch": (
                    "diff --git a/sklearn/cluster/_kmeans.py b/sklearn/cluster/_kmeans.py\n"
                    "--- a/sklearn/cluster/_kmeans.py\n"
                    "+++ b/sklearn/cluster/_kmeans.py\n"
                    "@@ -890,7 +890,7 @@\n"
                    "-        n_init = 10 if self.n_init == 'auto' else self.n_init\n"
                    "+        n_init = self._resolve_n_init()\n"
                ),
                "files": ["sklearn/cluster/_kmeans.py"],
            },
            {
                "repo": "matplotlib/matplotlib",
                "issue": (
                    "plt.savefig() with bbox_inches='tight' raises ValueError "
                    "when figure has no axes."
                ),
                "patch": (
                    "diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py\n"
                    "--- a/lib/matplotlib/figure.py\n"
                    "+++ b/lib/matplotlib/figure.py\n"
                    "@@ -3201,6 +3201,9 @@\n"
                    "     def get_tightbbox(self, renderer):\n"
                    "+        if not self.axes:\n"
                    "+            return self.bbox\n"
                ),
                "files": ["lib/matplotlib/figure.py"],
            },
            {
                "repo": "flask/flask",
                "issue": (
                    "Blueprint.teardown_request handler not called when "
                    "exception occurs during request processing."
                ),
                "patch": (
                    "diff --git a/src/flask/app.py b/src/flask/app.py\n"
                    "--- a/src/flask/app.py\n"
                    "+++ b/src/flask/app.py\n"
                    "@@ -1423,6 +1423,10 @@\n"
                    "     def process_response(self, response):\n"
                    "+        try:\n"
                    "+            self.do_teardown_request()\n"
                    "+        except Exception:\n"
                    "+            pass\n"
                ),
                "files": ["src/flask/app.py"],
            },
            {
                "repo": "requests/requests",
                "issue": (
                    "Session.send() does not properly handle redirect "
                    "with fragment in URL."
                ),
                "patch": (
                    "diff --git a/requests/sessions.py b/requests/sessions.py\n"
                    "--- a/requests/sessions.py\n"
                    "+++ b/requests/sessions.py\n"
                    "@@ -185,6 +185,8 @@\n"
                    "     def resolve_redirects(self, resp, req):\n"
                    "+        if '#' in url:\n"
                    "+            url = url.split('#')[0]\n"
                ),
                "files": ["requests/sessions.py"],
            },
        ]

        n = min(self.max_samples or len(issues), len(issues))
        selected = rng.sample(issues, n)

        samples: List[BenchmarkSample] = []
        for i, issue in enumerate(selected):
            prompt = (
                f"Fix the following GitHub issue by generating a unified diff patch.\n\n"
                f"Repository: {issue['repo']}\n\n"
                f"Issue:\n{issue['issue']}\n\n"
                f"Generate a patch (unified diff format) that resolves this issue:"
            )

            samples.append(BenchmarkSample(
                sample_id=f"swe_synth_{i}",
                prompt=prompt,
                choices=None,
                correct_answer=issue["patch"],
                category=issue["repo"].split("/")[-1],
                metadata={
                    "synthetic": True,
                    "repo": issue["repo"],
                    "gold_patch": issue["patch"],
                    "target_files": issue["files"],
                },
            ))

        logger.info(f"Generated {len(samples)} synthetic SWE-bench samples")
        self._samples = samples
        return samples
