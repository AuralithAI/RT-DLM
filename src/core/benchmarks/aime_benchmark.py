"""
AIME Benchmark Evaluator

Evaluates the model on AIME (American Invitational Mathematics Examination)
problems. AIME answers are always integers between 000 and 999 inclusive.

Scoring: Exact integer match.
Target: AIME 2024 and 2025 problem sets.
"""

import logging
import random
from typing import Any, List, Optional

from core.benchmarks.base_benchmark import BenchmarkBase, BenchmarkSample

logger = logging.getLogger(__name__)


class AIMEBenchmark(BenchmarkBase):
    """AIME (American Invitational Mathematics Examination) benchmark.

    AIME problems have integer answers in the range [0, 999].
    We evaluate exact numeric match after stripping whitespace.

    Usage:
        bench = AIMEBenchmark(year=2024, max_samples=30)
        result = bench.evaluate(model_fn, params, state, rng)
    """

    # HuggingFace dataset for AIME problems
    DATASET_ID = "di-dimitrov/aime-problems"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
        year: Optional[int] = None,
        use_huggingface: bool = False,
    ):
        """
        Args:
            data_dir: Local cache directory
            split: Dataset split
            max_samples: Cap number of samples
            seed: Random seed
            year: Filter to specific year (None → all available years)
            use_huggingface: If True, attempt to download from HuggingFace Hub.
                            Default is False — uses built-in curated problems.
        """
        super().__init__(data_dir, split, max_samples, seed)
        self.year = year
        self.use_huggingface = use_huggingface

    @property
    def name(self) -> str:
        suffix = f"_{self.year}" if self.year else ""
        return f"aime{suffix}"

    def score_sample(
        self,
        sample: BenchmarkSample,
        prediction: Any,
    ) -> bool:
        """AIME-specific scoring: exact integer match.

        AIME answers are integers in [0, 999]. We parse the prediction
        as an integer and compare.
        """
        try:
            # Parse prediction to integer
            pred_str = str(prediction).strip()
            # Handle potential decimal answers (e.g., "42.0")
            pred_val = int(float(pred_str))
        except (ValueError, TypeError):
            # If we can't parse, try to extract digits
            digits = "".join(c for c in str(prediction) if c.isdigit())
            if digits:
                pred_val = int(digits) % 1000  # AIME range
            else:
                return False

        try:
            correct_val = int(sample.correct_answer)
        except (ValueError, TypeError):
            return False

        return pred_val == correct_val

    def load_data(self) -> List[BenchmarkSample]:
        """Load AIME problems.

        Default: uses built-in curated AIME-style problems (no network needed).
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
            logger.info(f"Loaded AIME dataset: {len(ds)} problems")
        except Exception as e:
            logger.warning(f"Could not load AIME from HuggingFace: {e}. " f"Using synthetic fallback data.")
            return self._synthetic_fallback()

        samples: List[BenchmarkSample] = []

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            # Try common column names
            problem = row.get("problem", "") or row.get("question", "") or row.get("Problem", "")
            answer = row.get("answer", "") or row.get("Answer", "") or row.get("solution", "")
            year = row.get("year", row.get("Year", None))

            # Filter by year if specified
            if self.year is not None and year is not None:
                try:
                    if int(year) != self.year:
                        continue
                except (ValueError, TypeError):
                    pass

            # Build prompt
            prompt = (
                f"Solve the following AIME problem. "
                f"The answer is an integer between 0 and 999 inclusive.\n\n"
                f"Problem: {problem}\n\n"
                f"Answer (integer only):"
            )

            category = f"aime_{year}" if year else "aime"

            samples.append(
                BenchmarkSample(
                    sample_id=f"aime_{idx}",
                    prompt=prompt,
                    choices=None,  # Open-ended (integer answer)
                    correct_answer=str(answer).strip(),
                    category=category,
                    metadata={
                        "year": year,
                        "problem_number": row.get("problem_number", idx + 1),
                        "index": idx,
                    },
                )
            )

        logger.info(f"Prepared {len(samples)} AIME problems")
        self._samples = samples
        return samples

    def _synthetic_fallback(self) -> List[BenchmarkSample]:
        """Generate synthetic AIME-style problems for testing/CI.

        Returns math problems with known integer answers.
        """
        rng = random.Random(self.seed)

        # Curated AIME-style problems with known answers
        problems = [
            (
                "Find the number of positive integers n ≤ 100 such that " "n² + n + 1 is divisible by 3.",
                "67",
            ),
            (
                "Let S be the sum of all positive integers n such that "
                "n divides 2024. Find the remainder when S is divided by 1000.",
                "640",
            ),
            (
                "How many integers between 1 and 1000 inclusive are " "divisible by neither 3 nor 5?",
                "533",
            ),
            (
                "If the sum of the first n positive integers is 5050, " "find n.",
                "100",
            ),
            (
                "Find the remainder when 7^2024 is divided by 100.",
                "1",
            ),
            (
                "How many 3-element subsets of {1,2,...,10} contain " "no two consecutive integers?",
                "56",
            ),
            (
                "Let f(x) = x³ - 3x + 1. How many real roots does " "f have?",
                "3",
            ),
            (
                "Find the last three digits of 2^100.",
                "376",
            ),
            (
                "How many ways can 12 be written as an ordered sum "
                "of positive integers where each part is at most 3?",
                "12",
            ),
            (
                "Find the number of lattice points (x,y) with " "x² + y² ≤ 25.",
                "81",
            ),
            (
                "What is the sum of the digits of 99^2?",
                "18",
            ),
            (
                "How many positive divisors does 720 have?",
                "30",
            ),
            (
                "Find the smallest prime p such that p² > 500.",
                "23",
            ),
            (
                "The perimeter of a right triangle with legs 5 and 12 is:",
                "30",
            ),
            (
                "How many perfect squares are between 100 and 999 inclusive?",
                "22",
            ),
        ]

        n = min(self.max_samples or 15, len(problems))
        selected = rng.sample(list(enumerate(problems)), n)

        samples: List[BenchmarkSample] = []
        for rank, (orig_idx, (problem, answer)) in enumerate(selected):
            prompt = (
                f"Solve the following AIME problem. "
                f"The answer is an integer between 0 and 999 inclusive.\n\n"
                f"Problem: {problem}\n\n"
                f"Answer (integer only):"
            )

            samples.append(
                BenchmarkSample(
                    sample_id=f"aime_synth_{rank}",
                    prompt=prompt,
                    choices=None,
                    correct_answer=answer,
                    category="aime_synthetic",
                    metadata={
                        "synthetic": True,
                        "original_index": orig_idx,
                    },
                )
            )

        logger.info(f"Generated {len(samples)} synthetic AIME problems")
        self._samples = samples
        return samples
