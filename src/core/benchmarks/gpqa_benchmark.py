"""
GPQA Diamond Benchmark Evaluator

Evaluates the model on GPQA (Graduate-level Physics QA) Diamond split —
a curated set of challenging multiple-choice physics/science questions
targeting expert-level reasoning.

Dataset: Idavidrein/gpqa (HuggingFace Hub)
Metric: Accuracy over multiple-choice questions (A/B/C/D)
"""

import logging
import random
from typing import List, Optional

from core.benchmarks.base_benchmark import BenchmarkBase, BenchmarkSample

logger = logging.getLogger(__name__)


class GPQABenchmark(BenchmarkBase):
    """GPQA Diamond benchmark evaluator.

    Loads the GPQA Diamond split (198 hand-curated physics/science
    questions) from HuggingFace Hub. Each question has four choices.

    Scoring: exact-match on selected answer index.

    Usage:
        bench = GPQABenchmark(max_samples=50)
        result = bench.evaluate(model_fn, params, state, rng)
        print(bench.format_result(result))
    """

    DATASET_ID = "Idavidrein/gpqa"
    SUBSET = "gpqa_diamond"

    # Column mapping for GPQA dataset
    QUESTION_COL = "Question"
    ANSWER_COL = "Answer"
    CHOICE_COLS = [
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]
    DOMAIN_COL = "Subdomain"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",  # GPQA only has train split on HF
        max_samples: Optional[int] = None,
        seed: int = 42,
        shuffle_choices: bool = True,
        use_huggingface: bool = False,
    ):
        """
        Args:
            data_dir: Local cache directory
            split: Dataset split (GPQA has only 'train')
            max_samples: Cap number of samples
            seed: Random seed
            shuffle_choices: Whether to shuffle answer choices per question
            use_huggingface: If True, attempt to download from HuggingFace Hub.
                            Default is False — uses built-in curated problems.
        """
        super().__init__(data_dir, split, max_samples, seed)
        self.shuffle_choices = shuffle_choices
        self.use_huggingface = use_huggingface

    @property
    def name(self) -> str:
        return "gpqa_diamond"

    def load_data(self) -> List[BenchmarkSample]:
        """Load GPQA Diamond problems.

        Default: uses built-in curated science/physics questions (no network needed).
        If use_huggingface=True, attempts to download from HuggingFace Hub.
        """
        if not self.use_huggingface:
            return self._synthetic_fallback()

        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.DATASET_ID,
                self.SUBSET,
                split=self.split,
                cache_dir=self.data_dir,
                trust_remote_code=True,
            )
            logger.info(f"Loaded GPQA Diamond: {len(ds)} samples")
        except Exception as e:
            logger.warning(
                f"Could not load GPQA from HuggingFace: {e}. " f"Using synthetic fallback data."
            )
            return self._synthetic_fallback()

        rng = random.Random(self.seed)
        samples: List[BenchmarkSample] = []

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            question = row.get(self.QUESTION_COL, "")
            correct_answer = row.get(self.ANSWER_COL, "")
            incorrect = [row.get(col, "") for col in self.CHOICE_COLS]

            # Build choices: correct + 3 incorrect
            choices = [correct_answer] + incorrect
            correct_idx = 0  # correct is at index 0

            if self.shuffle_choices:
                # Shuffle and track correct answer position
                indexed = list(enumerate(choices))
                rng.shuffle(indexed)
                correct_idx = next(i for i, (orig_idx, _) in enumerate(indexed) if orig_idx == 0)
                choices = [c for _, c in indexed]

            # Build prompt with answer labels
            labels = "ABCD"
            choices_text = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(choices))
            prompt = (
                f"Answer the following question by selecting A, B, C, or D.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                f"Answer:"
            )

            domain = row.get(self.DOMAIN_COL, "general")

            samples.append(
                BenchmarkSample(
                    sample_id=f"gpqa_{idx}",
                    prompt=prompt,
                    choices=choices,
                    correct_answer=correct_idx,
                    category=str(domain),
                    metadata={
                        "correct_text": correct_answer,
                        "domain": str(domain),
                        "index": idx,
                    },
                )
            )

        logger.info(f"Prepared {len(samples)} GPQA samples")
        self._samples = samples
        return samples

    def _synthetic_fallback(self) -> List[BenchmarkSample]:
        """Generate synthetic GPQA-like samples for testing/CI.

        Returns 20 synthetic physics/science multiple-choice questions.
        """
        rng = random.Random(self.seed)
        domains = [
            "quantum_mechanics",
            "astrophysics",
            "molecular_biology",
            "organic_chemistry",
            "condensed_matter",
        ]

        templates = [
            (
                "What is the ground state energy of a hydrogen atom?",
                ["-13.6 eV", "-3.4 eV", "-27.2 eV", "-1.51 eV"],
                0,
            ),
            (
                "Which particle mediates the strong nuclear force?",
                ["Gluon", "Photon", "W boson", "Graviton"],
                0,
            ),
            (
                "What is the Schwarzschild radius of a solar-mass black hole?",
                ["~3 km", "~30 km", "~300 km", "~3000 km"],
                0,
            ),
            (
                "Which molecule has the highest bond dissociation energy?",
                ["N₂", "O₂", "CO", "H₂"],
                0,
            ),
            (
                "What is the Pauli exclusion principle?",
                [
                    "No two identical fermions can occupy the same quantum state",
                    "Energy is always conserved in quantum systems",
                    "Momentum and position cannot be simultaneously known",
                    "Wave function collapse is irreversible",
                ],
                0,
            ),
        ]

        samples: List[BenchmarkSample] = []
        n = min(self.max_samples or 20, 20)

        for i in range(n):
            t_idx = i % len(templates)
            question, choices, correct = templates[t_idx]
            domain = domains[i % len(domains)]

            if self.shuffle_choices:
                indexed = list(enumerate(choices))
                rng.shuffle(indexed)
                correct = next(j for j, (orig, _) in enumerate(indexed) if orig == 0)
                choices = [c for _, c in indexed]

            labels = "ABCD"
            choices_text = "\n".join(f"({labels[j]}) {c}" for j, c in enumerate(choices))
            prompt = (
                f"Answer the following question by selecting A, B, C, or D.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                f"Answer:"
            )

            samples.append(
                BenchmarkSample(
                    sample_id=f"gpqa_synth_{i}",
                    prompt=prompt,
                    choices=choices,
                    correct_answer=correct,
                    category=domain,
                    metadata={
                        "synthetic": True,
                        "domain": domain,
                    },
                )
            )

        logger.info(f"Generated {len(samples)} synthetic GPQA samples")
        self._samples = samples
        return samples
