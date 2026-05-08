"""Tests for MultimodalEvaluator and ModalityInterferenceReport."""

import unittest

import jax

from src.core.benchmark_evaluation import (
    ModalityInterferenceReport,
    MultimodalEvaluator,
)


def _make_task(score: float):
    def task(_apply_fn, _params, _rng, _sample):
        return {"score": score, "n": 1}

    return task


class TestMultimodalEvaluator(unittest.TestCase):
    def test_solo_vs_joint_interference(self):
        evaluator = MultimodalEvaluator(model_apply_fn=lambda *a, **kw: None)
        rng = jax.random.PRNGKey(0)

        modality_tasks = {
            "vision": _make_task(1.0),  # solo wraps below; we override per-call
            "audio": _make_task(0.8),
        }
        # Use distinct samples lists; scoring driven by closures.
        solo_samples = {"vision": [{}] * 4, "audio": [{}] * 4}
        joint_samples = {"vision": [{}] * 4, "audio": [{}] * 4}

        # Override: solo=1.0/0.8, joint=0.6/0.7 by swapping task fns mid-flight.
        # Easier: build two evaluators or pre-bake scoring.
        def vision_task(apply_fn, params, rng_, sample):
            return {"score": sample["score"], "n": 1}

        for s in solo_samples["vision"]:
            s["score"] = 1.0
        for s in joint_samples["vision"]:
            s["score"] = 0.6
        for s in solo_samples["audio"]:
            s["score"] = 0.8
        for s in joint_samples["audio"]:
            s["score"] = 0.7

        modality_tasks = {"vision": vision_task, "audio": vision_task}

        report = evaluator.evaluate_with_interference(
            params={},
            rng=rng,
            modality_tasks=modality_tasks,
            solo_samples=solo_samples,
            joint_samples=joint_samples,
        )
        self.assertIsInstance(report, ModalityInterferenceReport)
        self.assertAlmostEqual(report.per_modality_solo["vision"], 1.0, places=4)
        self.assertAlmostEqual(report.per_modality_joint["vision"], 0.6, places=4)
        self.assertAlmostEqual(report.interference["vision"], 0.4, places=4)
        self.assertAlmostEqual(report.interference["audio"], 0.1, places=4)
        self.assertGreater(report.aggregate_interference, 0.0)

    def test_run_benchmark_suite(self):
        evaluator = MultimodalEvaluator(model_apply_fn=lambda *a, **kw: None)
        rng = jax.random.PRNGKey(1)
        suite = {
            "vision:cls": (_make_task(0.9), [{}] * 5),
            "audio:cls": (_make_task(0.7), [{}] * 5),
        }
        results = evaluator.run_benchmark_suite(params={}, rng=rng, suite=suite)
        self.assertEqual(set(results.keys()), {"vision:cls", "audio:cls"})
        self.assertAlmostEqual(results["vision:cls"].accuracy, 0.9, places=4)
        self.assertAlmostEqual(results["audio:cls"].accuracy, 0.7, places=4)


if __name__ == "__main__":
    unittest.main()
