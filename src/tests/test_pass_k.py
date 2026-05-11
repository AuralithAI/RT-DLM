import math

import pytest

from src.core.benchmarks.pass_k import (
    aggregate_pass_at_k,
    best_of_n,
    maj_at_k,
    pass_at_k,
)


def test_pass_at_k_zero_correct():
    assert pass_at_k(0, 10, 1) == pytest.approx(0.0)


def test_pass_at_k_all_correct():
    assert pass_at_k(10, 10, 1) == pytest.approx(1.0)


def test_pass_at_k_short_circuit_when_few_failures():
    assert pass_at_k(8, 10, 3) == pytest.approx(1.0)


def test_pass_at_k_half_correct_k1():
    assert pass_at_k(5, 10, 1) == pytest.approx(0.5)


def test_pass_at_k_known_value_c1_n10_k5():
    expected = 1.0 - math.comb(9, 5) / math.comb(10, 5)
    assert pass_at_k(1, 10, 5) == pytest.approx(expected, rel=1e-9)


def test_pass_at_k_monotone_in_k():
    vals = [pass_at_k(3, 20, k) for k in (1, 2, 5, 10)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_pass_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        pass_at_k(1, 10, 0)
    with pytest.raises(ValueError):
        pass_at_k(1, 10, -1)


def test_pass_at_k_rejects_bad_n():
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_pass_at_k_rejects_correct_out_of_range():
    with pytest.raises(ValueError):
        pass_at_k(-1, 10, 1)
    with pytest.raises(ValueError):
        pass_at_k(11, 10, 1)


def test_maj_at_k_majority_wins():
    assert maj_at_k(["a", "a", "b"], "a") == pytest.approx(1.0)


def test_maj_at_k_minority_loses():
    assert maj_at_k(["a", "a", "b"], "b") == pytest.approx(0.0)


def test_maj_at_k_empty_raises():
    with pytest.raises(ValueError):
        maj_at_k([], "a")


def test_best_of_n_picks_highest_scorer():
    cands = ["a", "bb", "ccc"]
    score = len
    assert best_of_n(cands, score, "ccc") == pytest.approx(1.0)
    assert best_of_n(cands, score, "a") == pytest.approx(0.0)


def test_best_of_n_with_is_correct_predicate():
    cands = [1, 2, 3]
    assert best_of_n(cands, lambda x: x, gold=10, is_correct=lambda b, g: b * 3 == g + 1) == pytest.approx(0.0)
    assert best_of_n(cands, lambda x: x, gold=3, is_correct=lambda b, g: b == g) == pytest.approx(1.0)


def test_best_of_n_empty_raises():
    with pytest.raises(ValueError):
        best_of_n([], len, "x")


def test_aggregate_pass_at_k_averages_across_problems():
    results = [(10, 10), (0, 10), (5, 10)]
    out = aggregate_pass_at_k(results, ks=[1])
    assert out["pass@1"] == pytest.approx((1.0 + 0.0 + 0.5) / 3.0)


def test_aggregate_pass_at_k_multiple_ks():
    results = [(2, 10), (3, 10)]
    out = aggregate_pass_at_k(results, ks=[1, 5])
    assert "pass@1" in out and "pass@5" in out
    assert 0.0 <= out["pass@1"] <= 1.0
    assert out["pass@5"] >= out["pass@1"] - 1e-12


def test_aggregate_pass_at_k_empty_raises():
    with pytest.raises(ValueError):
        aggregate_pass_at_k([], ks=[1])
