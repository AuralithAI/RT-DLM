import math

import numpy as np
import pytest

from src.core.observability.goodput import ExpertUtilizationTracker, GoodputTracker


def test_goodput_basic_ratio():
    t = GoodputTracker(window=10)
    t.record(useful_seconds=5.0, wall_seconds=10.0)
    assert t.goodput() == pytest.approx(0.5)


def test_goodput_averages_over_window():
    t = GoodputTracker(window=10)
    t.record(2.0, 4.0)
    t.record(3.0, 6.0)
    assert t.goodput() == pytest.approx(5.0 / 10.0)


def test_goodput_empty_returns_zero():
    t = GoodputTracker(window=10)
    assert t.goodput() == pytest.approx(0.0)


def test_goodput_window_eviction():
    t = GoodputTracker(window=3)
    for _ in range(5):
        t.record(1.0, 2.0)
    summary = t.summary()
    assert summary["samples"] == pytest.approx(3.0)
    assert summary["wall_seconds_total"] == pytest.approx(6.0)


def test_goodput_rejects_nonpositive_wall():
    t = GoodputTracker()
    with pytest.raises(ValueError):
        t.record(0.0, 0.0)
    with pytest.raises(ValueError):
        t.record(1.0, -1.0)


def test_goodput_rejects_useful_gt_wall():
    t = GoodputTracker()
    with pytest.raises(ValueError):
        t.record(5.0, 1.0)


def test_goodput_rejects_negative_useful():
    t = GoodputTracker()
    with pytest.raises(ValueError):
        t.record(-0.1, 1.0)


def test_goodput_reset():
    t = GoodputTracker()
    t.record(1.0, 2.0)
    t.reset()
    assert t.goodput() == pytest.approx(0.0)
    assert t.summary()["samples"] == pytest.approx(0.0)


def test_expert_util_rejects_invalid_num_experts():
    with pytest.raises(ValueError):
        ExpertUtilizationTracker(num_experts=0)


def test_expert_util_uniform_entropy():
    n = 4
    t = ExpertUtilizationTracker(num_experts=n)
    t.update(np.array([0, 1, 2, 3] * 25, dtype=np.int64))
    p = t.utilization()
    assert np.allclose(p, np.full(n, 1.0 / n))
    assert t.entropy() == pytest.approx(math.log(n), rel=1e-6)
    assert t.imbalance() == pytest.approx(0.0)


def test_expert_util_one_hot_zero_entropy():
    t = ExpertUtilizationTracker(num_experts=4)
    t.update(np.zeros(100, dtype=np.int64))
    assert t.entropy() == pytest.approx(0.0)
    assert t.imbalance() == pytest.approx(1.0)


def test_expert_util_empty_state():
    t = ExpertUtilizationTracker(num_experts=3)
    assert np.allclose(t.utilization(), 0.0)
    assert t.entropy() == pytest.approx(0.0)
    s = t.summary()
    assert s["total_tokens"] == pytest.approx(0.0)


def test_expert_util_empty_update_is_noop():
    t = ExpertUtilizationTracker(num_experts=3)
    t.update(np.array([], dtype=np.int64))
    assert t.summary()["total_tokens"] == pytest.approx(0.0)


def test_expert_util_accumulates_across_updates():
    t = ExpertUtilizationTracker(num_experts=2)
    t.update(np.array([0, 0, 0]))
    t.update(np.array([1]))
    p = t.utilization()
    assert p[0] == pytest.approx(0.75)
    assert p[1] == pytest.approx(0.25)


def test_expert_util_reset():
    t = ExpertUtilizationTracker(num_experts=2)
    t.update(np.array([0, 1, 1]))
    t.reset()
    assert t.summary()["total_tokens"] == pytest.approx(0.0)
    assert np.allclose(t.utilization(), 0.0)


def test_expert_util_handles_2d_input():
    t = ExpertUtilizationTracker(num_experts=3)
    t.update(np.array([[0, 1], [2, 1]]))
    p = t.utilization()
    assert p[1] == pytest.approx(0.5)
    assert p[0] == pytest.approx(0.25)
    assert p[2] == pytest.approx(0.25)
