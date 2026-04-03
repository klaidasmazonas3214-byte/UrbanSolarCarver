"""Unit tests for scoring: head-tail threshold."""
import numpy as np
import pytest
from urbansolarcarver.scoring import headtail_threshold


# --- Head-tail threshold ---

def test_headtail_empty():
    result = headtail_threshold(np.array([]))
    assert result == 0.0


def test_headtail_uniform():
    """Uniform distribution should converge to a reasonable value."""
    scores = np.arange(100, dtype=float)
    thr = headtail_threshold(scores, max_iterations=20)
    assert 0 < thr < 100


def test_headtail_single():
    result = headtail_threshold(np.array([7.0]))
    assert result == pytest.approx(7.0)


def test_headtail_heavy_tail():
    """Heavy-tailed distribution: threshold should be above the median."""
    rng = np.random.default_rng(42)
    bulk = rng.uniform(0, 10, 900)
    tail = rng.uniform(50, 100, 100)
    scores = np.concatenate([bulk, tail])
    thr = headtail_threshold(scores)
    assert thr > np.median(scores), "Head-tail should split above median for heavy-tailed data"


def test_headtail_all_equal():
    """All-equal scores should return that value (no head to split)."""
    scores = np.full(200, 5.0)
    result = headtail_threshold(scores)
    assert result == pytest.approx(5.0)
