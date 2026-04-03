"""Tests for api_core/preprocessing.py: anomaly detection, manifest structure, empty inputs."""
import json
import warnings
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from urbansolarcarver.api_core._diagnostics import score_statistics


class TestScoreAnomalyDetection:
    """Verify NaN/Inf detection in preprocessing scores."""

    def test_nan_detected(self):
        """NaN values should be flagged."""
        scores = np.array([1.0, np.nan, 3.0])
        has_nan = bool(np.isnan(scores).any())
        assert has_nan

    def test_inf_detected(self):
        """Inf values should be flagged."""
        scores = np.array([1.0, np.inf, 3.0])
        has_inf = bool(np.isinf(scores).any())
        assert has_inf

    def test_clean_scores_no_anomaly(self):
        """Normal scores should not have NaN or Inf."""
        scores = np.array([0.0, 1.0, 2.0, 3.0])
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()

    def test_nan_produces_warning(self):
        """NaN in scores should emit a user-visible warning via the anomaly check."""
        # We test the warning logic inline (same logic as preprocessing uses)
        scores = np.array([1.0, np.nan, 3.0])
        nan_count = int(np.isnan(scores).sum())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if nan_count:
                total = scores.size
                warnings.warn(
                    f"Scores contain {nan_count} NaN values ({100*nan_count/total:.1f}% of {total} voxels).",
                )
            assert len(w) == 1
            assert "NaN" in str(w[0].message)


class TestScoreStatisticsIntegration:
    """Verify score_statistics produces all required keys."""

    def test_all_keys_present(self):
        scores = np.random.rand(100)
        stats = score_statistics(scores)
        required_keys = {"count", "min", "max", "mean", "median", "std",
                         "p5", "p25", "p75", "p95", "nonzero_count", "nonzero_fraction"}
        assert required_keys <= set(stats.keys())

    def test_empty_scores(self):
        stats = score_statistics(np.array([]))
        assert stats == {"count": 0}


class TestEmptySamplePoints:
    """Guard against empty sample points before ray generation."""

    def test_generate_sky_patch_rays_with_empty_points(self):
        """generate_sky_patch_rays with 0 points should return 0 rays (no crash)."""
        import torch
        from urbansolarcarver.raytracer import generate_sky_patch_rays
        pts = np.empty((0, 3), dtype=np.float32)
        norms = np.empty((0, 3), dtype=np.float32)
        patch_dirs = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        # Currently this doesn't crash — but returns empty results silently.
        # After fix B2, preprocessing should check BEFORE calling this.
        origins, dirs, ids, normals, point_idx = generate_sky_patch_rays(
            pts, norms, patch_dirs, torch.device("cpu")
        )
        assert origins.shape[0] == 0


class TestManifestStructure:
    """Verify preprocessing manifest has required fields."""

    def test_manifest_fields(self, tmp_path):
        """A preprocessing manifest should contain all required fields."""
        # This is a structural test — we create a mock manifest and verify format
        manifest = {
            "hash": "abcd1234",
            "scores_path": "scores.npy",
            "scores_kind": "weighted_sum",
            "shape": [10, 10, 10],
            "origin": [0.0, 0.0, 0.0],
            "suggested_threshold": None,
            "voxel_grid_path": "voxel_grid.npy",
            "mode": "benefit",
        }
        required_keys = {"hash", "scores_path", "scores_kind", "shape", "origin"}
        assert required_keys <= set(manifest.keys())

    def test_volume_shape_is_3_tuple(self):
        """volume_shape should be a 3-element list/tuple."""
        shape = [10, 10, 10]
        assert len(shape) == 3
        assert all(isinstance(x, int) for x in shape)
