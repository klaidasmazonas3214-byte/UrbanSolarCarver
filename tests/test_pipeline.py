"""Integration smoke test: full pipeline on a tiny cube mesh."""
import json
import os
import pytest
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# EPW path resolution — set USC_EPW_PATH env var on any machine to point at a
# local EPW file.  No hardcoded machine-specific paths live here.
# ---------------------------------------------------------------------------
_EPW = os.environ.get("USC_EPW_PATH", "")
_EPW_AVAILABLE = bool(_EPW) and Path(_EPW).is_file()
_SKIP_EPW = pytest.mark.skipif(
    not _EPW_AVAILABLE,
    reason="EPW file not available — set USC_EPW_PATH env var to enable",
)


@_SKIP_EPW
def test_full_pipeline_benefit(tmp_path, tiny_cube_mesh, tiny_surface_mesh):
    """Run preprocessing → thresholding → exporting on a tiny cube."""
    from urbansolarcarver import load_config, preprocessing, thresholding, exporting

    # Write test meshes
    vol_path = tmp_path / "max_volume.ply"
    srf_path = tmp_path / "test_surface.ply"
    tiny_cube_mesh.export(str(vol_path))
    tiny_surface_mesh.export(str(srf_path))

    # Write a minimal config
    cfg_dict = {
        "max_volume_path": str(vol_path),
        "test_surface_path": str(srf_path),
        "epw_path": _EPW,
        "out_dir": str(tmp_path / "out"),
        "mode": "benefit",
        "voxel_size": 2.0,
        "grid_step": 2.0,
        "ray_length": 50.0,
        "ray_batch_size": 50000,
        "threshold": "carve_fraction",
        "carve_fraction": 0.5,
        "start_month": 1, "start_day": 1, "start_hour": 8,
        "end_month": 1, "end_day": 1, "end_hour": 16,
        "apply_smoothing": False,
        "min_voxels": 1,
        "min_face_count": 1,
        "device": "cpu",
    }
    import yaml
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

    cfg = load_config(str(cfg_path))
    base = tmp_path / "out"

    # Stage 1: preprocessing
    pre = preprocessing(cfg, base / "preprocessing")
    assert pre.volume_path.exists(), "scores.npy not written"
    assert (pre.out_dir / "manifest.json").exists(), "preprocessing manifest missing"
    manifest = json.loads((pre.out_dir / "manifest.json").read_text())
    assert "scores_path" in manifest
    assert "shape" in manifest

    # Stage 2: thresholding
    thr = thresholding(pre, cfg, base / "thresholding")
    assert thr.mask_path.exists(), "mask.npy not written"
    assert (thr.out_dir / "manifest.json").exists(), "thresholding manifest missing"

    # Stage 3: exporting
    result = exporting(thr, cfg, base / "exporting")
    assert result.export_path.exists(), "exported mesh not written"

    # Check diagnostics exist
    pre_diag = pre.out_dir / "diagnostics"
    assert (pre_diag / "diagnostic.json").exists(), "preprocessing diagnostics missing"
    thr_diag = thr.out_dir / "diagnostics"
    assert (thr_diag / "diagnostic.json").exists(), "thresholding diagnostics missing"

    # Check preprocessing diagnostic has expected fields
    pre_summary = json.loads((pre_diag / "diagnostic.json").read_text())
    assert "score_stats" in pre_summary
    assert "mode" in pre_summary
    assert "voxel_size" in pre_summary
    assert "timings" in pre_summary

    # Check thresholding diagnostic has expected fields
    thr_summary = json.loads((thr_diag / "diagnostic.json").read_text())
    assert "threshold_method" in thr_summary
    assert "voxels_kept" in thr_summary
    assert "retention_pct" in thr_summary
    assert "timings" in thr_summary


@_SKIP_EPW
def test_run_pipeline_convenience(tmp_path, tiny_cube_mesh, tiny_surface_mesh):
    """Test the run_pipeline() convenience wrapper."""
    from urbansolarcarver import run_pipeline, load_config

    vol_path = tmp_path / "max_volume.ply"
    srf_path = tmp_path / "test_surface.ply"
    tiny_cube_mesh.export(str(vol_path))
    tiny_surface_mesh.export(str(srf_path))

    import yaml
    cfg_dict = {
        "max_volume_path": str(vol_path),
        "test_surface_path": str(srf_path),
        "epw_path": _EPW,
        "out_dir": str(tmp_path / "out"),
        "mode": "benefit",
        "voxel_size": 2.0,
        "grid_step": 2.0,
        "ray_length": 50.0,
        "ray_batch_size": 50000,
        "threshold": "carve_fraction",
        "carve_fraction": 0.5,
        "start_month": 1, "start_day": 1, "start_hour": 8,
        "end_month": 1, "end_day": 1, "end_hour": 16,
        "apply_smoothing": False,
        "min_voxels": 1,
        "min_face_count": 1,
        "device": "cpu",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

    cfg = load_config(str(cfg_path))
    result = run_pipeline(cfg, tmp_path / "pipeline_out")
    assert result.export_path.exists()


# ---------------------------------------------------------------------------
# Score smoothing unit tests (no GPU / EPW needed)
# ---------------------------------------------------------------------------

class TestScoreSmoothing:
    """Tests for Gaussian score smoothing before thresholding."""

    def _make_preprocessing_output(self, tmp_path, scores, kind="weighted_sum", voxel_size=1.0):
        """Write a fake preprocessing output dir with scores and manifest."""
        pre_dir = tmp_path / "preprocessing"
        pre_dir.mkdir(parents=True, exist_ok=True)
        scores_path = pre_dir / "scores.npy"
        np.save(scores_path, scores, allow_pickle=False)
        from urbansolarcarver.pydantic_schemas import PreprocessingManifest, schema_to_json
        pm = PreprocessingManifest(
            hash="test1234",
            scores_path=str(scores_path),
            scores_kind=kind,
            shape=tuple(scores.shape),
            origin=(0.0, 0.0, 0.0),
            voxel_size=voxel_size,
            mode="irradiance" if kind == "weighted_sum" else "time-based",
        )
        manifest_path = pre_dir / "manifest.json"
        manifest_path.write_text(schema_to_json(pm), encoding="utf-8")
        return pre_dir

    def test_smoothing_explicit_value(self, tmp_path):
        """Explicit score_smoothing value should apply Gaussian blur."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 100, size=(16, 16, 16)).astype(np.float32)
        pre_dir = self._make_preprocessing_output(tmp_path, scores, voxel_size=0.5)

        from urbansolarcarver.load_config import user_config
        cfg = user_config(
            max_volume_path="dummy", test_surface_path="dummy", out_dir=str(tmp_path),
            mode="irradiance", epw_path="dummy",
            start_month=1, start_day=1, start_hour=0,
            end_month=12, end_day=31, end_hour=23,
            score_smoothing=1.0,  # 1m → 2 voxels at 0.5m resolution
            threshold=50.0,
        )
        from urbansolarcarver.api_core.thresholding import thresholding
        result = thresholding(pre_dir, cfg, tmp_path / "thr_out")

        diag = json.loads((result.out_dir / "diagnostics" / "diagnostic.json").read_text())
        assert diag["score_smooth_applied"] is True
        assert diag["score_smoothing_sigma_voxels"] > 0

    def test_smoothing_auto_default(self, tmp_path):
        """score_smoothing=None should auto-compute as 1.1 × voxel_size."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 100, size=(16, 16, 16)).astype(np.float32)
        pre_dir = self._make_preprocessing_output(tmp_path, scores, voxel_size=2.0)

        from urbansolarcarver.load_config import user_config
        cfg = user_config(
            max_volume_path="dummy", test_surface_path="dummy", out_dir=str(tmp_path),
            mode="irradiance", epw_path="dummy",
            start_month=1, start_day=1, start_hour=0,
            end_month=12, end_day=31, end_hour=23,
            # score_smoothing left as None (default)
            threshold=50.0,
        )
        from urbansolarcarver.api_core.thresholding import thresholding
        result = thresholding(pre_dir, cfg, tmp_path / "thr_out_auto")

        diag = json.loads((result.out_dir / "diagnostics" / "diagnostic.json").read_text())
        assert diag["score_smooth_applied"] is True
        # Auto: 1.1 * 2.0 = 2.2m → 2.2/2.0 = 1.1 voxels
        assert 1.0 <= diag["score_smoothing_sigma_voxels"] <= 1.2
        assert abs(diag["score_smoothing_m"] - 2.2) < 0.01

    def test_smoothing_skipped_for_violation_count(self, tmp_path):
        """Violation-count scores (time-based) should NOT be smoothed."""
        scores = np.ones((8, 8, 8), dtype=np.float32) * 3.0
        pre_dir = self._make_preprocessing_output(tmp_path, scores, kind="violation_count", voxel_size=1.0)

        from urbansolarcarver.load_config import user_config
        cfg = user_config(
            max_volume_path="dummy", test_surface_path="dummy", out_dir=str(tmp_path),
            mode="time-based", epw_path="dummy",
            start_month=1, start_day=1, start_hour=0,
            end_month=12, end_day=31, end_hour=23,
            score_smoothing=2.0,
            threshold=0,
        )
        from urbansolarcarver.api_core.thresholding import thresholding
        result = thresholding(pre_dir, cfg, tmp_path / "thr_out2")

        diag = json.loads((result.out_dir / "diagnostics" / "diagnostic.json").read_text())
        assert diag["score_smooth_applied"] is False

    def test_smoothing_zero_disables(self, tmp_path):
        """score_smoothing=0 should disable smoothing entirely."""
        rng = np.random.default_rng(99)
        scores = rng.uniform(0, 50, size=(8, 8, 8)).astype(np.float32)
        pre_dir = self._make_preprocessing_output(tmp_path, scores, voxel_size=1.0)

        from urbansolarcarver.load_config import user_config
        cfg = user_config(
            max_volume_path="dummy", test_surface_path="dummy", out_dir=str(tmp_path),
            mode="irradiance", epw_path="dummy",
            start_month=1, start_day=1, start_hour=0,
            end_month=12, end_day=31, end_hour=23,
            score_smoothing=0.0,
            threshold=25.0,
        )
        from urbansolarcarver.api_core.thresholding import thresholding
        result = thresholding(pre_dir, cfg, tmp_path / "thr_out3")

        diag = json.loads((result.out_dir / "diagnostics" / "diagnostic.json").read_text())
        assert diag["score_smooth_applied"] is False
        assert diag["score_smoothing_sigma_voxels"] == 0.0
