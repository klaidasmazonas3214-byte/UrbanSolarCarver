"""Public facade. Re-export the staged API and config helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Union

from .load_config import load_config, user_config
from .api_core import (
    preprocessing, thresholding, exporting,
    PreprocessingResult, ThresholdingResult, ExportingResult,
)


def run_pipeline(
    cfg: Union[user_config, str, Path],
    out_dir: Union[str, Path],
) -> ExportingResult:
    """Run the full carving pipeline in one call.

    Convenience wrapper around the three stages. For finer control
    (e.g. re-thresholding without re-carving), call each stage directly.

    Returns the ExportingResult with the final mesh path.
    """
    if not isinstance(cfg, user_config):
        cfg = load_config(str(cfg))
    base = Path(out_dir)
    pre = preprocessing(cfg, base / "preprocessing")
    thr = thresholding(pre, cfg, base / "thresholding")
    return exporting(thr, cfg, base / "exporting")


__all__ = [
    "run_pipeline",
    "preprocessing",
    "thresholding",
    "exporting",
    "PreprocessingResult",
    "ThresholdingResult",
    "ExportingResult",
    "load_config",
    "user_config",
]