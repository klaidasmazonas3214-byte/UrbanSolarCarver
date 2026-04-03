# Pipeline

The pipeline is USC's main execution model. It runs in three stages -- preprocessing, thresholding, exporting -- each producing a manifest that the next stage consumes. This design lets you re-run thresholding with different parameters without recomputing scores.

`run_pipeline()` chains all three stages in one call. For finer control, call each stage individually.

```python
from urbansolarcarver import load_config, run_pipeline

# One-shot
result = run_pipeline(load_config("config.yaml"), out_dir="outputs")

# Decomposed
from urbansolarcarver import preprocessing, thresholding, exporting
from pathlib import Path

cfg = load_config("config.yaml")
pre  = preprocessing(cfg, Path("outputs/preprocessing"))
thr  = thresholding(pre, cfg, Path("outputs/thresholding"))
exp  = exporting(thr, cfg, Path("outputs/exporting"))
```

## run_pipeline

::: urbansolarcarver.api
    options:
      members: [run_pipeline]

## Preprocessing

Voxelizes the max volume, samples the test surfaces, generates rays toward sky patches (or sun vectors), traces them through the voxel grid, and writes per-voxel obstruction scores to `scores.npy`.

::: urbansolarcarver.api_core.preprocessing
    options:
      members: [preprocessing, PreprocessingResult]

## Thresholding

Converts the continuous score array into a binary carving mask. Supports head-tail breaks, a fixed carve-fraction, or a numeric threshold. Writes `mask.npy`.

::: urbansolarcarver.api_core.thresholding
    options:
      members: [thresholding, ThresholdingResult]

## Exporting

Reconstructs a triangle mesh from the binary mask -- either cubic voxel faces or SDF-smoothed marching cubes with Taubin polishing. Writes the final `carved_mesh.ply`.

::: urbansolarcarver.api_core.exporting
    options:
      members: [exporting, ExportingResult]
