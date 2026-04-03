# Urban Solar Carver

GPU-accelerated generation of maximum developable building volumes that protect solar and daylight access in urban environments.

USC follows a **subtractive logic**: a theoretical maximum volume is progressively carved by casting rays from protected surfaces toward the sky, removing voxels that obstruct solar/daylight access beyond a configurable threshold.

![USC pipeline overview](assets/explanation.png)

## Overview

- **6 carving modes** covering solar access, passive heating, daylighting, and radiative cooling
- **3-stage pipeline** (preprocessing, thresholding, exporting) -- each stage independently re-runnable
- **GPU-accelerated** ray tracing via NVIDIA Warp (DDA kernel), with CPU fallback
- **Grasshopper plugin** (15 components) for Rhino 3D integration
- **Python API**, **CLI**, and **Jupyter notebooks**

## Pipeline

```
Input (PLY meshes, EPW weather, YAML config)
    |
    v
Preprocessing ── scores.npy, voxel_grid.npy
    |
    v
Thresholding ─── mask.npy (binary carving mask)
    |
    v
Exporting ────── carved_mesh.ply
```

Each stage reads the previous stage's manifest, so you can re-run thresholding with different parameters without re-computing scores.

## Documentation

- **Getting Started**
    - [Installation](getting-started/installation.md) -- setup via `setup_env.py` or manual install
    - [Quick Start](getting-started/quick-start.md) -- Python API, CLI, and notebook tutorials
    - [Configuration](getting-started/configuration.md) -- YAML config reference with per-mode examples
- **[Carving Modes](modes.md)** -- the 6 analysis modes explained (tilted_plane, time-based, irradiance, benefit, daylight, radiative_cooling)
- **[Grasshopper Plugin](grasshopper.md)** -- Rhino 3D integration, component list, daemon setup
- **[Project Map](project-map.md)** -- every file and directory in the repository
- **[API Reference](api/index.md)** -- full function signatures, parameters, and return types (auto-generated from docstrings)
- **[Changelog](https://github.com/avarth/UrbanSolarCarver/blob/main/CHANGELOG.md)**
