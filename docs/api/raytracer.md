# Ray Tracer

The ray-tracing engine at the core of USC's carving pipeline. Implements a fused DDA (Digital Differential Analyser) kernel that traverses rays through the voxel grid and accumulates per-voxel hit scores in a single pass.

- **`trace_and_score_dda`** -- the main workhorse. Traces batches of rays through the grid, incrementing each voxel's score by the ray's weight on every intersection. Runs on GPU via Warp or falls back to CPU.
- **`generate_sky_patch_rays`** / **`generate_sun_rays`** -- expand sample points × directions into flat ray arrays (origins + directions) ready for batch tracing.
- **`auto_batch_size`** -- estimates the maximum ray batch that fits in GPU memory given the grid resolution, to avoid OOM errors on large scenes.

Rays are processed in batches whose size adapts to available VRAM. The batch loop is the primary computational bottleneck; see [Performance](../getting-started/configuration.md) for tuning guidance.

::: urbansolarcarver.raytracer
