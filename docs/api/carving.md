# Carving

The carving module implements the three geometric strategies used to remove voxels from the maximum buildable volume:

- **Sun-ray carving** (`carve_with_sun_rays`) -- traces rays along actual sun vectors at specific dates/times. Used by the `time-based` mode.
- **Sky-patch carving** (`carve_with_sky_patch_rays`) -- traces rays toward Tregenza sky patches weighted by irradiance, heating benefit, daylight luminance, or radiative cooling potential. Used by `irradiance`, `benefit`, `daylight`, and `radiative_cooling` modes.
- **Plane carving** (`carve_with_planes`) -- removes voxels above a tilted cut plane at a fixed angle from each test surface. Used by the `tilted_plane` mode.

All three accept a voxel grid, surface sample points with normals, and mode-specific parameters. They return a per-voxel score array that downstream thresholding converts into a binary carving mask.

::: urbansolarcarver.carving
