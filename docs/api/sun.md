# Sun Vectors

Computes solar position vectors for specified dates and times using Ladybug's sunpath model. Used by the `time-based` carving mode.

- **`get_sun_vectors`** -- returns unit direction vectors pointing toward the sun for each hour in the analysis period, filtered by a minimum altitude threshold (to exclude hours when the sun is too low to matter).
- **`warm_up`** -- pre-loads the EPW file and sunpath data into memory to reduce latency on the first real call. Called automatically by the daemon on startup.

::: urbansolarcarver.sun
