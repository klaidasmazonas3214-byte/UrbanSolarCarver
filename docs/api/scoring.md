# Scoring

Handles the conversion from raw per-voxel obstruction counts to a binary carving decision:

- **Weight dispatch** (`get_weights`) -- routes to the correct sky-patch weighting function based on the selected mode (Perez irradiance, heating benefit, CIE daylight luminance, or radiative cooling).
- **Normalization** (`normalize_scores`) -- rescales raw scores to [0, 1] using min-max or other strategies.
- **Thresholding** -- three automatic methods:
    - `otsu_threshold` -- minimizes intra-class variance (good general default).
    - `headtail_threshold` -- iterative head-tail breaks for heavy-tailed score distributions.
    - `carve_fraction` -- removes a fixed percentage of the scored volume.

::: urbansolarcarver.scoring
