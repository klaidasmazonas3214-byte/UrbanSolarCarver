"""USC Postprocess — Control mesh cleanup and smoothing.

Generates override strings for the exporting stage's mesh extraction.
Connect the output to USC Config's ``overrides`` input.

Inputs
------
apply_smoothing : bool, optional
    True = SDF smooth + marching cubes.  False = raw cubic mesh (default).
smooth_iters : int, optional
    Taubin polish passes (only with smoothing).  Default: 2.
min_voxels : int, optional
    Minimum connected voxel cluster to keep.  Default: 300.
min_faces : int, optional
    Minimum face count to keep mesh fragment.  Default: 100.

Outputs
-------
overrides : str
    Semicolon-joined key=value pairs. Connect to USC Export's
    ``postprocess_overrides`` input — changes here only re-run Export,
    not Preprocessing or Thresholding.
"""

try:
    ghenv.Component.Name = "USC Postprocess"
    ghenv.Component.NickName = "USC_PostProc"
    ghenv.Component.Description = "Controls mesh postprocessing: smoothing method, iteration count, and minimum cluster sizes for noise removal."
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    for i, (n, d) in enumerate([
        ("apply_smoothing", "True = smooth the voxel result using SDF + marching cubes (produces organic-looking surfaces). False = keep raw cubic voxel geometry (faster, preserves exact voxel boundaries). Default: False."),
        ("smooth_iters", "Number of Taubin mesh polishing passes (only when smoothing is enabled). More iterations = smoother but may lose detail. Default: 2. Range: 0-10."),
        ("min_voxels", "Minimum number of connected voxels in a cluster to keep. Smaller clusters are discarded as noise. Default: 300. Reduce for small-scale models."),
        ("min_faces", "Minimum number of mesh faces in a fragment to keep. Small disconnected mesh pieces below this count are removed. Default: 100."),
    ]):
        if i < len(ii):
            ii[i].Name, ii[i].Description = n, d
    oo[0].Name, oo[0].Description = "overrides", "Postprocessing settings as a semicolon-joined override string. Connect to USC_Export's 'postprocess_overrides' input — changes here will only re-run Export, not Preprocessing or Thresholding."
except Exception:
    pass

_items = []
if apply_smoothing is not None:
    _items.append(f"apply_smoothing={'true' if bool(apply_smoothing) else 'false'}")
for key, val in [
    ("smooth_iters", smooth_iters),
    ("min_voxels", min_voxels),
    ("min_face_count", min_faces),
]:
    if val is not None:
        _items.append(f"{key}={int(val)}")
overrides = ";".join(_items) if _items else None
