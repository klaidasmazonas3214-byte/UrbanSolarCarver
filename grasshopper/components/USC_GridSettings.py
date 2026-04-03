"""USC Grid Settings — Control voxel grid resolution and padding.

Inputs
------
voxel_size : float, optional
    Voxel edge length in meters. Smaller = finer but slower. Default: 1.0
grid_step : float, optional
    Surface sampling spacing in meters. Default: 1.0
margin_frac : float, optional
    Padding fraction around geometry bounding box. Default: 0.01

Outputs
-------
overrides : list of str
    Key=value pairs for grid parameters.
"""

try:
    ghenv.Component.Name = "USC Grid Settings"
    ghenv.Component.NickName = "USC_Grid"
    ghenv.Component.Description = "Controls voxel grid resolution and bounding-box padding. Smaller voxel_size = finer detail but slower. Start coarse (2-3 m) and refine."
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    for i, (n, d) in enumerate([
        ("voxel_size", "Edge length of each voxel cube in meters. Smaller values = finer resolution but slower computation. Start with 2.0 for testing, use 1.0 or 0.5 for final results. Typical range: 0.5 to 5.0 m."),
        ("grid_step", "Spacing between sample points on the test surfaces, in meters. Controls how densely rays are cast. Should be similar to or smaller than voxel_size. Typical: 1.0 m."),
        ("margin_frac", "Padding added around the geometry as a fraction of the bounding box. Prevents edge artifacts. Default 0.01 (1%). Increase to 0.05 if you see clipping at boundaries."),
    ]):
        if i < len(ii):
            ii[i].Name, ii[i].Description = n, d
    oo[0].Name, oo[0].Description = "overrides", "Grid settings formatted for USC_Config. Connect to Config's 'overrides' input."
except Exception:
    pass

_items = []
for key, val in [
    ("voxel_size", voxel_size),
    ("grid_step", grid_step),
    ("margin_frac", margin_frac),
]:
    if val is not None:
        _items.append(f"{key}={float(val)}")
overrides = ";".join(_items) if _items else None
