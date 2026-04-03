"""USC Diagnostics — Enable/disable diagnostic output for all pipeline stages.

When enabled, each stage writes score statistics and wall/CPU timings to
a ``diagnostics/`` subdirectory alongside the stage artifacts.

Optionally, plot images (histograms, sky patch visualizations) can be
generated.  These are rendered in a background thread so they do not
block the pipeline, but they do add matplotlib overhead on the first
call.  Image paths are surfaced by the Preprocess and ThresholdStage
components via their ``diag_images`` output, which can be fed directly
to Ladybug's Image Viewer component.

Inputs
------
enable : bool
    True = write JSON diagnostics (stats, timings).
    False = skip diagnostics entirely (faster). Default: False.
plots : bool, optional
    True = also generate diagnostic plot images (score histograms,
    sky patch weight/intensity plots, threshold histograms).
    Plots render in a background thread and do not block the pipeline.
    Default: False (no matplotlib overhead).

Outputs
-------
overrides : str
    Override string for USC Config.
    Connect to the ``overrides`` input of USC Config.
"""

try:
    ghenv.Component.Name = "USC Diagnostics"
    ghenv.Component.NickName = "USC_Diag"
    ghenv.Component.Description = "Toggles diagnostic output (statistics, timings) and optional plot images (histograms, sky dome plots). Plots render in a background thread. Enable for inspection and validation. Disable for faster production runs."
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    if len(ii) > 0:
        ii[0].Name, ii[0].Description = "enable", "True = write JSON diagnostic outputs (statistics, timings) for each pipeline stage. False = skip diagnostics for faster execution. Default: False."
    if len(ii) > 1:
        ii[1].Name, ii[1].Description = "plots", "True = also generate diagnostic plot images (score histograms, sky dome plots). Rendered in a background thread — does not block the pipeline. Default: False (no matplotlib overhead)."
    if len(oo) > 0:
        oo[0].Name, oo[0].Description = "overrides", "Diagnostics toggle formatted for USC_Config. Connect to Config's 'overrides' input."
except Exception:
    pass

parts = []
if enable is not None:
    parts.append(f"diagnostics={'true' if bool(enable) else 'false'}")
if plots is not None:
    parts.append(f"diagnostic_plots={'true' if bool(plots) else 'false'}")
elif enable is not None:
    # When plots is not connected, follow the enable toggle
    parts.append(f"diagnostic_plots={'true' if bool(enable) else 'false'}")

overrides = ";".join(parts) if parts else None
