"""USC Analysis Period — Define the time window for sun/sky analysis.

Required for: time-based, irradiance, benefit, radiative_cooling modes.
Optional for: daylight (CIE overcast is time-invariant; defaults to full year).
Not used by: tilted_plane.

Inputs
------
start_month : int (1-12)
start_day : int (1-31)
start_hour : int (0-23)
end_month : int (1-12)
end_day : int (1-31)
end_hour : int (0-23)

Outputs
-------
overrides : list of str
    Key=value pairs for the analysis period.
"""

try:
    ghenv.Component.Name = "USC Analysis Period"
    ghenv.Component.NickName = "USC_Period"
    ghenv.Component.Description = "Defines the time window for solar analysis. Required for time-based, irradiance, benefit, and radiative_cooling modes. Optional for daylight (time-invariant). Not needed for tilted_plane."
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    for i, (n, d) in enumerate([
        ("start_month", "First month of the analysis window (1=January, 12=December). For heating-season studies, try 10 (October)."),
        ("start_day", "First day of the start month (1-31). Use 1 for the beginning of the month."),
        ("start_hour", "First hour of day to include (0-23, where 0=midnight, 12=noon). For daytime-only analysis, use 7 or 8."),
        ("end_month", "Last month of the analysis window (1-12). For heating-season studies, try 3 (March) or 4 (April)."),
        ("end_day", "Last day of the end month (1-31). Use 31 for the end of the month."),
        ("end_hour", "Last hour of day to include (0-23). For daytime-only analysis, use 17 or 18."),
    ]):
        if i < len(ii):
            ii[i].Name, ii[i].Description = n, d
    oo[0].Name, oo[0].Description = "overrides", "Time period settings formatted for USC_Config. Connect to Config's 'overrides' input."
except Exception:
    pass

# Output a single semicolon-delimited string so GH doesn't iterate
_items = []
for key, val in [
    ("start_month", start_month), ("start_day", start_day),
    ("start_hour", start_hour), ("end_month", end_month),
    ("end_day", end_day), ("end_hour", end_hour),
]:
    if val is not None:
        _items.append(f"{key}={int(val)}")
overrides = ";".join(_items) if _items else None
