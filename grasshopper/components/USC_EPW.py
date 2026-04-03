"""USC EPW Path — Provide weather file location.

Required for: time-based, irradiance, benefit, radiative_cooling modes.
Not needed for: daylight, tilted_plane.

Inputs
------
epw_path : str
    Full file path to an EnergyPlus Weather (.epw) file.
    Accepts a text string or a File Path parameter.

Outputs
-------
overrides : str
    epw_path=<value> override for USC_Config.
"""

import os

try:
    ghenv.Component.Name = "USC EPW Path"
    ghenv.Component.NickName = "USC_EPW"
    ghenv.Component.Description = (
        "Provides the path to an EnergyPlus Weather (.epw) file. "
        "Required for time-based, irradiance, benefit, and radiative_cooling "
        "modes. Not needed for daylight or tilted_plane. "
        "Download EPW files from climate.onebuilding.org or ladybug.tools."
    )
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    ii[0].Name, ii[0].Description = "epw_path", (
        "Full path to an EnergyPlus Weather (.epw) file. "
        "Right-click and choose 'Set One File Path', or connect a "
        "text panel / File Path parameter. "
        "Example: C:\\Weather\\GRC_Athens.epw"
    )
    oo[0].Name, oo[0].Description = "overrides", (
        "EPW path formatted as an override string. "
        "Connect to USC_Config 'overrides' input."
    )
except Exception:
    pass

overrides = None

if epw_path is not None:
    _path = str(epw_path).strip()
    if _path:
        if not _path.lower().endswith(".epw"):
            try:
                from Grasshopper.Kernel import GH_RuntimeMessageLevel
                ghenv.Component.AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "File does not have .epw extension: {}".format(_path)
                )
            except Exception:
                pass
        if not os.path.isfile(_path):
            try:
                from Grasshopper.Kernel import GH_RuntimeMessageLevel
                ghenv.Component.AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "File not found: {}".format(_path)
                )
            except Exception:
                pass
        overrides = "epw_path={}".format(_path)
