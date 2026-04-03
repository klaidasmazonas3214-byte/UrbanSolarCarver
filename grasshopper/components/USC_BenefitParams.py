"""USC Benefit Parameters — Control benefit-mode balance point.

Only used when mode = "benefit". The balance temperature is the
outdoor air temperature at which a building is 'free-running' —
it needs neither heating nor cooling. Below this point, solar
radiation through windows reduces heating demand (benefit > 0).
Above it, solar radiation increases cooling load (benefit < 0).

Inputs
------
balance_temperature : float, optional
    Balance-point temperature in degrees Celsius. Default: 20.0
balance_offset : float, optional
    Transition band width in degrees Celsius. Default: 2.0

Outputs
-------
overrides : str
    Semicolon-joined key=value pairs for Config's overrides input.
"""

try:
    ghenv.Component.Name = "USC Benefit Params"
    ghenv.Component.NickName = "USC_Benefit"
    ghenv.Component.Description = "Sets the balance-point temperature for benefit mode. The balance point is the outdoor temperature at which the building is free-running (no heating or cooling). Below it, solar access reduces heating demand."
    ii = ghenv.Component.Params.Input
    oo = ghenv.Component.Params.Output
    for i, (n, d) in enumerate([
        ("balance_temperature", "Balance-point temperature (deg C): the outdoor temperature at which the building is free-running (no heating or cooling needed). Below this, solar gain reduces heating demand. Above it, solar gain increases cooling load. Typical values: 15-18 deg C for well-insulated buildings, 18-22 deg C for older stock."),
        ("balance_offset", "Transition band (deg C) around the balance point. Within this band, the benefit weight transitions smoothly from positive to negative, avoiding a hard cutoff. Default: 2.0 deg C. A wider band creates a gentler transition."),
    ]):
        if i < len(ii):
            ii[i].Name, ii[i].Description = n, d
    oo[0].Name, oo[0].Description = "overrides", "Benefit mode settings formatted for USC_Config. Connect to Config's 'overrides' input. Only relevant when mode='benefit'."
except Exception:
    pass

_items = []
if balance_temperature is not None:
    _items.append(f"balance_temperature={float(balance_temperature)}")
if balance_offset is not None:
    _items.append(f"balance_offset={float(balance_offset)}")
overrides = ";".join(_items) if _items else None
