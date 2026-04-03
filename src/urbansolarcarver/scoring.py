"""
UrbanSolarCarver Scoring Module

Provides sky-patch weight computation and threshold selection for voxel carving.

1. get_weights(mode, device, epw_file, hoys, …) → torch.Tensor
   Per-patch weight vector for the specified mode.  Delegates to
   compute_EPW_based_weights (sky_patches.py) for climate-based modes
   and compute_radiative_cooling_weights for radiative_cooling.

2. headtail_threshold(scores) → float
   Head/tail breaks (Jiang 2013): iteratively partitions right-skewed
   score distributions at the arithmetic mean.  Well-suited for solar
   obstruction scores which are typically heavy-tailed.
"""
import numpy as np
import torch
from .sky_patches import fetch_tregenza_patch_directions, compute_EPW_based_weights, compute_radiative_cooling_weights
from typing import Sequence

def get_weights(
    mode: str,
    device: torch.device = torch.device('cuda'),
    epw_file: str = None,
    hoys: Sequence[int] = None,
    dew_point_celsius: float = 10.0,
    bliss_k: float = 1.8,
    ground_reflectance: float = 0.2,
    balance_temperature: float = 15.0,
    balance_offset: float = 2.0,
    north_deg: float = 0.0,
) -> torch.Tensor:
    """
    Build a vector of weights for each sky patch, according to a chosen analysis mode.

    Explanation:
    - We represent the sky as discrete patches. This function assigns a numeric weight to each patch.
    - Modes include simple uniform weighting, solar irradiance, passive solar benefit, daylighting and radiative cooling.
    - Solar irradiance uses EPW data, Passive solar benefit uses Ladybug's method that employs a balance temp and offset.
    - Daylight uses CIE overcast approximation. Radiative cooling uses dew point temperature to estimate cooling potential of a night sky.
    Args:
      mode: Analysis type (e.g., 'time-based', 'irradiance', 'benefit', 'daylight', 'radiative_cooling').
      device: Compute device identifier (e.g., 'cuda' or 'cpu').
      epw_file: Path to weather data file; required for certain modes.
      hoys : Hour-of-year indices for EPW sampling.
      dew_point_celsius: Parameter for radiative cooling estimation.
      bliss_k: Angular-attenuation constant for radiative cooling.
      ground_reflectance: Reflectivity coefficient of ground surface.
      balance_temperature: Indoor-outdoor temperature threshold for comfort benefit.
      balance_offset: Temperature offset for comfort benefit.

    Returns:
      Tensor of length V (number of sky patches) with the computed weight for each patch.
    """
    if not isinstance(mode, str) or not mode.strip():
        raise TypeError(f"get_weights: mode must be a non-empty string, got {mode!r}")
    mode_key = mode.strip().lower()

    # Radiative cooling mode uses dew point to compute cooling potential per patch.
    if mode_key == 'radiative_cooling':
        return compute_radiative_cooling_weights(
            dew_point_celsius, bliss_k, torch.device(device)
        )

    # Time-based mode: assign a weight of 1 to every patch (uniform importance). Carving based on user-defined HOYs.
    if mode_key == 'time-based':
        sky_dirs = fetch_tregenza_patch_directions(torch.device(device))
        count_patches = sky_dirs.shape[0]
        return torch.ones(count_patches, dtype=torch.float32, device=device)

    # Daylight mode: CIE overcast sky — geometry only, no EPW needed.
    if mode_key == 'daylight':
        return compute_EPW_based_weights(
            mode_key,
            None,              # epw_path not used by daylight
            None,              # hoys not used by daylight
            torch.device(device),
            ground_reflectance,
            balance_temperature,
            balance_offset,
            north=north_deg,
        )

    # Modes requiring weather data: irradiance, benefit
    if mode_key in {'irradiance', 'benefit'}:
        if not epw_file or not hoys:
            raise ValueError(
                f"get_weights: epw_file and hoys list are required for mode '{mode}'"
            )
        return compute_EPW_based_weights(
            mode_key,
            epw_file,
            hoys,
            torch.device(device),
            ground_reflectance,
            balance_temperature,
            balance_offset,
            north=north_deg,
        )

    # Raise error for unsupported modes
    raise ValueError(f"get_weights: unsupported mode '{mode}'")




def headtail_threshold(
    scores: np.ndarray,
    max_iterations: int = 10
) -> float:
    """
    Head-tail breaks (Jiang, 2013): iteratively partition scores into 'head'
    (above mean) and 'tail' (below mean). Each iteration re-computes the mean
    of the head subset. The process stops when the head mean drops below the
    overall mean, indicating the distribution's heavy tail has been isolated.
    This is well-suited for right-skewed score distributions typical of solar
    exposure data.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of non-negative voxel scores.
    max_iterations : int
        Safety cap on recursion depth.

    Returns
    -------
    float
        Threshold value (the last computed mean).
    """
    if scores.size == 0:
        return 0.0
    current_set = scores.ravel()
    for _ in range(max_iterations):
        mean_val = float(current_set.mean())
        head_set = current_set[current_set > mean_val]
        if head_set.size == 0:
            return mean_val
        # Jiang (2013): stop when the head proportion drops below 40%,
        # meaning the heavy tail has been isolated from the bulk.
        head_pct = head_set.size / current_set.size
        if head_pct <= 0.40:
            return mean_val
        current_set = head_set
    return float(current_set.mean())
