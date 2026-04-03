"""
UrbanSolarCarver core package.

Implementations live in submodules:
 - preprocessing.py
 - thresholding.py
 - exporting.py
Support helpers live in `urbansolarcarver.grid` and `urbansolarcarver.carving`.
"""

# Staged API
from .preprocessing import preprocessing, PreprocessingResult  # noqa: F401
from .thresholding import thresholding, ThresholdingResult     # noqa: F401
from .exporting import exporting, ExportingResult              # noqa: F401

__all__ = [
    "preprocessing",
    "thresholding",
    "exporting",
    "PreprocessingResult",
    "ThresholdingResult",
    "ExportingResult",
]