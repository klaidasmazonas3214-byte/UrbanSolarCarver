# Sky Patches

Implements the Tregenza hemisphere subdivision (145 patches) used by all weighted carving modes. Each patch represents a solid-angle region of the sky dome; the scoring module assigns a weight to each patch depending on the analysis mode.

- **Tregenza geometry** -- 145 direction vectors and their solid angles, following the standard subdivision used in climate-based daylight modelling.
- **EPW-based weights** (`compute_EPW_based_weights`) -- aggregates hourly direct and diffuse irradiance from an EPW weather file onto each patch using the Perez all-weather sky model. Supports filtering by heating-benefit hours (outdoor temperature below balance point) and CIE overcast luminance.
- **Radiative cooling weights** (`compute_radiative_cooling_weights`) -- models long-wave sky emissivity per patch using the Martin-Berdahl clear-sky model with Bliss angular attenuation.

::: urbansolarcarver.sky_patches
