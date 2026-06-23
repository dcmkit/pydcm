# Perfusion (DCE-MRI)

`pydcm.dce` fits a pharmacokinetic model to a dynamic contrast-enhanced MRI
series and gives you per-voxel `Ktrans`, `ve` and `vp` maps. The fitting runs in
the native engine; this is a thin NumPy surface over it.

## The arterial input function

Use a population AIF, or measure one from a vessel ROI:

```python
from pydcm import dce
import numpy as np

times_min = np.linspace(0, 5, 40)                     # acquisition times, minutes
aif = dce.parker_aif(times_min)                       # Parker population AIF
# or: dce.population_aif(times_min, model="parker", hct=0.42)
# or: dce.measure_aif(series, vessel_mask, tr_s, fa_deg)   # patient-specific
```

## Fit a whole study

`fit_series` loads the dynamic series, converts signal → concentration with the
spoiled-GRE equation, and fits every slice — returning per-voxel maps:

```python
maps = dce.fit_series(
    "dce_study/", times_min, model="ext_tofts",
    input="spgr", tr_s=0.005, fa_deg=25.0, t1_0_s=1.4,   # spoiled-GRE signal → concentration
    aif=aif, n_baseline=5,
)
maps["ktrans"], maps["ve"], maps["vp"]   # per-voxel parameter maps
maps["rmse"]                             # per-voxel fit residual
```

Models: `tofts`, `ext_tofts` (Extended Tofts), `patlak`. Pass `input="concentration"`
if your series is already concentration. For a per-patient T1 baseline, build one
from a variable-flip-angle set with `dce.t1_map_vfa(...)` and pass it as `t1_map=`.

## Fit an in-memory slice

If you already hold a slice time-course as an array, fit it directly — the input
is `[T, H, W]` (time first):

```python
maps = dce.fit(slice_series, times_min, model="ext_tofts",
               input="concentration", aif=aif)
```

A 4-D acquisition is assembled with [`load_4d`](../quickstart.md) (`[T, Z, Y, X]`);
take a slice's time-course as `series[:, z]`.

## Write parameter maps as DICOM

```python
dce.write_param_maps("dce_study/", maps, params=("ktrans", "ve", "vp"),
                     output_dir="./paramaps")   # one DICOM Parametric Map per parameter
```

!!! note "Scope"
    The validated slice is the Parker population AIF, the spoiled-GRE
    signal→concentration conversion, and the Tofts / Extended-Tofts / Patlak
    tissue models fitted per voxel. Outputs are for research and engineering
    only.
