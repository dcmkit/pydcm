# RT dose &amp; DVH

Read an RT Dose grid, compute a dose-volume histogram against an RT Structure
Set, and author an RT Dose file — validated for full ROI
DVH coverage) and conformance-clean output.

## Read a dose grid

```python
import pydcm

grid = pydcm.read_rtdose("rtdose.dcm")
grid.dose            # 3-D float array, real-world dose (Gy)
grid.affine          # 4x4 voxel->world
grid.spacing         # (z, y, x) mm
grid.max_dose, grid.dose_units, grid.dose_summation_type
```

## Compute a DVH

```python
dvh = pydcm.dvhcalc("rtstruct.dcm", "rtdose.dcm", roi_number)
dvh.volume                  # cm3
dvh.min, dvh.mean, dvh.max  # Gy
dvh.counts                  # per-bin differential histogram
dvh.cumulative              # cumulative DVH
dvh.bins                    # bin edges (Gy), len(counts)+1
```

`dvhcalc` is a base-path DVH computation (no in-plane
interpolation / structure-extents knobs); the rasterisation, dose-plane
interpolation, histogram and statistics run in the native engine and match
ROI-for-ROI. `limit` caps the histogram in cGy.

## Author an RT Dose

Typical use: a predicted or accumulated dose grid → a file a TPS or viewer can
import.

```python
pydcm.write_rtdose(
    dose,                       # 3-D float array (Gy)
    affine=grid.affine,         # voxel→world (read_rtdose / load_series convention)
    dose_units="GY",
    dose_summation_type="PLAN",
    output="dose_out.dcm",
)
```

Pass geometry as `affine`, or as `origin` + `orientation` + `spacing`. Omit
`output` to get Part-10 `bytes`. The export quantises to integers with a
self-consistent `DoseGridScaling` and is IOD-conformance clean.
