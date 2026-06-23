# RT dose &amp; DVH

Read an RT Dose grid, sample the dose at points, compute a dose-volume histogram
against an RT Structure Set, rasterise an ROI to a mask, and author an RT Dose
file. Quantitative geometry and coverage checks run in the shared native RT
engine; IOD conformance remains available through the shared validator.

## Read a dose grid

```python
import pydcm

grid = pydcm.read_rtdose("rtdose.dcm")
grid.dose            # 3-D float array, real-world dose (Gy)
grid.affine          # 4x4 voxel->world
grid.spacing         # (z, y, x) mm
grid.frame_offsets_mm  # authoritative double-precision frame offsets
grid.origin_lps, grid.column_step_lps, grid.row_step_lps
grid.max_dose, grid.dose_units, grid.dose_summation_type
```

For one-frame or nonuniform dose grids, `has_uniform_affine` is false and no
fake z step is inserted into `affine`; use `frame_offsets_mm` for resampling or
round-trip writing. Stored DVHs retain all ROI INCLUDE/EXCLUDE relationships,
volume units and optional statistics. A malformed optional stored DVH is
reported by `stored_dvh_error` without hiding an otherwise usable dose grid.

## Point dose

`dose_at` reads the dose at arbitrary patient coordinates — what a prescription
point or a measured location received, rather than what a structure received.

```python
values, inside = pydcm.dose_at("rtdose.dcm", points)   # points: (N, 3)
```

`points` is `(N, 3)` patient coordinates **in the dose object's own Frame of
Reference** (a single `(3,)` point is accepted too). `values` is `(N,)` Gy and
`inside` is `(N,)` bool. Interpolation is trilinear.

The `inside` flag is what a bare number cannot say: a point the grid does not
cover comes back with `inside=False` and a value of `0`, while a point where the
dose genuinely is zero comes back with `inside=True`. They are the same number
and mean opposite things, so filter on the flag before trusting a zero:

```python
covered = values[inside]        # drop the off-grid points
```

Points measured in another Frame of Reference must be transformed **first** —
nothing here treats a mismatched frame as identity, because that would sample
the right grid at the wrong place and report a plausible number:

```python
reg = pydcm.read_registration("reg.dcm")
xform = reg.transform(source_for, grid.frame_of_reference_uid)   # (4, 4)
pts_in_dose_frame = points @ xform[:3, :3].T + xform[:3, 3]
values, inside = pydcm.dose_at("rtdose.dcm", pts_in_dose_frame)
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

`dvhcalc` uses the engine's fixed base policy: fractional contour coverage,
patient-space dose interpolation and compensated histogram/statistics. `limit`
truncates the returned histogram in cGy without truncating full-volume
integration or dose statistics.

`samples_per_axis` (1..32) sets the fractional-coverage sampling density — a
higher value trades speed for a finer partial-pixel estimate; the default is the
engine's. `require_full_dose_coverage=True` refuses a structure the dose grid
does not fully reach, rather than reporting it with a coverage note:

```python
dvh = pydcm.dvhcalc("rtstruct.dcm", "rtdose.dcm", roi_number,
                    samples_per_axis=16, require_full_dose_coverage=False)
```

When the grid does not cover the whole structure, the volume splits: `dvh.volume`
is the total ROI (with the default `calculate_full_volume=True`), while
`dvh.covered_volume` and `dvh.uncovered_volume` are the part the dose grid
actually produced a histogram for, and `dvh.notes` says so. Per-cent readings
are of the **covered** volume, because that is what the histogram is of.

## Read dose constraints

The D(V) / V(D) readings come from the engine's own bin search — the same
grammar the DVH constraint parser accepts — so a constraint reads the same
here, in the CLI and in a viewer. The common ones are plain attributes:

```python
dvh.D95      # dose 95% of the volume receives (Gy)
dvh.D2cc     # dose the hottest 2 cm3 receives (Gy)
dvh.V20Gy    # volume receiving at least 20 Gy (cm3)
```

Each is a `DVHValue`: it compares and formats as its number — `dvh.D95 > 60`,
`f"{dvh.D95:.1f}"`, `float(dvh.D2cc)` — while its `repr` keeps the units visible.
A reading with no answer (the structure is smaller than the volume asked about)
is `NaN`, never a misleading `0`.

Names that are not valid Python attributes — the `%` forms — go through
`statistic`:

```python
dvh.statistic("V100%")   # volume receiving at least 100% of the prescription
dvh.statistic("D0.5cc")  # dose the hottest 0.5 cm3 receives
```

Or spell the two readings out directly, choosing the units:

```python
dvh.dose_constraint(95)          # D95  — volume as a % (the default)
dvh.dose_constraint(2, "cc")     # D2cc — volume in cm3 ("cc" or "cm3")

dvh.volume_constraint(20)        # V20Gy — dose in Gy (the default)
dvh.volume_constraint(2000, "cGy")
dvh.volume_constraint(100, "%")  # a % of rx_dose (needs a prescription)
```

The `%` forms need a prescription. Set `rx_dose` in Gy, or hand `dvhcalc` an
RT Plan (or a path to one) to read it from the plan's largest target
prescription:

```python
dvh = pydcm.dvhcalc("rtstruct.dcm", "rtdose.dcm", roi_number, rx_dose="rtplan.dcm")
dvh.statistic("V95%")            # 95% of the plan's prescription
```

`describe()` prints the constraints a plan review reads first (volume, the
covered/uncovered split, min/mean/max, D100…D2cc, and the `%` volumes when a
prescription is set) and returns them as a dict:

```python
report = dvh.describe()
```

## RTSTRUCT ROI to a mask

`roi_mask` rasterises one ROI of an RT Structure Set to a boolean volume on a
reference image series' grid:

```python
mask, meta = pydcm.rt.roi_mask("rtstruct.dcm", reference_series, roi_number)
mask.shape            # (planes, rows, cols) bool — one plane per reference slice
meta["roi_name"]
meta["set_voxels"]    # how many voxels the ROI set
meta["findings"]      # what the mask could not simply state (see below)
```

`reference_series` is the image series the mask should land on — a directory, a
file, or a list of instance paths. Its slices are the target planes in the same
order the authoring writers use, so a mask and a Segmentation authored from the
same series share one grid. `roi_number` is the ROI **Number** (the identifier
`dvhcalc` also takes), one ROI per call. A pixel is in when its centre is.

When the structure set and the reference series live in different Frames of
Reference, pass the fusion `transform`; omitting it there is refused rather than
treated as identity:

```python
reg = pydcm.read_registration("reg.dcm")
xform = reg.transform(struct_for, series_for)             # (4, 4) RTSTRUCT->series
mask, meta = pydcm.rt.roi_mask("rtstruct.dcm", reference_series, roi_number,
                               transform=xform)
```

`meta["findings"]` explains any placement the mask cannot state on its own —
`contour_on_no_plane` (a contour reaches past the series), `same_plane_nested`
(nested `CLOSED_PLANAR` contours unioned, since the standard defines no hole for
them), `same_plane_partial_overlap`, and others. Each is reported rather than
silently applied.

For how a registration composes two Frames of Reference — and where `struct_for`
and `series_for` come from — see the [spatial-registration how-to](spatial-registration.md).

## Author an RT Dose

Typical use: a predicted or accumulated dose grid → a file a TPS or viewer can
import.

```python
pydcm.write_rtdose(
    dose,                       # 3-D float array (Gy)
    affine=grid.affine,         # voxel→world (read_rtdose / load_series convention)
    grid_frame_offsets=grid.frame_offsets_mm,  # preserves nonuniform geometry
    dose_units="GY",
    dose_summation_type="PLAN",
    output="dose_out.dcm",
)
```

Pass geometry as `affine`, or as `origin` + `orientation` + `spacing`. Omit
`output` to get Part-10 `bytes`. The export quantises to integers with a
self-consistent `DoseGridScaling`. For PLAN/BEAM-style summation, also pass an
RT Plan reference (or `ref_plan_uid`) when a conformant relationship is
required.
