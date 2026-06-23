# Diffusion tensor imaging (DTI)

`pydcm.dti` estimates the diffusion tensor from a DWI series and turns it into
the usual scalar maps — `FA`, `MD`, `AD`, `RD`, a directionally-encoded colour
map — and into deterministic streamlines. The tensor fit and the tracking run in
the native engine; this is a thin NumPy surface that orchestrates the series,
splits the baselines, and refuses a couple of inputs the engine cannot tell are
wrong.

Units follow the b-values: with b in s/mm², the diffusivities (`MD`, `AD`, `RD`)
are mm²/s and `FA` is dimensionless. Streamline coordinates are millimetres —
patient mm from `track_series`, and patient mm are what any DICOM output needs.

## Fit a series

`fit_series` reads a DWI directory, separates the b=0 baselines from the
diffusion directions using the series' own Manufacturer, fits the tensor, and
returns the maps with the geometry attached:

```python
from pydcm import dti

res = dti.fit_series("ep2d_diff/")     # default maps ("FA", "MD")
res["FA"], res["MD"]                    # (Z, Y, X) float32
res.affine                             # voxel -> patient, row-major 4x4
res.spacing                            # (col, row, slice) mm
res.shape                              # (depth, rows, cols)
```

`res` is a dict of `{map name: array}` with the grid carried alongside — a map
cannot be written or tracked without its affine, so it travels with the maps.
Ask for more maps by name:

```python
res = dti.fit_series("ep2d_diff/",
                     maps=("FA", "MD", "AD", "RD", "DEC", "CL", "CP", "CS"))
res["DEC"]                             # (Z, Y, X, 4) uint8 RGBA colour map
```

Every map name the engine accepts:

| name | shape / dtype | what it is |
|------|---------------|------------|
| `FA` | float32 | fractional anisotropy |
| `MD` | float32 | mean diffusivity (mm²/s) |
| `AD` | float32 | axial diffusivity, λ₁ (mm²/s) |
| `RD` | float32 | radial diffusivity, (λ₂+λ₃)/2 (mm²/s) |
| `DEC` | uint8 RGBA | direction-encoded colour, principal axis × FA |
| `CL` `CP` `CS` | float32 | Westin-1997 linearity / planarity / sphericity (÷λ₁) |
| `linearity` `planarity` `sphericity` | float32 | the trace-normalised form of the same three |

Pass `wls=True` to weight the fit by signal² instead of ordinary least squares.

## Masking — why an unmasked FA map peaks in air

Outside the head the signal is noise. The log-linear fit is degenerate there,
and the eigen decomposition returns an arbitrary direction with **FA near 1** — so
an unmasked FA map has its maximum in the background, and a histogram over the
whole volume describes mostly air. `fit_series` returns the fit as measured by
default (`mask=False`); pass `mask=True` to zero the maps outside a head mask:

```python
res = dti.fit_series("ep2d_diff/", mask=True)   # or mask=your_uint8_array
res["FA"].max()                                  # now inside the head, not in air
```

`mask=True` builds the mask from the mean baseline; you can also pass your own
`(Z, Y, X)` array. `head_mask` is the same routine, exposed directly for
inspection or reuse:

```python
mask, kept = dti.head_mask(b0)        # b0: 3-D (Z, Y, X) baseline; mask uint8, kept voxels
```

It median-filters, thresholds with Otsu, and dilates. `cleanup=True`
additionally removes islands and fills pockets — better for seeding, but a
departure from the reference behaviour it was matched against.

## Tractography

`track_series` fits the tensor and traces deterministic RK4 streamlines,
returning them in **patient** coordinates:

```python
tracks = dti.track_series("ep2d_diff/")   # list of (P, 3) float64 arrays, patient mm
```

Tracking applies a head mask before seeding by default, for the same reason the
FA map needs one — an unmasked run spends most of its budget on the degenerate
high-FA background. The stopping criteria are keyword arguments:

```python
tracks = dti.track_series(
    "ep2d_diff/",
    fa_threshold=0.15,      # stop below this FA
    angle_deg=45.0,         # max turn per step, in DEGREES (converted to a cosine for you)
    step_size=0.5,          # RK4 step, in VOXEL units
    seed_fa_min=0.3,        # only seed where FA is at least this
)
```

`angle_deg` is genuinely degrees — the engine takes a cosine internally, and the
wrapper converts, so passing `45` cannot silently become "no angular limit".

## Write streamlines to DICOM

`write_tracts` stores the streamlines as a DICOM Tractography Results object.
The tracks must be in patient coordinates — exactly what `track_series` returns:

```python
tracks = dti.track_series("ep2d_diff/")
dti.write_tracts("ep2d_diff/", tracks, "tracts.dcm")
```

The reference series supplies the demographics and the Frame of Reference UID the
coordinates are expressed in, so it should be the series the tracks were computed
from. Anatomy and diffusion-model codes default to White Matter and Single
Tensor. Omit the output path to get the Part-10 `bytes` instead of writing a file.

## The two guards

Two inputs the engine would accept and quietly mis-fit are refused here instead,
because nothing downstream reveals the mistake:

**b-vectors must be unit length.** The design matrix carries |g|², so a
gradient table normalised to anything other than 1 scales every diffusivity by
1/|g|² while leaving FA untouched — invisible to any FA comparison. A non-unit
table raises.

**baselines are not `bvals == 0`.** UIH writes 1.25 for its b=0 volume and
Siemens reports 50 for volumes that are baselines in every other respect, so the
split goes through a per-manufacturer threshold. `baseline_mask` exposes it:

```python
dti.baseline_mask(bvals, manufacturer="SIEMENS")   # True where a volume is a baseline
dti.baseline_mask(bvals, threshold=100.0)           # override the threshold entirely
```

`fit_series` reads the Manufacturer off the series for you; pass `threshold=` if
a series encodes its baseline unusually.

## Working from arrays

If you already hold a 4-D stack — for instance from `pydcm.load_dwi`, which
returns `(volumes, bvals, bvecs, affine)` — `fit` and `track` take it directly.
They accept either the `(3, V)` gradient layout `load_dwi` hands out or `(V, 3)`,
handle the baseline split, and carry an `affine` into the result:

```python
volumes, bvals, bvecs, affine = pydcm.load_dwi("ep2d_diff/")
res = dti.fit(volumes, bvals, bvecs, maps=("FA", "MD"), affine=affine, mask=True)
tracks = dti.track(volumes, bvals, bvecs, affine=affine)   # patient mm by default
```

!!! note "Scope"
    Single-tensor estimation (`FA` / `MD` / `AD` / `RD` / `DEC` / Westin
    measures), a head mask, and deterministic RK4 tractography with DICOM
    Tractography output. For research and engineering use only.
