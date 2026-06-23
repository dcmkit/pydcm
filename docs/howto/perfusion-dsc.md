# Perfusion (DSC-MRI)

`pydcm.dsc` recovers cerebral perfusion from a dynamic-susceptibility-contrast
MRI series — the bolus of contrast transiently drops the T2\*-weighted signal,
and deconvolving the tissue concentration against an arterial input gives
per-voxel `CBF`, `CBV`, `MTT`, `Tmax` and `TTP` maps. The deconvolution runs in
the native engine; this is a thin NumPy surface over it.

Units are **seconds** for time and MTT/Tmax; CBF/CBV are **relative**
(calibration-free) — multiply by your absolute scaling (`k·ρ·(1−Hct)`, and the
s→min / fraction→mL·100mL⁻¹ factors) if you need physical units.

## The arterial input function

DSC has no population AIF — measure one from an arterial ROI (e.g. an MCA voxel
cluster). `measure_aif` averages the ROI signal per frame and converts it to
ΔR2\*:

```python
from pydcm import dsc
import numpy as np

aif = dsc.measure_aif(series, artery_mask, te_s=0.030)   # ROI signal → ΔR2* AIF
```

## Fit a whole study

`fit_series` assembles the dynamic series, reads the acquisition times (and the
echo time, for the signal → ΔR2\* step) off the DICOM tags, and deconvolves every
slice — the AIF is factorised once and the slices run across cores:

```python
maps = dsc.fit_series(
    "dsc_study/", aif=aif, method="osvd", input="raw",   # raw T2*-weighted signal
)
maps["cbf"], maps["cbv"], maps["mtt"]    # per-voxel perfusion maps (Z, H, W)
maps["tmax"], maps["ttp"]                # residue peak time / enhancement peak time
```

Deconvolution methods:

| `method` | matrix | when |
|----------|--------|------|
| `ssvd`   | causal Toeplitz             | no bolus delay (least peak damping) |
| `csvd`   | block-circulant             | **delay-insensitive** — robust to bolus arrival differences |
| `osvd`   | block-circulant, adaptive   | per-voxel oscillation-index threshold (default) |

Pass `input="concentration"` if your series is already ΔR2\*. `times_s` and
`te_s` are read from the tags when omitted; pass them for a raw array.

## Tumour leakage correction

Where the blood–brain barrier is disrupted (tumour), contrast extravasates and
biases CBV. Enable the Boxerman–Schmainda correction, which fits each voxel
against the whole-brain mean reference and removes the leakage term:

```python
maps = dsc.fit_series("dsc_study/", aif=aif, method="osvd", input="raw",
                      leakage=True, mask=brain_mask)   # mask should cover brain
```

The reference is the mean over the masked tissue, so `mask` should cover brain,
not just the enhancing lesion.

## Fit an in-memory slice or a single curve

```python
# one slice time-course, [T, H, W] (time first)
maps = dsc.fit(slice_series, times_s, aif, method="csvd")

# a single tissue curve → parameters + the recovered residue
r = dsc.deconvolve(ct, aif, dt_s=1.5, method="osvd")
r["cbf"], r["cbv"], r["mtt"], r["tmax"], r["residue"]
```

## Single-curve building blocks

When you assemble your own tumour workflow — or want the brain-tumour rCBV
(Boxerman/ASFNR) that needs no AIF at all — the pieces the map pipeline uses are
exposed per curve:

```python
area  = dsc.cumtrapz(ct, dt_s=1.5)            # running trapezoidal integral, starts at 0
rcbv  = dsc.cbv(ct, ref, dt_s=1.5)            # blood volume as ∫ct / ∫ref
lc    = dsc.leakage_correct(ct, ref, dt_s=1.5)
lc["corrected"], lc["k2"]                     # curve with the leakage term added back, and K2
```

`cbv` is just that area ratio — no deconvolution, and `dt` cancels (it is taken
only so the call states its units). What `ref` is decides the meaning: an **AIF**
gives CBV in the Østergaard/stroke sense (equal to `CBF·MTT`), a **normal-appearing
white-matter** ROI mean gives rCBV in the brain-tumour sense.

`leakage_correct` is the Boxerman–Schmainda–Weisskoff fit: it regresses
`ct ≈ K1·ref − K2·∫ref` against a **non-leaking** reference and returns the curve
with the extravasation term added back. The reference must be leakage-free — the
whole-brain mean works in vivo (leaking voxels are a small minority), but an
unmasked reference is destroyed by air, where the log floor drives ΔR2\* to
hundreds of s⁻¹; for a phantom or a masked volume pass the NAWM ROI mean.

## Choose the first-pass window

Recirculation — the second, lower bump as the bolus comes round again — is not
part of the first transit and inflates every area that includes it.
`first_pass_end` reads how many leading samples make up the first pass off the
curve's own pre-contrast scatter, so you can integrate the first pass only:

```python
n = dsc.first_pass_end(ct, n_baseline=10)     # count of first-pass samples
if n < 0:
    n = len(ct)                               # inputs can't support a reading → full curve
```

It takes the ΔR2\* concentration curve (not raw signal); `n_baseline` must cover
pre-contrast frames and be at least 2, and `k_sigma=0` selects the library
default of 4. It returns the whole length when the curve never turns back up, and
`-1` when the inputs can't support the reading — guard for that.

!!! warning "One window feeds both steps"
    The count `first_pass_end` returns must be the **same** one you pass to
    `leakage_correct` and to `cbv`. K2 is fitted to whatever stretch of curve it
    is shown, so correcting over one range while integrating over another throws
    away more than half the benefit of correcting at all.

```python
n = dsc.first_pass_end(ct, n_baseline=10)
if n < 0:
    n = len(ct)
lc   = dsc.leakage_correct(ct[:n], ref[:n], dt_s=1.5)   # same window ...
rcbv = dsc.cbv(lc["corrected"], ref[:n], dt_s=1.5)      # ... in both
```

## Write parameter maps as DICOM

```python
dsc.write_param_maps("dsc_study/", maps,
                     params=("cbf", "cbv", "mtt", "tmax", "ttp"),
                     output_dir="./paramaps")   # one DICOM Parametric Map per parameter
```

!!! note "Scope"
    sSVD / cSVD / oSVD deconvolution with a measured AIF, ΔR2\* signal
    conversion, CBF / CBV / MTT / Tmax / TTP maps and optional leakage
    correction. CBF/CBV are relative unless you apply the absolute calibration.
    Outputs are for research and engineering only.
