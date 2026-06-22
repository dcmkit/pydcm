# DICOM → NIfTI, BIDS &amp; DWI

Turn a DICOM series into the files a neuroimaging or ML pipeline expects:
a NIfTI volume, a BIDS JSON sidecar, and — for diffusion — an FSL gradient table.

## A series → NIfTI

```python
import pydcm

vol = pydcm.load_series("ct_series/")
vol.pixels          # [depth, rows, cols] float32 HU, spatially sorted
vol.spacing         # (z, y, x) mm
vol.affine          # 4×4 voxel→world
path = vol.to_nifti("ct.nii.gz")
```

The affine is validated including the gantry-tilt case (the
column-2 increment follows the true Image-Position-Patient step, so tilted CT
does not drift). Non-tilted series are bit-identical to the straightforward
construction.

## BIDS sidecar

```python
meta = pydcm.bids_sidecar("ep2d_diff/0001.dcm")   # dict from one instance, standard BIDS fields
meta["PhaseEncodingDirection"]               # e.g. "j-"
meta["SliceTiming"]                          # per-slice acquisition times
meta["EffectiveEchoSpacing"], meta["TotalReadoutTime"]
```

Write it next to the NIfTI yourself:

```python
import json
with open("ep2d_diff.json", "w") as f:
    json.dump(meta, f, indent=2)
```

## Diffusion (DWI) → FSL `.bval` / `.bvec`

```python
import glob

files = glob.glob("ep2d_diff/*.dcm")
bvals, bvecs = pydcm.diffusion_table(files, output_prefix="dwi")   # writes dwi.bval / dwi.bvec
```

- `rotate=True` (default) rotates each gradient from the patient (LPS) frame
  into the image/voxel frame. The bvecs come out in the DICOM image frame; FSL
  works relative to the NIfTI (RAS) axes, so validate the sign against your
  pipeline — an axis flip may still be needed.
- Vendor coverage is built in: Siemens CSA + mosaic, enhanced multi-frame, and
  the GE / Philips / UIH private encodings.

To get the 4-D diffusion volume *and* the table together — `load_dwi` returns a
4-tuple:

```python
data, bvals, bvecs, affine = pydcm.load_dwi("ep2d_diff/")   # data [V, Z, Y, X], grouped by gradient
```

Or write the NIfTI **and** the FSL table in one call (returns the three paths):

```python
nii, bval, bvec = pydcm.save_dwi("ep2d_diff/", "dwi")       # dwi.nii.gz / dwi.bval / dwi.bvec
```

The output feeds FSL, MRtrix or dipy directly — pydcm produces the gradient
table; the downstream analysis stays in those tools.
