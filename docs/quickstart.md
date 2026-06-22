# pydcm quickstart

`import pydcm as pydicom` mostly just works. Beyond that, one wheel mirrors a
whole stack of specialist tools — each section below is a runnable recipe.

## Read &amp; decode

```python
import pydcm

ds = pydcm.dcmread("scan.dcm")
ds.PatientName              # PersonName
ds.Rows, ds.PixelSpacing    # 64, [0.3125, 0.3125]
ds.file_meta.TransferSyntaxUID
px = ds.pixel_array         # NumPy — ANY transfer syntax, no codec plugins
```

`decode()` is the one-call path straight to an array:

```python
arr = pydcm.decode("scan.dcm")             # [frames, rows, cols(, samples)]
hu  = pydcm.decode("ct.dcm", rescale=True) # float32 Hounsfield Units (modality LUT)
arr, meta = pydcm.decode("scan.dcm", with_meta=True)   # geometry sidecar, no extra read
```

!!! tip "Every transfer syntax built in — JPEG-2000, JPEG-XL, HTJ2K and more"
    The decoder is compiled into the wheel (openjpeg / charls / openjph /
    libjxl statically linked), so `pixel_array` / `decode()` read **any**
    transfer syntax with **no plugins** — **JPEG-2000**, **JPEG-LS**, **RLE**,
    baseline / lossless **JPEG**, and the newest **JPEG-XL** and
    **HTJ2K / High-Throughput JPEG 2000**. Plugin-less readers raise on these;
    pydcm just returns the array. And it **encodes** them too:

    ```python
    import pydcm
    from pydcm.uid import JPEGXLLossless, HTJ2KLossless

    ds = pydcm.dcmread("ct.dcm")
    ds.compress(JPEGXLLossless)        # transcode in place to JPEG-XL (lossless)
    ds.save_as("ct_jxl.dcm")

    ds = pydcm.dcmread("ct.dcm")
    ds.compress(HTJ2KLossless)         # …or High-Throughput JPEG 2000
    ds.save_as("ct_htj2k.dcm")
    ```

!!! tip "Reads real-world vendor text — where strict decoders give up"
    Character-set handling is where exports from different countries and vendors
    diverge most: inconsistent `SpecificCharacterSet` declarations, malformed
    ISO 2022 escape sequences, non-standard charset aliases, mixed or missing
    encodings. pydcm's native text engine covers the full range — Latin /
    Cyrillic / Greek /
    Arabic / Hebrew / Thai, **Japanese** (Shift-JIS, ISO 2022 IR 87 / 159),
    **Korean** (EUC-KR), **Chinese** (GB18030 / GBK), UTF-8 — and decodes it
    **adaptively**: lenient alias matching, tolerance of broken escape sequences,
    and a fault-tolerant fallback. A `PatientName` that a strict iconv-based
    reader rejects comes back as a correct Python
    string. Every text value is decoded to UTF-8 at read time, so you never
    touch encodings by hand.

    ```python
    ds = pydcm.dcmread("vendor_jp.dcm")
    str(ds.PatientName)        # 山田^太郎  — already decoded, whatever the vendor declared
    ```

## Edit — byte-verbatim save

`save_as` patches the *original* file bytes, so Transfer Syntax, PixelData
(including compressed J2K / RLE), private tags and every untouched element
survive byte-for-byte:

```python
ds = pydcm.dcmread("ct.dcm")
ds.PatientName = "Anon^Patient"
del ds.PatientBirthDate
ds.save_as("ct_anon.dcm")   # only the named tags change
```

!!! tip "Editing patches the file, it doesn't rewrite it"
    Many DICOM writers re-encode the whole dataset on save — which can subtly change
    compressed PixelData, private tags or VR representation. pydcm edits the
    **original bytes in place**, so anything you didn't touch is byte-identical
    on disk. Ideal for PHI redaction: change the named tags, guarantee nothing
    else moved.

See [Behaviour notes](divergences.md) for the handful of deliberate
behavioural differences.

## A directory → 3D volume → NIfTI

```python
vol = pydcm.load_series("ct_series/")
vol.pixels          # [depth, rows, cols] float32 HU, spatially sorted
vol.spacing         # (z, y, x) mm
vol.affine          # 4×4 voxel→world
vol.to_nifti("ct.nii.gz")          # validated affine (incl. gantry tilt)
```

!!! tip "Real-world values and geometry, computed in the engine"
    `decode(rescale=True)` returns the modality-LUT output — **HU for CT** — as
    float32, and `decode(with_meta=True)` hands back the geometry the engine
    already parsed (rescale, pixel spacing, image position / orientation, slice
    thickness, window) with no extra read. `load_series` sorts slices by
    position (IOP clustering + IPP projection) and returns the largest coherent
    volume, so a stray localizer can't corrupt the stack — all in native code,
    nothing re-derived in Python.

## A dynamic series → 4-D volume

`load_4d` assembles a multi-phase acquisition into one `[T, Z, Y, X]` stack and
**discovers the 4th axis on its own** — time / b-value / gradient direction /
echo / cardiac phase / stack / energy:

```python
vol = pydcm.load_4d("multiecho_mr/")
vol.shape           # (T, Z, Y, X)
vol.n_volumes       # T — number of 3-D volumes
vol.dimensions      # [Axis(kind='echo', size=3, values=[10.0, 20.0, 40.0])]
vol.volume_frame(0) # the first 3-D volume
vol.to_nifti("dynamic.nii.gz")     # 4-D NIfTI
```

A DCE perfusion series flows straight into pharmacokinetic fitting — each slice
time-series → per-voxel parameter maps:

```python
from pydcm import dce
import numpy as np

times_min = np.linspace(0, 5, vol.n_volumes)
aif  = dce.parker_aif(times_min)                       # Parker population AIF
maps = dce.fit(slice_series, times_min, model="ext_tofts",   # slice_series: [T, H, W]
               input="signal", tr_s=0.005, fa_deg=25.0, aif=aif)
maps["ktrans"], maps["ve"], maps["vp"]                 # per-voxel maps
```

For a DWI series where you also want the FSL `.bval`/`.bvec` gradient table, use
`pydcm.load_dwi`.

## A directory → PyTorch

```python
from torch.utils.data import DataLoader

ds = pydcm.DICOMDataset("study_dir/", to_torch=True)   # torch stays optional
for batch in DataLoader(ds, batch_size=8, num_workers=4, shuffle=True):
    ...   # [B, H, W] or [B, H, W, C]
```

## DICOMweb (QIDO / WADO / STOW)

```python
from pydcm import dicomweb

studies = dicomweb.search_studies("https://pacs.example.com", matches={"PatientID": "42"})
for part10 in dicomweb.iter_study("https://pacs.example.com", study_uid):
    ...                                    # streaming retrieve, bounded memory
dicomweb.store_instances("https://pacs.example.com", [open("ct.dcm", "rb").read()])
```

## DIMSE networking

```python
import pydcm.dimse as pynetdicom

ae = pynetdicom.AE(ae_title="PYDCM")
assoc = ae.associate("pacs.local", 11112, ae_title="ANY-SCP")
assoc.send_c_echo()
assoc.send_c_store(pydcm.dcmread("ct.dcm"))   # persistent: many ops, one association
assoc.release()
```

`AE.start_server` runs the SCP side with the same `EVT_C_STORE` / `ECHO` /
`FIND` / `GET` / `MOVE` event model.

## Preprocessing

```python
from pydcm import transforms as T

out = T.resample_cubic(vol, out_shape)      # bit-exact B-spline order-3
seg = T.sliding_window_inference(vol.pixels, roi_size=(96, 96, 96), predictor=model)
```

Tier-1 ops are **bit-exact** for the classic B-spline convention; Tier-2 ops
match the deep-learning (grid-sample) convention to ≤ 1 float32 ULP — same
numbers in training and serving.

## RT dosimetry

```python
grid = pydcm.read_rtdose("rtdose.dcm")
dvh  = pydcm.dvhcalc("rtstruct.dcm", "rtdose.dcm", roi_number)   # ROI-for-ROI DVH
pydcm.write_rtdose(dose, affine=grid.affine, output="out.dcm")   # conformance-clean
```

## Whole-slide imaging

```python
from pydcm import wsi

slide = wsi.open_slide("wsi_dir/")
region = slide.read_region((x, y), level=0, size=(512, 512))   # RGBA, level-0 coordinates
slide.associated_images["LABEL"]
```

For the complete capability map (radiomics, SEG / Parametric Map / TID 1500,
waveforms, FHIR / HL7, MCP) see the [home page](index.md) and the
[API reference](api.md).
