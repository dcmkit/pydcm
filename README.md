# pydcm

**The complete DICOM toolkit for Python.** A native reader/writer with a
built-in decoder for **every transfer syntax** (no codec plugins), plus
zero-copy **NumPy / PyTorch** pixel access — a compiled extension does the work
in-process.

- Decodes **JPEG, JPEG-2000, JPEG-LS, RLE, JPEG-XL and HTJ2K** out of the box,
  no plugins, and returns Hounsfield units and 3-D volumes.
- **Reads text from any vendor or locale** — full **Japanese / Korean / Chinese**
  multibyte and every ISO 2022 escape, decoded adaptively to correct UTF-8 where
  a strict reader would error or garble it.
- **One wheel replaces a whole stack of single-purpose tools:** read / decode /
  write every transfer syntax; DIMSE (`pydcm.dimse`) and DICOMweb
  (`pydcm.dicomweb`) networking; NIfTI / BIDS / DWI (`Volume.to_nifti`);
  segmentations, parametric maps and structured reports; radiomics
  (`pydcm.radiomics`); whole-slide imaging (`pydcm.wsi`); RT dose & DVH
  (`pydcm.rt`); waveforms; FHIR / HL7 bridges. See the
  [capability map](#beyond-the-core--capability-map).
- The API is clean and Pythonic, and most existing Python DICOM code runs
  against it unchanged. SimpleITK images are accepted / returned where natural
  (`radiomics`, `seg` masks) without being a dependency.

> **Not a medical device.** pydcm is **not** intended or cleared for clinical or
> diagnostic use. Decoded pixels and HU are for research/engineering only;
> validate any output for your intended use.

## Install

```bash
pip install pydcm
```

A platform wheel ships the compiled extension — no build step, no DICOM library
on your side. Wheels are published for macOS (arm64, x86_64) and Linux
(aarch64, x86_64; manylinux_2_28, i.e. glibc 2.28+ — RHEL 8+, Ubuntu 20.04+,
Debian 11+). One wheel per platform covers CPython 3.12+ (stable ABI).

## Use

```python
import pydcm

arr = pydcm.decode("scan.dcm")             # ndarray [frames, rows, cols(, samples)]
frame = pydcm.decode("ct.dcm", frame=2)    # 1-based; 0 (default) = all frames
arr, meta = pydcm.decode("scan.dcm", with_meta=True)
```

## Read & edit

`dcmread` returns a familiar `Dataset`. Attribute, item and sequence access,
`PersonName`, `file_meta`, and a lazy `pixel_array` all behave the way Python
DICOM users expect — so much that `import pydcm as pydicom` is usually enough:

```python
ds = pydcm.dcmread("scan.dcm")
ds.PatientName              # PersonName — .family_name / .given_name / .ideographic
ds.Rows, ds.PixelSpacing    # 64, [0.3125, 0.3125]   (MultiValue)
ds[0x0010, 0x0010].value    # element access by tag / keyword / (group, element)
ds.file_meta.TransferSyntaxUID
for elem in ds: ...         # iterate in tag order
px = ds.pixel_array         # NumPy (any transfer syntax, no plugins)
```

**Editing preserves everything.** `save_as` patches the *original* file bytes via
the native editor, so the Transfer Syntax, **PixelData** (including compressed
J2K / RLE), private tags and every untouched element survive byte-for-byte:

```python
ds = pydcm.dcmread("ct.dcm")
ds.PatientName = "Anon^Patient"
del ds.PatientBirthDate
ds.save_as("ct_anon.dcm")   # pixels + TS intact; only the named tags change
```

Verified for near-total element-value fidelity over a large real-world corpus
(the lone difference is a private sequence pydcm parses where others leave it
opaque — pydcm exposes *more*). The keyword↔tag↔VR dictionary holds 17,699
entries — a strict superset of the standard dictionaries — so attribute names
resolve broadly. *Known limits:* `file_meta` surfaces the three
mandatory UIDs (not optional group-0002 elements); a from-scratch (no source
file) `save_as` uses a metadata-only path that omits pixels.

### HU / real-world values

`rescale=False` (default) returns the **stored** integers (lossless). `rescale=True`
returns the modality-LUT output — real-world values, i.e. **HU for CT** — as
float32 (per-frame rescale applied for Enhanced multi-frame):

```python
hu = pydcm.decode("ct.dcm", rescale=True)         # float32 Hounsfield Units
ds = pydcm.DICOMDataset("ct_series/", rescale=True)
```

### Geometry sidecar

`with_meta=True` also returns the geometry the engine parsed — no extra read:

```python
_, m = pydcm.decode("ct.dcm", with_meta=True)
m["rescale_slope"], m["rescale_intercept"]
m["pixel_spacing"]                  # [row, col] mm
m["image_position_patient"]         # (0020,0032) [x, y, z]
m["image_orientation_patient"]      # (0020,0037) 6-vector
m["slice_thickness"], m["window_center"], m["window_width"]
m["modality"], m["series_instance_uid"], m["sop_instance_uid"]
```

### A directory → PyTorch

`DICOMDataset` walks a directory and decodes one image per item. It is
DataLoader-compatible **without importing torch** (torch stays optional):

```python
from torch.utils.data import DataLoader

ds = pydcm.DICOMDataset("study_dir/", to_torch=True)   # finds .dcm + extension-less DICOM
for batch in DataLoader(ds, batch_size=8, num_workers=4, shuffle=True):
    ...   # [B, H, W] or [B, H, W, C]
```

- One sample = one file. Single-frame files yield `[H, W]` / `[H, W, C]`;
  multi-frame files yield `[frames, H, W(, C)]`.
- `transform=fn` reshapes each sample (e.g. `[C, H, W]`, windowing, scaling).
- `pattern="*.dcm"` selects by name; the default also detects DICOM by the
  `DICM` preamble (catching extension-less clinical exports).

### A directory → one 3D volume

`load_series` assembles a directory of slices into a single **spatially-ordered
3D HU volume** (IOP clustering + IPP-projection sort, all in the native engine):

```python
vol = pydcm.load_series("ct_series/")
vol.pixels          # ndarray [depth, rows, cols], float32 HU, sorted by position
vol.spacing         # (z, y, x) mm — slice spacing computed from IPP deltas
vol.affine          # 4×4 voxel→world
vol.series_instance_uid
```

The largest coherent volume in the directory is returned, so a stray localizer
or second series does not corrupt the stack.

## Beyond the core — capability map

Everything below ships in the same wheel, over the same native engine, with
Python kept to thin marshalling — each area verified for correctness against
reference data.

| Area | Import | What it does |
|---|---|---|
| DIMSE networking | `pydcm.dimse` | SCU + full SCP, persistent associations |
| DICOMweb client | `pydcm.dicomweb` | QIDO / WADO / STOW / DELETE, streaming, TS negotiation |
| DICOM ↔ NIfTI | `Volume.to_nifti` / `from_nifti` | gantry-tilt-correct affine, vendor quirks handled; NIfTI → DICOM too |
| DWI / diffusion | `load_dwi` / `save_dwi` / `diffusion_table` | **FSL** `.bval`/`.bvec` (feeds FSL / MRtrix / dipy); Siemens CSA + mosaic, enhanced-MF, GE / Philips / UIH private |
| BIDS sidecar | `bids_sidecar` | BIDS `.json` (PhaseEncodingDirection, SliceTiming, EffectiveEchoSpacing…) |
| Preprocessing transforms | `pydcm.transforms` | resample / normalize / sliding-window; Tier 1 bit-exact (B-spline convention), Tier 2 ≤ 1 ULP (deep-learning convention) |
| Whole-slide imaging | `pydcm.wsi` | tile / region reads on the DICOM WSI pyramid + viewer tiles / total pixel matrix; bit-exact multi-vendor |
| RT dosimetry | `pydcm.rt` | `read_rtdose` / `write_rtdose` / `dvhcalc` — full ROI DVH coverage |
| Radiomics | `pydcm.radiomics` | 104 IBSI features / 7 classes, both aggregation conventions |
| SEG | `write_seg` / `write_seg_fractional` / `read_seg` | coded SEG, binary + fractional, SEG → labelmap |
| Parametric Map | `write_paramap` / `read_paramap` | author / read float parametric maps |
| Constructor-style object classes | `pydcm.sc` / `seg` / `pm` / `ko` / `pr` / `ann` / `sr` | `SCImage`, `Segmentation`, `ParametricMap`, KO, GSPS, the `MeasurementReport` content-tree classes, `MicroscopyBulkSimpleAnnotations` |
| SR / TID 1500 | `write_sr` / `write_report` / `read_report` / `sr_validate` | generic content trees + TID 1500 measurement reports + conformance checks |
| KO / PR / annotations | `write_ko` / `write_pr` / `read_ann` | Key Object Selection, Presentation State, Bulk Annotations |
| Encapsulated documents | `write_encapsulated` / `read_encapsulated` | PDF / CDA / STL / OBJ / MTL (PS3.3 A.104) |
| Waveforms (ECG / EEG) | `pydcm.waveforms` | 12-lead ECG / EEG read & write; arrays ready for analysis tools (MNE / neurokit2) |
| FHIR / HL7 | `pydcm.fhir` / `pydcm.hl7` | DICOM → FHIR R4 `ImagingStudy`; HL7 v2.5 parse + ORU^R01 build |
| Agent / MCP | `pydcm.mcp` | in-process MCP server over live pydcm objects |
| File sets | `pydcm.fileset` | read a DICOMDIR / File-set, iterate instances |

### DIMSE networking

```python
import pydcm.dimse as pynetdicom          # drop-in module shape

ae = pynetdicom.AE(ae_title="PYDCM")
assoc = ae.associate("pacs.local", 11112, ae_title="ANY-SCP")
assoc.send_c_echo()
assoc.send_c_store(pydcm.dcmread("ct.dcm"))   # persistent: many ops, one association
assoc.release()
```

`AE.start_server` runs the SCP side — `EVT_C_STORE` / `ECHO` / `FIND` / `GET` /
`MOVE` handlers plus the DIMSE-N set.

### DICOMweb client

```python
from pydcm import dicomweb

studies = dicomweb.search_studies("https://pacs.example.com", matches={"PatientID": "42"})
for part10 in dicomweb.iter_study("https://pacs.example.com", study_uid):
    ...                                    # streaming retrieve, bounded memory
dicomweb.store_instances("https://pacs.example.com", [open("ct.dcm", "rb").read()])
```

### DICOM ↔ NIfTI, BIDS, DWI

```python
vol = pydcm.load_series("ct_series/")
vol.to_nifti("ct.nii.gz")                  # validated affine (incl. gantry tilt)
meta = pydcm.bids_sidecar("ep2d_diff/")    # standard BIDS .json fields
dwi = pydcm.load_dwi("ep2d_diff/")         # 4D stack + b-values/b-vectors (all vendors)
```

### Preprocessing transforms

```python
from pydcm import transforms as T

out = T.resample_cubic(vol, out_shape)     # bit-exact B-spline order-3
seg = T.sliding_window_inference(vol.pixels, roi_size=(96, 96, 96), predictor=model)
```

Tier 1 ops are **bit-exact** for the classic B-spline convention; Tier 2 ops
match the deep-learning (grid-sample) convention to ≤ 1 float32 ULP — same
numbers in training and serving, no Python image stack required.

### Whole-slide imaging

```python
from pydcm import wsi

slide = wsi.open_slide("wsi_dir/")         # DICOM WSI pyramid (one or many files)
region = slide.read_region((x, y), level=0, size=(512, 512))   # RGBA, level-0 coordinates
slide.associated_images["LABEL"]
```

### RT dosimetry

```python
grid = pydcm.read_rtdose("rtdose.dcm")     # dose grid + scaling + grid geometry
dvh = pydcm.dvhcalc("rtstruct.dcm", "rtdose.dcm", roi_number)   # ROI-for-ROI DVH
pydcm.write_rtdose(dose, affine=grid.affine, output="out.dcm")  # conformance-clean
```

## License

pydcm is licensed under **Apache-2.0** (see [LICENSE](LICENSE) / [NOTICE](NOTICE)).
The high-performance DICOM engine ships as a compiled binary inside the extension.
Third-party components linked into the extension are listed in
[THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) — all permissive (BSD / MIT / Zlib /
Apache / IJG) except **FFmpeg**, which is included under **LGPL-2.1** (full text in
[LGPL-2.1.txt](LGPL-2.1.txt)) for embedded-video DICOM decode, with a §6 relink
offer.

pydcm distributes as **wheels only** (no sdist) — the engine ships as a
compiled binary inside the extension, and parts of it are not open source.
