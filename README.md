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
  (`pydcm.dicomweb`) networking; NIfTI / NRRD / MetaImage / BIDS / DWI volume
  export; segmentations, parametric maps and structured reports; **PS3.15
  de-identification** and **IOD / SR conformance validation**; **nnU-Net / MONAI
  preprocessing** (`pydcm.transforms`, bit-exact); radiomics (`pydcm.radiomics`);
  whole-slide imaging (`pydcm.wsi`); RT dose & DVH (`pydcm.rt`); waveforms;
  FHIR / HL7 bridges; and an **in-process MCP server** (`pydcm.mcp`) for agents.
  See the [capability map](#beyond-the-core--capability-map).
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
| DICOM → NRRD / MetaImage | `Volume.to_nrrd` / `Volume.to_metaimage` | 3D Slicer (`.nrrd`) + ITK / nnU-Net (`.mha`) on-ramps; double-faithful LPS geometry |
| DWI / diffusion | `load_dwi` / `save_dwi` / `diffusion_table` | **FSL** `.bval`/`.bvec` (feeds FSL / MRtrix / dipy); Siemens CSA + mosaic, enhanced-MF, GE / Philips / UIH private |
| BIDS sidecar | `bids_sidecar` | BIDS `.json` (PhaseEncodingDirection, SliceTiming, EffectiveEchoSpacing…) |
| Preprocessing transforms | `pydcm.transforms` | resample / normalize / sliding-window; Tier 1 bit-exact (B-spline convention), Tier 2 ≤ 1 ULP (deep-learning convention) |
| Whole-slide imaging | `pydcm.wsi` | tile / region reads on the DICOM WSI pyramid + viewer tiles / total pixel matrix; bit-exact multi-vendor |
| RT dose &amp; structures | `pydcm.rt` | `read_rtdose` / `write_rtdose` / `dvhcalc` (full ROI DVH coverage) + `write_rtstruct` — author RT Structure Sets (ROI contours over a reference series) |
| Tractography | `write_mktract` | author Tractography Results (66.6) from track sets — points + per-track measurements / statistics |
| Radiomics | `pydcm.radiomics` | the full IBSI set — 135 features / 10 classes; both aggregation conventions; register custom Python features |
| SEG | `write_seg` / `write_seg_fractional` / `read_seg` | coded SEG, binary + fractional, SEG → labelmap |
| Parametric Map | `write_paramap` / `read_paramap` | author / read float parametric maps |
| Constructor-style object classes | `pydcm.sc` / `seg` / `pm` / `ko` / `pr` / `ann` / `sr` | `SCImage`, `Segmentation`, `ParametricMap`, KO, GSPS, the `MeasurementReport` content-tree classes, `MicroscopyBulkSimpleAnnotations` |
| SR / TID 1500 | `write_sr` / `write_report` / `read_report` / `sr_validate` | a full SR object model (Comprehensive / Enhanced / 3D SR, ROI measurement groups, every content-item type) — author from a content tree or as TID 1500 reports, read them back, validate structure + PS3.16 coded concepts, render SR → HTML / JSON |
| KO / PR / annotations | `write_ko` / `write_pr` / `read_ann` | Key Object Selection, Presentation State, Bulk Annotations |
| Encapsulated documents | `write_encapsulated` / `read_encapsulated` | PDF / CDA / STL / OBJ / MTL (PS3.3 A.104) |
| Surface meshes | `read_surface` | read Surface Segmentation (66.5) — points + every mesh primitive (triangle / strip / fan / facet) as `(N,3)` / `(M,3)` arrays |
| Waveforms (ECG / EEG) | `pydcm.waveforms` | 12-lead ECG / EEG read & write; arrays ready for analysis tools (MNE / neurokit2) |
| FHIR / HL7 | `pydcm.fhir` / `pydcm.hl7` | DICOM → FHIR R4 `ImagingStudy`; HL7 v2.x parse (2.3 – 2.8) + ORU^R01 build |
| De-identification | `deident` / `deidentify` / `deidentify_series` | PS3.15 Table E.1-1 Basic Profile (617 attributes) **+ every option profile** (clean graphics / descriptors / structured content, retain longitudinal / patient-characteristics / device / UIDs / institution); consistent UID remap, dates shifted, CID 7050 method codes stamped in (0012,0064) |
| Validation | `validate` / `iod_validate` / `sr_validate` | `validate` is the full report — element-level layers (VR, VM, enumerated values, per-VR format, SpecificCharacterSet, pixel geometry, LUT) **plus** per-SOP-Class IOD conformance (Type 1/2 mandatory modules, top-level **and** nested), SR structural / coded / TID content-template checks, and RT cross-reference integrity; `iod_validate` is the narrower IOD-only view |
| Rendering / overlays | `render_overlay` | render a frame to 8-bit RGB and burn GSPS / SR / SEG / RTSTRUCT markup onto it — the frame an agent "sees" |
| Agent / MCP | `pydcm.mcp` | in-process MCP server over live pydcm objects |
| File sets | `pydcm.fileset` | read a DICOMDIR / File-set, iterate instances |

### DIMSE networking

```python
import pydcm.dimse as dimse               # drop-in for pynetdicom

ae = dimse.AE(ae_title="PYDCM")
assoc = ae.associate("localhost", 4242, ae_title="ORTHANC")   # dcm4chee: 11112, "DCM4CHEE"
assoc.send_c_echo()
assoc.send_c_store(pydcm.dcmread("ct.dcm"))   # persistent: many ops, one association
assoc.release()
```

`AE.start_server` runs the SCP side — `EVT_C_STORE` / `ECHO` / `FIND` / `GET` /
`MOVE` handlers plus the DIMSE-N set.

### DICOMweb client

Address a server by an **origin** plus a **base path** — the prefix differs per product
(Orthanc `/dicom-web`, dcm4chee-arc `/dcm4chee-arc/aets/DCM4CHEE/rs`, a root-mounted server `""`):

```python
from pydcm import dicomweb

S, BP = "http://localhost:8042", "/dicom-web"      # Orthanc
studies = dicomweb.search_studies(S, base_path=BP, matches={"PatientID": "42"})
for part10 in dicomweb.iter_study(S, study_uid, base_path=BP):
    ...                                    # streaming retrieve, bounded memory
dicomweb.store_instances(S, [open("ct.dcm", "rb").read()], base_path=BP)
```

See the DICOMweb and DIMSE how-tos for per-server connection tables, C-FIND/GET/MOVE and TLS.

### DICOM ↔ NIfTI, BIDS, DWI

```python
vol = pydcm.load_series("ct_series/")
vol.to_nifti("ct.nii.gz")                  # validated affine (incl. gantry tilt)
vol.to_nrrd("ct.nrrd", gzip=True)          # 3D Slicer (double-faithful LPS space)
vol.to_metaimage("ct.mha", compress=True)  # single-file MetaImage for ITK / nnU-Net
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

### Radiomics (IBSI)

```python
from pydcm.radiomics import radiomics

feats = radiomics("ct.dcm", mask="roi.dcm")    # pixels → HU, spacing from geometry
feats["firstorder_Mean"], feats["glcm_Contrast"], feats["shape_VoxelVolume"]
# 135 IBSI features / 10 classes; tune bins= / resample= / distances=,
# and @pydcm.radiomics.feature("firstorder") registers your own over the same grid.
```

### Segmentation (SEG)

```python
pydcm.write_seg("ct.dcm", labelmap, segments, output="seg.dcm")  # coded SEG from a labelmap
labels = pydcm.read_seg("seg.dcm")               # → labelmap (voxel = Segment Number)
masks  = pydcm.read_seg("seg.dcm", masks=True)   # → per-segment binary / fractional masks
```

### Waveforms (ECG / EEG)

```python
from pydcm import waveforms

wf  = waveforms.read_waveform("ecg.dcm")         # leads, units, sample rate, annotations
arr = waveforms.multiplex_array("ecg.dcm")       # [channels, samples] ndarray
raw = waveforms.to_mne("eeg.dcm")                # hand straight to MNE / neurokit2
```

### FHIR

```python
study = pydcm.fhir.imaging_study("study_dir/")   # whole study → FHIR R4 ImagingStudy (dict)
```

### Agent / MCP server

```console
$ python -m pydcm.mcp        # in-process MCP server (tools/list + tools/call over stdio)
```

`pydcm.mcp` exposes **79 tools** over the live engine — *more* than the command-line
surface, because it drives in-process objects (transforms / `Volume` / WSI / SR &amp;
SEG authoring with write + validate). For an agent learning the toolkit, the **Agent
Skill** (`docs/skill.md`) maps a task → the right `pydcm` call.

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
