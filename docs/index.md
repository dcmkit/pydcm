# pydcm

**The complete DICOM toolkit for Python.** One wheel **opens any DICOM — from any
scanner, vendor or country**. It decodes **every transfer syntax** (JPEG,
JPEG-2000, JPEG-LS, RLE, embedded video, and the modern **JPEG-XL / HTJ2K** most
toolchains still can't read) with no codec plugins, and reads **every character
set** correctly — full **Japanese / Korean / Chinese** multibyte and every
ISO 2022 escape, decoded adaptively across vendors where strict readers mangle
the text. From there it hands you NumPy / PyTorch arrays zero-copy and carries
you the whole way: 3-D volumes, NIfTI, RT dose &amp; DVH, radiomics, whole-slide
tiles, segmentations, structured reports, waveforms, and DICOM networking. One
install, one native engine, the entire pipeline.

!!! tip "No “unsupported” file"
    **Every pixel format, every encoding.** Compressed or not — including
    JPEG-XL and HTJ2K — pixels decode with nothing to install. Text from any
    vendor or locale comes back as correct UTF-8 (Latin / Cyrillic / Greek /
    Arabic / Hebrew / Thai, Shift-JIS, EUC-KR, GB18030, …), handled adaptively
    where a strict reader would error or garble it.

```python
import pydcm

ds  = pydcm.dcmread("scan.dcm")
px  = ds.pixel_array                     # any transfer syntax, no plugins
vol = pydcm.load_series("ct_series/")    # sorted 3-D HU volume + affine
vol.to_nifti("ct.nii.gz")
```

The engine is compiled into the wheel — nothing to assemble, no plugins, no
codec packages, no version matrix — with the same fast native path whether
you're decoding a frame, building a volume, or running a transform. The API is
clean and Pythonic, and most existing Python DICOM code runs against it
unchanged.

!!! warning "Not a medical device"
    pydcm is not intended or cleared for clinical or diagnostic use. Decoded
    pixels, HU, dose and derived values are for research and engineering only.

## One wheel, the whole pipeline

Every capability below ships in the same wheel, over the same native engine —
each verified for correctness against reference data where exactness matters:

| Area | pydcm | What it does |
|---|---|---|
| Read / write / decode | `dcmread`, `pixel_array`, `save_as` | **every transfer syntax + every character set** decoded; byte-verbatim editing; near-total element fidelity |
| De-identification | `deidentify`, `deidentify_series`, `clean_pixel_data` | PS3.15 Annex E profile, consistent UID remap across a study, burned-in-pixel blackout |
| Validation | `iod_validate`, `sr_validate` | per-SOP-Class IOD mandatory-module Type-1/2 conformance (nested too) + RT cross-reference integrity &amp; identifier uniqueness + conditional-presence rules (palette / modality LUT / VOI window / lossy / pixel-padding); SR structural + coded + TID content-template conformance |
| DIMSE networking | `pydcm.dimse` | SCU + SCP, all DIMSE services, persistent associations |
| DICOMweb | `pydcm.dicomweb` | QIDO / WADO / STOW + UPS-RS + delete against a remote server; Bearer / Basic auth |
| 3-D / 4-D volumes | `load_series`, `load_4d` | spatially-sorted HU volume; 4-D `[T, Z, Y, X]` stacks (cine / multi-echo / dynamic) |
| DICOM ↔ NIfTI / BIDS / DWI | `Volume.to_nifti`, `from_nifti`, `bids_sidecar`, `load_dwi`, `save_dwi` | spatially-correct affine incl. gantry tilt; FSL `.bval`/`.bvec`; NIfTI → DICOM too |
| DICOM → NRRD / MetaImage | `Volume.to_nrrd`, `Volume.to_metaimage` | 3D Slicer (`.nrrd`) + ITK / nnU-Net (`.mha`); double-faithful LPS geometry |
| DICOMDIR / file-sets | `pydcm.FileSet` | read a DICOMDIR, iterate / `find` instances |
| Legacy Converted Enhanced | `write_legacy_converted` | classic single-frame CT/MR/PET → enhanced multi-frame |
| SR coding (PS3.16) | `sr_code_meaning`, `sr_validate_code`, `sr_cid_has` | DICOM code table + context-group membership / validation |
| Preprocessing | `pydcm.transforms` | resample / normalize / sliding-window; **bit-exact** spatial ops, two interpolation conventions |
| Whole-slide imaging | `pydcm.wsi` | **read + write** the DICOM WSI pyramid — tile/region reads, author a pyramid from RGB levels; bit-exact multi-vendor |
| RT dosimetry | `pydcm.rt`, `dvhcalc` | dose read / write + point dose (`dose_at`) + DVH with full ROI coverage |
| Spatial registration | `pydcm.registration` | read + author the 4x4 transform between two Frames of Reference (66.1) |
| Tumour response | `pydcm.recist` | RECIST 1.1 / iRECIST / mRECIST / Cheson-Lugano / RANO / PCWG3 + Feret / volume off a SEG mask |
| Perfusion (DCE-MRI) | `pydcm.dce` | pharmacokinetic modelling — Tofts / Ext-Tofts / Patlak, Parker / population AIF, VFA T1 maps |
| Perfusion (DSC-MRI) | `pydcm.dsc` | SVD deconvolution — CBF / CBV / MTT / Tmax / TTP, sSVD / cSVD / oSVD, measured AIF, leakage correction |
| Diffusion tensor (DTI) | `pydcm.dti` | tensor fit → FA / MD / DEC / Westin maps + deterministic RK4 tractography from a DWI series |
| Radiomics | `pydcm.radiomics` | the full IBSI set — 135 features across 10 classes; register custom Python features over the same grid |
| Semantic content | `pydcm.content` | one reader for SEG / RT / Presentation State / Waveform / Ophthalmic Visual Field / Surface Segmentation / Structured Report → structured JSON |
| SEG / Parametric Map / SR | `write_seg`, `write_seg_fractional`, `write_paramap`, `write_report` | coded segmentations (binary + fractional, or from a model prediction) / parametric maps / measurement reports, lossless round-trip |
| KO / GSPS / annotations | `write_ko`, `write_pr`, `read_ann` | Key Object Selection / Presentation State / bulk annotations |
| Encapsulated documents | `write_encapsulated`, `read_encapsulated` | PDF / CDA / STL / OBJ / MTL ↔ DICOM |
| Surface meshes | `read_surface` | Surface Segmentation (66.5) → `(N,3)` points / `(M,3)` triangles, all primitives triangulated |
| Waveforms | `pydcm.waveforms` | ECG / EEG read &amp; write; arrays ready for analysis tools |
| Ophthalmic visual field | `pydcm.opv` | static perimetry → pandas / JSON; IOD conformance |
| FHIR / HL7 bridges | `pydcm.fhir`, `pydcm.hl7` | DICOM → FHIR R4 `ImagingStudy`; HL7 v2 parse / `ORU^R01` build |
| Agent / MCP server | `pydcm.mcp` | in-process MCP — **90+ tools spanning the whole toolkit** for an LLM agent |

## Where to go

- [Install](install.md)
- [Quickstart](quickstart.md) — decode, volumes, networking, RT, WSI in ten minutes
- [Behaviour notes](divergences.md) — deliberate behaviours worth knowing, and migration tips
- [How-to recipes](howto/index.md) — task-focused guides for every capability above
- [Agent / MCP server](howto/mcp-agent.md) — drive pydcm's live-object tools from an LLM agent
- [API reference](api.md) — generated from docstrings
- [Transforms — precision &amp; references](transforms_references.md) — why "bit-exact"
  is defined per framework, and what each op is checked against
