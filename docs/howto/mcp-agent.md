# Agent / MCP server

`pydcm.mcp` is an in-process [Model Context Protocol](https://modelcontextprotocol.io)
server: a **self-contained** agent surface over pydcm — analysis, conversion, reading,
authoring, de-identification and networking, all in-process — including the live-object things that work on
DICOM *objects* in memory, not just files — to any MCP-aware agent runtime, over
stdio.

## Run it

```console
$ python -m pydcm.mcp          # MCP stdio server; -v logs JSON-RPC to stderr
```

Or embed it in your own process:

```python
import pydcm.mcp
pydcm.mcp.serve(verbose=True)   # serves over stdin/stdout until EOF
```

## Connect an MCP client

Point any MCP client (Claude Desktop, Claude Code, your own runtime) at the
module entry point:

```json
{
  "mcpServers": {
    "pydcm": {
      "command": "python",
      "args": ["-m", "pydcm.mcp"]
    }
  }
}
```

Use the interpreter that has pydcm installed (e.g. the absolute path to a venv's
`python`). Restart the client; the tools below appear, callable by name with
schema-checked arguments.

## Tools

The server covers pydcm end to end — seventy-plus tools, grouped here by job. Each
advertises a JSON Schema (`inputSchema`) over `tools/list`, so the agent gets
argument validation for free.

**Inspect & render**

| Tool | Does |
|---|---|
| `dicom_metadata` | Header → flat `{keyword: value}` JSON — the quick "what is this file" |
| `render_image` | Render a frame to PNG (VOI/window applied, MONOCHROME1 inverted) |
| `render_animation` | Multi-frame / cine / decoded video → animated GIF |
| `extract_video` | Stream-copy the embedded MPEG / H.264 / HEVC out to a video file |
| `scan_directory` | Inventory a folder — file count, distinct studies / series, modality histogram |
| `dicom_content` | Semantic JSON of a structured object (SEG / RT Struct / RT Plan / RT Dose / PS / Waveform / OPV / SR content tree) |

**Volumes & conversion**

| Tool | Does |
|---|---|
| `volume_info` / `volume_to_nifti` | Assemble a series → geometry, or write it to NIfTI |
| `resample_volume_to_spacing` | Resample a series to a target mm spacing → NIfTI |
| `assemble_4d_nifti` | Dynamic / multi-echo / cine series → 4-D `[T,Z,Y,X]` NIfTI |
| `dwi_to_nifti` / `diffusion_table` | DWI → 4-D NIfTI + FSL `.bval`/`.bvec`, or just the gradient table |
| `bids_sidecar` | BIDS JSON sidecar (acquisition parameters) from a series |
| `legacy_to_enhanced` | Classic single-frame CT/MR/PET series → Legacy Converted Enhanced multi-frame |
| `segmentation_to_nifti` | Rasterise a SEG to a label-map NIfTI (voxel = segment number) |

**Analysis**

| Tool | Does |
|---|---|
| `radiomics_features` | IBSI radiomic features (135 / 10 classes) over an image + mask |
| `dvh` | Dose-Volume Histogram for an ROI from RT Struct + RT Dose |
| `dce_parameter_maps` | DCE-MRI fit (Tofts / Patlak, Parker AIF) → Ktrans / ve / vp Parametric Maps |
| `validate_iod` | IOD conformance findings for a file |
| `validate_sr` | SR conformance — structural + coded + TID content-template (TID 1500) findings |
| `sr_code_lookup` | Coded-concept meaning (PS3.16) + Context-Group membership |

**Read structured objects**

| Tool | Does |
|---|---|
| `read_sr` | Structured Report (TID 1500) → JSON |
| `read_segmentation` | SEG metadata — segments, labels, coded properties, geometry |
| `read_paramap` | Parametric Map — quantity / units / geometry + value summary |
| `read_key_object` / `read_presentation_state` / `read_annotations` | KO / GSPS / bulk-annotation → JSON |
| `rtdose_info` | RT Dose grid summary (units, scaling, geometry, stored DVHs) |
| `wsi_info` | Whole-slide pyramid: levels, dimensions, microns-per-pixel |
| `waveform_info` | Waveform (ECG / EEG) leads / units / sampling / annotations |
| `ophthalmic_visual_field` | Static-perimetry (Sup-146) content as JSON |
| `extract_encapsulated` | Pull the embedded PDF / CDA / STL / … out of an Encapsulated Document |

**Author & convert out**

| Tool | Does |
|---|---|
| `write_report` | Author a TID 1500 Measurement Report (SR) from a measurements structure |
| `write_sr` | Author an arbitrary SR from a content-tree document (the general writer) |
| `write_key_object` | Author a Key Object Selection (KO) document referencing instances |
| `write_segmentation` | Author a binary SEG from a label-map NIfTI + reference (inverse of segmentation_to_nifti) |
| `write_fractional_segmentation` | Author a fractional (probability / occupancy) SEG from a maps NIfTI + reference |
| `write_parametric_map` | Author a Parametric Map from a values NIfTI + reference |
| `write_rtdose` | Author an RT Dose grid from a values NIfTI + reference |
| `write_presentation_state` | Author a Softcopy Presentation State (GSPS) — window / level + identity |
| `write_waveform` | Author an ECG / EEG / hemodynamic / audio Waveform from a signals `.npy` |
| `write_annotations` | Author a Microscopy Bulk Annotation (ANN) from annotation groups + source |
| `image_to_dicom` | Wrap a PNG / JPEG / TIFF raster into a Secondary Capture DICOM |
| `tiff_to_wsi` | Build a DICOM WSI pyramid from a pyramidal TIFF |
| `encapsulate` | Wrap a PDF / CDA / STL / OBJ / MTL into an Encapsulated Document DICOM |

**Edit, convert & file-sets**

| Tool | Does |
|---|---|
| `edit_tags` | Set and / or delete tags, write the result |
| `transcode` | Re-encode to another transfer syntax (compress / decompress) |
| `dicom_compare` | Element-by-element diff of two files |
| `dicom_to_json` / `json_to_dicom` | DICOM ↔ DICOM JSON model (PS3.18 Annex F) |
| `build_dicomdir` | Build a DICOMDIR file-set from a directory |
| `extract_raw_pixels` | Dump decoded pixels to a raw binary file |
| `dicom_to_tiff` | Write decoded pixels to a TIFF (multi-frame → multi-page) |
| `sr_to_html` | Render a Structured Report to standalone HTML |

**Signing**

| Tool | Does |
|---|---|
| `sign_dicom` / `verify_signature` | Apply / verify a PS3.15 digital signature |

**De-identification**

| Tool | Does |
|---|---|
| `deidentify_file` / `deidentify_series` | De-identify one file / a whole directory (PS3.15 Annex E, consistent UID remap) |
| `clean_burned_in_pixels` | Black out burned-in annotations in the pixel data |

**Networking**

| Tool | Does |
|---|---|
| `dimse_echo` / `dimse_store` / `dimse_find` / `dimse_move` / `dimse_get` | C-ECHO / C-STORE / C-FIND / C-MOVE / C-GET a DIMSE SCP |
| `dicomweb_search` | QIDO-RS query (studies / series / instances) |
| `dicomweb_retrieve` / `dicomweb_retrieve_metadata` / `dicomweb_retrieve_rendered` / `dicomweb_wado_uri` | WADO-RS instances / metadata / rendered image, or WADO-URI single object |
| `dicomweb_store` / `dicomweb_delete` | STOW-RS store / delete a study / series / instance |

**Workflow & print (DIMSE-N)**

| Tool | Does |
|---|---|
| `storage_commitment_scu` | Storage Commitment (N-ACTION) for a set of instances |
| `mpps_scu` | Modality Performed Procedure Step — N-CREATE / N-SET |
| `ups_scu` | Unified Procedure Step — N-CREATE / N-ACTION a workitem |
| `ian_scu` | Instance Availability Notification (N-CREATE) |
| `print_scu` | Basic Grayscale Print — film session / box / image box / print |

**EHR bridges**

| Tool | Does |
|---|---|
| `dicom_to_fhir` | DICOM instance or study → FHIR R4 `ImagingStudy` |
| `hl7_parse` / `hl7_build_oru` | Parse an HL7 v2 message / build an `ORU^R01` result |

## When to use it

`pydcm.mcp` runs **inside your Python process**, so it hands back live results —
an assembled volume's geometry, a resampled NIfTI on disk, a parsed RT Dose grid
— the things that are Python objects, not just files. Reach for it when the
agent is already working in Python and you want it to drive pydcm's
volume / transform / RT / EHR capabilities directly.
