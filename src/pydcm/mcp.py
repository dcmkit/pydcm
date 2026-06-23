# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — agent-facing MCP server (``pydcm.mcp``).

The Model Context Protocol projection of pydcm, over the shared native agent engine. A self-contained agent surface: it dispatches **in-process** to Python and covers
pydcm end to end — analysis (radiomics / DVH / validation), perfusion (DCE),
volume / 4-D / DWI assembly, WSI, waveforms, the structured-content reader,
authoring, de-identification, and DIMSE / DICOMweb networking.

Run it as an MCP stdio server (what an agent / Claude Desktop / `mcp` client connects to)::

    python -m pydcm.mcp

Each tool's handler returns a string: a JSON object (surfaced as ``structuredContent``), plain
text, or a ``data:image/*;base64,…`` data URI (surfaced as an image block). Register more tools
with the :func:`tool` decorator. Requires the ``_agent`` extension (included in builds with agent support).
"""
from __future__ import annotations

import json
import sys

try:
    from . import _agent
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.mcp requires the native _agent extension, which is not "
        "included in this build."
    ) from _e


# --- tool registry ----------------------------------------------------------------------------

#: Registered tools: list of {name, description, schema, fn}. fn(args: dict) -> str.
TOOLS: list[dict] = []


def tool(name: str, description: str, schema: dict):
    """Register an MCP tool. ``schema`` is the JSON Schema for ``arguments``; the decorated
    ``fn(args: dict) -> str`` returns the tool's output (JSON object string / text / data URI)."""
    def deco(fn):
        TOOLS.append({"name": name, "description": description, "schema": schema, "fn": fn})
        return fn
    return deco


def _str(x) -> str:
    return x if isinstance(x, str) else json.dumps(x)


def _dispatch(name: str, args):
    """The native ToolInvoker target: route a tools/call to its handler.

    Returns ``(code, output)``; a non-zero code is reported by the engine as an in-band MCP
    error (isError), not a protocol error."""
    args = args or {}
    for t in TOOLS:
        if t["name"] == name:
            try:
                return (0, _str(t["fn"](args)))
            except Exception as e:                   # surfaced in-band as isError
                return (1, f"{type(e).__name__}: {e}")
    return (1, f"unknown tool: {name}")


def _mcp_def(t: dict) -> str:
    return json.dumps({"name": t["name"], "description": t["description"], "inputSchema": t["schema"]})


def serve(verbose: bool = False) -> int:
    """Run the MCP stdio server until stdin EOF. Returns the engine exit code."""
    defs = [(t["name"], _mcp_def(t), "") for t in TOOLS]
    return _agent.serve_stdio(defs, _dispatch, "pydcm", "0.1", verbose)


# --- tools (pydcm capability surface, in-process) ---------------------------------------------

_PATH = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}


@tool("dicom_content",
      "Semantic content of a structured DICOM object (Segmentation / RT Structure Set / RT Plan "
      "/ RT Dose / Presentation State / Waveform / Ophthalmic Visual Field / Structured Report) "
      "as JSON; null if not a structured object.",
      {"type": "object",
       "properties": {"path": {"type": "string"},
                      "contours": {"type": "boolean", "description": "RTSTRUCT: include contour points"},
                      "control_points": {"type": "boolean", "description": "RTPLAN: include every control point"}},
       "required": ["path"]})
def _content(a):
    from .content import content
    return json.dumps(content(a["path"], contours=a.get("contours", False),
                              control_points=a.get("control_points", False)))


@tool("dicom_to_fhir",
      "Convert a DICOM instance to a FHIR R4 ImagingStudy resource (JSON).", _PATH)
def _to_fhir(a):
    from . import fhir
    return json.dumps(fhir.imaging_study(a["path"]))


@tool("hl7_parse",
      "Parse an HL7 v2 message into a list of segments [{id, fields}].",
      {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
def _hl7_parse(a):
    from . import hl7
    return json.dumps(hl7.parse(a["text"]))


@tool("hl7_build_oru",
      "Build an HL7 v2.5 ORU^R01 radiology-result message (ER7 text) from config / context / "
      "observations dicts.",
      {"type": "object",
       "properties": {"config": {"type": "object"}, "context": {"type": "object"},
                      "observations": {"type": "array", "items": {"type": "object"}}},
       "required": ["config", "context", "observations"]})
def _hl7_oru(a):
    from . import hl7
    return hl7.build_oru(a.get("config", {}), a.get("context", {}), a.get("observations", []))


@tool("rtdose_info",
      "Read an RT Dose file: grid geometry + dosimetry summary (units, max dose, DVH count) as "
      "JSON. Does not return the voxel grid.", _PATH)
def _rtdose(a):
    from . import rt
    g = rt.read_rtdose(a["path"])
    return json.dumps({"shape": list(g.shape), "dose_units": g.dose_units, "dose_type": g.dose_type,
                       "dose_summation_type": g.dose_summation_type, "max_dose": g.max_dose,
                       "spacing": list(g.spacing), "affine": list(g.affine), "num_dvhs": len(g.dvhs),
                       "sop_instance_uid": g.sop_instance_uid})


@tool("volume_info",
      "Assemble a DICOM series directory into a 3D volume; return geometry (shape, spacing, 4x4 "
      "affine, series UID) as JSON.",
      {"type": "object",
       "properties": {"path": {"type": "string", "description": "directory of DICOM slices"},
                      "recursive": {"type": "boolean"}},
       "required": ["path"]})
def _volume_info(a):
    import numpy as np
    from . import volume
    v = volume.load_series(a["path"], recursive=a.get("recursive", True))
    return json.dumps({"shape": list(v.shape), "spacing": list(v.spacing),
                       "affine": np.asarray(v.affine).tolist(),
                       "series_instance_uid": v.series_instance_uid})


@tool("volume_to_nifti",
      "Assemble a DICOM series directory into a 3D volume and write it to NIfTI-1.",
      {"type": "object",
       "properties": {"path": {"type": "string"}, "output": {"type": "string"},
                      "recursive": {"type": "boolean"}},
       "required": ["path", "output"]})
def _volume_to_nifti(a):
    from . import volume
    v = volume.load_series(a["path"], recursive=a.get("recursive", True))
    return json.dumps({"output": v.to_nifti(a["output"]), "shape": list(v.shape)})


@tool("resample_volume_to_spacing",
      "Assemble a DICOM series, resample to a target voxel spacing (z,y,x mm), and write NIfTI-1.",
      {"type": "object",
       "properties": {"path": {"type": "string"}, "output": {"type": "string"},
                      "spacing": {"type": "array", "items": {"type": "number"},
                                  "description": "(z, y, x) mm"},
                      "interp": {"type": "string", "enum": ["linear", "nearest", "cubic"]},
                      "is_label": {"type": "boolean"}},
       "required": ["path", "output", "spacing"]})
def _resample(a):
    from . import transforms, volume
    v = volume.load_series(a["path"], recursive=a.get("recursive", True))
    rv = transforms.resample_to_spacing(v, tuple(a["spacing"]),
                                        interp=a.get("interp", "linear"), is_label=a.get("is_label"))
    return json.dumps({"output": rv.to_nifti(a["output"]), "shape": list(rv.shape),
                       "spacing": list(rv.spacing)})


@tool("bids_sidecar",
      "Extract a BIDS JSON sidecar (acquisition parameters) from a DICOM series/file.", _PATH)
def _bids(a):
    from . import volume
    return json.dumps(volume.bids_sidecar(a["path"]))


# ── Analysis / conversion over the live native engine — the in-process complement
#    to the file-based CLI tools, so an agent reaches pydcm's full capability surface.

@tool("radiomics_features",
      "Compute the IBSI radiomic feature set (135 features, 10 classes) over an ROI from an image + "
      "mask DICOM; returns the feature dict.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "image DICOM (file or series dir)"},
          "mask": {"type": "string", "description": "ROI mask DICOM / SEG"},
          "bins": {"type": "integer"},
          "distances": {"type": "array", "items": {"type": "integer"}, "description": "GLCM neighbour distances"}},
       "required": ["path", "mask"]})
def _radiomics(a):
    from .radiomics import radiomics
    kw = {}
    if "bins" in a: kw["bins"] = int(a["bins"])
    if "distances" in a: kw["distances"] = a["distances"]
    feats = radiomics(a["path"], mask=a["mask"], **kw)
    return json.dumps({k: float(v) for k, v in feats.items()})


@tool("validate_iod",
      "IOD / module conformance for the file's SOP Class: every mandatory module's Type-1/2 "
      "attributes present (+ IOD-independent conditional rules), descending into nested sequences. "
      "Returns the findings list ([] = conformant). Narrow view — use 'validate' for the full report.",
      _PATH)
def _validate_iod(a):
    from ._core import iod_validate
    return json.dumps(iod_validate(a["path"]))


@tool("validate",
      "Full DICOM conformance check: IOD / module / conditional PLUS the element-level layers (VR, "
      "value multiplicity, enumerated values, value format, SpecificCharacterSet, pixel geometry, "
      "LUT) and the SR content tree. Returns {severity, tag, keyword, rule, message} ([] = conformant).",
      _PATH)
def _validate_full(a):
    from ._core import validate
    return json.dumps(validate(a["path"]))


@tool("dvh",
      "Compute a Dose-Volume Histogram for one ROI from an RT Structure Set + RT Dose; returns "
      "volume (cm³) and dose statistics (min / max / mean, Gy).",
      {"type": "object", "properties": {
          "structure": {"type": "string"}, "dose": {"type": "string"}, "roi": {"type": "integer"}},
       "required": ["structure", "dose", "roi"]})
def _dvh(a):
    from .rt import dvhcalc
    d = dvhcalc(a["structure"], a["dose"], int(a["roi"]))
    return json.dumps({"name": d.name, "volume": d.volume, "min": d.min,
                       "max": d.max, "mean": d.mean, "notes": d.notes})


@tool("assemble_4d_nifti",
      "Assemble a dynamic / multi-phase series (cine / multi-echo / DCE / cardiac phase) into a 4-D "
      "[T,Z,Y,X] NIfTI — the engine discovers the 4th axis on its own.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"}, "recursive": {"type": "boolean"}},
       "required": ["path", "output"]})
def _assemble_4d(a):
    from .volume import load_4d
    v = load_4d(a["path"], recursive=a.get("recursive", True))
    out = v.to_nifti(a["output"])
    axes = [{"kind": ax.kind, "size": len(ax.values)} for ax in getattr(v, "dimensions", [])]
    return json.dumps({"output": str(out), "volumes": v.n_volumes,
                       "shape": list(v.shape), "axes": axes})


@tool("dwi_to_nifti",
      "Assemble a DWI series into a 4-D NIfTI and write the FSL .bval / .bvec gradient table.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output_prefix": {"type": "string"}},
       "required": ["path", "output_prefix"]})
def _dwi(a):
    from .diffusion import save_dwi
    nii, bval, bvec = save_dwi(a["path"], a["output_prefix"])
    return json.dumps({"nifti": nii, "bval": bval, "bvec": bvec})


@tool("dce_parameter_maps",
      "DCE-MRI pharmacokinetic fit: assemble a dynamic series, fit a tissue model per voxel against a "
      "Parker AIF, and write Ktrans / ve / vp maps as DICOM Parametric Maps.",
      {"type": "object", "properties": {
          "path": {"type": "string", "description": "dynamic series directory"},
          "times_min": {"type": "array", "items": {"type": "number"}, "description": "acquisition times, minutes"},
          "output_dir": {"type": "string"},
          "model": {"type": "string", "enum": ["tofts", "ext_tofts", "patlak"]},
          "input": {"type": "string", "enum": ["signal", "concentration"]},
          "tr_s": {"type": "number"}, "fa_deg": {"type": "number"}},
       "required": ["path", "times_min", "output_dir"]})
def _dce(a):
    from . import dce
    import numpy as np
    times = np.asarray(a["times_min"], float)
    kw = {"model": a.get("model", "ext_tofts"), "input": a.get("input", "signal"),
          "aif": dce.parker_aif(times)}
    if "tr_s" in a: kw["tr_s"] = a["tr_s"]
    if "fa_deg" in a: kw["fa_deg"] = a["fa_deg"]
    maps = dce.fit_series(a["path"], times, **kw)
    written = dce.write_param_maps(a["path"], maps, output_dir=a["output_dir"])
    files = {k: str(v) for k, v in written.items()} if isinstance(written, dict) else None
    return json.dumps({"output_dir": a["output_dir"],
                       "params": list(files) if files else None, "files": files})


@tool("ophthalmic_visual_field",
      "Read an Ophthalmic Visual Field (static perimetry, Supplement 146) DICOM into structured JSON "
      "(test parameters, reliability, global results, per-point measurements).", _PATH)
def _opv(a):
    from .opv import read_visual_field
    return json.dumps(read_visual_field(a["path"]))


@tool("waveform_info",
      "Read a Waveform SOP instance (ECG / EEG / …): leads, units, sampling rate, filters and "
      "annotations as JSON (metadata only, not the raw sample arrays).", _PATH)
def _waveform(a):
    from . import waveforms
    w = waveforms.read_waveform(a["path"])
    out = {k: v for k, v in w.items() if k not in ("raw", "signals", "annotations")}
    out["num_annotations"] = len(w.get("annotations") or [])
    return json.dumps(out, default=str)


@tool("wsi_info",
      "Open a DICOM Whole-Slide Image pyramid and report its base dimensions, per-level dimensions, "
      "level count and microns-per-pixel.", _PATH)
def _wsi(a):
    from . import wsi
    s = wsi.open_slide(a["path"])
    props = s.properties
    return json.dumps({
        "dimensions": list(s.dimensions),
        "level_count": s.level_count,
        "level_dimensions": [list(d) for d in s.level_dimensions],
        "mpp_x": props.get(wsi.PROPERTY_NAME_MPP_X),
        "mpp_y": props.get(wsi.PROPERTY_NAME_MPP_Y),
        "vendor": props.get(wsi.PROPERTY_NAME_VENDOR)})


@tool("extract_encapsulated",
      "Extract the embedded document (PDF / CDA / STL / OBJ / …) from an Encapsulated Document DICOM "
      "to a file; returns its MIME type and title.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _extract_encap(a):
    from .encapdoc import read_encapsulated
    doc = read_encapsulated(a["path"])
    with open(a["output"], "wb") as f:
        f.write(doc.payload)
    return json.dumps({"output": a["output"], "mime": doc.mime, "type": doc.type, "title": doc.title})


# ── De-identification, authoring and networking — so pydcm.mcp is a self-contained
#    agent surface (no companion CLI needed).

@tool("deidentify_file",
      "De-identify a DICOM file (PS3.15 Annex E) and write the result — consistent UID remap, "
      "configurable retain / clean options.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "profile": {"type": "string"}, "retain_dates": {"type": "boolean"},
          "clean_descriptors": {"type": "boolean"}, "shift_dates_days": {"type": "integer"}},
       "required": ["path", "output"]})
def _deident(a):
    from .deident import deidentify
    kw = {k: a[k] for k in ("profile", "retain_dates", "clean_descriptors", "shift_dates_days") if k in a}
    with open(a["path"], "rb") as f:
        out = deidentify(f.read(), **kw)
    with open(a["output"], "wb") as f:
        f.write(out)
    return json.dumps({"output": a["output"]})


@tool("write_report",
      "Author a TID 1500 Measurement Report (DICOM SR) from a measurements structure; writes the SR file.",
      {"type": "object", "properties": {
          "measurements": {"type": "object", "description": "the TID 1500 measurements content"},
          "reference": {"type": "string", "description": "a DICOM file to copy patient/study identity from"},
          "output": {"type": "string"}},
       "required": ["measurements", "output"]})
def _report(a):
    from .sr import write_report
    out = write_report(a["measurements"], reference=a.get("reference"), output=a["output"])
    return json.dumps({"output": str(out)})


@tool("dimse_echo",
      "DIMSE C-ECHO (verification) against a remote SCP — confirm connectivity.",
      {"type": "object", "properties": {
          "host": {"type": "string"}, "port": {"type": "integer"},
          "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}},
       "required": ["host", "port"]})
def _dimse_echo(a):
    from . import dimse
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        st = assoc.send_c_echo()
        return json.dumps({"established": True, "status": getattr(st, "Status", 0) if st else None})
    finally:
        assoc.release()


@tool("dimse_store",
      "DIMSE C-STORE — send DICOM files to a remote SCP over one association.",
      {"type": "object", "properties": {
          "host": {"type": "string"}, "port": {"type": "integer"},
          "files": {"type": "array", "items": {"type": "string"}},
          "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}},
       "required": ["host", "port", "files"]})
def _dimse_store(a):
    from . import dimse
    from ._dicom import dcmread
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        sent = 0
        for f in a["files"]:
            assoc.send_c_store(dcmread(f))
            sent += 1
        return json.dumps({"established": True, "sent": sent})
    finally:
        assoc.release()


@tool("dicomweb_search",
      "DICOMweb QIDO-RS — query a remote server for studies / series / instances.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "level": {"type": "string", "enum": ["studies", "series", "instances"]},
          "matches": {"type": "object"}, "study_uid": {"type": "string"}, "series_uid": {"type": "string"},
          "auth": {"type": "string"}},
       "required": ["server"]})
def _dwq(a):
    from . import dicomweb
    level = a.get("level", "studies")
    kw = {"matches": a.get("matches"), "auth": a.get("auth", "")}
    if level == "studies":
        res = dicomweb.search_studies(a["server"], **kw)
    elif level == "series":
        res = dicomweb.search_series(a["server"], a["study_uid"], **kw)
    else:
        res = dicomweb.search_instances(a["server"], a["study_uid"], a["series_uid"], **kw)
    return json.dumps(res)


@tool("dicomweb_retrieve",
      "DICOMweb WADO-RS — retrieve a study's instances and write them to a directory.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "study_uid": {"type": "string"}, "output_dir": {"type": "string"},
          "auth": {"type": "string"}},
       "required": ["server", "study_uid", "output_dir"]})
def _dwr(a):
    import os
    from . import dicomweb
    os.makedirs(a["output_dir"], exist_ok=True)
    parts = dicomweb.retrieve_study(a["server"], a["study_uid"], auth=a.get("auth", ""))
    paths = []
    for i, buf in enumerate(parts):
        p = os.path.join(a["output_dir"], f"instance_{i:04d}.dcm")
        with open(p, "wb") as f:
            f.write(buf)
        paths.append(p)
    return json.dumps({"count": len(paths), "output_dir": a["output_dir"]})


@tool("dicomweb_store",
      "DICOMweb STOW-RS — store DICOM files to a remote server.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}},
          "auth": {"type": "string"}},
       "required": ["server", "files"]})
def _dws(a):
    from . import dicomweb
    blobs = []
    for f in a["files"]:
        with open(f, "rb") as fh:
            blobs.append(fh.read())
    res = dicomweb.store_instances(a["server"], blobs, auth=a.get("auth", ""))
    return json.dumps(res)


@tool("dimse_find",
      "DIMSE C-FIND (Query/Retrieve) against a remote SCP — match keys go in, matching identifiers "
      "(DICOM JSON) come back. Empty-string keys are return keys.",
      {"type": "object", "properties": {
          "host": {"type": "string"}, "port": {"type": "integer"},
          "level": {"type": "string", "enum": ["PATIENT", "STUDY", "SERIES", "IMAGE"]},
          "matches": {"type": "object", "description": "keyword → value, e.g. {PatientID: '42', StudyDate: ''}"},
          "model": {"type": "string", "enum": ["study", "patient"]},
          "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}},
       "required": ["host", "port"]})
def _dimse_find(a):
    from . import dimse, sop_class
    from ._dicom import Dataset
    model = (sop_class.PatientRootQueryRetrieveInformationModelFind if a.get("model") == "patient"
             else sop_class.StudyRootQueryRetrieveInformationModelFind)
    q = Dataset()
    q.QueryRetrieveLevel = a.get("level", "STUDY")
    for k, v in (a.get("matches") or {}).items():
        setattr(q, k, v)
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        results = [ident.to_json_dict() for status, ident in assoc.send_c_find(q, model)
                   if ident is not None]
        return json.dumps({"established": True, "count": len(results), "matches": results})
    finally:
        assoc.release()


# ── Inspection, rendering, reading and conversion — the everyday operations, so the
#    surface matches pydcm's breadth (not just its distinctive analysis tools).

def _default(o):
    """JSON fallback for reader outputs that carry numpy arrays / engine objects."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return {"_ndarray": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (bytes, bytearray)):
        return f"<{len(o)} bytes>"
    if hasattr(o, "_asdict"):
        return o._asdict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def _dump(obj):
    return json.dumps(obj, default=_default)


@tool("dicom_metadata",
      "Read a DICOM header into a flat {keyword: value} JSON (top-level non-bulk elements) — the quick "
      "'what is this file' inspector.",
      {"type": "object", "properties": {"path": {"type": "string"}, "include_private": {"type": "boolean"}},
       "required": ["path"]})
def _metadata(a):
    from . import dcmread
    ds = dcmread(a["path"])
    out = {}
    for el in ds:
        if el.VR in ("SQ", "OB", "OW", "OF", "OD", "OV", "UN") or el.keyword == "PixelData":
            continue
        if getattr(el, "is_private", False) and not a.get("include_private"):
            continue
        v = el.value
        out[el.keyword or str(el.tag)] = v if isinstance(v, (str, int, float)) else str(v)
    return json.dumps(out, default=str)


@tool("render_image",
      "Render a DICOM frame to a PNG — VOI/window applied (from tags, or pass window_center/width), "
      "MONOCHROME1 inverted. Requires Pillow.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"}, "frame": {"type": "integer"},
          "window_center": {"type": "number"}, "window_width": {"type": "number"}},
       "required": ["path", "output"]})
def _render(a):
    import numpy as np
    from . import dcmread
    from .pixels import apply_voi_lut
    try:
        from PIL import Image
    except ImportError:
        return json.dumps({"error": "render_image requires Pillow (pip install pillow)"})
    ds = dcmread(a["path"])
    arr = ds.pixel_array
    if arr.ndim >= 3 and int(ds.get("SamplesPerPixel", 1)) == 1:
        arr = arr[int(a.get("frame", 0))]
    if arr.ndim == 2:
        wl = arr.astype(np.float64)
        if a.get("window_width"):
            c, w = float(a.get("window_center", 0.0)), float(a["window_width"])
            lo, hi = c - w / 2.0, c + w / 2.0
        else:
            try:
                wl = apply_voi_lut(arr, ds).astype(np.float64)
            except Exception:
                pass
            lo, hi = float(wl.min()), float(wl.max())
        u8 = np.clip((wl - lo) / (hi - lo + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
            u8 = 255 - u8
        img = Image.fromarray(u8, "L")
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    img.save(a["output"])
    return json.dumps({"output": a["output"], "width": img.width, "height": img.height, "mode": img.mode})


@tool("scan_directory",
      "Scan a directory for DICOM files and summarise the inventory — file count, distinct studies / "
      "series, and a modality histogram.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "recursive": {"type": "boolean"},
          "max_read": {"type": "integer", "description": "cap on headers read (default 2000)"}},
       "required": ["path"]})
def _scan(a):
    from . import scan, dcmread
    paths = scan(a["path"], recursive=a.get("recursive", True))
    cap = int(a.get("max_read", 2000))
    studies, series, mods = set(), set(), {}
    for p in paths[:cap]:
        try:
            ds = dcmread(str(p))
        except Exception:
            continue
        studies.add(ds.get("StudyInstanceUID"))
        series.add(ds.get("SeriesInstanceUID"))
        m = ds.get("Modality", "?") or "?"
        mods[m] = mods.get(m, 0) + 1
    return json.dumps({"files": len(paths), "read": min(len(paths), cap),
                       "studies": len(studies - {None}), "series": len(series - {None}),
                       "modalities": mods})


@tool("deidentify_series",
      "De-identify every DICOM file under a directory (consistent UID remap across the set) and write "
      "the results to an output directory.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output_dir": {"type": "string"},
          "profile": {"type": "string"}, "retain_dates": {"type": "boolean"}, "recursive": {"type": "boolean"}},
       "required": ["path", "output_dir"]})
def _deident_series(a):
    import os
    from . import deidentify_series, scan
    files = [str(p) for p in scan(a["path"], recursive=a.get("recursive", True))]
    kw = {k: a[k] for k in ("profile", "retain_dates") if k in a}
    out = deidentify_series(files, **kw)
    os.makedirs(a["output_dir"], exist_ok=True)
    for i, buf in enumerate(out):
        with open(os.path.join(a["output_dir"], f"deid_{i:04d}.dcm"), "wb") as f:
            f.write(buf)
    return json.dumps({"input_files": len(files), "written": len(out), "output_dir": a["output_dir"]})


@tool("read_sr", "Read a Structured Report (TID 1500 measurement report) into structured JSON.", _PATH)
def _read_sr(a):
    from . import read_report
    return _dump(read_report(a["path"]))


@tool("read_segmentation",
      "Read a DICOM Segmentation's metadata — segments, labels, coded properties, geometry.", _PATH)
def _read_seg(a):
    from . import read_seg
    return _dump(read_seg(a["path"]))


@tool("read_paramap",
      "Read a Parametric Map — quantity / units / geometry metadata and value summary.", _PATH)
def _read_pm(a):
    from . import read_paramap
    return _dump(read_paramap(a["path"]))


@tool("read_key_object", "Read a Key Object Selection (KO) document into JSON.", _PATH)
def _read_ko(a):
    from . import read_ko
    return _dump(read_ko(a["path"]))


@tool("read_presentation_state",
      "Read a Grayscale / Color Softcopy Presentation State (GSPS) into JSON.", _PATH)
def _read_pr(a):
    from . import read_pr
    return _dump(read_pr(a["path"]))


@tool("read_annotations", "Read a Microscopy / Bulk Annotation (ANN) object into JSON.", _PATH)
def _read_ann(a):
    from . import read_ann
    return _dump(read_ann(a["path"]))


@tool("encapsulate",
      "Wrap a document (PDF / CDA / STL / OBJ / MTL) into an Encapsulated Document DICOM instance.",
      {"type": "object", "properties": {
          "src": {"type": "string"}, "output": {"type": "string"},
          "type": {"type": "string"}, "title": {"type": "string"},
          "reference": {"type": "string", "description": "a DICOM file to inherit patient/study identity"}},
       "required": ["src", "output"]})
def _encap(a):
    from . import write_encapsulated
    write_encapsulated(a["src"], type=a.get("type", "auto"), title=a.get("title"),
                       reference=a.get("reference"), output=a["output"])
    return json.dumps({"output": a["output"]})


@tool("legacy_to_enhanced",
      "Convert a classic single-frame CT / MR / PET series into one Legacy Converted Enhanced "
      "multi-frame instance.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _legacy(a):
    from . import write_legacy_converted, scan
    files = [str(p) for p in scan(a["path"], recursive=True)]
    write_legacy_converted(files, output=a["output"])
    return json.dumps({"input_frames": len(files), "output": a["output"]})


@tool("clean_burned_in_pixels",
      "Black out burned-in annotations in a DICOM image's pixel data and write the cleaned file.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "regions": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}},
                      "description": "optional [(x, y, w, h), …] boxes; omit to use the built-in rules"}},
       "required": ["path", "output"]})
def _clean_px(a):
    from . import clean_pixel_data
    with open(a["path"], "rb") as f:
        out = clean_pixel_data(f.read(), regions=a.get("regions"))
    with open(a["output"], "wb") as f:
        f.write(out)
    return json.dumps({"output": a["output"]})


@tool("diffusion_table",
      "Extract a DWI series' gradient table (b-values + b-vectors); optionally write FSL .bval/.bvec.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output_prefix": {"type": "string"}, "recursive": {"type": "boolean"}},
       "required": ["path"]})
def _difftable(a):
    import numpy as np
    from . import diffusion_table, scan
    files = [str(p) for p in scan(a["path"], recursive=a.get("recursive", True))]
    res = diffusion_table(files, output_prefix=a.get("output_prefix"))
    if isinstance(res, tuple) and len(res) == 2:
        bval, bvec = res
        return json.dumps({"num_volumes": int(np.asarray(bval).size),
                           "bvals": np.asarray(bval).ravel().tolist(),
                           "bvecs": np.asarray(bvec).tolist(),
                           "output_prefix": a.get("output_prefix")})
    return _dump(res)


@tool("sr_code_lookup",
      "Look up a coded concept's meaning (PS3.16) and optionally test membership in a Context Group.",
      {"type": "object", "properties": {
          "scheme": {"type": "string", "description": "e.g. SCT, DCM, LN"}, "value": {"type": "string"},
          "cid": {"type": "integer", "description": "Context Group ID to test membership"}},
       "required": ["scheme", "value"]})
def _sr_code(a):
    from . import sr_code_meaning, sr_cid_has
    out = {"scheme": a["scheme"], "value": a["value"],
           "meaning": sr_code_meaning(a["scheme"], a["value"])}
    if "cid" in a:
        out["in_cid"] = bool(sr_cid_has(int(a["cid"]), a["scheme"], a["value"]))
    return json.dumps(out)


# ── Editing, conversion, signing, file-sets and the rest of the DICOMweb / DIMSE
#    verbs — closing the gap with the file-based toolset so pydcm.mcp stands alone.

@tool("transcode",
      "Re-encode a DICOM file to another transfer syntax (compress or decompress); writes the result.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "transfer_syntax": {"type": "string", "description": "target Transfer Syntax UID"}},
       "required": ["path", "output", "transfer_syntax"]})
def _transcode(a):
    from . import dcmread
    ds = dcmread(a["path"])
    ts = a["transfer_syntax"]
    if ts in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"):
        ds.decompress()
        ds.file_meta.TransferSyntaxUID = ts
    else:
        ds.compress(ts)
    ds.save_as(a["output"], write_like_original=False)
    return json.dumps({"output": a["output"], "transfer_syntax": ts})


@tool("edit_tags",
      "Modify a DICOM file — set tag values and / or delete tags — and write the result.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "set": {"type": "object", "description": "{keyword: value} to assign"},
          "delete": {"type": "array", "items": {"type": "string"}, "description": "keywords to remove"}},
       "required": ["path", "output"]})
def _edit(a):
    from . import dcmread
    ds = dcmread(a["path"])
    for k, v in (a.get("set") or {}).items():
        setattr(ds, k, v)
    for k in (a.get("delete") or []):
        if k in ds:
            delattr(ds, k)
    ds.save_as(a["output"], write_like_original=False)
    return json.dumps({"output": a["output"], "set": list(a.get("set") or {}), "deleted": a.get("delete") or []})


@tool("dicom_compare",
      "Compare two DICOM files element-by-element; returns the differing / added / removed tags.",
      {"type": "object", "properties": {"path_a": {"type": "string"}, "path_b": {"type": "string"}},
       "required": ["path_a", "path_b"]})
def _compare(a):
    from . import dcmread
    da, db = dcmread(a["path_a"]), dcmread(a["path_b"])
    ka = {el.tag: el for el in da if el.VR != "SQ" and el.keyword != "PixelData"}
    kb = {el.tag: el for el in db if el.VR != "SQ" and el.keyword != "PixelData"}
    diffs = []
    for t in sorted(set(ka) | set(kb)):
        va = str(ka[t].value) if t in ka else None
        vb = str(kb[t].value) if t in kb else None
        if va != vb:
            kw = (ka.get(t) or kb.get(t)).keyword or str(t)
            diffs.append({"tag": kw, "a": va, "b": vb})
    return json.dumps({"differences": len(diffs), "diffs": diffs[:500]})


@tool("dicom_to_json",
      "Convert a DICOM file to the standard DICOM JSON model (PS3.18 Annex F).",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "include_pixels": {"type": "boolean"}}, "required": ["path"]})
def _to_json(a):
    from . import dcmread
    ds = dcmread(a["path"])
    j = ds.to_json_dict()
    if not a.get("include_pixels"):
        j.pop("7FE00010", None)
    if a.get("output"):
        with open(a["output"], "w") as f:
            json.dump(j, f)
        return json.dumps({"output": a["output"], "elements": len(j)})
    return json.dumps(j)


@tool("json_to_dicom",
      "Build a DICOM file from the DICOM JSON model (an inline object or a .json file).",
      {"type": "object", "properties": {
          "json": {"type": "object"}, "path": {"type": "string", "description": "a .json file (if json not inline)"},
          "output": {"type": "string"}}, "required": ["output"]})
def _from_json(a):
    from . import Dataset
    data = a.get("json")
    if data is None:
        with open(a["path"]) as f:
            data = json.load(f)
    ds = Dataset.from_json(data)
    ds.save_as(a["output"], write_like_original=False)
    return json.dumps({"output": a["output"]})


@tool("build_dicomdir",
      "Build a DICOMDIR file-set from a directory of DICOM files.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output_dir": {"type": "string"},
          "file_set_id": {"type": "string"}, "recursive": {"type": "boolean"}},
       "required": ["path", "output_dir"]})
def _mkdicomdir(a):
    from . import FileSet, scan
    fs = FileSet()
    n = 0
    for p in scan(a["path"], recursive=a.get("recursive", True)):
        fs.add(str(p))
        n += 1
    out = fs.write(a["output_dir"], file_set_id=a.get("file_set_id", "PYDCM_FILESET"))
    return json.dumps({"added": n, "output": str(out)})


@tool("write_key_object",
      "Author a Key Object Selection (KO) document referencing a set of instances.",
      {"type": "object", "properties": {
          "references": {"type": "array", "description": "referenced instances (files or reference dicts)"},
          "title": {"type": "string"}, "output": {"type": "string"},
          "patient_id": {"type": "string"}, "study_uid": {"type": "string"}},
       "required": ["references", "output"]})
def _write_ko(a):
    from . import write_ko
    write_ko(a["references"], title=a.get("title"), patient_id=a.get("patient_id", ""),
             study_uid=a.get("study_uid", ""), output=a["output"])
    return json.dumps({"output": a["output"], "references": len(a["references"])})


@tool("sign_dicom",
      "Apply a PS3.15 digital signature to a DICOM file (needs a PEM private key + certificate).",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "key": {"type": "string", "description": "PEM private-key file"},
          "cert": {"type": "string", "description": "PEM certificate file"},
          "mac": {"type": "string", "enum": ["SHA256", "SHA384", "SHA512"]}},
       "required": ["path", "output", "key", "cert"]})
def _sign(a):
    from .dsig import sign
    with open(a["path"], "rb") as f:
        data = f.read()
    with open(a["key"], "rb") as f:
        key = f.read()
    with open(a["cert"], "rb") as f:
        cert = f.read()
    out = sign(data, key, cert, mac=a.get("mac", "SHA256"))
    with open(a["output"], "wb") as f:
        f.write(out)
    return json.dumps({"output": a["output"]})


@tool("verify_signature",
      "Verify the PS3.15 digital signatures in a DICOM file; returns per-signature results.", _PATH)
def _verify(a):
    from .dsig import verify
    with open(a["path"], "rb") as f:
        return _dump(verify(f.read()))


@tool("dimse_move",
      "DIMSE C-MOVE — instruct an SCP to send matching instances to a destination AE title.",
      {"type": "object", "properties": {
          "host": {"type": "string"}, "port": {"type": "integer"}, "destination_ae": {"type": "string"},
          "level": {"type": "string", "enum": ["PATIENT", "STUDY", "SERIES", "IMAGE"]},
          "matches": {"type": "object"}, "model": {"type": "string", "enum": ["study", "patient"]},
          "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}},
       "required": ["host", "port", "destination_ae"]})
def _dimse_move(a):
    from . import dimse, sop_class
    from ._dicom import Dataset
    model = (sop_class.PatientRootQueryRetrieveInformationModelMove if a.get("model") == "patient"
             else sop_class.StudyRootQueryRetrieveInformationModelMove)
    q = Dataset()
    q.QueryRetrieveLevel = a.get("level", "STUDY")
    for k, v in (a.get("matches") or {}).items():
        setattr(q, k, v)
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        completed = failed = 0
        for status, ident in assoc.send_c_move(q, a["destination_ae"], model):
            if status is not None:
                completed = getattr(status, "NumberOfCompletedSuboperations", None) or completed
                failed = getattr(status, "NumberOfFailedSuboperations", None) or failed
        return json.dumps({"established": True, "completed": completed, "failed": failed})
    finally:
        assoc.release()


@tool("dicomweb_delete",
      "Delete a study / series / instance from a DICOMweb server.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "level": {"type": "string", "enum": ["study", "series", "instance"]},
          "study_uid": {"type": "string"}, "series_uid": {"type": "string"}, "instance_uid": {"type": "string"},
          "auth": {"type": "string"}},
       "required": ["server", "study_uid"]})
def _dwdel(a):
    from . import dicomweb
    lvl, auth = a.get("level", "study"), a.get("auth", "")
    if lvl == "study":
        st = dicomweb.delete_study(a["server"], a["study_uid"], auth=auth)
    elif lvl == "series":
        st = dicomweb.delete_series(a["server"], a["study_uid"], a["series_uid"], auth=auth)
    else:
        st = dicomweb.delete_instance(a["server"], a["study_uid"], a["series_uid"], a["instance_uid"], auth=auth)
    return json.dumps({"level": lvl, "status": st})


@tool("dicomweb_retrieve_metadata",
      "DICOMweb WADO-RS metadata — a study's or series' metadata as DICOM JSON (no pixels).",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "study_uid": {"type": "string"}, "series_uid": {"type": "string"},
          "auth": {"type": "string"}}, "required": ["server", "study_uid"]})
def _dwmeta(a):
    from . import dicomweb
    if a.get("series_uid"):
        r = dicomweb.retrieve_series_metadata(a["server"], a["study_uid"], a["series_uid"], auth=a.get("auth", ""))
    else:
        r = dicomweb.retrieve_study_metadata(a["server"], a["study_uid"], auth=a.get("auth", ""))
    return _dump(r)


@tool("dicomweb_retrieve_rendered",
      "DICOMweb WADO-RS rendered — fetch a server-rendered image (JPEG / PNG) and write it.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "study_uid": {"type": "string"}, "series_uid": {"type": "string"},
          "instance_uid": {"type": "string"}, "output": {"type": "string"},
          "window": {"type": "string", "description": "e.g. '40,400' center,width"}, "auth": {"type": "string"}},
       "required": ["server", "study_uid", "output"]})
def _dwrend(a):
    from . import dicomweb
    data = dicomweb.retrieve_rendered(a["server"], a["study_uid"], a.get("series_uid", ""),
                                      a.get("instance_uid", ""), window=a.get("window"), auth=a.get("auth", ""))
    with open(a["output"], "wb") as f:
        f.write(data)
    return json.dumps({"output": a["output"], "bytes": len(data)})


@tool("dimse_get",
      "DIMSE C-GET — retrieve matching instances over the association and write them to a directory "
      "(the SCP streams them back via C-STORE sub-operations).",
      {"type": "object", "properties": {
          "host": {"type": "string"}, "port": {"type": "integer"}, "output_dir": {"type": "string"},
          "level": {"type": "string", "enum": ["PATIENT", "STUDY", "SERIES", "IMAGE"]},
          "matches": {"type": "object"}, "model": {"type": "string", "enum": ["study", "patient"]},
          "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}},
       "required": ["host", "port", "output_dir"]})
def _dimse_get(a):
    import os
    from . import dimse, sop_class
    from ._dicom import Dataset
    model = (sop_class.PatientRootQueryRetrieveInformationModelGet if a.get("model") == "patient"
             else sop_class.StudyRootQueryRetrieveInformationModelGet)
    os.makedirs(a["output_dir"], exist_ok=True)
    saved = []

    def handle_store(event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        p = os.path.join(a["output_dir"], f"{ds.SOPInstanceUID}.dcm")
        ds.save_as(p, write_like_original=False)
        saved.append(p)
        return 0x0000

    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    ae.add_requested_context(model)
    roles = []
    for cx in dimse.StoragePresentationContexts[:120]:
        ae.add_requested_context(cx.abstract_syntax)
        roles.append(dimse.build_role(cx.abstract_syntax, scp_role=True))
    q = Dataset()
    q.QueryRetrieveLevel = a.get("level", "STUDY")
    for k, v in (a.get("matches") or {}).items():
        setattr(q, k, v)
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"),
                         ext_neg=roles, evt_handlers=[(dimse.evt.EVT_C_STORE, handle_store)])
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        completed = failed = 0
        for status, ident in assoc.send_c_get(q, model):
            if status is not None:
                completed = getattr(status, "NumberOfCompletedSuboperations", None) or completed
                failed = getattr(status, "NumberOfFailedSuboperations", None) or failed
        return json.dumps({"established": True, "saved": len(saved), "completed": completed,
                           "failed": failed, "output_dir": a["output_dir"]})
    finally:
        assoc.release()


@tool("segmentation_to_nifti",
      "Rasterise a DICOM Segmentation to a label-map NIfTI (each voxel = its segment number), with the "
      "geometry-correct affine.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _seg2nii(a):
    import numpy as np
    from . import read_seg
    from .volume import _write_nifti
    masks, meta = read_seg(a["path"], masks=True)
    masks = np.asarray(masks)
    if masks.ndim == 3:
        masks = masks[None]
    segnums = meta.get("segment_numbers") or list(range(1, masks.shape[0] + 1))
    nseg = len(segnums)
    if masks.shape[0] != nseg and masks.shape[1] == nseg:
        masks = np.moveaxis(masks, 1, 0)
    labelmap = np.zeros(masks.shape[1:], dtype=np.int16)
    for i, num in enumerate(segnums):
        labelmap[masks[i] > 0] = int(num)
    _write_nifti(labelmap, np.asarray(meta["affine"], dtype=float), a["output"])
    return json.dumps({"output": a["output"], "shape": list(labelmap.shape), "segments": list(segnums)})


@tool("extract_video",
      "Stream-copy the embedded video (MPEG-2 / MPEG-4 / H.264 / HEVC) out of a DICOM into an "
      "elementary video file — no re-encoding.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _extract_video(a):
    from . import dcmread
    from .encaps import defragment_data
    ds = dcmread(a["path"])
    ts = str(ds.file_meta.TransferSyntaxUID)
    if not ts.startswith("1.2.840.10008.1.2.4.10"):     # video range .100–.108
        return json.dumps({"error": f"not an embedded-video transfer syntax: {ts}"})
    stream = defragment_data(ds.PixelData)
    with open(a["output"], "wb") as f:
        f.write(stream)
    return json.dumps({"output": a["output"], "bytes": len(stream), "transfer_syntax": ts,
                       "frames": int(ds.get("NumberOfFrames", 1) or 1)})


@tool("dicomweb_wado_uri",
      "DICOMweb WADO-URI — retrieve a single object (DICOM or rendered) by study/series/instance UID.",
      {"type": "object", "properties": {
          "server": {"type": "string"}, "study_uid": {"type": "string"}, "series_uid": {"type": "string"},
          "instance_uid": {"type": "string"}, "output": {"type": "string"},
          "content_type": {"type": "string", "description": "e.g. application/dicom or image/jpeg"},
          "transfer_syntax": {"type": "string"}, "frame_number": {"type": "integer"}, "auth": {"type": "string"}},
       "required": ["server", "study_uid", "series_uid", "instance_uid", "output"]})
def _wadouri(a):
    from . import dicomweb
    data = dicomweb.retrieve_instance_wado_uri(
        a["server"], a["study_uid"], a["series_uid"], a["instance_uid"],
        content_type=a.get("content_type", "application/dicom"),
        transfer_syntax=a.get("transfer_syntax", ""), frame_number=int(a.get("frame_number", 0)),
        auth=a.get("auth", ""))
    with open(a["output"], "wb") as f:
        f.write(data)
    return json.dumps({"output": a["output"], "bytes": len(data)})


@tool("render_animation",
      "Render a multi-frame DICOM (cine loop or decoded video) to an animated GIF — VOI/window applied "
      "per frame, frame rate from FrameTime / CineRate. Requires Pillow.",
      {"type": "object", "properties": {
          "path": {"type": "string"}, "output": {"type": "string"},
          "window_center": {"type": "number"}, "window_width": {"type": "number"},
          "fps": {"type": "number"}, "max_frames": {"type": "integer"}},
       "required": ["path", "output"]})
def _animate(a):
    import numpy as np
    from . import dcmread
    from .pixels import apply_voi_lut
    try:
        from PIL import Image
    except ImportError:
        return json.dumps({"error": "render_animation requires Pillow (pip install pillow)"})
    ds = dcmread(a["path"])
    px = ds.pixel_array
    color = int(ds.get("SamplesPerPixel", 1)) == 3
    if px.ndim < 3 or (px.ndim == 3 and color):
        return json.dumps({"error": "not a multi-frame image"})
    if not color:
        if a.get("window_width"):
            c, w = float(a.get("window_center", 0.0)), float(a["window_width"])
            vol, lo, hi = px.astype(np.float64), c - w / 2.0, c + w / 2.0
        else:
            try:
                vol = apply_voi_lut(px, ds).astype(np.float64)
            except Exception:
                vol = px.astype(np.float64)
            lo, hi = float(vol.min()), float(vol.max())
        inv = ds.get("PhotometricInterpretation", "") == "MONOCHROME1"
    else:
        vol = px
    n = px.shape[0]
    cap = int(a.get("max_frames", 0) or n)
    imgs = []
    for i in range(min(n, cap)):
        fr = vol[i]
        if color:
            imgs.append(Image.fromarray(fr.astype(np.uint8)))
        else:
            u8 = np.clip((fr - lo) / (hi - lo + 1e-9) * 255.0, 0, 255).astype(np.uint8)
            if inv:
                u8 = 255 - u8
            imgs.append(Image.fromarray(u8, "L"))
    if a.get("fps"):
        dur = 1000.0 / float(a["fps"])
    elif ds.get("FrameTime"):
        dur = float(ds.FrameTime)
    else:
        dur = 1000.0 / float(ds.get("CineRate", 15) or 15)
    imgs[0].save(a["output"], save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    return json.dumps({"output": a["output"], "frames": len(imgs), "duration_ms": dur})


@tool("write_segmentation",
      "Author a binary DICOM Segmentation from a label-map NIfTI + a reference series — the inverse of "
      "segmentation_to_nifti. The engine aligns the mask to the reference geometry, so a model "
      "prediction on its own grid (saved as a labelled NIfTI) is resampled in automatically. For "
      "probability / occupancy maps use write_fractional_segmentation.",
      {"type": "object", "properties": {
          "reference": {"type": "string", "description": "reference image series (dir or file)"},
          "mask": {"type": "string", "description": "label-map NIfTI"},
          "segments": {"type": "array", "description": "[{label, number, ...}] segment descriptions"},
          "output": {"type": "string"}},
       "required": ["reference", "mask", "segments", "output"]})
def _write_seg(a):
    from .seg import seg_from_nifti
    seg_from_nifti(a["reference"], a["mask"], a["segments"], output=a["output"])
    return json.dumps({"output": a["output"], "segments": len(a["segments"])})


@tool("extract_raw_pixels",
      "Dump decoded pixel data to a raw binary file (row-major, native dtype); reports shape & dtype.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _raw(a):
    import numpy as np
    from . import dcmread
    arr = np.asarray(dcmread(a["path"]).pixel_array)
    with open(a["output"], "wb") as f:
        f.write(arr.tobytes())
    return json.dumps({"output": a["output"], "shape": list(arr.shape), "dtype": str(arr.dtype)})


@tool("dicom_to_tiff",
      "Write a DICOM image's decoded pixels to a TIFF file (multi-frame → multi-page).",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _to_tiff(a):
    import numpy as np
    import tifffile
    from . import dcmread
    arr = np.asarray(dcmread(a["path"]).pixel_array)
    tifffile.imwrite(a["output"], arr)
    return json.dumps({"output": a["output"], "shape": list(arr.shape)})


@tool("image_to_dicom",
      "Wrap a raster image (PNG / JPEG / TIFF) into a Secondary Capture DICOM instance.",
      {"type": "object", "properties": {
          "src": {"type": "string"}, "output": {"type": "string"},
          "patient_id": {"type": "string"}, "patient_name": {"type": "string"}, "modality": {"type": "string"}},
       "required": ["src", "output"]})
def _img2dcm(a):
    import numpy as np
    from PIL import Image
    from . import Dataset, uid, dataset as _dsmod
    im = Image.open(a["src"])
    if im.mode == "RGBA":
        im = im.convert("RGB")
    elif im.mode not in ("L", "RGB"):
        im = im.convert("L")
    rgb = im.mode == "RGB"
    arr = np.asarray(im)
    ds = Dataset()
    ds.PatientID = a.get("patient_id", "")
    ds.PatientName = a.get("patient_name", "")
    ds.Modality = a.get("modality", "OT")
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"          # Secondary Capture Image Storage
    ds.SOPInstanceUID = uid.generate_uid()
    ds.StudyInstanceUID = uid.generate_uid()
    ds.SeriesInstanceUID = uid.generate_uid()
    ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
    ds.SamplesPerPixel = 3 if rgb else 1
    ds.PhotometricInterpretation = "RGB" if rgb else "MONOCHROME2"
    if rgb:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    fm = _dsmod.FileMetaDataset()
    fm.TransferSyntaxUID = uid.ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = ds.SOPClassUID
    fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta = fm
    ds.save_as(a["output"], write_like_original=False)
    return json.dumps({"output": a["output"], "rows": ds.Rows, "columns": ds.Columns, "rgb": rgb})


# ── The long tail: DIMSE-N services + a few format converters. Niche, but they keep
#    pydcm.mcp at parity with the CLI for everything that isn't a daemon or XML-bound.

def _build_ds(d):
    """Build a Dataset from a {keyword: value} dict; list-of-dicts → a sequence."""
    from ._dicom import Dataset
    ds = Dataset()
    for k, v in (d or {}).items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            setattr(ds, k, [_build_ds(x) for x in v])
        else:
            setattr(ds, k, v)
    return ds


def _associate(a, ctx):
    from . import dimse
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    ae.add_requested_context(ctx)
    return ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))


_N_PROPS = {"host": {"type": "string"}, "port": {"type": "integer"},
            "called_ae": {"type": "string"}, "calling_ae": {"type": "string"}}


@tool("storage_commitment_scu",
      "DIMSE Storage Commitment (N-ACTION) — ask an SCP to commit to safekeeping a set of instances.",
      {"type": "object", "properties": {**_N_PROPS,
          "instances": {"type": "array", "items": {"type": "object"},
                        "description": "[{sop_class_uid, sop_instance_uid}, …]"}},
       "required": ["host", "port", "instances"]})
def _stgcmt(a):
    from . import uid
    from ._dicom import Dataset
    SC, SC_INST = "1.2.840.10008.1.20.1", "1.2.840.10008.1.20.1.1"
    ds = Dataset()
    ds.TransactionUID = uid.generate_uid()
    seq = []
    for r in a["instances"]:
        it = Dataset()
        it.ReferencedSOPClassUID = r["sop_class_uid"]
        it.ReferencedSOPInstanceUID = r["sop_instance_uid"]
        seq.append(it)
    ds.ReferencedSOPSequence = seq
    assoc = _associate(a, SC)
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        status, _ = assoc.send_n_action(ds, 1, SC, SC_INST)
        return json.dumps({"established": True, "status": getattr(status, "Status", None) if status else None,
                           "transaction_uid": str(ds.TransactionUID)})
    finally:
        assoc.release()


@tool("mpps_scu",
      "DIMSE Modality Performed Procedure Step — N-CREATE to start a PPS or N-SET to update / complete it.",
      {"type": "object", "properties": {**_N_PROPS,
          "operation": {"type": "string", "enum": ["create", "set"]},
          "sop_instance_uid": {"type": "string", "description": "required for set; generated for create"},
          "dataset": {"type": "object", "description": "the PPS attributes"}},
       "required": ["host", "port", "dataset"]})
def _mpps(a):
    from . import sop_class, uid
    MPPS = sop_class.ModalityPerformedProcedureStep
    ds = _build_ds(a.get("dataset"))
    assoc = _associate(a, MPPS)
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        if a.get("operation", "create") == "create":
            inst = a.get("sop_instance_uid") or uid.generate_uid()
            status, _ = assoc.send_n_create(ds, MPPS, inst)
        else:
            inst = a["sop_instance_uid"]
            status, _ = assoc.send_n_set(ds, MPPS, inst)
        return json.dumps({"established": True, "status": getattr(status, "Status", None) if status else None,
                           "sop_instance_uid": str(inst)})
    finally:
        assoc.release()


@tool("ups_scu",
      "DIMSE Unified Procedure Step — N-CREATE a workitem or N-ACTION to change its state.",
      {"type": "object", "properties": {**_N_PROPS,
          "operation": {"type": "string", "enum": ["create", "action"]},
          "sop_instance_uid": {"type": "string"}, "action_type": {"type": "integer"},
          "dataset": {"type": "object"}},
       "required": ["host", "port"]})
def _ups(a):
    from . import sop_class, uid
    UPS = sop_class.UnifiedProcedureStepPush
    ds = _build_ds(a.get("dataset"))
    assoc = _associate(a, UPS)
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        if a.get("operation", "create") == "create":
            inst = a.get("sop_instance_uid") or uid.generate_uid()
            status, _ = assoc.send_n_create(ds, UPS, inst)
        else:
            inst = a["sop_instance_uid"]
            status, _ = assoc.send_n_action(ds, int(a.get("action_type", 1)), UPS, inst)
        return json.dumps({"established": True, "status": getattr(status, "Status", None) if status else None,
                           "sop_instance_uid": str(inst)})
    finally:
        assoc.release()


@tool("ian_scu",
      "DIMSE Instance Availability Notification (N-CREATE) — notify an SCP that instances are available.",
      {"type": "object", "properties": {**_N_PROPS, "dataset": {"type": "object"}},
       "required": ["host", "port", "dataset"]})
def _ian(a):
    from . import sop_class, uid
    IAN = sop_class.InstanceAvailabilityNotification
    ds = _build_ds(a.get("dataset"))
    assoc = _associate(a, IAN)
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        inst = uid.generate_uid()
        status, _ = assoc.send_n_create(ds, IAN, inst)
        return json.dumps({"established": True, "status": getattr(status, "Status", None) if status else None,
                           "sop_instance_uid": str(inst)})
    finally:
        assoc.release()


@tool("print_scu",
      "DIMSE Basic Grayscale Print — N-CREATE a film session + film box, N-SET the image box, N-ACTION "
      "to print. Sends one image per film box.",
      {"type": "object", "properties": {**_N_PROPS,
          "image": {"type": "string", "description": "a DICOM image file to print"},
          "film_session": {"type": "object"}, "film_box": {"type": "object"}},
       "required": ["host", "port", "image"]})
def _print(a):
    from . import dcmread, dimse
    from ._dicom import Dataset
    FS, FB, IB = ("1.2.840.10008.5.1.1.1", "1.2.840.10008.5.1.1.2", "1.2.840.10008.5.1.1.4")
    META = "1.2.840.10008.5.1.1.9"                       # Basic Grayscale Print Management Meta
    ae = dimse.AE(ae_title=a.get("calling_ae", "PYDCM"))
    for c in (FS, FB, IB):
        ae.add_requested_context(c)
    assoc = ae.associate(a["host"], int(a["port"]), ae_title=a.get("called_ae", "ANY-SCP"))
    try:
        if not assoc.is_established:
            return json.dumps({"established": False})
        fs_st, fs = assoc.send_n_create(_build_ds(a.get("film_session")), FS, None, meta_uid=META)
        fb_ds = _build_ds(a.get("film_box"))
        fb_ds.ImageDisplayFormat = (a.get("film_box") or {}).get("ImageDisplayFormat", "STANDARD\\1,1")
        fb_ds.ReferencedFilmSessionSequence = [_ref(fs, FS)] if fs else []
        fb_st, fb = assoc.send_n_create(fb_ds, FB, None, meta_uid=META)
        img = dcmread(a["image"])
        ib_uid = _imagebox_uid(fb)
        ibx = Dataset()
        ibx.ImageBoxPosition = 1
        ibx.BasicGrayscaleImageSequence = [img]
        assoc.send_n_set(ibx, IB, ib_uid, meta_uid=META)
        pr_st, _ = assoc.send_n_action(Dataset(), 1, FB, str(fb.SOPInstanceUID), meta_uid=META)
        return json.dumps({"established": True, "printed": True,
                           "status": getattr(pr_st, "Status", None) if pr_st else None})
    finally:
        assoc.release()


def _ref(ds, cls):
    from ._dicom import Dataset
    it = Dataset()
    it.ReferencedSOPClassUID = cls
    it.ReferencedSOPInstanceUID = str(ds.SOPInstanceUID)
    return it


def _imagebox_uid(film_box):
    seq = getattr(film_box, "ReferencedImageBoxSequence", None)
    return str(seq[0].ReferencedSOPInstanceUID) if seq else None


@tool("sr_to_html",
      "Render any DICOM Structured Report to a standalone HTML file.",
      {"type": "object", "properties": {"path": {"type": "string"}, "output": {"type": "string"}},
       "required": ["path", "output"]})
def _sr2html(a):
    from . import sr_to_html
    doc = sr_to_html(a["path"])
    with open(a["output"], "w") as f:
        f.write(doc)
    return json.dumps({"output": a["output"], "bytes": len(doc)})


@tool("tiff_to_wsi",
      "Build a DICOM Whole-Slide pyramid from a pyramidal TIFF (each TIFF level → a pyramid level).",
      {"type": "object", "properties": {
          "src": {"type": "string"}, "output_dir": {"type": "string"},
          "tile": {"type": "integer"}, "mpp": {"type": "number"},
          "transfer_syntax": {"type": "string"}}, "required": ["src", "output_dir"]})
def _tiff2wsi(a):
    import os
    import numpy as np
    import tifffile
    from . import wsi
    with tifffile.TiffFile(a["src"]) as tf:
        series = tf.series[0]
        lv = getattr(series, "levels", None)
        if lv:
            levels = [np.asarray(level.asarray()) for level in lv]
        else:
            levels = [np.asarray(series.asarray())]
    levels = [x[..., :3] if x.ndim == 3 and x.shape[-1] >= 3 else
              np.repeat(x[..., None], 3, -1) if x.ndim == 2 else x for x in levels]
    parts = wsi.write_slide(levels, tile=int(a.get("tile", 256)), mpp=float(a.get("mpp", 0.25)),
                            transfer_syntax=a.get("transfer_syntax", "1.2.840.10008.1.2.4.50"))
    os.makedirs(a["output_dir"], exist_ok=True)
    paths = []
    for i, buf in enumerate(parts):
        p = os.path.join(a["output_dir"], f"level{i}.dcm")
        with open(p, "wb") as f:
            f.write(buf)
        paths.append(p)
    return json.dumps({"levels": len(levels), "output_dir": a["output_dir"], "files": len(paths)})


@tool("write_parametric_map",
      "Author a DICOM Parametric Map from a values NIfTI + a reference series. The NIfTI must match the "
      "reference geometry (e.g. one produced from the same series).",
      {"type": "object", "properties": {
          "reference": {"type": "string"}, "values": {"type": "string", "description": "values NIfTI"},
          "output": {"type": "string"}, "units": {"type": "string"}, "quantity": {"type": "string"},
          "label": {"type": "string"}}, "required": ["reference", "values", "output"]})
def _write_pm(a):
    import numpy as np
    import nibabel
    from . import write_paramap
    arr = np.asanyarray(nibabel.load(a["values"]).dataobj)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))              # NIfTI (X,Y,Z) → (slices, rows, cols)
    write_paramap(a["reference"], arr, units=a.get("units"), quantity=a.get("quantity"),
                  label=a.get("label"), output=a["output"])
    return json.dumps({"output": a["output"], "shape": list(arr.shape)})


# ── Authoring / validation verbs that complete the read↔write symmetry of the
#    surface (SR validation, RT-dose / GSPS / waveform / ANN authoring, generic
#    SR, fractional SEG). Each wraps the same pydcm function the file tools use.

@tool("validate_sr",
      "Validate a DICOM SR document's conformance — structural (value types / relationships), coded "
      "concepts (PS3.16 code tables), and TID content-template rules (TID 1500 family). Returns a list "
      "of {severity, location, message}; no `error` severity means it is structurally well-formed.", _PATH)
def _validate_sr(a):
    from . import sr_validate
    return json.dumps(sr_validate(a["path"]))


@tool("write_sr",
      "Author an arbitrary DICOM Structured Report from a content-tree document dict (title code + a "
      "`content` list of code / num / text / image / container items) — the general SR writer, vs the "
      "TID-1500-only write_report.",
      {"type": "object", "properties": {
          "document": {"type": "object",
                       "description": "SR content tree: patient_name / patient_id / study_uid / title / content[]"},
          "output": {"type": "string"}},
       "required": ["document", "output"]})
def _write_sr(a):
    from . import write_sr
    return json.dumps({"output": str(write_sr(a["document"], output=a["output"]))})


@tool("write_fractional_segmentation",
      "Author a FRACTIONAL DICOM Segmentation (probability / occupancy) from a maps NIfTI + a reference "
      "series — the fractional counterpart of write_segmentation. The maps NIfTI is 4-D (one volume per "
      "segment) of floats in [0, 1].",
      {"type": "object", "properties": {
          "reference": {"type": "string", "description": "reference image series (dir or file)"},
          "maps": {"type": "string", "description": "4-D float maps NIfTI (X, Y, Z, segment)"},
          "segments": {"type": "array", "description": "[{label, labelID, category, type, …}] per segment"},
          "type": {"type": "string", "enum": ["probability", "occupancy"]},
          "output": {"type": "string"}},
       "required": ["reference", "maps", "segments", "output"]})
def _write_seg_frac(a):
    import os
    import numpy as np
    import nibabel
    from . import write_seg_fractional, scan
    arr = np.asanyarray(nibabel.load(a["maps"]).dataobj)
    if arr.ndim == 4:
        arr = np.transpose(arr, (3, 2, 1, 0))           # (X,Y,Z,seg) → segment-major (seg, slices, rows, cols)
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))[None]        # single segment
    ref = a["reference"]                                 # a dir → its instance list (write_seg_fractional wants files)
    if os.path.isdir(ref):
        ref = [str(p) for p in scan(ref, recursive=True)]
    write_seg_fractional(ref, arr, a["segments"],
                         type=a.get("type", "probability"), output=a["output"])
    return json.dumps({"output": a["output"], "segments": len(a["segments"])})


@tool("write_rtdose",
      "Author an RT Dose grid from a values NIfTI + a reference series. Geometry (affine) and "
      "Patient / Study / Frame-of-Reference identity are taken from the reference; the values NIfTI must "
      "match its geometry. The read counterpart is rtdose_info.",
      {"type": "object", "properties": {
          "reference": {"type": "string", "description": "reference series (dir or file): geometry + identity"},
          "values": {"type": "string", "description": "dose-values NIfTI matching the reference geometry"},
          "output": {"type": "string"},
          "dose_units": {"type": "string", "enum": ["GY", "RELATIVE"]},
          "dose_type": {"type": "string"}, "dose_summation_type": {"type": "string"}},
       "required": ["reference", "values", "output"]})
def _write_rtdose(a):
    import numpy as np
    import nibabel
    from . import write_rtdose, load_series, scan
    arr = np.asanyarray(nibabel.load(a["values"]).dataobj)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    v = load_series(a["reference"])
    files = [str(p) for p in scan(a["reference"], recursive=True)]
    write_rtdose(arr, affine=v.affine, reference=files[0] if files else a["reference"],
                 dose_units=a.get("dose_units", "GY"), dose_type=a.get("dose_type", "PHYSICAL"),
                 dose_summation_type=a.get("dose_summation_type", "PLAN"), output=a["output"])
    return json.dumps({"output": a["output"], "shape": list(arr.shape)})


@tool("write_presentation_state",
      "Author a Softcopy Presentation State (GSPS by default) for one or more referenced images — "
      "window / level (VOI) and identity. The read counterpart is read_presentation_state.",
      {"type": "object", "properties": {
          "references": {"type": "array", "items": {"type": "string"},
                         "description": "image files the presentation state applies to"},
          "output": {"type": "string"},
          "kind": {"type": "string", "enum": ["GSPS", "CSPS", "PCSPS"]},
          "window": {"type": "array", "items": {"type": "number"},
                     "description": "[center, width] VOI window"},
          "content_label": {"type": "string"}},
       "required": ["references", "output"]})
def _write_pr(a):
    from . import write_pr
    w = a.get("window")
    write_pr(a["references"], kind=a.get("kind", "GSPS"), window=tuple(w) if w else None,
             content_label=a.get("content_label", "PS"), output=a["output"])
    return json.dumps({"output": a["output"], "references": len(a["references"])})


@tool("write_waveform",
      "Author a Waveform SOP instance (12-lead ECG / EEG / hemodynamic / audio) from a signals .npy "
      "file ([samples, channels] or [channels, samples]) at a given sampling frequency. The read "
      "counterpart is waveform_info.",
      {"type": "object", "properties": {
          "signals": {"type": "string", "description": "path to a .npy array of samples"},
          "output": {"type": "string"},
          "sampling_frequency": {"type": "number"},
          "kind": {"type": "string", "description": "ecg12 / eeg / hemodynamic / audio"},
          "units": {"type": "string"}, "patient_id": {"type": "string"}},
       "required": ["signals", "output", "sampling_frequency"]})
def _write_waveform(a):
    import numpy as np
    from . import waveforms
    sig = np.load(a["signals"])
    waveforms.write_waveform(a["output"], sig, sampling_frequency=a["sampling_frequency"],
                             kind=a.get("kind", "ecg12"), units=a.get("units", "mV"),
                             patient_id=a.get("patient_id", ""))
    return json.dumps({"output": a["output"], "shape": list(np.asarray(sig).shape)})


@tool("write_annotations",
      "Author a Microscopy Bulk Annotation (ANN) object from annotation groups + a source image. The "
      "read counterpart is read_annotations.",
      {"type": "object", "properties": {
          "source": {"type": "string", "description": "source / reference DICOM image"},
          "groups": {"type": "array",
                     "description": "[{number, label, generation_type, property_category, property_type, "
                                    "graphic_type, annotations, …}] per group"},
          "coordinate_type": {"type": "string", "enum": ["2D", "3D"]},
          "output": {"type": "string"}},
       "required": ["source", "groups", "output"]})
def _write_ann(a):
    from .ann import write_ann
    write_ann(a["source"], a["groups"], coordinate_type=a.get("coordinate_type", "2D"), output=a["output"])
    return json.dumps({"output": a["output"], "groups": len(a["groups"])})


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(serve(verbose="--verbose" in sys.argv or "-v" in sys.argv))

__all__ = ['serve', 'tool']
