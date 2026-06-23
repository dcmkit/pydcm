# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Grayscale Softcopy Presentation State authoring + reading (``pydcm.pr``).

The ``pr`` capability (GSPS): author a presentation state that records how
to display referenced images — window/level, Presentation LUT shape, rotate/flip,
displayed area, and graphic/text annotations on named layers — over the native
presentation-state engine. Reading reuses the existing PS content reader
(``pydcm.content``) — a native authoring capability.
"""
from __future__ import annotations

from . import _core
from .ko import _ref_of   # shared reference-extraction (dict | path | Dataset)


def write_pr(references, *, kind="GSPS", patient_name="", patient_id="", study_uid="", study_date="",
             sop_instance_uid="", series_instance_uid="",
             content_label="PS", content_description="", content_creator="",
             window=None, voi_luts=None, presentation_lut_shape="IDENTITY",
             rotation=0, h_flip=False, displayed_areas=None,
             graphic_layers=None, graphic_annotations=None,
             palette=None, icc_profile=None, color_space="", mask=None,
             blending=None, blending_display=None, output=None):
    """Author a Softcopy Presentation State for `references`.

    kind: "GSPS" (Grayscale, default), "COLOR" (Color SC PS, for RGB images — adds an
        ICC profile, drops the grayscale VOI/Presentation-LUT pipeline),
        "PSEUDO_COLOR" (Pseudo-Color SC PS, for grayscale images — adds a Palette
        Color LUT mapping stored values to RGB), "XAXRF" (XA/XRF Grayscale SC PS, the
        grayscale pipeline + Mask Subtraction), or "ADVANCED_BLENDING" (Advanced
        Blending SC PS, 11.8 — blend N pseudo-color/color inputs into true color).
    mask: XA/XRF Mask Subtraction — ``{operation:"AVG_SUB"|"TID"|"REV_TID",
        mask_frames?:[...], applicable_range?:[start,end], sub_pixel_shift?:[row,col],
        tid_offset?}``.
    blending: ADVANCED_BLENDING inputs — ``[{input_number, study_uid, series_uid,
        references:[{sop_class_uid, sop_instance_uid}], palette:{red,green,blue,
        first_mapped?}}]`` (each input is pseudo-colored via its palette).
    blending_display: how the inputs combine — ``[{mode:"EQUAL"|"FOREGROUND",
        inputs:[input_number,...], relative_opacity?}]``.
    references: a list of reference dicts (``{series_uid, sop_class_uid, sop_instance_uid,
        frame_numbers?}``) or DICOM paths / ``Dataset`` objects (identifiers extracted).
    window: convenience for one Softcopy VOI LUT — ``(center, width)`` or a dict
        ``{window_center, window_width, function?, explanation?}``. Use `voi_luts` for
        several. function: "LINEAR" (default) / "LINEAR_EXACT" / "SIGMOID".
    presentation_lut_shape: "IDENTITY" (default) or "INVERSE" (GSPS only).
    rotation / h_flip: spatial transform (0/90/180/270; flip horizontally).
    displayed_areas: list of ``{tlhc:[x,y], brhc:[x,y], size_mode?, magnification?,
        pixel_spacing?:[x,y]}``. If omitted, a SCALE-TO-FIT area covering the first
        path/Dataset reference's full extent is added automatically.
    graphic_layers: ``[{name, order?, description?, cielab?:[L,a,b]}]``.
    graphic_annotations: ``[{layer, texts?:[...], graphics?:[...]}]``.
    palette: PSEUDO_COLOR Palette Color LUT — ``{red:[...], green:[...], blue:[...],
        first_mapped?}``; each channel an equal-length list of 16-bit values (the entry
        count and 16-bit depth are taken from the data — there is nothing else to set).
    icc_profile: COLOR ICC profile bytes (0028,2000); color_space: defined term
        (0028,2002), e.g. "SRGB".
    output: write the PS there and return ``None``; if omitted, return Part-10 bytes.
    """
    if kind == "PSEUDO_COLOR" and palette is None:
        raise ValueError("kind='PSEUDO_COLOR' requires a palette (Palette Color LUT is mandatory)")
    if kind == "COLOR" and icc_profile is None:
        raise ValueError("kind='COLOR' requires icc_profile bytes (the Color SC PS ICC Profile)")
    refs = [_ref_of(r) for r in references]
    doc = {"ps_type": kind, "patient_name": patient_name, "patient_id": patient_id,
           "study_uid": study_uid, "study_date": study_date, "content_label": content_label or "PS",
           "content_description": content_description, "content_creator": content_creator,
           "presentation_lut_shape": presentation_lut_shape, "rotation": int(rotation),
           "h_flip": bool(h_flip), "references": refs}
    if palette is not None:
        p = dict(palette)
        p["red"], p["green"], p["blue"] = list(p["red"]), list(p["green"]), list(p["blue"])
        doc["palette"] = p
    if icc_profile is not None:
        doc["icc_profile"] = bytes(icc_profile)
    if color_space:
        doc["color_space"] = color_space
    if mask is not None:
        doc["mask"] = dict(mask)
    if blending:
        doc["blending"] = [dict(b) for b in blending]
    if blending_display:
        doc["blending_display"] = [dict(b) for b in blending_display]

    vois = list(voi_luts) if voi_luts else []
    if window is not None:
        if isinstance(window, dict):
            vois.append(window)
        else:
            c, w = window
            vois.append({"window_center": float(c), "window_width": float(w)})
    if vois:
        doc["voi_luts"] = vois
    if graphic_layers:
        doc["graphic_layers"] = list(graphic_layers)
    if graphic_annotations:
        doc["graphic_annotations"] = list(graphic_annotations)

    # First path/Dataset reference: inherit identity + supply a default displayed area.
    src = next((r for r in references if not isinstance(r, dict)), None)
    ds = None
    if src is not None:
        from . import dcmread
        from ._dicom import Dataset
        ds = src if isinstance(src, Dataset) else dcmread(str(src), stop_before_pixels=True, force=True)
        for key, attr in (("patient_name", "PatientName"), ("patient_id", "PatientID"),
                          ("study_uid", "StudyInstanceUID"), ("study_date", "StudyDate")):
            if not doc[key]:
                v = getattr(ds, attr, None)
                if v not in (None, ""):
                    doc[key] = str(v)

    if displayed_areas:
        doc["displayed_areas"] = list(displayed_areas)
    elif ds is not None and getattr(ds, "Rows", None) and getattr(ds, "Columns", None):
        doc["displayed_areas"] = [{"tlhc": [1, 1], "brhc": [int(ds.Columns), int(ds.Rows)],
                                   "size_mode": "SCALE TO FIT"}]

    # This Presentation State's own identity. Empty keeps the engine's
    # DETERMINISTIC counter, which is reset per export — so two states built in
    # one process claim the SAME SOP Instance UID. Right for one self-contained
    # file, a global-uniqueness violation for a producer that mints many.
    if sop_instance_uid:
        doc["sop_instance_uid"] = str(sop_instance_uid)
    if series_instance_uid:
        doc["series_instance_uid"] = str(series_instance_uid)
    return _core.write_pr(doc, str(output) if output else "")


def read_pr(path):
    """Read a Presentation State's semantic content (referenced images, presentation LUT
    shape, displayed areas, graphic layers, annotations, …) as a dict, or ``None`` when
    `path` is not a presentation state. Reuses the shared PS content reader."""
    from . import content
    c = content(str(path))
    return c if c and c.get("type") == "presentation_state" else None


# ── class API (over write_pr) ────────────────────────
def _layer_to_dict(L):
    """A GraphicLayer Dataset -> write_pr layer dict."""
    if isinstance(L, dict):
        return L
    d = {"name": str(L.GraphicLayer)}
    if "GraphicLayerOrder" in L:
        d["order"] = int(L.GraphicLayerOrder)
    if "GraphicLayerDescription" in L:
        d["description"] = str(L.GraphicLayerDescription)
    if "GraphicLayerRecommendedDisplayCIELabValue" in L:
        d["cielab"] = [int(v) for v in L.GraphicLayerRecommendedDisplayCIELabValue]
    return d


def _annotation_to_dict(A):
    """A GraphicAnnotation Dataset -> write_pr annotation dict."""
    if isinstance(A, dict):
        return A
    a = {"layer": str(A.GraphicLayer)}
    texts = []
    for t in A.get("TextObjectSequence", []) or []:
        td = {"text": str(t.UnformattedTextValue)}
        units = str(t.get("BoundingBoxAnnotationUnits", t.get("AnchorPointAnnotationUnits", "PIXEL")))
        td["units"] = "DISPLAY" if units == "DISPLAY" else "PIXEL"
        if "BoundingBoxTopLeftHandCorner" in t and "BoundingBoxBottomRightHandCorner" in t:
            tl, br = t.BoundingBoxTopLeftHandCorner, t.BoundingBoxBottomRightHandCorner
            td["bounding_box"] = [float(tl[0]), float(tl[1]), float(br[0]), float(br[1])]
        if "AnchorPoint" in t:
            ap = t.AnchorPoint
            td["anchor"] = [float(ap[0]), float(ap[1])]
        texts.append(td)
    graphics = []
    for g in A.get("GraphicObjectSequence", []) or []:
        graphics.append({
            "graphic_type": str(g.GraphicType),
            "units": "DISPLAY" if str(g.GraphicAnnotationUnits) == "DISPLAY" else "PIXEL",
            "filled": str(g.get("GraphicFilled", "N")) == "Y",
            "points": [float(x) for x in g.GraphicData],
        })
    if texts:
        a["texts"] = texts
    if graphics:
        a["graphics"] = graphics
    return a


def _make_ps(kind, referenced_images, series_instance_uid, series_number, sop_instance_uid,
             instance_number, manufacturer, manufacturer_model_name, software_versions,
             device_serial_number, content_label, content_description=None,
             graphic_annotations=None, graphic_layers=None, presentation_lut_shape=None,
             **_kwargs):
    import os as _os
    import tempfile as _tempfile
    from . import dcmread
    from .seg import _SEG_TMPS
    kw = {}
    if graphic_layers:
        kw["graphic_layers"] = [_layer_to_dict(L) for L in graphic_layers]
    if graphic_annotations:
        kw["graphic_annotations"] = [_annotation_to_dict(a) for a in graphic_annotations]
    if presentation_lut_shape is not None:
        kw["presentation_lut_shape"] = str(presentation_lut_shape)
    fd, tmp = _tempfile.mkstemp(suffix=".dcm"); _os.close(fd)
    _SEG_TMPS.append(tmp)
    write_pr(list(referenced_images), kind=kind, content_label=content_label,
             content_description=content_description or "", output=tmp, **kw)
    ds = dcmread(tmp)
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = int(series_number)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = int(instance_number)
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = manufacturer_model_name
    ds.SoftwareVersions = (software_versions if isinstance(software_versions, str)
                           else list(software_versions))
    ds.DeviceSerialNumber = device_serial_number
    return ds


def GrayscaleSoftcopyPresentationState(referenced_images, series_instance_uid, series_number,
                                       sop_instance_uid, instance_number, manufacturer,
                                       manufacturer_model_name, software_versions,
                                       device_serial_number, content_label, **kwargs):
    """Constructor (GSPS)."""
    return _make_ps("GSPS", referenced_images, series_instance_uid, series_number,
                    sop_instance_uid, instance_number, manufacturer, manufacturer_model_name,
                    software_versions, device_serial_number, content_label, **kwargs)


def ColorSoftcopyPresentationState(referenced_images, series_instance_uid, series_number,
                                   sop_instance_uid, instance_number, manufacturer,
                                   manufacturer_model_name, software_versions,
                                   device_serial_number, content_label, **kwargs):
    """Constructor (Color SC PS)."""
    return _make_ps("COLOR", referenced_images, series_instance_uid, series_number,
                    sop_instance_uid, instance_number, manufacturer, manufacturer_model_name,
                    software_versions, device_serial_number, content_label, **kwargs)


def PseudoColorSoftcopyPresentationState(referenced_images, series_instance_uid, series_number,
                                         sop_instance_uid, instance_number, manufacturer,
                                         manufacturer_model_name, software_versions,
                                         device_serial_number, content_label, **kwargs):
    """Constructor."""
    return _make_ps("PSEUDO_COLOR", referenced_images, series_instance_uid, series_number,
                    sop_instance_uid, instance_number, manufacturer, manufacturer_model_name,
                    software_versions, device_serial_number, content_label, **kwargs)


__all__ = ["write_pr", "read_pr", "GrayscaleSoftcopyPresentationState",
           "ColorSoftcopyPresentationState", "PseudoColorSoftcopyPresentationState"]
