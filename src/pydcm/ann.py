# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — Microscopy Bulk Simple Annotations reading + authoring (``pydcm.ann``).

The ``ann`` capability: read AND write a Microscopy Bulk Simple Annotations
object (SOP 1.2.840.10008.5.1.4.1.1.91.1) — the compact format for huge numbers of
whole-slide annotations (cells / nuclei / regions). Both directions live in the
native engine (read + build over the shared Part-10 emit/parse primitives); these
are the thin marshalling wrappers.
"""
from __future__ import annotations

import numpy as np

from . import _core, _native

# Points per annotation for the fixed-size graphic types (variable types —
# POLYLINE / POLYGON — use the Long Primitive Point Index List instead).
_FIXED_POINTS = {"POINT": 1, "RECTANGLE": 4, "ELLIPSE": 4}


def read_ann(path):
    """Read a Microscopy Bulk Simple Annotations file.

    Returns a dict ``{coordinate_type, groups:[...]}`` or ``None`` if `path` is not
    a Bulk Annotations object. Each group carries its identity (number/uid/label/
    generation_type), coded ``property_category`` / ``property_type``,
    ``graphic_type``, ``num_annotations``, and ``annotations``: a list of
    ``(n_points, dim)`` float64 arrays decoded from the bulk coordinates (dim is 2
    for a "2D" coordinate type, 3 for "3D"). Each group's ``measurements`` is a list
    of ``{name, unit, values, annotation_index}`` — `values` a float64 array (one per
    annotation, or per `annotation_index` when sparse).
    """
    doc = _core.read_ann(str(path))
    if doc is None:
        return None
    for g in doc["groups"]:
        dim = g["dimensionality"]
        coords = np.frombuffer(g.pop("coords"), dtype=np.float64)
        pts = coords.reshape(-1, dim) if coords.size else np.empty((0, dim), np.float64)
        n = int(g["num_annotations"])
        idx = g.pop("index_list")
        anns = []
        if g["graphic_type"] in _FIXED_POINTS:
            ppa = _FIXED_POINTS[g["graphic_type"]]
            anns = [pts[i * ppa:(i + 1) * ppa] for i in range(n)]
        elif idx:                       # variable graphics: 1-based value offsets,
            for i in range(len(idx)):   # one per annotation (index list is authoritative)
                start = (idx[i] - 1) // dim
                end = ((idx[i + 1] - 1) // dim) if i + 1 < len(idx) else len(pts)
                anns.append(pts[start:end])
        elif n:                          # one variable-length annotation, no index list
            anns = [pts]
        g["annotations"] = anns
        for m in g["measurements"]:      # bulk float32 → float64 array, one value per annotation
            m["values"] = np.frombuffer(m.pop("values"), dtype=np.float64)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
#  Authoring — thin marshalling over the native annotation builder (the exact
#  inverse of read; the IOD authoring + bulk encoding lives natively, reused by
#  the CLI / server). These helpers just shape Python inputs into the call.
# ─────────────────────────────────────────────────────────────────────────────
def _ct(x):
    s = str(x).upper()
    return "3D" if ("3D" in s or s == "SCOORD3D") else "2D"


def _code_dict(c):
    if hasattr(c, "value") and hasattr(c, "scheme_designator"):     # Code / CodedConcept
        return {"value": str(c.value), "scheme": str(c.scheme_designator), "meaning": str(c.meaning)}
    return {"value": str(c[0]), "scheme": str(c[1]), "meaning": str(c[2])}   # (value, scheme, meaning)


def _group_dict(g):
    import pydcm
    out = {"number": int(g.get("number", 1)), "uid": g.get("uid") or pydcm.generate_uid(),
           "label": g.get("label", ""), "generation_type": str(g.get("generation_type", "MANUAL")),
           "graphic_type": str(g["graphic_type"]),
           "property_category": _code_dict(g["property_category"]),
           "property_type": _code_dict(g["property_type"]),
           # each annotation flattened to [x, y, (z,) …]; native concatenates + indexes.
           "annotations": [np.asarray(a, dtype="float64").ravel().tolist() for a in g["annotations"]]}
    if g.get("measurements"):
        out["measurements"] = [
            {"name": _code_dict(m["name"]), "unit": _code_dict(m["unit"]),
             "values": [float(v) for v in m["values"]],
             "annotation_index": list(m.get("annotation_index") or [])}
            for m in g["measurements"]]
    return out


def write_ann(source, groups, *, coordinate_type="2D", series_instance_uid=None,
              series_number=1, sop_instance_uid=None, instance_number=1,
              manufacturer="pydcm", manufacturer_model_name=None, software_versions=None,
              device_serial_number=None, output=None):
    """Author a Microscopy Bulk Simple Annotations object (native annotation engine).

    source: a source-image path / Dataset (or a list) — identity, Frame of Reference and
        referenced-image links are taken from it.
    groups: list of dicts ``{number, label, generation_type, property_category,
        property_type, graphic_type, annotations, measurements?}`` where ``annotations``
        is a list of ``(n_points, dim)`` arrays and the codes are ``(value, scheme,
        meaning)`` tuples or :class:`~pydcm.sr.Code`.
    """
    import pydcm
    from .dataset import Dataset
    srcs = source if isinstance(source, (list, tuple)) else [source]
    heads = [s if isinstance(s, Dataset)
             else pydcm.dcmread(str(s), stop_before_pixels=True, force=True) for s in srcs]
    h0 = heads[0]

    def s(obj, attr, default=""):
        v = getattr(obj, attr, None)
        return str(v) if v not in (None, "") else default

    sv = ((software_versions if isinstance(software_versions, str) else " ".join(software_versions))
          if software_versions else "")
    doc = {
        "coordinate_type": _ct(coordinate_type),
        "patient_name": s(h0, "PatientName"), "patient_id": s(h0, "PatientID"),
        "study_uid": s(h0, "StudyInstanceUID", pydcm.generate_uid()),
        "study_date": s(h0, "StudyDate"), "study_time": s(h0, "StudyTime"),
        "study_id": s(h0, "StudyID"), "accession_number": s(h0, "AccessionNumber"),
        "series_uid": str(series_instance_uid or pydcm.generate_uid()),
        "series_number": str(series_number),
        "sop_uid": str(sop_instance_uid or pydcm.generate_uid()),
        "instance_number": str(instance_number),
        "frame_of_reference_uid": s(h0, "FrameOfReferenceUID"),
        "manufacturer": manufacturer, "manufacturer_model": manufacturer_model_name or "",
        "software_versions": sv, "device_serial": device_serial_number or "",
        "references": [{"sop_class": s(h, "SOPClassUID"), "sop_instance": s(h, "SOPInstanceUID")}
                       for h in heads],
        "groups": [_group_dict(g) for g in groups],
    }
    return _native.write_ann(doc, str(output) if output else "")


# ── class API (over the native write_ann) ──────────
class Measurements:
    """Measured quantity over a group."""
    def __init__(self, name, unit, values, annotation_index=None):
        self.name = name
        self.unit = unit
        self.values = values
        self.annotation_index = annotation_index


class AnnotationGroup:
    """Annotation group."""
    def __init__(self, number, uid, label, annotated_property_category,
                 annotated_property_type, graphic_type, graphic_data, algorithm_type,
                 algorithm_identification=None, measurements=None, description=None,
                 anatomic_regions=None, primary_anatomic_structures=None, display_color=None):
        self.number = number
        self.uid = uid
        self.label = label
        self.annotated_property_category = annotated_property_category
        self.annotated_property_type = annotated_property_type
        self.graphic_type = graphic_type
        self.graphic_data = graphic_data
        self.algorithm_type = algorithm_type
        self.measurements = measurements

    def _to_dict(self):
        d = {"number": self.number, "uid": self.uid, "label": self.label,
             "generation_type": str(self.algorithm_type),
             "property_category": self.annotated_property_category,
             "property_type": self.annotated_property_type,
             "graphic_type": str(self.graphic_type), "annotations": list(self.graphic_data)}
        if self.measurements:
            d["measurements"] = [{"name": m.name, "unit": m.unit, "values": m.values,
                                  "annotation_index": m.annotation_index} for m in self.measurements]
        return d


def MicroscopyBulkSimpleAnnotations(source_images, annotation_coordinate_type, annotation_groups,
                                    series_instance_uid, series_number, sop_instance_uid,
                                    instance_number, manufacturer, manufacturer_model_name=None,
                                    software_versions=None, device_serial_number=None, **_kwargs):
    """Constructor — returns a
    pydcm Dataset built over the native ``write_ann``."""
    import pydcm
    bytes_or_none = write_ann(
        list(source_images), [g._to_dict() for g in annotation_groups],
        coordinate_type=annotation_coordinate_type, series_instance_uid=series_instance_uid,
        series_number=series_number, sop_instance_uid=sop_instance_uid,
        instance_number=instance_number, manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name, software_versions=software_versions,
        device_serial_number=device_serial_number)
    # write_ann returns Part-10 bytes (output omitted) -> a Dataset, like the other shims.
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".dcm"); os.close(fd)
    with open(tmp, "wb") as f:
        f.write(bytes_or_none)
    from .seg import _SEG_TMPS
    _SEG_TMPS.append(tmp)
    return pydcm.dcmread(tmp)


__all__ = ["read_ann", "write_ann", "MicroscopyBulkSimpleAnnotations",
           "AnnotationGroup", "Measurements"]
