# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm Structured Reporting (``pydcm.sr``) — compat surface + native engine.

* **compat** — `Code` (a coded-concept shape) and `content_json`,
  for ``from pydcm.sr import …`` imports.
* **pydcm-native** (over the shared native SR engines): author any Comprehensive SR
  from a content-tree dict (`write_sr`); author / read a TID 1500 Measurement Report
  (`write_report` / `read_report`);
  validate an SR content tree (`sr_validate`); and look up the PS3.16 coded-concept
  table (`sr_code_meaning` / `sr_validate_code` / `sr_cid_has`).
"""
from __future__ import annotations

from collections import namedtuple

from . import _core


# ---- surface --------------------------------------------
# PS3.3 coded concept (scheme, value, meaning) — a coded concept.
Code = namedtuple("Code", ["value", "scheme_designator", "meaning",
                           "scheme_version"])
Code.__new__.__defaults__ = (None,)


def content_json(ds):
    """The semantic SR content tree (reuses the native content engine via
    :func:`pydcm.content`)."""
    from .content import content
    return content(ds)


# ---- pydcm-native authoring / validation / codes ---------------------------
def write_sr(document, output=None):
    """Author a Comprehensive DICOM Structured Report from a content-tree dict.

    The general SR writer (vs. a fixed template): build any tree of content items.

    document: a dict with ``patient_name`` / ``patient_id`` / ``study_uid`` /
        ``study_date`` / ``series_uid`` (+ optional ``sop_class_uid`` /
        ``completion_flag`` / ``verification_flag``), a ``title`` code
        ``{value, scheme, meaning}`` (the root CONTAINER's Concept Name), and a
        ``content`` list. Each content item: ``relationship`` ("CONTAINS", …),
        ``value_type`` ("CODE"/"NUM"/"TEXT"/"CONTAINER"/"IMAGE"/"SCOORD"/…),
        ``concept`` code, and per-type fields — ``text``; ``code`` ({value,scheme,
        meaning}); ``value`` + ``unit`` (NUM); ``datetime``; ``ref_sop_class`` /
        ``ref_sop_instance`` (IMAGE); ``graphic_type`` + ``graphic_data`` (SCOORD)
        — plus a nested ``content`` list for children.
    output: write the SR there and return ``None``; if omitted, return Part-10 bytes.
    """
    return _core.write_sr(dict(document), str(output) if output else "")


def write_report(measurements, *, reference=None, patient_name="", patient_id="",
                 study_uid="", study_date="", series_uid="", output=None):
    """Author a TID 1500 Measurement Report SR from a list of measurements, over the
    same native SR-export engine.

    measurements: a list of measurement dicts, each with ``concept_value`` /
        ``concept_scheme`` / ``concept_meaning`` (the measured quantity's code, e.g.
        ``"103355008", "SCT", "Width"``), ``value`` (float), ``unit_code`` /
        ``unit_meaning`` (UCUM, e.g. ``"mm"``), and optionally ``ref_sop_class_uid`` /
        ``ref_sop_instance_uid`` (the measured image), ``graphic_type`` (``"POINT"`` /
        ``"POLYLINE"`` / ``"CIRCLE"`` / ``"ELLIPSE"``) and ``scoord``
        (``[col0, row0, col1, row1, …]`` pixel coordinates). ``scoord`` is recorded
        only when ``ref_sop_instance_uid`` is also given — spatial coordinates are
        stored relative to their referenced image. May instead be a full document
        dict (``{patient_…, study_…, series_uid, measurements: […]}``, the shape
        :func:`read_report` returns) so ``write_report(read_report(x))`` round-trips.
    reference: a DICOM path to inherit patient + study identity from (the report
        attaches to that study); explicit keyword args take precedence. Study/Series
        UIDs are content-derived when neither given nor inherited.
    output: write the SR there and return ``None``; if omitted, return Part-10 bytes.
    """
    if isinstance(measurements, dict) and "measurements" in measurements:
        src = measurements
        doc = {"patient_name": patient_name or src.get("patient_name", ""),
               "patient_id":   patient_id   or src.get("patient_id", ""),
               "study_uid":    study_uid    or src.get("study_uid", ""),
               "study_date":   study_date   or src.get("study_date", ""),
               "series_uid":   series_uid   or src.get("series_uid", ""),
               "measurements": list(src["measurements"])}
    else:
        doc = {"patient_name": patient_name, "patient_id": patient_id,
               "study_uid": study_uid, "study_date": study_date, "series_uid": series_uid,
               "measurements": list(measurements)}
    if reference is not None:
        from . import dcmread
        ds = dcmread(str(reference), stop_before_pixels=True, force=True)
        for key, attr in (("patient_name", "PatientName"), ("patient_id", "PatientID"),
                          ("study_uid", "StudyInstanceUID"), ("study_date", "StudyDate")):
            if not doc[key]:
                v = getattr(ds, attr, None)
                if v not in (None, ""):
                    doc[key] = str(v)
    return _core.write_report(doc, str(output) if output else "")


def read_report(path):
    """The measurements of a TID 1500 Measurement Report SR. Returns ``{patient_name, patient_id, study_uid,
    study_date, series_uid, measurements: [...]}`` round-tripping :func:`write_report`'s
    input (``measurements`` is empty when `path` carries no SR content)."""
    return _core.read_report(str(path))


def sr_to_html(path):
    """Render a DICOM Structured Report to clinical-readable HTML (a ``str``).

    Renders any SR to standalone, clinical-readable markup — not only the
    TID 1500 measurement-report shape.
    """
    return _core.sr_to_html(str(path))


def write_measurement_report(document, output=None):
    """Author a TYPED TID 1500 Measurement Report — the standard TID 1500
    measurement-report capability, over the native SR authoring engine (structure
    cross-validated against reference SR implementations). Unlike :func:`write_report` (a flat
    list of measurements) this builds the full TID 1500 structure: observation context,
    measurement groups (TID 1411/1501), each with tracking identity, finding +
    finding sites, an optional ROI region, NUM measurements (TID 300, with method /
    derivation / per-measurement finding sites) and qualitative evaluations.

    document: a dict with ``patient_name`` / ``patient_id`` / ``study_uid`` /
        ``study_date`` / ``series_uid``; optional ``observer`` =
        ``{type: "device"|"person", name, uid}``, ``procedure_reported`` (a code),
        ``language`` (default "en-US"); and ``groups`` — each
        ``{tracking_id, tracking_uid, finding?, finding_sites?[], roi?, measurements[],
        qualitative_evaluations?[]}``. A code is ``{value, scheme, meaning}``. A
        measurement is ``{name, value, unit, method?, derivation?, finding_sites?[]}``.
        A roi is ``{graphic_type, scoord:[...], is_3d?, frame_of_reference_uid?,
        ref_sop_class_uid?, ref_sop_instance_uid?}``. A qualitative evaluation is
        ``{name, value}`` (both codes).
    output: write the SR there and return ``None``; if omitted, return Part-10 bytes.
    """
    return _core.write_measurement_report(dict(document), str(output) if output else "")


def read_measurement_report(path):
    """The typed TID 1500 Measurement Report of an SR — ``{patient/study, observer,
    procedure_reported?, groups: [...]}`` round-tripping :func:`write_measurement_report`
    (empty ``groups`` when `path` is not a measurement report). Reads third-party
    SR reports, not just pydcm's own output."""
    return _core.read_measurement_report(str(path))


def sr_code_meaning(scheme, value):
    """Code Meaning for a coded concept ``(scheme, value)`` from the DICOM PS3.16
    Content Mapping Resource (the most complete public set), or ``None``
    if the code is unknown. E.g. ``sr_code_meaning("DCM", "126000")`` →
    ``"Imaging Measurement Report"``."""
    return _core.sr_code_meaning(str(scheme), str(value))


def sr_validate_code(scheme, value, meaning=None):
    """True if ``(scheme, value)`` is a known coded concept and — when `meaning` is
    given — its Code Meaning matches it (a typo / wrong-meaning check)."""
    return _core.sr_validate_code(str(scheme), str(value), str(meaning) if meaning else "")


def sr_cid_has(cid, scheme, value):
    """True if the coded concept ``(scheme, value)`` is a member of Context Group
    ``cid`` (e.g. ``sr_cid_has(7469, "SCT", "103339001")``)."""
    return _core.sr_cid_has(int(cid), str(scheme), str(value))


def sr_validate(path):
    """Validate an SR file's content tree — structural well-formedness (root is a
    CONTAINER, valid value types / relationships, NUM has units, CODE has a value,
    …), coded-concept conformance against the PS3.16 table, AND TID content-template
    conformance (measurement-group mandatory rows + cardinality, value-type per row,
    value-set-per-concept, container nesting, and conditional Observer / Subject /
    Algorithm-Identification rows) — returning a list of ``{severity, location,
    message}`` findings (empty = a conformant SR)."""
    return _core.sr_validate(str(path))


# ═════════════════════════════════════════════════════════════════════════════
#  TID 1500 class tree (over the native write_measurement_report).
#  The SR authoring engine is native; these classes are the
#  ergonomic layer that assembles the document dict the engine consumes.
# ═════════════════════════════════════════════════════════════════════════════
CodedConcept = Code                       # CodedConcept: alias for pydcm.sr.Code


def _c(code):
    """A code (Code / CodedConcept / (value,scheme,meaning) / dict) -> {value,scheme,meaning}."""
    if isinstance(code, dict):
        return code
    if hasattr(code, "value") and hasattr(code, "scheme_designator"):
        return {"value": str(code.value), "scheme": str(code.scheme_designator),
                "meaning": str(code.meaning)}
    return {"value": str(code[0]), "scheme": str(code[1]), "meaning": str(code[2])}


class TrackingIdentifier:
    """A measurement group's tracking identity."""
    def __init__(self, uid=None, identifier=None):
        self.uid = uid
        self.identifier = identifier


class FindingSite:
    """An anatomic location code."""
    def __init__(self, anatomic_location, laterality=None, topographical_modifier=None):
        self.anatomic_location = anatomic_location

    def _code(self):
        return _c(self.anatomic_location)


class Measurement:
    """One numeric measurement (name/value/unit + qualifiers)."""
    def __init__(self, name, value, unit, qualifier=None, tracking_identifier=None,
                 algorithm_id=None, derivation=None, method=None, finding_sites=None,
                 properties=None, referenced_images=None, **_kw):
        self.name, self.value, self.unit = name, value, unit
        self.method, self.derivation, self.finding_sites = method, derivation, finding_sites

    def _to_dict(self):
        d = {"name": _c(self.name), "value": float(self.value), "unit": _c(self.unit)}
        if self.method is not None:
            d["method"] = _c(self.method)
        if self.derivation is not None:
            d["derivation"] = _c(self.derivation)
        if self.finding_sites:
            d["finding_sites"] = [fs._code() if isinstance(fs, FindingSite) else _c(fs)
                                  for fs in self.finding_sites]
        return d


class QualitativeEvaluation:
    """A coded name/value evaluation."""
    def __init__(self, name, value):
        self.name, self.value = name, value

    def _to_dict(self):
        return {"name": _c(self.name), "value": _c(self.value)}


class SourceImageForRegion:
    """The image a 2D region is drawn on."""
    def __init__(self, referenced_sop_class_uid, referenced_sop_instance_uid, **_kw):
        self.sop_class = referenced_sop_class_uid
        self.sop_instance = referenced_sop_instance_uid


class ImageRegion:
    """A 2D ROI (SCOORD) on a source image."""
    def __init__(self, graphic_type, graphic_data, source_image, **_kw):
        self.graphic_type, self.graphic_data, self.source_image = graphic_type, graphic_data, source_image

    def _to_roi(self):
        import numpy as np
        return {"graphic_type": str(self.graphic_type),
                "scoord": [float(x) for x in np.asarray(self.graphic_data).ravel()],
                "is_3d": False, "ref_sop_class_uid": str(self.source_image.sop_class),
                "ref_sop_instance_uid": str(self.source_image.sop_instance)}


class ImageRegion3D:
    """A 3D ROI (SCOORD3D) in a Frame of Reference."""
    def __init__(self, graphic_type, graphic_data, frame_of_reference_uid, **_kw):
        self.graphic_type, self.graphic_data = graphic_type, graphic_data
        self.frame_of_reference_uid = frame_of_reference_uid

    def _to_roi(self):
        import numpy as np
        return {"graphic_type": str(self.graphic_type),
                "scoord": [float(x) for x in np.asarray(self.graphic_data).ravel()],
                "is_3d": True, "frame_of_reference_uid": str(self.frame_of_reference_uid)}


class MeasurementsAndQualitativeEvaluations:
    """Measurement group (no ROI)."""
    def __init__(self, tracking_identifier, measurements=None, qualitative_evaluations=None,
                 finding_type=None, finding_sites=None, **_kw):
        self.tracking = tracking_identifier
        self.measurements = list(measurements or [])
        self.quals = list(qualitative_evaluations or [])
        self.finding = finding_type
        self.finding_sites = finding_sites
        self.roi = None

    def _to_group(self):
        import pydcm
        g = {"tracking_id": self.tracking.identifier or "",
             "tracking_uid": self.tracking.uid or pydcm.generate_uid()}
        if self.finding is not None:
            g["finding"] = _c(self.finding)
        if self.finding_sites:
            g["finding_sites"] = [fs._code() if isinstance(fs, FindingSite) else _c(fs)
                                  for fs in self.finding_sites]
        if self.roi is not None:
            g["roi"] = self.roi._to_roi()
        g["measurements"] = [m._to_dict() for m in self.measurements]
        if self.quals:
            g["qualitative_evaluations"] = [q._to_dict() for q in self.quals]
        return g


class PlanarROIMeasurementsAndQualitativeEvaluations(MeasurementsAndQualitativeEvaluations):
    """Planar (2D) ROI measurement group."""
    def __init__(self, tracking_identifier, referenced_region=None, referenced_segment=None,
                 finding_type=None, finding_sites=None, measurements=None,
                 qualitative_evaluations=None, **_kw):
        super().__init__(tracking_identifier, measurements, qualitative_evaluations,
                         finding_type, finding_sites)
        self.roi = referenced_region


class VolumetricROIMeasurementsAndQualitativeEvaluations(
        PlanarROIMeasurementsAndQualitativeEvaluations):
    """Volumetric (3D) ROI measurement group."""


class PersonObserverIdentifyingAttributes:
    """Person observer identifying attributes."""
    def __init__(self, name, login_name=None, organization_name=None, **_kw):
        self._obs = {"type": "person", "name": str(name), "uid": ""}


class DeviceObserverIdentifyingAttributes:
    """Device observer identifying attributes."""
    def __init__(self, uid, name=None, manufacturer_name=None, model_name=None, **_kw):
        self._obs = {"type": "device", "name": str(name or model_name or ""), "uid": str(uid)}


class ObserverContext:
    """Wraps a person/device observer."""
    def __init__(self, observer_type, observer_identifying_attributes):
        self._attrs = observer_identifying_attributes


class ObservationContext:
    """Observer (+ subject) context."""
    def __init__(self, observer_person_context=None, observer_device_context=None,
                 subject_context=None):
        ctx = observer_device_context or observer_person_context
        self._observer = ctx._attrs._obs if ctx is not None else None


class MeasurementReport:
    """The TID 1500 root (observation context +
    procedure + measurement groups)."""
    def __init__(self, observation_context, procedure_reported, imaging_measurements=None,
                 title=None, language_of_content_item_and_descendants=None, **_kw):
        self.observation_context = observation_context
        self.procedure_reported = procedure_reported
        self.imaging_measurements = list(imaging_measurements or [])

    def _to_document(self):
        doc = {}
        if self.observation_context is not None and self.observation_context._observer:
            doc["observer"] = self.observation_context._observer
        if self.procedure_reported is not None:
            # A single Code is itself a tuple subclass; only a real list/sequence of codes
            # should be indexed (take the first reported procedure).
            pr = (self.procedure_reported[0] if isinstance(self.procedure_reported, list)
                  else self.procedure_reported)
            doc["procedure_reported"] = _c(pr)
        doc["groups"] = [g._to_group() for g in self.imaging_measurements]
        return doc


# ── generic CONTENT ITEM classes (build ANY SR tree, over write_sr) ──
class ContentItem:
    """Base for the SR content-item primitives — a typed SR tree node."""
    def __init__(self, value_type, name, relationship_type=None):
        self._vt = value_type
        self._concept = name
        self._rel = str(relationship_type) if relationship_type else "CONTAINS"
        self._extra: dict = {}
        self.children: list = []

    def append(self, item):
        """Add a child content item."""
        self.children.append(item)
        return self

    @property
    def ContentSequence(self):
        return self.children

    @ContentSequence.setter
    def ContentSequence(self, items):
        self.children = list(items)

    def _to_node(self):
        n = {"value_type": self._vt, "relationship": self._rel}
        if self._concept is not None:
            n["concept"] = _c(self._concept)
        n.update(self._extra)
        if self.children:
            n["content"] = [c._to_node() for c in self.children]
        return n


class CodeContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("CODE", name, relationship_type)
        self._extra["code"] = _c(value)


class NumContentItem(ContentItem):
    def __init__(self, name, value, unit, qualifier=None, relationship_type=None, **_kw):
        super().__init__("NUM", name, relationship_type)
        self._extra["value"] = float(value)
        self._extra["unit"] = _c(unit)


class TextContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("TEXT", name, relationship_type)
        self._extra["text"] = str(value)


class ContainerContentItem(ContentItem):
    def __init__(self, name, is_content_continuous=True, template_id=None, relationship_type=None):
        super().__init__("CONTAINER", name, relationship_type)
        self._extra["continuity"] = "CONTINUOUS" if is_content_continuous else "SEPARATE"


class _ScoordBase(ContentItem):
    def __init__(self, value_type, name, graphic_type, graphic_data, relationship_type, **extra):
        import numpy as np
        super().__init__(value_type, name, relationship_type)
        self._extra["graphic_type"] = str(graphic_type)
        self._extra["graphic_data"] = [float(x) for x in np.asarray(graphic_data).ravel()]
        self._extra.update(extra)


class ScoordContentItem(_ScoordBase):
    def __init__(self, name, graphic_type, graphic_data, pixel_origin_interpretation=None,
                 relationship_type=None, **_kw):
        super().__init__("SCOORD", name, graphic_type, graphic_data, relationship_type)


class Scoord3DContentItem(_ScoordBase):
    def __init__(self, name, graphic_type, graphic_data, frame_of_reference_uid,
                 relationship_type=None, **_kw):
        super().__init__("SCOORD3D", name, graphic_type, graphic_data, relationship_type,
                         frame_of_reference_uid=str(frame_of_reference_uid))


class _RefBase(ContentItem):
    def __init__(self, value_type, name, sop_class, sop_instance, relationship_type):
        super().__init__(value_type, name, relationship_type)
        self._extra["ref_sop_class"] = str(sop_class)
        self._extra["ref_sop_instance"] = str(sop_instance)


class ImageContentItem(_RefBase):
    def __init__(self, name, referenced_sop_class_uid, referenced_sop_instance_uid,
                 relationship_type=None, **_kw):
        super().__init__("IMAGE", name, referenced_sop_class_uid, referenced_sop_instance_uid,
                         relationship_type)


class CompositeContentItem(_RefBase):
    def __init__(self, name, referenced_sop_class_uid, referenced_sop_instance_uid,
                 relationship_type=None, **_kw):
        super().__init__("COMPOSITE", name, referenced_sop_class_uid,
                         referenced_sop_instance_uid, relationship_type)


class WaveformContentItem(_RefBase):
    def __init__(self, name, referenced_sop_class_uid, referenced_sop_instance_uid,
                 relationship_type=None, **_kw):
        super().__init__("WAVEFORM", name, referenced_sop_class_uid,
                         referenced_sop_instance_uid, relationship_type)


class UIDRefContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("UIDREF", name, relationship_type)
        self._extra["text"] = str(value)


class PnameContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("PNAME", name, relationship_type)
        self._extra["text"] = str(value)


class DateContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("DATE", name, relationship_type)
        self._extra["datetime"] = str(value)


class TimeContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("TIME", name, relationship_type)
        self._extra["datetime"] = str(value)


class DateTimeContentItem(ContentItem):
    def __init__(self, name, value, relationship_type=None):
        super().__init__("DATETIME", name, relationship_type)
        self._extra["datetime"] = str(value)


class TcoordContentItem(ContentItem):
    def __init__(self, name, temporal_range_type=None, relationship_type=None, **_kw):
        super().__init__("TCOORD", name, relationship_type)


class ContentSequence(list):
    """An ordered list of content items."""
    def __init__(self, items=None, is_root=False, is_sr=True):
        super().__init__(items or [])


def _sr_document(evidence, content, series_instance_uid, series_number, sop_instance_uid,
                 instance_number, manufacturer, sop_class_uid=None):
    """Shared body for the Comprehensive(3D)SR / EnhancedSR document constructors.

    Branches on the content: a :class:`MeasurementReport` uses the TID 1500 writer; a
    generic content-item tree (ContentSequence / ContainerContentItem) uses write_sr."""
    import os
    import tempfile
    import pydcm
    from .dataset import Dataset
    from .seg import _SEG_TMPS

    is_measurement = hasattr(content, "_to_document")
    if is_measurement:
        doc = content._to_document()
    else:
        items = list(content) if isinstance(content, (list, tuple)) else [content]
        if len(items) == 1 and isinstance(items[0], ContainerContentItem):
            root = items[0]                                    # the root CONTAINER
            title = root._concept
            nodes = [c._to_node() for c in root.children]
        else:
            title = Code("126000", "DCM", "Imaging Measurement Report")
            nodes = [it._to_node() for it in items]
        doc = {"title": _c(title), "content": nodes}

    ev0 = (evidence[0] if isinstance(evidence, (list, tuple)) and evidence else evidence)
    head = (ev0 if isinstance(ev0, Dataset)
            else pydcm.dcmread(str(ev0), stop_before_pixels=True, force=True)) if ev0 is not None else None
    if head is not None:
        doc.setdefault("patient_name", str(getattr(head, "PatientName", "") or ""))
        doc.setdefault("patient_id", str(getattr(head, "PatientID", "") or ""))
        doc.setdefault("study_uid", str(getattr(head, "StudyInstanceUID", "") or pydcm.generate_uid()))
        doc.setdefault("study_date", str(getattr(head, "StudyDate", "") or ""))
    doc["series_uid"] = str(series_instance_uid)

    fd, tmp = tempfile.mkstemp(suffix=".dcm"); os.close(fd)
    _SEG_TMPS.append(tmp)
    (write_measurement_report if is_measurement else write_sr)(doc, output=tmp)
    ds = pydcm.dcmread(tmp)
    ds.SeriesNumber = int(series_number)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = int(instance_number)
    if manufacturer is not None:
        ds.Manufacturer = manufacturer
    return ds


def Comprehensive3DSR(evidence, content, series_instance_uid, series_number, sop_instance_uid,
                      instance_number, manufacturer=None, **_kw):
    """Constructor — a TID 1500 SR Dataset."""
    return _sr_document(evidence, content, series_instance_uid, series_number,
                        sop_instance_uid, instance_number, manufacturer)


# ComprehensiveSR / EnhancedSR share the TID 1500 structure (the engine emits a
# Comprehensive 3D SR); provided so ported code constructs successfully.
ComprehensiveSR = Comprehensive3DSR
EnhancedSR = Comprehensive3DSR


__all__ = ["Code", "CodedConcept", "content_json", "write_sr", "write_report", "read_report",
           "write_measurement_report", "read_measurement_report",
           "sr_code_meaning", "sr_validate_code", "sr_cid_has", "sr_validate",
           # TID 1500 class tree
           "TrackingIdentifier", "FindingSite", "Measurement", "QualitativeEvaluation",
           "SourceImageForRegion", "ImageRegion", "ImageRegion3D",
           "MeasurementsAndQualitativeEvaluations",
           "PlanarROIMeasurementsAndQualitativeEvaluations",
           "VolumetricROIMeasurementsAndQualitativeEvaluations",
           "PersonObserverIdentifyingAttributes", "DeviceObserverIdentifyingAttributes",
           "ObserverContext", "ObservationContext", "MeasurementReport",
           "Comprehensive3DSR", "ComprehensiveSR", "EnhancedSR",
           # generic content-item primitives (build any SR tree)
           "ContentItem", "ContentSequence", "CodeContentItem", "NumContentItem",
           "TextContentItem", "ContainerContentItem", "ScoordContentItem",
           "Scoord3DContentItem", "ImageContentItem", "CompositeContentItem",
           "WaveformContentItem", "UIDRefContentItem", "PnameContentItem",
           "DateContentItem", "TimeContentItem", "DateTimeContentItem", "TcoordContentItem"]
