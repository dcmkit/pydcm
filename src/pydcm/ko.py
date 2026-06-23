# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Key Object Selection authoring + reading (``pydcm.ko``).

The ``ko`` capability: flag instances as "key objects" (a KOS document,
PS3.3 KOS IOD / PS3.16 TID 2010) and read one back, over the native KOS engine
— a native authoring capability.
"""
from __future__ import annotations

from . import _core


def _ref_of(item):
    """A reference dict {study/series/sop_class/sop_instance_uid} from an explicit
    dict, or extracted from a DICOM path / Dataset."""
    if isinstance(item, dict):
        return item
    from . import dcmread
    from ._dicom import Dataset
    ds = item if isinstance(item, Dataset) else dcmread(str(item), stop_before_pixels=True, force=True)
    return {"study_uid": str(getattr(ds, "StudyInstanceUID", "")),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "sop_class_uid": str(getattr(ds, "SOPClassUID", "")),
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", ""))}


def _title_of(title):
    if title is None:
        return None
    if isinstance(title, dict):
        return {"value": str(title.get("value", title.get("code", ""))),
                "scheme": str(title.get("scheme", "DCM")), "meaning": str(title.get("meaning", ""))}
    t = list(title)                                   # (value, scheme, meaning)
    return {"value": str(t[0]), "scheme": str(t[1]), "meaning": str(t[2])}


def write_ko(references, *, patient_name="", patient_id="", study_uid="", study_date="",
             study_time="", study_id="", accession_number="", title=None, output=None):
    """Author a Key Object Selection document flagging `references` as key objects.

    references: a list whose items are either reference dicts (``{study_uid, series_uid,
        sop_class_uid, sop_instance_uid}``) or DICOM paths / ``Dataset`` objects (their
        identifiers are extracted automatically — the common "flag these images" case).
    title: the Key Object Document Title — a ``{value, scheme, meaning}`` dict or a
        ``(value, scheme, meaning)`` tuple; defaults to ``(113000, DCM, "Of Interest")``.
    patient_* / study_*: identity for the KOS; when omitted they are inherited from the
        first path/Dataset reference (the study the KOS is filed under).
    output: write the KOS there and return ``None``; if omitted, return Part-10 bytes.
    """
    refs = [_ref_of(r) for r in references]
    doc = {"patient_name": patient_name, "patient_id": patient_id, "study_uid": study_uid,
           "study_date": study_date, "study_time": study_time, "study_id": study_id,
           "accession_number": accession_number, "references": refs}
    t = _title_of(title)
    if t is not None:
        doc["title"] = t
    # Inherit identity from the first non-dict (path/Dataset) reference, if any.
    if not study_uid or not patient_name:
        src = next((r for r in references if not isinstance(r, dict)), None)
        if src is not None:
            from . import dcmread
            from ._dicom import Dataset
            ds = src if isinstance(src, Dataset) else dcmread(str(src), stop_before_pixels=True, force=True)
            for key, attr in (("patient_name", "PatientName"), ("patient_id", "PatientID"),
                              ("study_uid", "StudyInstanceUID"), ("study_date", "StudyDate"),
                              ("study_time", "StudyTime"), ("accession_number", "AccessionNumber")):
                if not doc[key]:
                    v = getattr(ds, attr, None)
                    if v not in (None, ""):
                        doc[key] = str(v)
    return _core.write_ko(doc, str(output) if output else "")


def read_ko(path):
    """Read a Key Object Selection document -> ``{patient_name, patient_id, study_uid,
    series_uid, title, references: [{sop_class_uid, sop_instance_uid}, …]}`` (the IMAGE
    content items), or ``None`` when `path` is not a KOS."""
    return _core.read_ko(str(path))


# ── class API (over write_ko) ───────────────────────
class KeyObjectSelection:
    """Content: a document-title code +
    the objects flagged as key."""
    def __init__(self, document_title, referenced_objects, observer_person_context=None,
                 observer_device_context=None, description=None):
        self.document_title = document_title
        self.referenced_objects = list(referenced_objects)
        self.description = description


def KeyObjectSelectionDocument(evidence, content, series_instance_uid, series_number,
                               sop_instance_uid, instance_number, manufacturer=None,
                               institution_name=None, institutional_department_name=None,
                               requested_procedures=None, transfer_syntax_uid=None,
                               **_kwargs):
    """Constructor — returns a pydcm
    Dataset over the native ``write_ko``. ``content.referenced_objects`` are the flagged
    key objects, ``content.document_title`` the KOS title code."""
    import os as _os
    import tempfile as _tempfile
    from . import dcmread
    from .seg import _code_tuple, _SEG_TMPS
    t = content.document_title
    title = _code_tuple(t) if hasattr(t, "value") else t
    fd, tmp = _tempfile.mkstemp(suffix=".dcm"); _os.close(fd)
    _SEG_TMPS.append(tmp)
    write_ko(content.referenced_objects, title=title, output=tmp)
    ds = dcmread(tmp)
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = int(series_number)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = int(instance_number)
    if manufacturer is not None:
        ds.Manufacturer = manufacturer
    if institution_name is not None:
        ds.InstitutionName = institution_name
    return ds


__all__ = ["write_ko", "read_ko", "KeyObjectSelection", "KeyObjectSelectionDocument"]
