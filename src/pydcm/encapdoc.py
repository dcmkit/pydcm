# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — Encapsulated Documents (``pydcm.encapdoc``).

Wrap a PDF / CDA / STL / OBJ / MTL document into its Encapsulated Document
Storage instance (PS3.3 A.45/A.85), or extract one — over the shared
native encapsulation engine. The assembly, per-type module sets, detection
and MIME-aware extraction all run in C++; this wrapper only shuttles bytes
and copies identity from a reference dataset.
"""
from __future__ import annotations

from . import _core


class EncapsulatedDocument:
    """An extracted Encapsulated Document.

    Attributes:
        payload: the document bytes. Trailing OB pad NULs are stripped for
            pdf/text MIME types; model/* (binary STL legitimately ends in
            0x00) is verbatim, possibly with the single even-length pad byte.
        mime / title / sop_class_uid / sop_instance_uid: as recorded.
        type: ``"pdf" | "cda" | "stl" | "obj" | "mtl"`` when the SOP class is
            one of the five Encapsulated …Storage classes, else ``None``.
    """

    def __init__(self, d):
        self.payload = d["payload"]
        self.mime = d["mime"]
        self.title = d["title"]
        self.sop_class_uid = d["sop_class_uid"]
        self.sop_instance_uid = d["sop_instance_uid"]
        self.type = d["type"]

    def __repr__(self):
        kind = self.type if self.type else repr(self.mime)
        return (f"<EncapsulatedDocument {kind} "
                f"{len(self.payload)} bytes title={self.title!r}>")


_ID_KEYS = frozenset({
    "patient_name", "patient_id", "birth_date", "sex", "study_uid", "study_date",
    "study_time", "study_id", "accession", "referring", "series_uid",
    "frame_of_reference_uid", "charset",
})


def write_encapsulated(src, *, type="auto", output=None, title=None, mime=None,
                       units=None, reference=None, **ids):
    """Wrap a document into its Encapsulated Document DICOM instance.

    `src`: a file path, or raw ``bytes`` (then `type` must be explicit unless
    content magic identifies it). `type`: ``auto`` (file extension, then
    content magic) or ``pdf|cda|stl|obj|mtl``. `title` defaults to the file
    stem. `units`: 3D-model Measurement Units UCUM code (default ``um``).
    `reference`: a DICOM file whose Patient/Study identity
    is copied (the document joins that study). Extra keyword ids:
    ``patient_name, patient_id, birth_date, sex, study_uid, study_date,
    study_time, study_id, accession, referring, series_uid,
    frame_of_reference_uid, charset``.

    Returns the Part-10 ``bytes``, or writes `output` and returns its path.
    """
    import pathlib

    unknown = set(ids) - _ID_KEYS
    if unknown:
        raise TypeError(f"unknown identity keyword(s): {sorted(unknown)} "
                        f"(allowed: {sorted(_ID_KEYS)})")

    name = ""
    if isinstance(src, (bytes, bytearray, memoryview)):
        payload = bytes(src)
    else:
        p = pathlib.Path(src)
        payload = p.read_bytes()
        name = p.name
        if title is None:
            title = p.stem

    if type == "auto":
        detected = _core.encap_detect(name, payload[:4096] if len(payload) > 4096
                                      else payload)
        # binary-STL detection needs the exact size invariant — retry full
        if detected is None and len(payload) > 4096:
            detected = _core.encap_detect(name, payload)
        if detected is None:
            raise ValueError(
                "cannot detect the document type — pass type='pdf|cda|stl|obj|mtl'")
        type = detected

    if reference is not None:
        if not pathlib.Path(reference).is_file():
            raise FileNotFoundError(f"reference DICOM not found: {reference}")
        from . import dcmread
        ref = dcmread(str(reference))
        # NOTE: no charset copy — dcmread returns DECODED values and the engine
        # writes them back as UTF-8, so the only truthful (0008,0005) is the
        # engine's ISO_IR 192 default (copying a legacy charset claim would
        # mislabel the re-encoded bytes).
        for key, tag in [("patient_name", "PatientName"), ("patient_id", "PatientID"),
                         ("birth_date", "PatientBirthDate"), ("sex", "PatientSex"),
                         ("study_uid", "StudyInstanceUID"), ("study_date", "StudyDate"),
                         ("study_time", "StudyTime"), ("study_id", "StudyID"),
                         ("accession", "AccessionNumber"),
                         ("referring", "ReferringPhysicianName"),
                         ("frame_of_reference_uid", "FrameOfReferenceUID")]:
            if key not in ids:
                v = getattr(ref, tag, "")
                if v is not None and str(v):
                    ids[key] = str(v)

    blob = _core.encapsulate(payload, type, title=title or "", mime=mime or "",
                             units=units or "", ids=ids)
    if output is None:
        return blob
    out = pathlib.Path(output)
    out.write_bytes(blob)
    return out


def read_encapsulated(path):
    """Extract an Encapsulated Document instance → :class:`EncapsulatedDocument`."""
    return EncapsulatedDocument(_core.read_encapsulated(str(path)))


__all__ = ["EncapsulatedDocument", "write_encapsulated", "read_encapsulated"]
