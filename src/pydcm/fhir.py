# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""FHIR R4 bridge — DICOM → FHIR resources, over the native FHIR engine.

Turns a DICOM instance into a FHIR ImagingStudy resource (the imaging↔EHR seam), so an
agent or app can hand a study to a FHIR consumer without a separate mapping layer::

    study = pydcm.fhir.imaging_study("CT0001.dcm")   # -> dict (FHIR R4 ImagingStudy)

The field mapping (study/series/instance + a contained Patient) lives in the native
engine; this module is the thin marshaller. Requires the optional ``_fhir`` extension
(like ``_dimse`` for networking).
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import _fhir
except ImportError as _e:                            # pragma: no cover
    raise ImportError(
        "pydcm.fhir requires the optional native _fhir extension, "
        "which is not present in this build."
    ) from _e


def imaging_study(source) -> dict:
    """Build a FHIR R4 ``ImagingStudy`` (as a ``dict``) from one DICOM instance, **or a whole
    study** of many instances.

    ``source`` may be:

    * raw Part-10 ``bytes`` — a single instance;
    * a path to a ``.dcm`` file — a single instance;
    * a path to a **directory** — every readable instance under it is aggregated into one
      ImagingStudy, grouped by ``SeriesInstanceUID`` (the multi-series / multi-instance
      study form a FHIR consumer actually expects);
    * an iterable of paths / ``bytes`` — likewise aggregated.

    Study/series/instance identifiers, the modality set, and patient demographics are mapped
    into the FHIR ImagingStudy with a *contained* Patient (``subject.reference = "#patient-…"``).
    Raises if no readable instance is found or StudyInstanceUID is absent.
    """
    bufs = _gather(source)
    if not bufs:
        raise ValueError("pydcm.fhir.imaging_study: no DICOM instances found in source")
    return json.loads(_fhir.imaging_study_multi(bufs))


def _gather(source) -> list[bytes]:
    """Normalise `source` to a list of Part-10 byte buffers (one per instance)."""
    if isinstance(source, (bytes, bytearray)):
        return [bytes(source)]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return [f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file()]
        return [p.read_bytes()]
    out: list[bytes] = []           # iterable of paths / bytes
    for item in source:
        out.extend(_gather(item))
    return out

__all__ = ['imaging_study']
