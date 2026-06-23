# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — structured-object content reader (``pydcm.content``).

The interpreted (semantic) view of a derived DICOM object — Segmentation, RT
Structure Set, RT Plan (photon + ion), RT Dose, Presentation State, Waveform,
Ophthalmic Visual Field, or Structured Report (the full content tree) — over the
shared native content engine. One unified reader that auto-detects the
SOP class, organized by operation rather than object; the interpreted counterpart
to the raw element model.
"""
from __future__ import annotations

import json

from . import _core


def content(path, contours=False, control_points=False, meshes=False):
    """Semantic content of a structured DICOM object — Segmentation, RT Structure
    Set, RT Plan, RT Dose, Presentation State, Waveform, Ophthalmic Visual Field,
    Surface Segmentation, or Structured Report (the full content tree) — as a dict
    (coded concepts resolved), or ``None`` if `path` is not one of those.

    Raises ``RuntimeError`` when `path` is not decodable DICOM at all.

    `contours`: RT Structure Set only — include each contour's xyz point list.
    `control_points`: RT Plan only — include every control point (angles,
    meterset, leaf/jaw positions) instead of the first-CP summary.
    `meshes`: Surface Segmentation only — include each surface's full points /
    normals / triangles arrays instead of just counts (use ``pydcm.read_surface``
    for arrays as NumPy).
    """
    j = _core.content_json(str(path), contours=contours, control_points=control_points,
                           meshes=meshes)
    return json.loads(j) if j else None


__all__ = ["content"]
