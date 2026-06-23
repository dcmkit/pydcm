# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — Surface Segmentation reading (``pydcm.surface``).

Read a Surface Segmentation object (SOP 1.2.840.10008.5.1.4.1.1.66.5) — surface
meshes stored as native DICOM (points + mesh primitives) rather than an
encapsulated STL blob. Every primitive type is decoded by the native engine and
triangulated into one flat triangle list, so each surface comes back as a plain
``(points, triangles)`` pair ready for ``pydcm.mesh`` / rendering. This is the
thin marshalling wrapper; the parse lives in the native engine.

Authoring (the inverse ``write_surface``) is deliberately not implemented yet —
surface producers are rare, and STL encapsulation covers most mesh-export needs.
"""
from __future__ import annotations

import numpy as np

from . import _core


def read_surface(path):
    """Read a Surface Segmentation file.

    Returns a dict ``{surfaces:[...], segments:[...]}`` or ``None`` if `path` is
    not a Surface Segmentation object. Each surface carries its identity
    (``number``) and display hints (``finite_volume`` / ``manifold`` /
    ``recommended_type`` / ``recommended_opacity`` / ``cielab``) plus geometry:

    - ``points``   — ``(N, 3)`` float64 vertex coordinates (mm, patient space)
    - ``normals``  — ``(N, 3)`` float64 per-vertex normals, or ``None``
    - ``triangles``— ``(M, 3)`` uint32 vertex indices (0-based; triangle / strip /
      fan / facet primitives are all expanded into this single list)
    - ``lines`` / ``vertices`` — ``(L, 2)`` / ``(V,)`` uint32 indices for edge /
      line / vertex primitives (present only when the surface uses them)

    Each segment carries ``number``, ``label``, ``algorithm_type``, coded
    ``property_category`` / ``property_type``, the surface-generation
    ``algorithm_family`` / ``algorithm_name`` / ``algorithm_version``, and the
    ``referenced_surface_numbers`` linking it to its surface(s).
    """
    doc = _core.read_surface(str(path))
    if doc is None:
        return None
    for s in doc["surfaces"]:
        pts = np.frombuffer(s.pop("points"), dtype=np.float64)
        s["points"] = pts.reshape(-1, 3) if pts.size else np.empty((0, 3), np.float64)
        nrm = np.frombuffer(s.pop("normals"), dtype=np.float64)
        s["normals"] = nrm.reshape(-1, 3) if nrm.size else None
        tri = np.frombuffer(s.pop("triangles"), dtype=np.uint32)
        s["triangles"] = tri.reshape(-1, 3) if tri.size else np.empty((0, 3), np.uint32)
        ln = np.frombuffer(s.pop("lines"), dtype=np.uint32)
        s["lines"] = ln.reshape(-1, 2) if ln.size else None
        vx = np.frombuffer(s.pop("vertices"), dtype=np.uint32)
        s["vertices"] = vx if vx.size else None
    return doc


__all__ = ["read_surface"]
