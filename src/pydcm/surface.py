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

``write_surface`` is the inverse, over the same model: what :func:`read_surface`
returns is close to what it takes.
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


def write_surface(surfaces, segments, *, output=None,
                  patient_name="", patient_id="", study_uid="", study_date="",
                  series_uid="", frame_of_reference_uid="",
                  sop_instance_uid="", content_date="", content_time=""):
    """Author a Surface Segmentation file.

    surfaces: list of dicts, one per mesh.

        - ``points`` — ``(N, 3)`` float64 vertex coordinates in patient mm.
          Required. Written as Double Point Coordinates Data, so the
          coordinates come back at the precision they went in with; the
          32-bit form would round them to about seven digits.
        - ``triangles`` — ``(M, 3)`` uint32 vertex indices, **0-based**, or
          omitted for a point cloud. The 1-based wire form is the engine's
          business. An index outside ``points`` is refused rather than
          dropped, which is what a reader would do with it.
        - ``normals`` — ``(N, 3)`` float32 per-vertex normals, optional.
        - ``comments``, ``processing``, ``opacity``, ``rgb``
        - ``finite_volume`` / ``manifold`` — True / False / omitted. Omitted
          is UNKNOWN, which is the honest answer for an arbitrary mesh
          rather than a claim that it is neither.
        - ``presentation_type`` — omitted follows the geometry: SURFACE with
          triangles, POINTS without. A cloud presented as a surface draws
          as nothing.

    segments: list of dicts, one per segment. ``label``,
        ``algorithm_type``, ``algorithm_name``, ``algorithm_version``,
        ``property_category`` / ``property_type`` / ``algorithm_family`` as
        ``(value, scheme, meaning)``, and ``surfaces`` — **0-based indices
        into the surfaces list**, not wire Surface Numbers.

    output: path to write to. Omitted returns the bytes.

    Both sequences are Type 1: a document needs at least one surface and at
    least one segment.
    """
    blob = _core.write_surface(
        list(surfaces), list(segments),
        patient_name=patient_name, patient_id=patient_id,
        study_uid=study_uid, study_date=study_date, series_uid=series_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        sop_instance_uid=sop_instance_uid,
        content_date=content_date, content_time=content_time)
    if output is None:
        return blob
    with open(output, "wb") as fh:
        fh.write(blob)
    return output


__all__ = ["read_surface", "write_surface"]
