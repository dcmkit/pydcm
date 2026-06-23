# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Tractography Results (``pydcm.tract``): author and read.

* :func:`write_mktract` — author a Tractography Results Storage
  (SOP 1.2.840.10008.5.1.4.1.1.66.6) from track sets of polylines (streamlines)
  in Frame-of-Reference world coordinates, via the native ``dcm_tract_export``
  engine. The write counterpart of pydcm's tractography reader and the natural
  sink for ``dipy`` / ``MRtrix`` streamlines.
"""
from __future__ import annotations

import numpy as np

from . import _core


def _refs(reference):
    if reference is None:
        return []
    if isinstance(reference, (str, bytes)) or hasattr(reference, "__fspath__"):
        return [str(reference)]
    return [str(p) for p in reference]


def _code3(x, *, default_scheme="DCM"):
    """Normalize a coded concept to (value, scheme, meaning), or None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return (str(x.get("value", x.get("code", ""))),
                str(x.get("scheme", default_scheme)), str(x.get("meaning", "")))
    if isinstance(x, str):
        return ("", default_scheme, x)                  # meaning only
    t = list(x)
    if len(t) == 2:
        return (str(t[0]), default_scheme, str(t[1]))   # (value, meaning)
    return (str(t[0]), str(t[1]), str(t[2]))            # (value, scheme, meaning)


def _norm_track(track):
    arr = np.ascontiguousarray(track, dtype=np.float32)
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("each track must be (n, 3) — n points of (x, y, z)")
    if arr.shape[0] == 0:
        raise ValueError("a track has zero points")
    return arr


def _norm_codes(src, keys):
    """Copy a measurement/statistic dict, normalizing its coded-concept fields."""
    out = {}
    for k in keys:
        c = _code3(src.get(k))
        if c is not None:
            out[k] = c
    return out


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32).ravel()


def write_mktract(reference, track_sets, *, output=None,
                   sop_instance_uid="", content_date="", content_time=""):
    """Author a DICOM Tractography Results Storage from track sets.

    reference: source-series path / list of instance paths (demographics + Frame
        of Reference UID), or ``None`` to mint fresh identifiers. Track point
        coordinates are in that Frame of Reference (patient world mm).
    track_sets: a single track-set dict, or a list of them. Each::

        {
          "label": str, "description": str, "algorithm_name": str,
          "anatomy": coded concept,      # default SCT 389080008 "White Matter"
          "diffusion": coded concept,    # default DCM 113231 "Single Tensor"
          "line_thickness": float, "rgb": (r, g, b),
          "tracks": [ (n_i, 3) array of xyz, ... ],  # streamlines, world mm
          # optional per-track measurements + statistics (e.g. FA / ADC):
          "measurements": [
            {"concept": code, "units": code,         # what is measured + units
             "values": [ arr_track0, arr_track1, … ]}  # one value array per track
          ],
          "track_statistics": [
            {"concept": code, "modifier": code, "units": code,
             "values": arr}                          # one scalar per track
          ],
          "set_statistics": [
            {"concept": code, "modifier": code, "units": code, "value": float}
          ],
        }

        Coded concepts accept ``(value, scheme, meaning)``, ``(value, meaning)``,
        a plain meaning string, a dict, or ``None``. A measurement's ``concept``
        is required; ``units`` defaults to unitless; ``modifier`` is optional.
    output: write the file there and return ``None``; if omitted, return the
        Part-10 ``bytes``.

    ``sop_instance_uid`` / ``content_date`` (YYYYMMDD) / ``content_time``
    (HHMMSS): this object's own identity. Left empty, the SOP Instance
    UID is derived deterministically from the study, so two built for one
    study carry the same one — right for a single self-contained export, a
    DICOM global-uniqueness violation for a producer that mints many.

    """
    if isinstance(track_sets, dict):
        track_sets = [track_sets]
    norm = []
    for ts in track_sets:
        d = {}
        for k in ("label", "description", "algorithm_name"):
            if ts.get(k):
                d[k] = str(ts[k])
        if ts.get("line_thickness"):
            d["line_thickness"] = float(ts["line_thickness"])
        if ts.get("rgb") is not None:
            r, g, b = ts["rgb"]
            d["rgb"] = (int(r), int(g), int(b))
        anat = _code3(ts.get("anatomy"))
        if anat is not None:
            d["anatomy"] = anat
        diff = _code3(ts.get("diffusion"))
        if diff is not None:
            d["diffusion"] = diff
        d["tracks"] = [_norm_track(t) for t in ts.get("tracks", [])]
        # Optional measurements + statistics.
        if ts.get("measurements"):
            d["measurements"] = [
                {**_norm_codes(m, ("concept", "units")),
                 "values": [_f32(v) for v in m.get("values", [])]}
                for m in ts["measurements"]]
        if ts.get("track_statistics"):
            d["track_statistics"] = [
                {**_norm_codes(s, ("concept", "modifier", "units")),
                 "values": _f32(s.get("values", []))}
                for s in ts["track_statistics"]]
        if ts.get("set_statistics"):
            d["set_statistics"] = [
                {**_norm_codes(s, ("concept", "modifier", "units")),
                 "value": float(s.get("value", 0.0))}
                for s in ts["set_statistics"]]
        norm.append(d)
    return _core.write_mktract(_refs(reference), norm, str(output) if output else "",
                               sop_instance_uid or "", content_date or "",
                               content_time or "")


__all__ = ["write_mktract"]

def read_tract(path):
    """Read a Tractography Results (66.6) file — the inverse of ``write_mktract``.

    Returns ``{"track_sets": [...]}`` or ``None`` when `path` is not a
    Tractography Results object. Each track set carries its identity
    (``number``, ``label``, ``description``), the coded ``anatomy``, the
    ``diffusion_model`` and ``algorithm_name``, display hints (``rgb`` /
    ``cielab`` / ``line_thickness``) and:

    - ``tracks`` — list of ``{points: (N, 3) float32, rgb?, point_colors?}``.
      Coordinates stay float32 because that is the width the standard gives
      them ``(0066,0016 OF)`` and a streamline goes to a vertex buffer.
    - ``measurements`` — a quantity sampled ALONG the tracks (FA, ADC), one
      entry per track under ``tracks``. ``indices`` is present only when the
      values do not apply to every point in order; the standard pairs them one
      to one, but this reports what the file says, so zip them only after
      checking the lengths agree.
    - ``track_statistics`` — one float64 per track, e.g. mean FA per track.
    - ``set_statistics`` — one value for the whole set.

    Statistics come back as float64 and coordinates as float32 on purpose:
    a statistic is a measurement, a coordinate is geometry the viewer draws.
    """
    doc = _core.read_tract(str(path))
    if doc is None:
        return None
    for ts in doc["track_sets"]:
        for t in ts["tracks"]:
            pts = np.frombuffer(t.pop("points"), dtype=np.float32)
            t["points"] = pts.reshape(-1, 3) if pts.size else np.empty((0, 3), np.float32)
            if "point_colors" in t:
                pc = np.frombuffer(t["point_colors"], dtype=np.uint16)
                t["point_colors"] = pc.reshape(-1, 3)
        for m in ts["measurements"]:
            for per in m["tracks"]:
                per["values"] = np.frombuffer(per.pop("values"), dtype=np.float32)
                if "indices" in per:
                    per["indices"] = np.frombuffer(per["indices"], dtype=np.uint32)
        for st in ts["track_statistics"]:
            st["values"] = np.frombuffer(st.pop("values"), dtype=np.float64)
    return doc

