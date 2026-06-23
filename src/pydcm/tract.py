# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — DICOM Tractography Results authoring (``pydcm.tract``).

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


def write_mktract(reference, track_sets, *, output=None):
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
    return _core.write_mktract(_refs(reference), norm, str(output) if output else "")


__all__ = ["write_mktract"]
