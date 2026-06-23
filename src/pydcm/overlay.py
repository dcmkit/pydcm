# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""pydcm — render a DICOM frame and burn structured markup onto it (``pydcm.overlay``).

The headless, agent-vision rendering path: render one frame to an 8-bit display
image (window/level), then burn a presentation state's graphic annotations (GSPS)
and/or a structured report's SCOORD measurement regions (SR) onto the pixels — so a
vision model SEES the radiologist's markup. Only markup that references the rendered
image is drawn. All compute is native; this is a thin wrapper.
"""
from __future__ import annotations

import io
import os

from . import _core


def _part10(x) -> bytes:
    """Coerce a path / bytes / Dataset to Part-10 bytes."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, (str, os.PathLike)):
        with open(os.fspath(x), "rb") as f:
            return f.read()
    if hasattr(x, "save_as"):                       # a dataset-like object
        buf = io.BytesIO()
        x.save_as(buf)
        return buf.getvalue()
    raise TypeError(f"render_overlay: expected a path, bytes, or Dataset, got {type(x).__name__}")


def render_overlay(image, overlays=(), *, frame=1, window=None, max_dim=0, with_overlays=False):
    """Render ``image`` (one frame) to 8-bit RGB and burn ``overlays``' markup onto it.

    image: the DICOM image — path | bytes | Dataset.
    overlays: a GSPS/SR object or an iterable of them (path | bytes | Dataset).
        Each is auto-detected; only markup referencing ``image``'s SOP Instance
        is drawn (GSPS annotations in their layer colour, SR SCOORDs in green).
    frame: 1-based frame number (default 1).
    window: ``(center, width)`` window/level, or ``None`` for the per-frame default.
    max_dim: aspect-preserving downscale so the largest side fits (0 = native).
        Markup is burned at native resolution and then downscaled with
        coverage preservation so thin contours remain visible.
    with_overlays: also return a structured description of the markup. The shapes
        (and the image's own 60xx overlay planes) are projected to pixel space with
        UTF-8 text / measurement values / labels — the "image + overlay JSON" form,
        so values/text need no in-pixel font.

    The result is a ``numpy.ndarray`` ``[H, W, 3]`` uint8 (RGB) — or, with
    ``with_overlays=True``, a tuple ``(ndarray, overlays)`` where ``overlays`` is a
    list of dicts (``source``, ``kind``, ``points``, ``color``, and optional
    ``text`` / ``label`` / ``value`` / ``unit`` / ``number`` / ``filled``).
    """
    if image is None:
        raise ValueError("render_overlay: image is required")
    if isinstance(overlays, (str, bytes, bytearray, os.PathLike)) or hasattr(overlays, "save_as"):
        overlays = (overlays,)                      # a single overlay, not an iterable
    if window is not None and len(window) != 2:
        raise ValueError("render_overlay: window must be (center, width)")
    wc, ww = window if window is not None else (None, None)
    return _core.render_overlay(_part10(image), [_part10(o) for o in overlays],
                                int(frame), wc, ww, int(max_dim), bool(with_overlays))
