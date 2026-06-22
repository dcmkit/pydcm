# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""overlay-plane access (``pydcm.overlays``)."""
from __future__ import annotations


def get_overlay_array(ds, group: int):
    """The (rows × cols) {0,1} overlay plane of ``group`` (60xx), PS3.3 C.9."""
    return ds.overlay_array(group)


__all__ = ["get_overlay_array"]
