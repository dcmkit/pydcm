# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""``from pydcm.filereader import dcmread`` + low-level helpers."""
from __future__ import annotations

from ._dicom import dcmread, _build_file_meta

read_file = dcmread                 # legacy alias


def read_preamble(fp, force: bool = False):
    """Read the 128-byte preamble + 'DICM' magic from a binary stream.

    Returns the preamble bytes, or ``None`` when absent and ``force=True``."""
    preamble = fp.read(128)
    if fp.read(4) != b"DICM":
        if force:
            fp.seek(0)
            return None
        from .errors import InvalidDicomError
        raise InvalidDicomError(
            "File is missing DICOM File Meta Information header ('DICM' magic) — "
            "use force=True to read it anyway")
    return preamble


def read_file_meta_info(filename):
    """The group-0002 :class:`~pydcm.dataset.FileMetaDataset` for a Part-10 file."""
    import os
    return _build_file_meta(os.fspath(filename))


__all__ = ["dcmread", "read_file", "read_preamble", "read_file_meta_info"]
