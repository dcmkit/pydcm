# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""``from pydcm.filewriter import dcmwrite`` + low-level helpers."""
from __future__ import annotations

from ._dicom import dcmwrite, Tag

write_file = dcmwrite                # legacy alias


def write_dataset(fp, dataset) -> int:
    """Write ``dataset`` as a complete Part-10 stream to a writable binary ``fp``.

    pydcm's writer is whole-file (it reuses the native Part-10 encoder), so this emits a
    full Part-10 buffer rather than a bare dataset body; returns the number of bytes
    written. For element-by-element control use :meth:`Dataset.save_as` to a path."""
    data = dataset._encode_part10()
    fp.write(data)
    return len(data)


def write_file_meta_info(fp, file_meta, enforce_standard: bool = True) -> int:
    """Write the group-0002 File Meta Information of ``file_meta`` to ``fp``.

    Reuses the native encoder via a carrier dataset; returns bytes written."""
    from ._dicom import Dataset
    carrier = Dataset()
    for t, e in file_meta._dict.items():
        carrier._dict[Tag(t)] = e
    data = carrier._encode_part10()
    fp.write(data)
    return len(data)


def tag_in_exception(tag):
    """No-op context manager that would annotate exceptions with ``tag``."""
    from .tag import tag_in_exception as _tie
    return _tie(tag)


__all__ = ["dcmwrite", "write_file", "write_dataset", "write_file_meta_info",
           "tag_in_exception"]
