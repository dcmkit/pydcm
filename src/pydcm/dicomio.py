# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""binary IO (``pydcm.dicomio``). pydcm's reader/writer are whole-file,
so this provides the common ``DicomBytesIO`` wrapper for ported code."""
from __future__ import annotations

import io

from ._dicom import dcmread, dcmwrite


class DicomBytesIO(io.BytesIO):
    """An in-memory binary buffer."""
    is_little_endian = True
    is_implicit_VR = False


DicomIO = DicomBytesIO
read_file = dcmread
write_file = dcmwrite
__all__ = ["DicomBytesIO", "DicomIO", "dcmread", "dcmwrite", "read_file", "write_file"]
