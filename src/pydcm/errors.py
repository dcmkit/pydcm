# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""exceptions (`pydcm.errors`)."""


class InvalidDicomError(Exception):
    """The file is not valid DICOM."""


class BytesLengthException(Exception):
    """A value's byte length is inconsistent with its VR."""
