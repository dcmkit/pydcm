# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""`from pydcm.dataset import Dataset, FileDataset, FileMetaDataset`.
(The PyTorch DICOMDataset moved to `pydcm.torchdata`.)"""
from ._dicom import Dataset, FileDataset, FileMetaDataset, DataElement


def validate_file_meta(file_meta=None, enforce_standard=True):
    """Accepted for source compatibility; pydcm validates leniently."""
    return None


__all__ = ["Dataset", "FileDataset", "FileMetaDataset", "DataElement", "validate_file_meta"]
