# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""`from pydcm.dataset import Dataset, FileDataset, FileMetaDataset`.
(The PyTorch DICOMDataset moved to `pydcm.torchdata`.)"""
from ._dicom import Dataset, FileDataset, FileMetaDataset, DataElement


def validate_file_meta(file_meta=None, enforce_standard=True):
    """Validate a File Meta Information dataset.

    pydcm keeps this public helper and delegates to
    :meth:`FileMetaDataset.validate`, which checks the group-0002 boundary using
    the shared Dataset/Tag model.
    """
    if file_meta is not None:
        FileMetaDataset.validate(file_meta)
    return None


__all__ = ["Dataset", "FileDataset", "FileMetaDataset", "DataElement", "validate_file_meta"]
