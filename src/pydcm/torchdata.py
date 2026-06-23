# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Directory → samples. A directory of DICOM files is, for PyTorch, just a list
of instances; ``DICOMDataset`` walks it and decodes one image per ``__getitem__``.

One sample = one file. Single-frame files yield ``[rows, cols(, samples)]``;
multi-frame files yield ``[frames, rows, cols(, samples)]``. To instead collapse
a directory into one spatially-ordered 3D volume, use :func:`pydcm.load_series`.
"""

from __future__ import annotations

from pathlib import Path

from . import _native

_DICOM_EXT = {".dcm", ".dicom", ".ima"}


def _is_dicom(p: Path) -> bool:
    if p.suffix.lower() in _DICOM_EXT:
        return True
    try:                                          # PS3.10 preamble: "DICM" at offset 128
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def scan(root, *, recursive: bool = True, pattern: str | None = None) -> list[Path]:
    """Discover DICOM instance files under ``root`` (a directory or a single file).

    ``pattern`` — a glob (e.g. ``"*.dcm"``) selects by name only. When ``None``,
    files are detected by extension OR the ``DICM`` preamble (also catching the
    extension-less files clinical exports often produce). Returns a sorted list.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if pattern is None:                           # discovery happens in the native engine
        return [Path(p) for p in _native.require().scan_dicom_dir(str(root), recursive)]
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(p for p in it if p.is_file())


class DICOMDataset:
    """Map-style dataset over the DICOM files under ``root``.

    DataLoader-compatible via ``__len__`` / ``__getitem__`` WITHOUT importing
    torch, so torch stays optional. ``__getitem__`` returns a NumPy array (or, with
    ``to_torch=True``, a ``torch.Tensor``); pass a ``transform`` to override that
    and shape each sample however your model wants. ``rescale=True`` yields HU.
    """

    def __init__(self, root, *, recursive: bool = True, pattern: str | None = None,
                 rescale: bool = False, transform=None, to_torch: bool = False):
        self.files = scan(root, recursive=recursive, pattern=pattern)
        if not self.files:
            raise FileNotFoundError(f"no DICOM files found under {root}")
        self.rescale = rescale
        self.transform = transform
        self.to_torch = to_torch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        arr, _ = _native.decode(self.files[i], 0, self.rescale)
        if arr.shape[0] == 1:                     # squeeze single frame → [H, W(, C)]
            arr = arr[0]
        if self.transform is not None:
            return self.transform(arr)
        if self.to_torch:
            import torch
            return torch.from_numpy(arr.copy())   # copy → writable + owns its memory
        return arr

    def __repr__(self) -> str:
        return f"DICOMDataset(n={len(self)})"

__all__ = ['DICOMDataset', 'scan']
