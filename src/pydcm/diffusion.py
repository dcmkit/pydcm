# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DWI gradient tables and volume loading.

The public table is produced by the same native series plan as :func:`load_dwi`:
one entry per assembled 3-D acquisition volume, with the real IPP-derived slice
axis. This matters for tilted stacks and also prevents a per-slice list from being
mistaken for an FSL per-volume ``.bval`` / ``.bvec`` table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _core, _native


def _write_fsl(prefix, bvals, bvecs):
    """Write FSL ``<prefix>.bval`` (one row) + ``<prefix>.bvec`` (3 rows). bvecs
    is [3, N]. Returns the two paths."""
    p = str(prefix)
    with open(p + ".bval", "w") as fh:
        fh.write(" ".join(f"{b:g}" for b in bvals) + "\n")
    with open(p + ".bvec", "w") as fh:
        for i in range(3):
            fh.write(" ".join(f"{v:.6f}" for v in bvecs[i]) + "\n")
    return p + ".bval", p + ".bvec"


def diffusion_table(files, *, output_prefix=None, rotate=True):
    """Collect DWI b-values + gradient directions into FSL ``.bval`` / ``.bvec``.

    files: a directory, a single file, or a list of instances. Entries are one
        per assembled 3-D diffusion acquisition and are ordered by acquisition
        (earliest InstanceNumber, with a stable frame-order tiebreak).
    rotate: take the gradient against the image axes — the native conversion,
        which projects patient-frame encodings, maps GE/Canon classic logical
        axes, and leaves UIH's image-frame private vector unprojected. The b0
        zero vector stays zero.
        ``False`` returns the vector as the vendor stored it, for callers doing
        their own geometry.
    output_prefix: also write ``<prefix>.bval`` and ``<prefix>.bvec``.

    Returns ``(bvals[N], bvecs[3, N])``.

    The rotated bvecs are in the VOXEL convention: they pair with pixel data in
    DICOM row order, which is what :func:`load_dwi` returns and what
    ``to_nifti`` / :func:`save_dwi` write (our NIfTI writer copies rows verbatim
    and puts LPS→RAS in the affine alone). Pairing them instead with a NIfTI
    written rows-bottom-up mirrors every tensor about the row axis and changes no
    scalar map — use ``_core.read_diffusion``'s ``gradient_fsl`` for that case.
    """
    _native.require()
    if isinstance(files, (str, Path)) and Path(files).is_dir():
        from .torchdata import scan
        files = [str(p) for p in scan(files, recursive=True)]
    elif isinstance(files, (str, Path)):
        files = [str(files)]
    else:
        files = [str(f) for f in files]

    # This intentionally shares assemble_dwi rather than reconstructing geometry
    # from independent instances. In particular, its affine third column is the
    # true inter-slice IPP increment for a tilted/sheared stack; row×column is not.
    _, meta = _core.assemble_dwi(files, "acquisition")
    bvals = np.asarray(meta["bvals"], float)
    source = "bvecs" if rotate else "stored_bvecs"
    bvecs = np.asarray(meta[source], float).T            # 3 × volumes

    if output_prefix:
        _write_fsl(output_prefix, bvals, bvecs)
    return bvals, bvecs


def load_dwi(series, *, recursive=True, order="gradient"):
    """Load a single-frame DWI series as a 4-D volume + gradient table.

    Groups the slices by their diffusion (b-value + gradient — the standard
    top-level tags 0018,9087 / 0018,9089, falling back to the Siemens CSA header;
    non-DWI frames are skipped), assembles each direction's 3-D volume, and stacks
    them. Returns ``(data[V, Z, Y, X], bvals[V], bvecs[3, V], affine)`` — bvecs
    rotated into the image/voxel frame, ready for ``dti_*``.

    order : "gradient" (default) sorts volumes by gradient, b0 first — fully
        deterministic; "acquisition" orders them by each direction's earliest
        InstanceNumber. The .bval/.bvec stay aligned either way.

    Enhanced-MF DWI is split through its per-frame MR Diffusion functional
    groups by the same native series engine.
    """
    core = _native.require()
    from pathlib import Path
    if isinstance(series, (str, Path)) and Path(series).is_dir():
        from .torchdata import scan
        files = [str(p) for p in scan(series, recursive=recursive)]
    elif isinstance(series, (str, Path)):
        files = [str(series)]
    else:
        files = [str(f) for f in series]

    # The grouping + 3D assembly + voxel-frame gradient rotation all live in
    # the shared native volume engine, so every consumer produces identical
    # tables.
    arr, meta = core.assemble_dwi(files, order)
    bvals = np.asarray(meta["bvals"], float)
    bvecs = np.asarray(meta["bvecs"], float).T                              # [3, V]
    affine = np.asarray(meta["affine"], np.float32).reshape(4, 4).T          # column-major → row-major
    return np.asarray(arr), bvals, bvecs, affine


def save_dwi(series, output_prefix, *, recursive=True, order="acquisition"):
    """Convert a single-frame DWI series to NIfTI + FSL .bval/.bvec (
    DWI deliverable). Writes ``<prefix>.nii.gz`` (4-D), ``<prefix>.bval`` and
    ``<prefix>.bvec``; returns their paths.

    order defaults to "acquisition" so the volume order is deterministic
    (pass "gradient" for deterministic b0-first ordering).
    """
    core = _native.require()
    data, bvals, bvecs, affine = load_dwi(series, recursive=recursive, order=order)
    p = str(output_prefix)
    arr = np.ascontiguousarray(data, dtype=np.float32)      # [V, Z, Y, X]
    affine_cm = np.asarray(affine).T.flatten().astype(np.float32).tolist()  # row-major → column-major LPS
    core.write_nifti_volume(arr, affine_cm, p + ".nii.gz")
    _write_fsl(p, bvals, bvecs)
    return p + ".nii.gz", p + ".bval", p + ".bvec"

__all__ = ['diffusion_table', 'load_dwi', 'save_dwi']
