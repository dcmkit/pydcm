# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""DWI gradient table — a DICOM series → FSL ``.bval`` / ``.bvec``.

Thin orchestration over the native ``read_diffusion`` (the STANDARD MR Diffusion
sequence first — the modern enhanced-MF path already parsed by the native core — then the
legacy Siemens CSA fallback). The collected (b-value, gradient) table is exactly
the input the native ``dcm_dti`` tensor engine consumes.
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

    files: a directory, a single file, or a list of instances. Multi-frame
        (enhanced-MF) files contribute one entry per frame; single-frame series
        one per file. Entries are ordered by (InstanceNumber, frame).
    rotate: rotate each gradient from the patient (LPS) frame into the image/voxel
        frame ``[row·g, col·g, normal·g]`` (the b0 zero vector is left as-is).
    output_prefix: also write ``<prefix>.bval`` and ``<prefix>.bvec``.

    Returns ``(bvals[N], bvecs[3, N])``.

    Note: bvecs are in the DICOM (LPS) image frame; FSL — relative to the NIfTI
    (RAS) axes — may need an axis sign flip. Validate against your pipeline.
    """
    _native.require()
    from . import dcmread   # deferred: avoid a circular import at package init
    if isinstance(files, (str, Path)) and Path(files).is_dir():
        from .torchdata import scan
        files = [str(p) for p in scan(files, recursive=True)]
    elif isinstance(files, (str, Path)):
        files = [str(files)]
    else:
        files = [str(f) for f in files]

    entries = []  # (instance, frame, bval, bvec3)
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        frames = _core.read_diffusion(raw)
        if not frames:
            continue
        ds = dcmread(f)
        iop = np.asarray(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
        row, col = iop[:3], iop[3:6]
        nrm = np.cross(row, col)
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        for fi, e in enumerate(frames):
            g = np.asarray(e["gradient"], float)
            if rotate and np.linalg.norm(g) > 1e-6:
                bvec = np.array([row @ g, col @ g, nrm @ g])
            else:
                bvec = g
            entries.append((inst, fi, float(e["b_value"]), bvec))

    if not entries:
        raise ValueError("no diffusion frames found in the given files")
    entries.sort(key=lambda x: (x[0], x[1]))
    bvals = np.array([e[2] for e in entries], float)
    bvecs = np.array([e[3] for e in entries], float).T   # 3 × N

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

    Note: this covers single-frame (per-file) DWI. Enhanced-MF DWI needs the
    native per-frame MR Diffusion parse (a separate change).
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
