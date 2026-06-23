# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Directory of slices → one spatially-ordered 3D volume.

Thin orchestration over the native volume engine: IOP clustering, IPP-projection
Z-sort, and N-D dimension discovery all happen in the compiled `_core` extension.
Nothing about the geometry is reimplemented here — Python only enumerates the
files, hands them to the engine, and wraps the result as NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native
from .torchdata import scan


def _resolve_files(path, recursive, pattern):
    """Normalise a directory / single path / list-of-paths into a non-empty list of
    file-path strings (shared by load_series / load_4d). Raises if nothing is found."""
    if isinstance(path, (str, Path)) and Path(path).is_dir():
        files = scan(path, recursive=recursive, pattern=pattern)
    elif isinstance(path, (str, Path)):
        files = [Path(path)]
    else:
        files = [Path(p) for p in path]
    if not files:
        raise FileNotFoundError(f"no DICOM files found under {path}")
    return [str(f) for f in files]


def _write_nifti(pixels, affine, path):
    """Write a 3-D or 4-D array + row-major LPS affine to NIfTI-1 via the native
    ``dcm_nifti`` N-D writer (``.nii.gz`` → gzip). The writer flips LPS→RAS and keeps
    float32/uint8/int16/uint16 dtypes (so integer label masks stay integer); any other
    dtype is cast to float32. Shared by Volume.to_nifti / Volume4D.to_nifti."""
    core = _native.require()
    arr = np.ascontiguousarray(pixels)
    if arr.dtype not in (np.float32, np.uint8, np.int16, np.uint16):
        arr = arr.astype(np.float32)                       # HU / real-world default
    affine_cm = np.asarray(affine).T.flatten().astype(np.float32).tolist()  # row-major → column-major LPS
    core.write_nifti_volume(arr, affine_cm, str(path))
    return str(path)


def _write_volexport(kind, pixels, affine, path, compress):
    """Write a 3-D/4-D array + row-major LPS affine to NRRD (``kind='nrrd'``) or
    single-file MetaImage (``kind='mha'``) via the native ``dcm_volexport`` engine.
    Both formats are LPS-native — the affine is passed straight through, no RAS
    flip (unlike :func:`_write_nifti`). Dtype handling matches NIfTI: float32/uint8/
    int16/uint16 are kept (label masks stay integer), anything else casts to float32."""
    core = _native.require()
    arr = np.ascontiguousarray(pixels)
    if arr.dtype not in (np.float32, np.uint8, np.int16, np.uint16):
        arr = arr.astype(np.float32)
    # full-precision affine: NRRD/MetaImage ASCII headers carry double (no float32 cast)
    affine_cm = np.asarray(affine).T.flatten().astype(np.float64).tolist()  # row-major → column-major LPS
    if kind == "nrrd":
        core.write_nrrd_volume(arr, affine_cm, str(path), compress)
    else:
        core.write_metaimage_volume(arr, affine_cm, str(path), compress)
    return str(path)


@dataclass
class Volume:
    """An assembled 3D volume. ``pixels`` is float32 Hounsfield/real-world values from
    :func:`load_series`; from :func:`from_nifti` it keeps the file's dtype (e.g. an
    integer label mask)."""

    pixels: np.ndarray            # [depth, rows, cols]; float32 HU (load_series) or file dtype
    spacing: tuple                # (z, y, x) mm  — slice, row, col
    affine: np.ndarray            # 4×4 voxel→world (row-major)
    series_instance_uid: str

    @property
    def shape(self):
        return self.pixels.shape

    def __repr__(self):
        return (f"Volume(shape={self.pixels.shape}, spacing={tuple(round(s, 3) for s in self.spacing)}, "
                f"series={self.series_instance_uid[:16]}…)")

    def to_nifti(self, path) -> str:
        """Write this volume to NIfTI-1. ``.nii.gz`` extension → gzip, else ``.nii``.

        Thin call into the native ``dcm_nifti`` engine — the voxel→world affine
        (LPS) is flipped to RAS inside the writer; nothing is recomputed here. The
        NIfTI datatype follows ``pixels``' dtype (float32 / uint8 / int16 / uint16,
        so integer label masks stay integer); other dtypes are cast to float32.
        """
        return _write_nifti(self.pixels, self.affine, path)

    def to_nrrd(self, path, *, gzip: bool = False) -> str:
        """Write this volume to single-file NRRD (``.nrrd``) — 3D Slicer's native
        format. LPS-native, so the voxel→world affine maps straight to NRRD
        ``space directions``/``space origin`` (no RAS flip). ``gzip=True`` →
        ``encoding: gzip``. Datatype follows ``pixels`` (label masks stay integer)."""
        return _write_volexport("nrrd", self.pixels, self.affine, path, gzip)

    def to_metaimage(self, path, *, compress: bool = False) -> str:
        """Write this volume to single-file MetaImage (``.mha``) — the ITK / nnU-Net /
        MONAI interchange format. LPS-native (direction cosines → ``TransformMatrix``,
        origin → ``Offset``). ``compress=True`` → ``CompressedData=True``."""
        return _write_volexport("mha", self.pixels, self.affine, path, compress)


def load_series(path, *, recursive: bool = True, pattern: str | None = None) -> Volume:
    """Assemble the DICOM slices under ``path`` into one ordered 3D HU volume.

    ``path`` is a directory (or list of files). Files are grouped/sorted by
    the native engine; the **largest coherent volume** is returned (so a stray
    localizer or mixed series does not corrupt the stack). For a plain CT/MR
    series directory that is simply the volume.
    """
    core = _native.require()
    arr, meta = core.assemble_volume(_resolve_files(path, recursive, pattern))
    affine = np.asarray(meta["affine"], dtype=np.float64).reshape(4, 4).T  # column-major → row-major (double-faithful)
    return Volume(arr, tuple(meta["spacing"]), affine, meta["series_instance_uid"])


@dataclass
class Axis:
    """One non-spatial axis of a :class:`Volume4D` — what the 4th dimension *means*.

    ``kind`` is the semantic label (``"temporal"``, ``"bvalue"``, ``"direction"``,
    ``"echo"``, ``"cardiac"``, ``"stack"``, ``"frametype"``, ``"velocity"``,
    ``"energy"``); ``values`` holds one sorted value per step (e.g. trigger times,
    b-values, echo times)."""

    kind: str                     # semantic name (temporal / bvalue / echo / …)
    values: np.ndarray            # [size] one value per step along this axis
    spacing: float = 0.0          # uniform step (0.0 = non-uniform)
    tag: int = 0                  # raw vol_dim_tag_t

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Axis(kind={self.kind!r}, size={len(self.values)}, values={np.asarray(self.values).tolist()})"


@dataclass
class Volume4D:
    """A 4-D stack: ``n_volumes`` co-registered 3-D volumes sharing one world grid.

    The 4th dimension is the non-spatial axis (or axes) that varies across the
    series — time (cardiac cine, perfusion, fMRI), b-value/direction (DWI), echo
    time (multi-echo), cardiac phase, stack, spectral energy. :attr:`dimensions`
    labels it; when more than one axis varies the volume index decomposes
    row-major (slowest-first) across them — see :meth:`coords`.
    """

    pixels: np.ndarray            # [T, depth, rows, cols] float32 (HU/real-world)
    spacing: tuple                # (z, y, x) mm — slice, row, col
    affine: np.ndarray           # 4×4 voxel→world (row-major), shared by every volume
    dimensions: list             # list[Axis], slowest-first; product of sizes == T
    series_instance_uid: str
    volume_path: list | None = None      # [T] representative source file per volume
    volume_frame: list | None = None     # [T] frame index within that file

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def n_volumes(self):
        return self.pixels.shape[0]

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, i) -> Volume:
        """Volume ``i`` of the stack as a 3-D :class:`Volume` (shares the grid)."""
        return Volume(self.pixels[i], self.spacing, self.affine, self.series_instance_uid)

    def coords(self, i) -> dict:
        """The non-spatial coordinate of volume ``i`` as ``{kind: value}`` (e.g.
        ``{"bvalue": 1000.0, "direction": 3.0}`` or ``{"temporal": 7.0}``)."""
        sizes = tuple(len(a) for a in self.dimensions) or (1,)
        idx = np.unravel_index(i, sizes)                      # row-major, slowest-first
        return {a.kind: float(a.values[j]) for a, j in zip(self.dimensions, idx)}

    def axis(self, kind: str) -> Axis | None:
        """The :class:`Axis` with the given ``kind`` (``"temporal"`` …), or ``None``."""
        return next((a for a in self.dimensions if a.kind == kind), None)

    def __repr__(self):
        dims = ", ".join(f"{a.kind}×{len(a)}" for a in self.dimensions) or "single"
        return (f"Volume4D(shape={self.pixels.shape}, dims=[{dims}], "
                f"spacing={tuple(round(s, 3) for s in self.spacing)}, "
                f"series={self.series_instance_uid[:16]}…)")

    def to_nifti(self, path) -> str:
        """Write the 4-D stack to NIfTI-1 (``.nii.gz`` → gzip). Thin call into the
        native ``dcm_nifti`` N-D writer — the LPS affine is flipped to RAS inside;
        the 4th axis becomes NIfTI's time/volume dimension."""
        return _write_nifti(self.pixels, self.affine, path)

    def to_nrrd(self, path, *, gzip: bool = False) -> str:
        """Write the 4-D stack to single-file NRRD (``.nrrd``). The 4th axis becomes
        a non-spatial NRRD axis (``kind: list``, no space direction). ``gzip=True`` →
        ``encoding: gzip``."""
        return _write_volexport("nrrd", self.pixels, self.affine, path, gzip)

    def to_metaimage(self, path, *, compress: bool = False) -> str:
        """Write the 4-D stack to single-file MetaImage (``.mha``); the 4th axis is the
        slowest MetaImage dimension. ``compress=True`` → ``CompressedData=True``."""
        return _write_volexport("mha", self.pixels, self.affine, path, compress)


def load_4d(path, *, recursive: bool = True, pattern: str | None = None) -> Volume4D:
    """Assemble the DICOM slices under ``path`` into one 4-D stack ``[T, Z, Y, X]``.

    The native engine clusters by orientation/grid, Z-sorts by IPP projection, and
    discovers the varying non-spatial axes (time / b-value / direction / echo /
    cardiac phase / stack / energy) — each becomes the 4th dimension and is
    labelled in :attr:`Volume4D.dimensions`. ``path`` may be a directory, a single
    enhanced multi-frame file, or a list of files. A plain 3-D series yields
    ``T == 1`` with an empty ``dimensions`` list.

    For a DWI series where you also need the FSL ``.bval``/``.bvec`` gradient table,
    use :func:`pydcm.load_dwi`; this returns the geometry + dimension semantics for
    *any* 4-D organisation, not just diffusion.
    """
    core = _native.require()
    arr, meta = core.assemble_4d(_resolve_files(path, recursive, pattern))
    affine = np.asarray(meta["affine"], dtype=np.float64).reshape(4, 4).T  # column-major → row-major (double-faithful)
    dims = [Axis(d["name"], np.asarray(d["values"], dtype=np.float32), float(d["spacing"]), int(d["tag"]))
            for d in meta.get("dimensions", [])]
    return Volume4D(arr, tuple(meta["spacing"]), affine, dims, meta["series_instance_uid"],
                    list(meta.get("volume_path", [])), list(meta.get("volume_frame", [])))


def from_nifti(path) -> Volume:
    """Read a NIfTI-1 file (``.nii``/``.nii.gz``) back into a :class:`Volume`.

    The file's RAS sform is flipped to our LPS convention by the native reader,
    so the returned ``affine`` matches what :func:`load_series` produces. Voxels
    keep the file's dtype (e.g. float32 image, integer label mask).
    """
    core = _native.require()
    arr, meta = core.read_nifti(str(path))
    if arr.ndim == 4:                                  # 4D (e.g. fMRI/DWI) — a Volume is 3D
        raise ValueError(
            f"4D NIfTI ({meta.get('time_points')} time points) cannot be a 3D Volume; "
            "read the raw array via pydcm._core.read_nifti(path) instead")
    affine = np.asarray(meta["affine"], dtype=np.float64).reshape(4, 4).T  # column-major → row-major (double-faithful)
    return Volume(arr, tuple(meta["spacing"]), affine, "")


def bids_sidecar(path) -> dict:
    """Extract a BIDS JSON sidecar (the standard BIDS metadata written next to a ``.nii``)
    from one DICOM instance — timing (in **seconds**), sequence, and geometry fields.

    Returns a ``dict`` of the present fields (e.g. ``RepetitionTime``, ``EchoTime``,
    ``FlipAngle``, ``Manufacturer``, ``ImageOrientationPatientDICOM``).

    ``PhaseEncodingDirection`` (BIDS ``i``/``i-``/``j``/``j-``) is emitted when the
    vendor records the polarity (Siemens CSA / 0021,111C; GE 0018,9034; UIH 0065,1058);
    its sign follows this writer's no-row-flip storage (no row flip), so
    the **i** sign equals the reference while the **j** sign is the *negation* — verified on
    the dcm_qa suite, where our volume equals the reference with the rows
    flipped, so each sidecar correctly describes its own array. Unknown polarity gives only
    the unsigned ``PhaseEncodingAxis``; a 3-D non-EPI scan emits neither.

    ``SliceTiming`` (seconds) is emitted for a Siemens mosaic from the CSA
    ``MosaicRefAcqTimes`` (the whole schedule lives in one instance).
    ``EffectiveEchoSpacing``/``TotalReadoutTime`` (seconds) are emitted for Siemens EPI
    (``1/(BW·N)`` and ``ES·(N−1)``, ``N``=NumberOfPhaseEncodingSteps)
    for full-resolution and phase-oversampled EPI.
    """
    import json
    core = _native.require()
    return json.loads(core.bids_sidecar(Path(path).read_bytes()))

__all__ = ['Volume', 'Volume4D', 'Axis', 'load_series', 'load_4d', 'from_nifti',
           'bids_sidecar']
