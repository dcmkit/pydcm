# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""Whole-slide imaging (DICOM VL Whole Slide Microscopy) reader — an surface over pydcm's native pyramid engine.

    from pydcm.wsi import open_slide
    s = open_slide("/path/to/slide_dir")           # a dir of the slide's .dcm levels
    s.level_count, s.level_dimensions, s.level_downsamples
    rgba = s.read_region((x, y), level, (w, h))     # (x,y) in LEVEL-0 coords → numpy

``read_region`` location is in level-0 reference coords, size in the
requested level's coords; default returns RGBA, edge/sparse-missing pixels are
transparent). ``rgba=False`` returns RGB. ``associated_images`` exposes DICOM label /
overview / thumbnail / localizer images as a lazy mapping. The decode / tile assembly /
pyramid all run in the shared native core — analysis (tiling for ML, stain
normalisation) stays interop: feed the returned NumPy arrays to your pipeline.
"""
from __future__ import annotations

from collections.abc import Mapping
import glob
import os

import numpy as np

from . import _core

# WSI summary properties are published under the neutral canonical ``wsi.*`` namespace.
# Every key is ALSO mirrored under ``openslide.*`` in the properties dict, so code written
# against the openslide-python property convention (``properties["openslide.mpp-x"]``) keeps
# working unchanged — these constants name the canonical form.
PROPERTY_NAME_VENDOR = "wsi.vendor"
PROPERTY_NAME_MPP_X = "wsi.mpp-x"
PROPERTY_NAME_MPP_Y = "wsi.mpp-y"
PROPERTY_NAME_OBJECTIVE_POWER = "wsi.objective-power"
PROPERTY_NAME_BOUNDS_X = "wsi.bounds-x"
PROPERTY_NAME_BOUNDS_Y = "wsi.bounds-y"
PROPERTY_NAME_BOUNDS_WIDTH = "wsi.bounds-width"
PROPERTY_NAME_BOUNDS_HEIGHT = "wsi.bounds-height"
_VL_WSI_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.6"
MISSING_FRAME_INDEX = int(np.iinfo(np.uint32).max)


def _convert_dtype(arr, dtype):
    if dtype is None:
        return arr
    return arr.astype(np.dtype(dtype), copy=False)


def _reject_unsupported_pixel_transforms(
    *,
    apply_real_world_transform=None,
    apply_modality_transform=None,
    apply_voi_transform=False,
    apply_palette_color_lut=None,
):
    unsupported = []
    if apply_real_world_transform:
        unsupported.append("apply_real_world_transform")
    if apply_modality_transform:
        unsupported.append("apply_modality_transform")
    if apply_voi_transform:
        unsupported.append("apply_voi_transform")
    if apply_palette_color_lut:
        unsupported.append("apply_palette_color_lut")
    if unsupported:
        names = ", ".join(unsupported)
        raise NotImplementedError(f"WSI pixel transforms are not implemented: {names}")


def _validate_icc_request(apply_icc_profile, profile, transform_available: bool) -> bool:
    if apply_icc_profile not in (None, True, False):
        raise TypeError("apply_icc_profile must be None, True, or False")
    if apply_icc_profile is False:
        return False
    if not profile:
        if apply_icc_profile is True:
            raise ValueError("ICC profile requested but the WSI level has no ICC profile")
        return False
    if not transform_available:
        raise RuntimeError("ICC profile transform requires an LCMS2-enabled pydcm build")
    return True


def _normalize_axis_interval(start, end, length: int, *, as_indices: bool, axis: str):
    def convert(value, *, default, is_end: bool):
        if value is None:
            return default
        value = int(value)
        if as_indices:
            index = length + value if value < 0 else value
        else:
            if value == 0:
                raise ValueError(f"{axis} positions are 1-based unless as_indices=True")
            index = length + value if value < 0 else value - 1
            if is_end and value == length + 1:
                index = length
        return index

    first = convert(start, default=0, is_end=False)
    last = convert(end, default=length, is_end=True)
    if first < 0 or last < 0 or first > length or last > length or first > last:
        raise ValueError(f"invalid {axis} interval")
    return first, last


def _normalize_frame_index(frame_number, frame_count: int, *, as_index: bool) -> int:
    frame_number = int(frame_number)
    if as_index:
        index = frame_count + frame_number if frame_number < 0 else frame_number
    else:
        if frame_number == 0:
            raise ValueError("frame numbers are 1-based unless as_index/as_indices=True")
        index = frame_count + frame_number if frame_number < 0 else frame_number - 1
    if index < 0 or index >= frame_count:
        raise ValueError("frame number out of range")
    return index


class _AssociatedImages(Mapping):
    """Lazy mapping of associated image name to uint8 NumPy image."""

    def __init__(self, slide: "Slide"):
        self._slide = slide

    def __getitem__(self, name: str):
        if name not in self._slide.associated_image_names:
            raise KeyError(name)
        return self._slide.read_associated_image(name)

    def __iter__(self):
        return iter(self._slide.associated_image_names)

    def __len__(self) -> int:
        return len(self._slide.associated_image_names)


class Slide:
    """A whole-slide pyramid (subset)."""

    def __init__(self, instances):
        """`instances`: paths to the DICOM files of ONE slide (levels + label/overview)."""
        paths = [os.fspath(p) for p in instances]
        if not paths:
            raise ValueError("no instances given")
        if hasattr(_core, "wsi_open_paths"):
            self._s = _core.wsi_open_paths(paths)
        else:
            data = []
            for p in paths:
                with open(p, "rb") as fh:
                    data.append(fh.read())
            self._s = _core.wsi_open(data)
        open_error = str(self._s.open_error())
        if open_error:
            raise NotImplementedError(open_error)
        if self._s.level_count() == 0:
            raise ValueError("no VL Whole Slide Microscopy VOLUME levels found")

    @property
    def level_count(self) -> int:
        return self._s.level_count()

    @property
    def level_dimensions(self):
        """``((cols, rows), ...)`` per level, largest (level 0) first."""
        return tuple(self._s.level_dimensions(i) for i in range(self.level_count))

    @property
    def level_tile_dimensions(self):
        """``((tile_cols, tile_rows), ...)`` per level."""
        return tuple(self._s.level_tile_dimensions(i) for i in range(self.level_count))

    @property
    def level_tile_counts(self):
        """``((tiles_x, tiles_y), ...)`` per level."""
        return tuple(self._s.level_tile_counts(i) for i in range(self.level_count))

    @property
    def level_frame_counts(self):
        """Stored frame count per level in source instance frame order."""
        return tuple(int(self._s.level_frame_count(i)) for i in range(self.level_count))

    def level_descriptor(self, level: int) -> dict:
        """Viewer-oriented metadata for one level without decoding any tile."""
        desc = self._s.level_descriptor(int(level))
        if desc is None:
            raise ValueError("invalid WSI level")
        return dict(desc)

    def viewer_level(self, level: int, *, include_ranges=False) -> dict:
        """Return the level descriptor plus source paths and optional range table.

        The level entry a tiled WSI viewer consumes directly:
        metadata is cheap, and the dense encoded tile range grid can be handed to
        a range loader / tile scheduler without decoding pixels in this call.
        """
        level = int(level)
        desc = self.level_descriptor(level)
        desc["source_paths"] = self.level_source_paths(level)
        if include_ranges:
            desc["tile_range_grid"] = self.level_tile_range_grid(level)
        return desc

    def viewer_levels(self, *, include_ranges=False) -> tuple[dict, ...]:
        """Return viewer-oriented descriptors for all pyramid levels."""
        return tuple(
            self.viewer_level(i, include_ranges=include_ranges)
            for i in range(self.level_count)
        )

    def level_concatenation(self, level: int) -> dict | None:
        """DICOM Concatenation metadata for a pyramid level, if present."""
        return self._s.level_concatenation(int(level))

    @property
    def level_downsamples(self):
        return tuple(self._s.level_downsample(i) for i in range(self.level_count))

    @property
    def properties(self) -> dict[str, str]:
        """and DICOM-derived slide metadata as string properties."""
        return dict(self._s.properties())

    @property
    def associated_image_names(self):
        """Names of associated non-pyramid images, e.g. ``label`` or ``overview``."""
        return tuple(self._s.associated_image_names())

    @property
    def associated_images(self):
        """Lazy mapping: name -> RGBA uint8 NumPy image."""
        return _AssociatedImages(self)

    def associated_image_dimensions(self, name: str):
        """``(cols, rows)`` for an associated image, or ``(0, 0)`` if absent."""
        return self._s.associated_image_dimensions(str(name))

    def read_associated_image(self, name: str, *, rgba=True, apply_icc_profile=False):
        """Return an associated image as ``(rows, cols, 4)`` RGBA or ``(rows, cols, 3)`` RGB."""
        name = str(name)
        apply_icc = _validate_icc_request(
            apply_icc_profile, self.associated_image_icc_profile(name),
            self.icc_transform_available)
        if apply_icc:
            arr = self._s.read_associated_image_srgb(name, bool(rgba))
            w, h = self.associated_image_dimensions(name)
            if arr.size == 0 and w > 0 and h > 0:
                raise RuntimeError("associated image read failed ICC transform")
            return arr
        return self._s.read_associated_image(name, bool(rgba))

    @property
    def icc_profile(self) -> bytes | None:
        """Raw DICOM ICC Profile bytes for the base pyramid level, if present."""
        return self.level_icc_profile(0)

    def level_icc_profile(self, level: int) -> bytes | None:
        """Raw DICOM ICC Profile bytes for one pyramid level, if present."""
        return self._s.level_color_profile(int(level))

    @property
    def icc_transform_available(self) -> bool:
        """Whether this build can apply WSI ICC profiles to sRGB via LCMS2."""
        return bool(self._s.icc_transform_available())

    def associated_image_icc_profile(self, name: str) -> bytes | None:
        """Raw DICOM ICC Profile bytes for an associated image, if present."""
        return self._s.associated_image_color_profile(str(name))

    @property
    def dimensions(self):
        """Level-0 (cols, rows)."""
        return self._s.level_dimensions(0)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the best pyramid level for `downsample`."""
        return int(self._s.best_level_for_downsample(float(downsample)))

    @property
    def tile_cache_capacity(self) -> int:
        """Decoded tile cache capacity in bytes. ``0`` disables retaining decoded tiles."""
        return int(self._s.tile_cache_capacity())

    def set_tile_cache_capacity(self, capacity: int) -> None:
        """Set decoded tile cache capacity in bytes for this slide."""
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("tile cache capacity must be non-negative")
        self._s.set_tile_cache_capacity(capacity)

    def read_region(self, location, level, size, *, rgba=True):
        """`location` = (x, y) top-left in LEVEL-0 coords; `size` = (w, h) in `level`
        coords. Returns a ``(h, w, 4)`` RGBA (or ``(h, w, 3)`` RGB) uint8 NumPy array."""
        x, y = location
        w, h = size
        if int(w) < 0 or int(h) < 0:
            raise ValueError("read_region size must be non-negative")
        return self._s.read_region(int(x), int(y), int(level), int(w), int(h), bool(rgba))

    def read_tile(self, level, tile, *, rgba=True, fill_missing=False):
        """Return one stored tile by zero-based ``tile`` = (tile_x, tile_y).

        Sparse-missing tiles return an empty array by default. With
        ``fill_missing=True``, an in-grid sparse-missing tile returns an all-zero tile
        instead (transparent in RGBA).
        """
        tx, ty = tile
        if int(tx) < 0 or int(ty) < 0:
            raise ValueError("tile coordinates must be non-negative")
        return self._s.read_tile(int(level), int(tx), int(ty), bool(rgba), bool(fill_missing))

    def read_tiles(self, level, tiles, *, rgba=True, fill_missing=False):
        """Return multiple stored tiles in input order.

        This is equivalent to repeated ``read_tile()`` calls, but crosses the
        Python/native boundary once for the whole batch.
        """
        norm_tiles = []
        for tile in tiles:
            tx, ty = tile
            if int(tx) < 0 or int(ty) < 0:
                raise ValueError("tile coordinates must be non-negative")
            norm_tiles.append((int(tx), int(ty)))
        return list(self._s.read_tiles(int(level), norm_tiles, bool(rgba), bool(fill_missing)))

    def read_tile_stack(self, level, tiles, *, rgba=True, fill_missing=False):
        """Return multiple full-size tiles as ``(n, tile_h, tile_w, channels)``.

        Unlike ``read_tiles()``, this is strict: every requested tile must produce a
        full tile. Sparse-missing tiles require ``fill_missing=True``.
        """
        norm_tiles = []
        for tile in tiles:
            tx, ty = tile
            if int(tx) < 0 or int(ty) < 0:
                raise ValueError("tile coordinates must be non-negative")
            norm_tiles.append((int(tx), int(ty)))
        return self._s.read_tile_stack(int(level), norm_tiles, bool(rgba), bool(fill_missing))

    def read_tile_grid(self, level, origin, shape, *, rgba=True, require_existing=False):
        """Read a rectangular tile grid as ``(tile_rows, tile_cols, tile_h, tile_w, channels)``.

        This uses one native ``read_region()`` call and returns a NumPy view over the
        region buffer. Sparse-missing tiles are transparent by default; set
        ``require_existing=True`` to reject grids containing missing sparse tiles.
        """
        level = int(level)
        tx0, ty0 = origin
        grid_w, grid_h = shape
        tx0, ty0, grid_w, grid_h = int(tx0), int(ty0), int(grid_w), int(grid_h)
        if tx0 < 0 or ty0 < 0:
            raise ValueError("tile coordinates must be non-negative")
        if grid_w < 0 or grid_h < 0:
            raise ValueError("tile grid shape must be non-negative")

        tile_w, tile_h = self._s.level_tile_dimensions(level)
        tiles_x, tiles_y = self._s.level_tile_counts(level)
        if tile_w <= 0 or tile_h <= 0:
            raise ValueError("invalid WSI level or zero tile dimensions")
        if tx0 + grid_w > tiles_x or ty0 + grid_h > tiles_y:
            raise ValueError("tile grid extends beyond the WSI level")

        if require_existing:
            for ty in range(ty0, ty0 + grid_h):
                for tx in range(tx0, tx0 + grid_w):
                    if not self.tile_exists(level, (tx, ty)):
                        raise RuntimeError("tile grid contains a missing tile")

        downsample = float(self._s.level_downsample(level))
        x0 = int(tx0 * tile_w * downsample + 0.5)
        y0 = int(ty0 * tile_h * downsample + 0.5)
        region = self.read_region(
            (x0, y0), level, (grid_w * tile_w, grid_h * tile_h), rgba=rgba)
        channels = 4 if rgba else 3
        return region.reshape(grid_h, tile_h, grid_w, tile_w, channels).swapaxes(1, 2)

    def level_frame_tile(self, level, frame_number, *, as_index=False):
        """Return ``(tile_x, tile_y)`` for a stored frame in source frame order.

        ``frame_number`` is DICOM 1-based by default. Set ``as_index=True`` for
        Python 0-based indexing.
        """
        level = int(level)
        frame_count = int(self._s.level_frame_count(level))
        index = _normalize_frame_index(frame_number, frame_count, as_index=bool(as_index))
        tile = self._s.level_frame_tile(level, index)
        if tile is None:
            raise ValueError("frame number out of range")
        return tuple(tile)

    def level_source_paths(self, level: int) -> tuple[str, ...]:
        """File-backed source path(s) for a level; memory-backed slides return ``()``."""
        return tuple(self._s.level_source_paths(int(level)))

    def level_tile_ranges(self, level: int):
        """Return encoded tile byte ranges for a level as ``uint64[n, 6]``.

        Columns are ``source_index, frame_index, tile_x, tile_y, offset, length``.
        The ranges point into the source Part-10 file and are intended for
        viewer-style range loading; this method does not decode pixels.
        """
        return self._s.level_tile_ranges(int(level))

    def level_tile_range_grid(self, level: int):
        """Return dense row-major encoded tile ranges as ``uint64[n, 6]``.

        The row-major index is ``tile_y * tile_count_x + tile_x``. Sparse-missing
        tiles have ``length == 0`` and ``frame_index == MISSING_FRAME_INDEX``.
        Columns are ``source_index, frame_index, tile_x, tile_y, offset, length``.
        """
        return self._s.level_tile_range_grid(int(level))

    def tile_range(self, level, tile) -> tuple[int, int, int, int, int, int] | None:
        """Return one encoded tile range or ``None`` for an absent/out-of-grid tile."""
        tx, ty = tile
        if int(tx) < 0 or int(ty) < 0:
            raise ValueError("tile coordinates must be non-negative")
        r = self._s.level_tile_range(int(level), int(tx), int(ty))
        return None if r is None else tuple(int(x) for x in r)

    def level_frame_range(
        self, level, frame_number, *, as_index=False
    ) -> tuple[int, int, int, int, int, int]:
        """Return encoded byte range for a stored frame in source frame order."""
        level = int(level)
        frame_count = int(self._s.level_frame_count(level))
        index = _normalize_frame_index(frame_number, frame_count, as_index=bool(as_index))
        r = self._s.level_frame_range(level, index)
        if r is None:
            raise ValueError("frame number out of range")
        return tuple(int(x) for x in r)

    def get_stored_frame(self, frame_number, *, level=0, as_index=False,
                         dtype=None, rgba=False, fill_missing=False,
                         apply_icc_profile=None):
        """stored frame access for one WSI level.

        The frame number follows DICOM 1-based numbering unless ``as_index=True``.
        The returned array is a full stored tile in source frame order.
        """
        level = int(level)
        tile = self.level_frame_tile(level, frame_number, as_index=as_index)
        apply_icc = _validate_icc_request(
            apply_icc_profile, self.level_icc_profile(level), self.icc_transform_available)
        if apply_icc:
            arr = self._s.read_tile_srgb(
                level, int(tile[0]), int(tile[1]), bool(rgba), bool(fill_missing))
        else:
            arr = self.read_tile(level, tile, rgba=rgba, fill_missing=fill_missing)
        if arr.size == 0:
            raise RuntimeError("stored frame is missing, failed to decode, or failed ICC transform")
        return _convert_dtype(arr, dtype)

    def get_stored_frames(self, frame_numbers=None, *, level=0, as_indices=False,
                          dtype=None, rgba=False, fill_missing=False,
                          apply_icc_profile=None):
        """stored frame batch access.

        ``frame_numbers`` are DICOM 1-based by default. Set ``as_indices=True`` for
        Python 0-based indexing. ``None`` reads all stored frames for the level.
        """
        level = int(level)
        frame_count = int(self._s.level_frame_count(level))
        if frame_numbers is None:
            indices = range(frame_count)
        else:
            indices = [
                _normalize_frame_index(fn, frame_count, as_index=bool(as_indices))
                for fn in frame_numbers
            ]
        tiles = []
        for index in indices:
            tile = self._s.level_frame_tile(level, int(index))
            if tile is None:
                raise ValueError("frame number out of range")
            tiles.append(tuple(tile))
        apply_icc = _validate_icc_request(
            apply_icc_profile, self.level_icc_profile(level), self.icc_transform_available)
        if apply_icc:
            if not tiles:
                arr = self.read_tile_stack(level, tiles, rgba=rgba, fill_missing=fill_missing)
            else:
                arr = np.stack([
                    self.get_stored_frame(
                        int(index), level=level, as_index=True, dtype=None, rgba=rgba,
                        fill_missing=fill_missing, apply_icc_profile=True)
                    for index in indices
                ], axis=0)
        else:
            arr = self.read_tile_stack(level, tiles, rgba=rgba, fill_missing=fill_missing)
        return _convert_dtype(arr, dtype)

    def get_frame(self, frame_number, *, level=0, as_index=False, dtype=None,
                  rgba=False, fill_missing=False,
                  apply_real_world_transform=None, apply_modality_transform=None,
                  apply_voi_transform=False, apply_palette_color_lut=None,
                  apply_icc_profile=None, **_kwargs):
        """frame access.

        For WSI this currently aliases stored-frame access. Pixel-transform keyword
        arguments are accepted for source compatibility; unsupported requested
        transforms raise ``NotImplementedError``.
        """
        _reject_unsupported_pixel_transforms(
            apply_real_world_transform=apply_real_world_transform,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            apply_palette_color_lut=apply_palette_color_lut,
        )
        return self.get_stored_frame(
            frame_number, level=level, as_index=as_index, dtype=dtype,
            rgba=rgba, fill_missing=fill_missing,
            apply_icc_profile=apply_icc_profile)

    def get_frames(self, frame_numbers=None, *, level=0, as_indices=False, dtype=None,
                   rgba=False, fill_missing=False,
                   apply_real_world_transform=None, apply_modality_transform=None,
                   apply_voi_transform=False, apply_palette_color_lut=None,
                   apply_icc_profile=None, **_kwargs):
        """batch frame access for WSI stored frames."""
        _reject_unsupported_pixel_transforms(
            apply_real_world_transform=apply_real_world_transform,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            apply_palette_color_lut=apply_palette_color_lut,
        )
        return self.get_stored_frames(
            frame_numbers, level=level, as_indices=as_indices, dtype=dtype,
            rgba=rgba, fill_missing=fill_missing,
            apply_icc_profile=apply_icc_profile)

    def get_total_pixel_matrix(self, *,
                               row_start=None, row_end=None,
                               column_start=None, column_end=None,
                               level=0, as_indices=False, dtype=None, rgba=False,
                               apply_real_world_transform=None,
                               apply_modality_transform=None,
                               apply_voi_transform=False,
                               apply_palette_color_lut=None,
                               apply_icc_profile=None,
                               **_kwargs):
        """total pixel matrix access.

        Row/column positions are DICOM 1-based by default, with ``row_end`` and
        ``column_end`` denoting the first position beyond the returned matrix.
        Set ``as_indices=True`` for Python 0-based intervals.
        """
        _reject_unsupported_pixel_transforms(
            apply_real_world_transform=apply_real_world_transform,
            apply_modality_transform=apply_modality_transform,
            apply_voi_transform=apply_voi_transform,
            apply_palette_color_lut=apply_palette_color_lut,
        )
        level = int(level)
        cols, rows = self._s.level_dimensions(level)
        r0, r1 = _normalize_axis_interval(
            row_start, row_end, int(rows), as_indices=bool(as_indices), axis="row")
        c0, c1 = _normalize_axis_interval(
            column_start, column_end, int(cols), as_indices=bool(as_indices), axis="column")
        downsample = float(self._s.level_downsample(level))
        x0 = int(c0 * downsample + 0.5)
        y0 = int(r0 * downsample + 0.5)
        apply_icc = _validate_icc_request(
            apply_icc_profile, self.level_icc_profile(level), self.icc_transform_available)
        if apply_icc:
            arr = self._s.read_region_srgb(
                x0, y0, level, c1 - c0, r1 - r0, bool(rgba))
            if arr.size == 0 and c1 > c0 and r1 > r0:
                raise RuntimeError("total pixel matrix read failed ICC transform")
        else:
            arr = self.read_region((x0, y0), level, (c1 - c0, r1 - r0), rgba=rgba)
        return _convert_dtype(arr, dtype)

    def tile_exists(self, level, tile) -> bool:
        """Return true when zero-based ``tile`` = (tile_x, tile_y) has stored pixel data."""
        tx, ty = tile
        if int(tx) < 0 or int(ty) < 0:
            raise ValueError("tile coordinates must be non-negative")
        return bool(self._s.tile_exists(int(level), int(tx), int(ty)))

    def get_thumbnail(self, size):
        """A downscaled RGB overview of the whole slide fitting within `size` = (w, h)."""
        lv = self.level_count - 1                      # smallest level
        cols, rows = self._s.level_dimensions(lv)
        full = self._s.read_region(0, 0, lv, cols, rows, False)   # (rows, cols, 3)
        tw, th = size
        sx, sy = max(1, cols // max(1, tw)), max(1, rows // max(1, th))
        return full[::sy, ::sx]                        # nearest-neighbour fit (no PIL dep)


def open_slide(path) -> Slide:
    """Open a slide from a directory of its ``.dcm`` instances, a list of paths, or a
    single multi-level file."""
    if isinstance(path, (list, tuple)):
        paths = [os.fspath(p) for p in path]
    elif os.path.isdir(path):
        groups = group_slide_paths(path)
        if len(groups) > 1:
            raise ValueError("multiple DICOM WSI slides found; use open_slides(path)")
        paths = next(iter(groups.values()), [])
    else:
        paths = [os.fspath(path)]
    return Slide(paths)


def open_slides(path) -> dict[str, Slide]:
    """Open every WSI slide found under a directory or path list.

    Returns ``{slide_key: Slide}``, where ``slide_key`` is usually the
    ``FrameOfReferenceUID``. Non-WSI files and WSI groups without a VOLUME instance are
    skipped.
    """
    return {key: Slide(group) for key, group in group_slide_paths(path).items()}


def group_slide_paths(path) -> dict[str, tuple[str, ...]]:
    """Group DICOM WSI file paths by slide key without opening pixel data."""
    if isinstance(path, (list, tuple)):
        paths = [os.fspath(p) for p in path]
    elif os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.dcm")))
    else:
        paths = [os.fspath(path)]
    return {key: tuple(group) for key, group in _group_wsi_paths(paths).items()}


def _image_type_tokens(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split("\\")
    else:
        parts = [str(v) for v in value]
    return tuple(part.strip().upper() for part in parts if part.strip())


def _group_wsi_paths(paths) -> dict[str, list[str]]:
    from ._dicom import dcmread

    groups: dict[str, list[str]] = {}
    has_volume: dict[str, bool] = {}
    for path in paths:
        try:
            ds = dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "SOPClassUID", "")) != _VL_WSI_SOP_CLASS_UID:
            continue
        key = str(getattr(ds, "FrameOfReferenceUID", "") or
                  getattr(ds, "StudyInstanceUID", "") or
                  getattr(ds, "SeriesInstanceUID", "") or
                  os.fspath(path))
        groups.setdefault(key, []).append(os.fspath(path))
        if "VOLUME" in _image_type_tokens(getattr(ds, "ImageType", None)):
            has_volume[key] = True
    return {key: groups[key] for key in sorted(groups) if has_volume.get(key)}


def write_slide(levels, *, tile=256, mpp=0.25, patient_id="", patient_name="",
                study_uid="", container_id="", specimen_id="",
                transfer_syntax="1.2.840.10008.1.2.4.50", quality=80):
    """Author a DICOM VL Whole Slide Microscopy Image **pyramid** from per-level RGB arrays.

    The write counterpart of :func:`open_slide` — give it the pyramid as a list of ``(H, W, 3)``
    ``uint8`` arrays (**biggest first**; ``levels[0]`` is the base) and get back one Part-10
    instance per level, sharing one Study / Series / Frame of Reference / specimen /
    Multi-Resolution Pyramid (TILED_FULL, IOD-conformant). Each level is tiled here into
    ``tile``×``tile`` frames and handed to the native engine.

    levels: list of ``(H, W, 3)`` uint8 RGB arrays, highest resolution first.
    tile: square tile size (DICOM Rows/Columns per frame).
    mpp: microns per pixel at the base level → Pixel Spacing.
    transfer_syntax: encapsulated TS UID (default lossy JPEG baseline; pass a lossless UID
        like ``1.2.840.10008.1.2.4.80`` for a bit-exact round-trip). quality: JPEG/J2K 1..100.

    Returns a ``list[bytes]`` (one Part-10 instance per level). Read them back with
    :func:`open_slide`. To author from a pyramidal TIFF, read tiles with ``tifffile`` and pass
    the level arrays here.
    """
    bufs, cols, rows, tsz = [], [], [], []
    for arr in levels:
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("write_slide: each level must be an (H, W, 3) uint8 RGB array")
        h, w = int(a.shape[0]), int(a.shape[1])
        nx, ny = (w + tile - 1) // tile, (h + tile - 1) // tile
        padded = np.zeros((ny * tile, nx * tile, 3), np.uint8)
        padded[:h, :w] = a
        # TILED_FULL raster order: (tile_y, tile_x, row, col, channel)
        t = padded.reshape(ny, tile, nx, tile, 3).transpose(0, 2, 1, 3, 4)
        bufs.append(np.ascontiguousarray(t).tobytes())
        cols.append(w); rows.append(h); tsz.append(int(tile))
    return _core.wsi_write_pyramid(bufs, cols, rows, tsz, float(mpp),
                                   patient_id, patient_name, study_uid, container_id,
                                   specimen_id, transfer_syntax, int(quality))


__all__ = [
    "Slide",
    "open_slide",
    "write_slide",
    "open_slides",
    "group_slide_paths",
    "PROPERTY_NAME_VENDOR",
    "PROPERTY_NAME_MPP_X",
    "PROPERTY_NAME_MPP_Y",
    "PROPERTY_NAME_OBJECTIVE_POWER",
    "PROPERTY_NAME_BOUNDS_X",
    "PROPERTY_NAME_BOUNDS_Y",
    "PROPERTY_NAME_BOUNDS_WIDTH",
    "PROPERTY_NAME_BOUNDS_HEIGHT",
]
