# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""Parametric Maps (`pydcm.pm`) — class API over the native
``write_paramap`` writer. Re-exports the functional ``write_paramap`` / ``read_paramap``."""

from __future__ import annotations

import os as _os
import tempfile as _tempfile

import numpy as _np

from .paramap import write_paramap, read_paramap
from .seg import _code_tuple, _source_paths, _SEG_TMPS   # shared hd-shim helpers


class RealWorldValueMapping:
    """Real-world value mapping."""
    def __init__(self, lut_label, lut_explanation, unit, value_range,
                 slope=None, intercept=None, lut_data=None, quantity_definition=None):
        self.lut_label = lut_label
        self.lut_explanation = lut_explanation
        self.unit = unit
        self.value_range = value_range
        self.slope = slope
        self.intercept = intercept
        self.lut_data = lut_data
        self.quantity_definition = quantity_definition


def _first_rwvm(rwvm):
    """Accept a flat sequence or a per-frame sequence-of-sequences; return the first."""
    seq = list(rwvm)
    if seq and isinstance(seq[0], (list, tuple)):
        seq = list(seq[0])
    return seq[0] if seq else None


def ParametricMap(source_images, pixel_array, series_instance_uid, series_number,
                  sop_instance_uid, instance_number, manufacturer, manufacturer_model_name,
                  software_versions, device_serial_number, contains_recognizable_visual_features,
                  real_world_value_mappings, window_center, window_width, *,
                  content_description=None, content_label=None, content_creator_name=None,
                  transfer_syntax_uid=None, **_kwargs):
    """Constructor — returns a pydcm Dataset.

    Built over the native ``write_paramap``: ``source_images`` supply geometry/demographics,
    ``pixel_array`` the real-valued (or stored) planes, and the first
    ``real_world_value_mappings`` entry the units / quantity / slope / intercept.
    extra kwargs are accepted for source compatibility.
    """
    refs = _source_paths(source_images)
    m = _first_rwvm(real_world_value_mappings)
    units = _code_tuple(m.unit) if m is not None and m.unit is not None else None
    quantity = (_code_tuple(m.quantity_definition)
                if m is not None and m.quantity_definition is not None else None)
    slope = getattr(m, "slope", None)
    intercept = getattr(m, "intercept", None)
    label = getattr(m, "lut_label", None)
    explanation = getattr(m, "lut_explanation", None)
    dtype = None if pixel_array.dtype.kind == "f" else str(pixel_array.dtype)

    fd, tmp = _tempfile.mkstemp(suffix=".dcm"); _os.close(fd)
    _SEG_TMPS.append(tmp)
    write_paramap(refs, pixel_array, units=units, quantity=quantity, slope=slope,
                  intercept=intercept, label=label, explanation=explanation,
                  dtype=dtype, output=tmp)
    from . import dcmread
    ds = dcmread(tmp)
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = int(series_number)
    ds.SOPInstanceUID = sop_instance_uid
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.InstanceNumber = int(instance_number)
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = manufacturer_model_name
    ds.SoftwareVersions = (software_versions if isinstance(software_versions, str)
                           else list(software_versions))
    ds.DeviceSerialNumber = device_serial_number
    ds.WindowCenter = float(window_center)
    ds.WindowWidth = float(window_width)
    if content_description is not None:
        ds.ContentDescription = content_description
    if content_label is not None:
        ds.ContentLabel = content_label
    if content_creator_name is not None:
        ds.ContentCreatorName = content_creator_name
    return ds


__all__ = ["ParametricMap", "RealWorldValueMapping", "write_paramap", "read_paramap"]
