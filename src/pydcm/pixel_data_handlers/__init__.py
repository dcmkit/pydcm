# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""Back-compat alias for the older ``pixel_data_handlers`` module name — re-exports ``pixels``."""
from ..pixels import (apply_modality_lut, apply_voi_lut, apply_windowing,
                      convert_color_space, pixel_array)
from . import util
__all__ = ["apply_modality_lut", "apply_voi_lut", "apply_windowing",
           "convert_color_space", "pixel_array", "util"]
