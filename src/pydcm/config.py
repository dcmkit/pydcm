# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.

"""configuration (``pydcm.config``).

pydcm parses leniently and decodes every transfer syntax natively, so most of
These switches are no-ops here — but ported code that reads/sets them
(``enforce_valid_values``, ``settings.reading_validation_mode``, …)
must not crash. These are accepted and, where they have a meaning for us, honoured.
"""

from __future__ import annotations

# Validation: pydcm is intentionally lenient (it reads leniently on malformed
# files), so these default off. Settable for source compatibility.
enforce_valid_values = False
datetime_conversion = False
use_DS_decimal = False
use_DS_numpy = False                     # pydcm returns DSfloat, never a numpy DS array
use_IS_numpy = False
allow_DS_float = True
replace_un_with_known_vr = True          # pydcm always resolves UN via its superset dict
convert_wrong_length_to_UN = True
assume_implicit_vr_switch = True
have_numpy = True                        # pydcm always links numpy (native decode → ndarray)
APPLY_J2K_CORRECTIONS = True
show_file_meta = True
pixel_data_handlers: list = []           # pydcm decodes natively — no pluggable handlers
data_element_callback = None
data_element_callback_kwargs: dict = {}

IGNORE = 0
WARN = 1
RAISE = 2


def DS_numpy(use_numpy: bool = True) -> None:
    """Accepted for source compatibility; pydcm always returns DSfloat (no numpy DS)."""
    global use_DS_numpy
    use_DS_numpy = bool(use_numpy)


def DS_decimal(use_Decimal_boolean: bool = True) -> None:
    """Accepted for source compatibility; pydcm always returns DSfloat."""
    global use_DS_decimal
    use_DS_decimal = bool(use_Decimal_boolean)


def strict_reading():
    """No-op context manager (pydcm reads leniently)."""
    return disable_value_validation()


class _Settings:
    """A settings object — accepted, mostly advisory here."""
    reading_validation_mode = WARN
    writing_validation_mode = WARN
    infer_sq_for_un_vr = True


settings = _Settings()
Settings = _Settings              # exposes the class as well as the instance

# validation / future-behavior flags (accepted; advisory under pydcm's lenient
# native read, but settable + queryable so ported code behaves identically).
use_none_as_empty_text_VR_value = False
INVALID_KEYWORD_BEHAVIOR = "WARN"   # "WARN" | "RAISE" | "IGNORE"
INVALID_KEY_BEHAVIOR = "WARN"       # "WARN" | "RAISE" | "IGNORE"
debugging = False
_use_future = False


def future_behavior(enable_future: bool = True) -> None:
    """Imitate the next major-version behavior (deprecations -> errors)."""
    global _use_future, INVALID_KEYWORD_BEHAVIOR
    _use_future = bool(enable_future)
    INVALID_KEYWORD_BEHAVIOR = "RAISE" if enable_future else "WARN"
    settings.writing_validation_mode = RAISE if enable_future else WARN


def reset_data_element_callback() -> None:
    """Reset the data-element read callback to the default."""
    global data_element_callback, data_element_callback_kwargs
    data_element_callback = None
    data_element_callback_kwargs = {}


def debug(debug_on: bool = True, default_handler: bool = True) -> None:
    """Accepted for source compatibility; pydcm has no global debug logger to toggle."""
    return None


def disable_value_validation():
    """Context-manager-free no-op (pydcm doesn't validate on read)."""
    class _Noop:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return _Noop()
