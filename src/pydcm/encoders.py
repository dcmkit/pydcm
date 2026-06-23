# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""pixel encoders (``pydcm.encoders``).

pydcm encodes via the native transcoder (see :meth:`pydcm.Dataset.compress`); this exposes
a registry of the supported encapsulated-lossless targets."""
from __future__ import annotations

from . import _native
from .uid import (RLELossless, JPEG2000Lossless, JPEGLSLossless, HTJ2KLossless, UID)

JPEGXLLossless = UID("1.2.840.10008.1.2.4.110")
_SUPPORTED = {str(x) for x in (RLELossless, JPEG2000Lossless, JPEGLSLossless,
                               HTJ2KLossless, JPEGXLLossless)}


class Encoder:
    """An encoder for one Transfer Syntax (thin over Dataset.compress)."""
    def __init__(self, uid):
        self.UID = UID(str(uid))

    def is_available(self) -> bool:
        return str(self.UID) in _SUPPORTED

    def encode(self, ds, **kw) -> bytes:
        out = _native.transcode(ds._encode_part10(), str(self.UID))
        return out                      # full Part-10; for the encapsulated PixelData use compress

    def __repr__(self):
        return f"<Encoder {self.UID} ({'available' if self.is_available() else 'unavailable'})>"


def get_encoder(uid) -> Encoder:
    return Encoder(uid)


RLELosslessEncoder = Encoder(RLELossless)
JPEG2000LosslessEncoder = Encoder(JPEG2000Lossless)
JPEGLSLosslessEncoder = Encoder(JPEGLSLossless)
HTJ2KLosslessEncoder = Encoder(HTJ2KLossless)

__all__ = ["Encoder", "get_encoder", "RLELosslessEncoder", "JPEG2000LosslessEncoder",
           "JPEGLSLosslessEncoder", "HTJ2KLosslessEncoder"]
