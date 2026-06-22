# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""`from pydcm.tag import Tag, BaseTag`."""
import contextlib

from . import _native
from ._dicom import Tag, BaseTag

TupleTag = Tag                      # alias
ItemTag = Tag(0xFFFEE000)
ItemDelimiterTag = Tag(0xFFFEE00D)
SequenceDelimiterTag = Tag(0xFFFEE0DD)


def tag_for_keyword(keyword):
    return _native.tag_for_keyword(keyword)


@contextlib.contextmanager
def tag_in_exception(tag):
    """Annotate any exception raised in the block with ``tag``."""
    try:
        yield
    except Exception as exc:                       # pragma: no cover - passthrough
        msg = exc.args[0] if exc.args else ""
        exc.args = (f"With tag {Tag(tag)} got exception: {msg}", *exc.args[1:])
        raise


__all__ = ["Tag", "BaseTag", "TupleTag", "tag_for_keyword", "tag_in_exception",
           "ItemTag", "ItemDelimiterTag", "SequenceDelimiterTag"]
