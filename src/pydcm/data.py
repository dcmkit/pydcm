# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""test-data loaders (``pydcm.data``).

pydcm ships no bundled corpus; this re-exports pydicom's data helpers when it is
installed (so ``pydcm.data.get_testdata_file('CT_small.dcm')`` works in ported tests).
"""
from __future__ import annotations


def _backend():
    try:
        import pydicom.data as _d
        return _d
    except ImportError as e:                      # pragma: no cover
        raise ModuleNotFoundError(
            "pydcm.data re-exports pydicom's bundled test files; install pydicom to use it"
        ) from e


def get_testdata_file(name, **kw):
    return _backend().get_testdata_file(name, **kw)


def get_testdata_files(pattern="*", **kw):
    return _backend().get_testdata_files(pattern, **kw)


def get_charset_files(pattern="*"):
    return _backend().get_charset_files(pattern)


def get_palette_files(pattern="*"):
    return _backend().get_palette_files(pattern)


__all__ = ["get_testdata_file", "get_testdata_files", "get_charset_files",
           "get_palette_files"]
