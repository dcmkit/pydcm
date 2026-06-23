# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fuli Wu
# NOT a medical device — not for clinical or diagnostic use.
"""environment diagnostics (``pydcm.env_info``)."""
from __future__ import annotations

import platform
import sys


def main() -> None:
    import numpy
    from . import __version__
    print(f"pydcm:   {__version__}")
    print(f"python:  {sys.version.split()[0]} ({platform.platform()})")
    print(f"numpy:   {numpy.__version__}")


if __name__ == "__main__":
    main()
