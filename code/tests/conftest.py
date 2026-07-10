"""Pytest configuration.

The numbered analysis directories under ``code/`` are run-by-path entry points,
not an importable package, so their directories are added to ``sys.path`` here
to let tests import the (now import-safe, ``main()``-guarded) data-prep modules
directly. The shared library is imported normally as ``adtopo`` (installed).
"""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
for _sub in ('01_data_prep',):
    p = str(_CODE / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)
