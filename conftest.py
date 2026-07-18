"""Make `scout` importable during tests without relying on the editable install.

On macOS, `uv` marks everything under `.venv` with the UF_HIDDEN flag, and
CPython's site.py skips hidden `.pth` files — which silently drops the editable
`scout` package from sys.path after a dependency sync. Anchoring the repo root
here keeps `pytest` working regardless.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
