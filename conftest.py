"""Make `scout` importable during tests without relying on the editable install.

On macOS, `uv` marks everything under `.venv` with the UF_HIDDEN flag, and
CPython's site.py skips hidden `.pth` files — which silently drops the editable
`scout` package from sys.path after a dependency sync. Anchoring the repo root
here keeps `pytest` working regardless.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# UI tests run with the shared-workspace guard on: a mutation of the cached
# (shared, read-only) workspace objects fails the test that caused it rather
# than leaking into every other session (scout/ui.py, _load_workspace).
os.environ.setdefault("SCOUT_CACHE_GUARD", "1")

# AppTest builds a fresh ScriptCache for every run, so ui.py is re-rewritten
# and recompiled on each `at.run()` (~260 ms). A real server keeps one cache
# per process; sharing one here does the same and takes ~a third off the
# suite. Test-only, and it patches a Streamlit internal — if that internal
# moves, the patch is skipped and the suite is merely slower again.
try:
    from streamlit.runtime.scriptrunner.script_cache import ScriptCache
    from streamlit.testing.v1 import local_script_runner as _lsr

    if hasattr(_lsr, "ScriptCache"):
        _SHARED_SCRIPT_CACHE = ScriptCache()
        _lsr.ScriptCache = lambda: _SHARED_SCRIPT_CACHE
except ImportError:  # pragma: no cover — streamlit absent or reorganised
    pass
