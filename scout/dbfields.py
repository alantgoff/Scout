"""Pure helpers behind the startup database's user-owned columns.

Kept out of ui.py so they're unit-testable without executing the Streamlit
script (AppTest can't simulate data_editor edits, so the diffing logic must
be testable directly). No Streamlit, no I/O.
"""

from __future__ import annotations

import math
import re

ATTR_TYPE_LABELS = {
    "select": "Single select", "multiselect": "Multi-select",
    "text": "Text", "number": "Number", "checkbox": "Checkbox",
}

# Column names the grid computes — user columns may not shadow these.
DB_COMPUTED_LABELS = {
    "handle", "Startup", "Score", "Q", "F", "S", "Band", "What they do",
    "Stage", "Sector", "Customers", "Status", "Adjusted", "Memo", "Followers",
    "First seen", "Runs", "Profile", "Notes",
}


def slugify_key(label: str, taken: set[str]) -> str:
    """Stable storage key from a column label, deduped against existing."""
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "field"
    key, n = base, 2
    while key in taken:
        key, n = f"{base}_{n}", n + 1
    return key


def canon(value):
    """Canonicalize a data_editor cell for comparison/storage: NaN, blank
    strings and empty lists → None; arrow arrays → plain lists of str."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if hasattr(value, "tolist"):  # numpy array from an arrow list column
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if str(v).strip()]
        return items or None
    return value


def editor_changes(
    original: list[dict], edited: list[dict], editable: list[str]
) -> list[tuple[str, str, object]]:
    """Diff data_editor rows → [(handle, column label, new value)].

    Rows are compared positionally (num_rows='fixed' guarantees alignment);
    values are canonicalized on both sides so cosmetic differences (NaN vs
    None, "" vs None, tuple vs list) never register as edits."""
    changes: list[tuple[str, str, object]] = []
    for orig, new in zip(original, edited):
        for key in editable:
            before, after = canon(orig.get(key)), canon(new.get(key))
            if before != after:
                changes.append((orig["handle"], key, after))
    return changes


def attr_display(value) -> str:
    """Read-only rendering of an attribute value for the Browse grid."""
    if value in (None, "", []) or value is False:
        return ""
    if value is True:
        return "✓"
    if isinstance(value, list):
        return " · ".join(str(v) for v in value)
    return str(value)
