"""Remote decision inbox — how the phone digest writes back to the store.

The GitHub Pages digest can't touch scout.db directly, so triage actions on
the phone become tiny JSON files committed to the PRIVATE code repo under
`remote/inbox/` (one file per decision — unique filenames mean no write
races and no lost updates). The next headless run (or the fast
`scout-apply` workflow) calls `scout inbox --apply --delete`, which folds
them into the pipeline table and removes the applied files.

Decision file shape (written by docs/index.html's JS):

    {"handle": "some_handle", "action": "shortlist", "tag": "", "note": "",
     "at": "2026-07-19T12:00:00Z"}

Actions: shortlist | pass | restore | tag | untag | note. Unknown actions
and malformed files are skipped and reported, never fatal — a bad file must
not wedge the pipeline. Applying is idempotent (statuses are absolute, tags
dedupe), so re-running after a partial failure is safe.

Pure parsing/apply logic — unit-tested offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from scout.store import Store, pipeline_tags

DEFAULT_INBOX_DIR = Path("remote/inbox")

ACTIONS = ("shortlist", "pass", "restore", "tag", "untag", "note")

# Absolute status per action — applying twice lands in the same state.
_STATUS_FOR = {"shortlist": "shortlisted", "pass": "passed", "restore": "new"}


class Decision(BaseModel):
    """One phone-side triage decision."""

    handle: str
    action: str
    tag: str = ""
    note: str = ""
    at: str = ""  # ISO timestamp stamped by the page (informational)


def load_decisions(
    inbox_dir: Path = DEFAULT_INBOX_DIR,
) -> tuple[list[tuple[Path, Decision]], list[Path]]:
    """Read every *.json decision in the inbox, oldest first (filenames are
    timestamp-prefixed by the page). Returns (parsed, malformed)."""
    if not inbox_dir.is_dir():
        return [], []
    parsed: list[tuple[Path, Decision]] = []
    malformed: list[Path] = []
    for path in sorted(inbox_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            decision = Decision.model_validate(data)
            decision.handle = decision.handle.lstrip("@").strip()
            if not decision.handle or decision.action not in ACTIONS:
                raise ValueError(f"bad handle/action: {data!r}")
            parsed.append((path, decision))
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            malformed.append(path)
    return parsed, malformed


def apply_decision(store: Store, decision: Decision) -> str:
    """Apply one decision to the pipeline table; returns a summary line."""
    handle = decision.handle
    if decision.action in _STATUS_FOR:
        store.set_pipeline(handle, status=_STATUS_FOR[decision.action])
        return f"@{handle}: status -> {_STATUS_FOR[decision.action]}"
    if decision.action == "tag":
        store.tag_lead(handle, decision.tag)
        return f"@{handle}: +tag '{decision.tag.strip()}'"
    if decision.action == "untag":
        store.untag_lead(handle, decision.tag)
        return f"@{handle}: -tag '{decision.tag.strip()}'"
    if decision.action == "note":
        existing = (store.get_pipeline(handle).get("notes") or "").strip()
        note = decision.note.strip()
        if note and note not in existing:  # idempotent on re-apply
            store.set_pipeline(
                handle, notes=(existing + "\n" + note).strip() if existing else note
            )
        return f"@{handle}: note appended"
    return f"@{handle}: unknown action '{decision.action}' skipped"


def apply_inbox(
    store: Store, inbox_dir: Path = DEFAULT_INBOX_DIR, delete: bool = False
) -> tuple[list[str], list[Path]]:
    """Apply every pending decision; optionally delete applied files.
    Returns (summary lines, malformed paths left in place for inspection)."""
    parsed, malformed = load_decisions(inbox_dir)
    lines: list[str] = []
    for path, decision in parsed:
        lines.append(apply_decision(store, decision))
        if delete:
            path.unlink(missing_ok=True)
    return lines, malformed


__all__ = [
    "ACTIONS",
    "DEFAULT_INBOX_DIR",
    "Decision",
    "apply_decision",
    "apply_inbox",
    "load_decisions",
    "pipeline_tags",
]
