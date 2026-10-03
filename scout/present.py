"""Small presentation helpers shared by the UI and anything that renders
for people — pure, importable without Streamlit, so they unit-test.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Within this many seconds either side of now, a time is just "just now".
_NOW_WINDOW_S = 90


def relative_time(ts: str | datetime | None, now: datetime | None = None) -> str:
    """Compact relative time, both directions: "5m ago" / "in 10h".

    Accepts an ISO string (store rows) or a datetime (pydantic models).
    The future matters as much as the past: schedules, leases and retries
    all carry times that have not happened yet, and rendering "tomorrow at
    06:00" as "just now" told the reader the opposite of the truth.
    """
    if not ts:
        return ""
    if isinstance(ts, datetime):
        then = ts
    else:
        try:
            then = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    seconds = (now - then).total_seconds()
    if abs(seconds) < _NOW_WINDOW_S:
        return "just now"
    span = abs(seconds)
    if span < 3600:
        amount = f"{int(span // 60)}m"
    elif span < 86400 * 2:
        amount = f"{int(span // 3600)}h"
    else:
        amount = f"{int(span // 86400)}d"
    return f"{amount} ago" if seconds > 0 else f"in {amount}"


# How a scan kind reads in a sentence ("Run in progress", "Preview finished").
_RUN_LABELS = {
    "run": "Run",
    "source": "Preview",
    "source preview": "Preview",
    "preview": "Preview",
    "reclassify": "Rescore",
    "verify": "Verification",
    "refresh": "Refresh",
    "resolve": "Resolve",
}


def run_label(kind: str | None) -> str:
    """The noun for a scan kind. Never "Run running": the verb is the
    banner's job ("in progress", "finished", "failed")."""
    return _RUN_LABELS.get((kind or "").lower(), (kind or "Run").capitalize())
