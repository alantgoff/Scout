"""The workspace's map: pages, the views inside them, and the aliases old
links still use. Pure data + resolution, importable without Streamlit, so
routing is unit-tested rather than discovered in a browser.

A route is (page, view): `view` is the rail switch inside a page
(Startups → Feed / Database). Every link into the app — a Slack deep link
`?p=<page>&s=<handle>`, a button that jumps to another page, a test setting
`nav` — goes through `resolve`, so a page that moved keeps answering to its
old name forever (Slack messages are permanent; the alias table is never
pruned).
"""

from __future__ import annotations

PAGES: list[str] = [
    "Thesis", "Startups", "Longlist", "Shortlist", "Memos", "Activity",
    "Graph", "Evidence", "Automation", "Settings",
]
LANDING = "Thesis"

# page → (session key of its view switch, the views in order). The first
# view is the default.
PAGE_VIEWS: dict[str, tuple[str, list[str]]] = {
    "Startups": ("startups_view", ["Feed", "Database"]),
    "Evidence": ("evidence_view", ["Results", "Signals", "Over time"]),
}

# Old (or shorthand) name → (page, view). Filled as pages consolidate.
ALIASES: dict[str, tuple[str, str | None]] = {
    "Feed": ("Startups", "Feed"),
    "Database": ("Startups", "Database"),
}

# Which session key holds the selected startup on a page that has a
# cockpit (list + dossier).
SELECTION_KEYS: dict[str, str] = {
    "Startups": "feed_selected",
    "Longlist": "ll_selected",
    "Shortlist": "sl_selected",
}


def resolve(name: str | None) -> tuple[str, str | None] | None:
    """A page name, alias or view name → (page, view); None if unknown."""
    if not name:
        return None
    if name in PAGES:
        return name, None
    if name in ALIASES:
        return ALIASES[name]
    lowered = name.strip().lower()
    for page in PAGES:
        if page.lower() == lowered:
            return page, None
    for alias, target in ALIASES.items():
        if alias.lower() == lowered:
            return target
    return None


def view_key(page: str) -> str | None:
    entry = PAGE_VIEWS.get(page)
    return entry[0] if entry else None


def views(page: str) -> list[str]:
    entry = PAGE_VIEWS.get(page)
    return list(entry[1]) if entry else []
