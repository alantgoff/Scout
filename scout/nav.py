"""The workspace's map: pages, the views inside them, and the aliases old
links still use. Pure data + resolution, importable without Streamlit, so
routing is unit-tested rather than discovered in a browser.

A route is (page, view): `view` is the rail switch inside a page
(Startups → Feed / Database / Graph). Every link into the app — a Slack deep
link `?p=<page>&s=<handle>`, a button that jumps to another page, a test
setting `nav` — goes through `resolve`, so a page that moved keeps answering
to its old name forever (Slack messages are permanent; the alias table is
never pruned).

Six pages, in the order the work happens: find startups, work the ones worth
pursuing, write them up, see what the firm did, then the setup pages a
member visits rarely. It used to be ten — three near-identical list pages,
and setup (Thesis) as the landing page.
"""

from __future__ import annotations

PAGES: list[str] = ["Startups", "Pipeline", "Memos", "Activity", "Thesis", "Settings"]
LANDING = "Startups"

# page → (session key of its view switch, the views in order). The first
# view is the default.
PAGE_VIEWS: dict[str, tuple[str, list[str]]] = {
    "Startups": ("startups_view", ["Feed", "Database", "Graph"]),
    "Pipeline": ("pipeline_stage", ["Longlist", "Shortlist", "In talks",
                                    "Allocated", "Passed"]),
    "Activity": ("activity_view", ["Feed", "Your taste"]),
    "Thesis": ("thesis_view", ["Define", "Tune", "Evidence"]),
    "Settings": ("settings_view", ["General", "Integrations", "Automation",
                                   "Workspace"]),
}

# Old (or shorthand) name → (page, view). The pages that were folded into
# another keep answering here: Slack digests and phone bookmarks carry them.
ALIASES: dict[str, tuple[str, str | None]] = {
    "Feed": ("Startups", "Feed"),
    "Database": ("Startups", "Database"),
    "Graph": ("Startups", "Graph"),
    "Longlist": ("Pipeline", "Longlist"),
    "Shortlist": ("Pipeline", "Shortlist"),
    "Evidence": ("Thesis", "Evidence"),
    "Automation": ("Settings", "Automation"),
    "Notifications": ("Settings", "Integrations"),
}

# Pipeline view → the statuses it lists. "Shortlist" is the old Shortlist
# page's first stage only; the conversation stages have their own view, so
# the shortlist is the decision queue rather than everything past it.
PIPELINE_STAGES: dict[str, list[str]] = {
    "Longlist": ["longlisted"],
    "Shortlist": ["shortlisted"],
    "In talks": ["contacted", "meeting", "diligence"],
    "Allocated": ["won"],
    "Passed": ["passed"],
}

# Which session key holds the selected startup on a page that has a
# cockpit (list + dossier).
SELECTION_KEYS: dict[str, str] = {
    "Startups": "feed_selected",
    "Pipeline": "pipe_selected",
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


def stage_for_status(status: str | None) -> str | None:
    """The Pipeline view a status lives in; None for a startup not in the
    pipeline (new / untriaged), which lives in the Startups feed."""
    for stage, statuses in PIPELINE_STAGES.items():
        if status in statuses:
            return stage
    return None
