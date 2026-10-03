"""Routing map (scout/nav.py): page names, views and aliases resolve the
same way for every way into the app — links, buttons, tests."""

from __future__ import annotations

from scout import nav


def test_every_page_resolves_to_itself():
    for page in nav.PAGES:
        assert nav.resolve(page) == (page, None)
        assert nav.resolve(page.lower()) == (page, None)


def test_aliases_and_views_resolve():
    assert nav.resolve("Feed") == ("Startups", "Feed")
    assert nav.resolve("database") == ("Startups", "Database")
    assert nav.resolve("") is None and nav.resolve("Nope") is None
    for alias, (page, view) in nav.ALIASES.items():
        assert page in nav.PAGES
        if view is not None:
            assert view in nav.views(page), alias


def test_view_keys():
    assert nav.view_key("Startups") == "startups_view"
    assert nav.views("Startups")[0] == "Feed"
    assert nav.view_key("Memos") is None and nav.views("Memos") == []


def test_six_pages_land_on_the_work():
    assert nav.PAGES == ["Startups", "Pipeline", "Memos", "Activity", "Thesis", "Settings"]
    assert nav.LANDING == "Startups"


def test_folded_pages_keep_answering_to_their_old_names():
    """Slack digests and bookmarks carry the ten-page names forever."""
    assert nav.resolve("Longlist") == ("Pipeline", "Longlist")
    assert nav.resolve("shortlist") == ("Pipeline", "Shortlist")
    assert nav.resolve("Graph") == ("Startups", "Graph")
    assert nav.resolve("Evidence") == ("Thesis", "Evidence")
    assert nav.resolve("Automation") == ("Settings", "Automation")


def test_every_status_has_one_pipeline_stage():
    from scout.status import FUNNEL_STAGES, STATUS_LABELS

    seen = [s for statuses in nav.PIPELINE_STAGES.values() for s in statuses]
    assert len(seen) == len(set(seen))  # no status in two stages
    assert set(seen) == set(FUNNEL_STAGES) | {"passed"}
    assert set(seen) | {"new"} == set(STATUS_LABELS)
    assert list(nav.PIPELINE_STAGES) == nav.views("Pipeline")
    assert nav.stage_for_status("meeting") == "In talks"
    assert nav.stage_for_status("won") == "Allocated"
    assert nav.stage_for_status("new") is None
    assert nav.stage_for_status(None) is None


def test_every_page_with_a_list_has_a_selection_key():
    assert set(nav.SELECTION_KEYS) <= set(nav.PAGES)
    assert len(set(nav.SELECTION_KEYS.values())) == len(nav.SELECTION_KEYS)
