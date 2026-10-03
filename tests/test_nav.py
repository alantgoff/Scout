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
