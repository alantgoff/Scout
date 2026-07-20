"""Pure grid helpers behind the startup database (scout/dbfields.py)."""

from __future__ import annotations

import math

from scout.dbfields import attr_display, canon, editor_changes, slugify_key


def test_slugify_key_slugs_and_dedupes() -> None:
    assert slugify_key("Warm intro!", set()) == "warm_intro"
    assert slugify_key("Warm intro", {"warm_intro"}) == "warm_intro_2"
    assert slugify_key("Warm intro", {"warm_intro", "warm_intro_2"}) == "warm_intro_3"
    assert slugify_key("///", set()) == "field"


def test_canon_normalizes_editor_noise() -> None:
    assert canon(float("nan")) is None
    assert canon("  ") is None
    assert canon(" x ") == "x"
    assert canon([]) is None
    assert canon(("A", " ", "B")) == ["A", "B"]
    assert canon(False) is False
    assert canon(3.0) == 3.0

    class FakeArray:  # numpy-ish arrow list cell
        def tolist(self):
            return ["A", "B"]

    assert canon(FakeArray()) == ["A", "B"]


def test_editor_changes_diffs_only_real_edits() -> None:
    original = [
        {"handle": "ada", "Status": "New", "Notes": "", "Vertical": None,
         "Use case": [], "Warm intro": False},
        {"handle": "bo", "Status": "Longlisted", "Notes": "hi",
         "Vertical": "Fintech", "Use case": ["Marketplace"], "Warm intro": True},
    ]
    edited = [
        {"handle": "ada", "Status": "Shortlisted", "Notes": "  ",
         "Vertical": "Security", "Use case": [], "Warm intro": False},
        {"handle": "bo", "Status": "Longlisted", "Notes": "hi",
         "Vertical": "Fintech", "Use case": ("Marketplace",), "Warm intro": True},
    ]
    editable = ["Status", "Notes", "Vertical", "Use case", "Warm intro"]
    changes = editor_changes(original, edited, editable)
    # Only ada's status + vertical actually changed; "" vs None notes and
    # tuple-vs-list multiselects are cosmetic, not edits.
    assert changes == [("ada", "Status", "Shortlisted"),
                       ("ada", "Vertical", "Security")]


def test_attr_display_renders_types() -> None:
    assert attr_display(None) == "" and attr_display([]) == ""
    assert attr_display(True) == "✓" and attr_display(False) == ""
    assert attr_display(["A", "B"]) == "A · B"
    assert attr_display(3.5) == "3.5"
