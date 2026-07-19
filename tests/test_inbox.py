"""Remote inbox tests: decision parsing (malformed tolerated), idempotent
application to the pipeline table, delete semantics. All offline."""

from __future__ import annotations

import json
from pathlib import Path

from scout.inbox import Decision, apply_decision, apply_inbox, load_decisions
from scout.store import Store, pipeline_tags


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "t.db")


def write_decision(inbox: Path, name: str, **payload) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- loading ------------------------------------------------------------------


def test_load_decisions_sorted_and_malformed_split(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    write_decision(inbox, "200-pass-b.json", handle="b", action="pass")
    write_decision(inbox, "100-shortlist-a.json", handle="@A", action="shortlist")
    (inbox / "150-broken.json").write_text("{not json", encoding="utf-8")
    write_decision(inbox, "300-bad-action.json", handle="c", action="explode")
    (inbox / "notes.txt").write_text("ignored", encoding="utf-8")  # non-json ignored

    parsed, malformed = load_decisions(inbox)
    assert [(d.handle, d.action) for _, d in parsed] == [("A", "shortlist"), ("b", "pass")]
    assert sorted(p.name for p in malformed) == ["150-broken.json", "300-bad-action.json"]


def test_load_decisions_missing_dir(tmp_path: Path) -> None:
    assert load_decisions(tmp_path / "nope") == ([], [])


# --- applying -----------------------------------------------------------------


def test_apply_status_tag_note_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    apply_decision(store, Decision(handle="ada", action="shortlist"))
    apply_decision(store, Decision(handle="ada", action="tag", tag="intro-ready"))
    apply_decision(store, Decision(handle="ada", action="note", note="met at demo day"))

    row = store.get_pipeline("ada")
    assert row["status"] == "shortlisted"
    assert pipeline_tags(row) == ["intro-ready"]
    assert row["notes"] == "met at demo day"

    apply_decision(store, Decision(handle="ada", action="untag", tag="Intro-Ready"))
    assert pipeline_tags(store.get_pipeline("ada")) == []
    apply_decision(store, Decision(handle="ada", action="pass"))
    assert store.get_pipeline("ada")["status"] == "passed"
    apply_decision(store, Decision(handle="ada", action="restore"))
    assert store.get_pipeline("ada")["status"] == "new"


def test_apply_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for _ in range(2):  # re-running after a partial failure must be safe
        apply_decision(store, Decision(handle="ada", action="shortlist"))
        apply_decision(store, Decision(handle="ada", action="tag", tag="healthcare"))
        apply_decision(store, Decision(handle="ada", action="note", note="ping intro"))
    row = store.get_pipeline("ada")
    assert row["status"] == "shortlisted"
    assert pipeline_tags(row) == ["healthcare"]  # not duplicated
    assert row["notes"].count("ping intro") == 1  # not appended twice


def test_apply_inbox_deletes_applied_keeps_malformed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    inbox = tmp_path / "inbox"
    write_decision(inbox, "1-shortlist-a.json", handle="a", action="shortlist")
    write_decision(inbox, "2-tag-a.json", handle="a", action="tag", tag="x-ray")
    bad = inbox / "3-broken.json"
    bad.write_text("nope", encoding="utf-8")

    lines, malformed = apply_inbox(store, inbox, delete=True)
    assert len(lines) == 2
    assert store.get_pipeline("a")["status"] == "shortlisted"
    assert not (inbox / "1-shortlist-a.json").exists()
    assert not (inbox / "2-tag-a.json").exists()
    assert bad.exists()  # left for inspection
    assert malformed == [bad]


def test_apply_inbox_without_delete_leaves_files(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    inbox = tmp_path / "inbox"
    path = write_decision(inbox, "1-pass-b.json", handle="b", action="pass")
    lines, _ = apply_inbox(store, inbox, delete=False)
    assert lines and path.exists()
