"""Collaboration logic: stance tallies, disagreement, mentions (pure), plus
the store's events spine, votes, comments and memo versioning.

The pure half needs no database (the dbfields.py convention); the store half
uses a tmp DB and no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scout.collab import (
    STANCES,
    contested_sort_key,
    disagreements,
    parse_mentions,
    vote_summary,
)
from scout.models import Vote
from scout.store import Store


def make_store(tmp_path: Path, actor: str = "alan@firm.com") -> Store:
    return Store(tmp_path / "collab.db", actor=actor)


def vote(handle: str, actor: str, stance: str, **kw) -> Vote:
    return Vote(handle=handle, actor=actor, stance=stance, **kw)


# --- stance tallies -------------------------------------------------------------


def test_vote_summary_none_without_votes() -> None:
    assert vote_summary([]) is None


def test_vote_summary_tallies_and_maps_actors() -> None:
    summary = vote_summary([
        vote("acme", "alan@firm.com", "strong_yes"),
        vote("acme", "sara@firm.com", "yes"),
    ])
    assert summary is not None
    assert summary.n_votes == 2
    assert summary.mean == 1.5  # (2 + 1) / 2
    assert summary.spread == 1.0
    assert summary.contested is False  # agreement, just different intensity
    assert summary.by_actor == {"alan@firm.com": "strong_yes",
                                "sara@firm.com": "yes"}


def test_contested_needs_two_voters_spanning_a_real_gap() -> None:
    # Yes vs pass = spread 3 → a genuine split.
    split = vote_summary([
        vote("acme", "alan@firm.com", "yes"),
        vote("acme", "sara@firm.com", "pass"),
    ])
    assert split is not None and split.contested is True
    assert split.spread == 3.0
    # Strong yes vs unsure = spread 2 → a hedge, not a fight.
    hedge = vote_summary([
        vote("acme", "alan@firm.com", "strong_yes"),
        vote("acme", "sara@firm.com", "unsure"),
    ])
    assert hedge is not None and hedge.contested is False
    # A lone dissenter cannot be "contested" — there's nobody to contest.
    solo = vote_summary([vote("acme", "alan@firm.com", "pass")])
    assert solo is not None and solo.contested is False and solo.spread == 0.0


def test_pass_weighs_like_a_strong_yes() -> None:
    """The asymmetry is deliberate: one conviction holder against one
    dissenter must read as a split, not a mild positive average."""
    assert STANCES["pass"] == -STANCES["strong_yes"]
    summary = vote_summary([
        vote("acme", "alan@firm.com", "strong_yes"),
        vote("acme", "sara@firm.com", "pass"),
    ])
    assert summary is not None
    assert summary.mean == 0.0 and summary.contested is True


def test_contested_sort_key_orders_widest_split_first() -> None:
    wide = vote_summary([vote("a", "x@f.com", "strong_yes"),
                         vote("a", "y@f.com", "pass")])
    narrow = vote_summary([vote("b", "x@f.com", "yes"),
                           vote("b", "y@f.com", "unsure")])
    ordered = sorted([narrow, wide], key=contested_sort_key)
    assert [s.handle for s in ordered] == ["a", "b"]
    # No votes sinks below everything that has them.
    assert contested_sort_key(None) > contested_sort_key(narrow)


def test_disagreements_reads_as_a_meeting_agenda() -> None:
    summaries = {
        "acme": vote_summary([vote("acme", "alan@firm.com", "strong_yes"),
                              vote("acme", "sara@firm.com", "pass")]),
        "calm": vote_summary([vote("calm", "alan@firm.com", "yes"),
                              vote("calm", "sara@firm.com", "yes")]),
    }
    out = disagreements(summaries, {"alan@firm.com": "Alan", "sara@firm.com": "Sara"})
    assert len(out) == 1  # only the contested one
    handle, sentence = out[0]
    assert handle == "acme"
    # Positive stance leads, both sides named.
    assert sentence == "Alan: Strong yes · Sara: Pass"


# --- mentions -------------------------------------------------------------------


def test_parse_mentions_matches_aliases_and_full_emails() -> None:
    users = [
        {"id": "sara.lin@firm.com", "name": "Sara Lin"},
        {"id": "alan@firm.com", "name": "Alan Goff"},
    ]
    assert parse_mentions("ping @sara.lin on this", users) == ["sara.lin@firm.com"]
    assert parse_mentions("@sara.lin@firm.com take a look", users) == ["sara.lin@firm.com"]
    assert parse_mentions("@alan and @sara.lin", users) == [
        "alan@firm.com", "sara.lin@firm.com"
    ]
    # Trailing punctuation, dedupe, and unknown handles.
    assert parse_mentions("@alan, @alan again", users) == ["alan@firm.com"]
    assert parse_mentions("@nobody here", users) == []
    assert parse_mentions("", users) == []


# --- votes (store) --------------------------------------------------------------


def test_set_vote_upserts_and_keeps_created_at(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_vote("@Acme", "yes", "clean wedge", thesis_id="novel-arch")
    first = store.votes_for("acme")[0]
    store.set_vote("acme", "strong_yes", "met the team")
    votes = store.votes_for("acme")
    assert len(votes) == 1  # one row per (handle, actor)
    assert votes[0].stance == "strong_yes"
    assert votes[0].rationale == "met the team"
    assert votes[0].created_at == first.created_at  # first vote's timestamp kept
    # Both stances are in the event history — the row is state, events are history.
    stances = [e.payload["stance"] for e in store.events(verbs=["vote_cast"])]
    assert stances == ["strong_yes", "yes"]  # newest first


def test_votes_from_two_partners_and_clearing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_vote("acme", "yes", actor="alan@firm.com")
    store.set_vote("acme", "pass", "crowded space", actor="sara@firm.com")
    summary = vote_summary(store.votes_for("acme"))
    assert summary is not None and summary.contested is True
    assert store.all_votes()["acme"][0].handle == "acme"
    store.set_vote("acme", "", actor="sara@firm.com")
    assert [v.actor for v in store.votes_for("acme")] == ["alan@firm.com"]


def test_vote_rejects_unknown_stance_and_missing_author(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.set_vote("acme", "maybe_later")
    anon = Store(tmp_path / "anon.db")  # no bound actor
    with pytest.raises(ValueError):
        anon.set_vote("acme", "yes")


# --- events spine ---------------------------------------------------------------


def test_status_change_emits_event_with_transition(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("acme", status="longlisted")
    store.set_pipeline("acme", status="shortlisted")
    events = store.events(verbs=["status_changed"])
    assert [(e.payload["old"], e.payload["new"]) for e in events] == [
        ("longlisted", "shortlisted"), ("new", "longlisted"),
    ]
    assert all(e.actor == "alan@firm.com" for e in events)
    # A no-op write emits nothing.
    store.set_pipeline("acme", status="shortlisted")
    assert len(store.events(verbs=["status_changed"])) == 2


def test_event_exists_only_if_the_state_write_committed(tmp_path: Path) -> None:
    """The spine's core invariant: events are appended inside the same
    transaction as the change they describe."""
    store = make_store(tmp_path)
    store.set_pipeline("acme", status="longlisted")
    before = len(store.events())
    with pytest.raises(RuntimeError):
        with store.write_tx():
            store.db["pipeline"].upsert(
                {"handle": "acme", "status": "won"}, pk="handle", alter=True)
            store._append_event("status_changed", handle="acme",
                                payload={"old": "longlisted", "new": "won"})
            raise RuntimeError("boom mid-transaction")
    # Neither the state change nor its event survived.
    assert store.get_pipeline("acme")["status"] == "longlisted"
    assert len(store.events()) == before


def test_events_filter_by_handle_and_actor(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_vote("acme", "yes", actor="alan@firm.com")
    store.set_vote("beta", "pass", actor="sara@firm.com")
    assert [e.handle for e in store.events(handle="acme")] == ["acme"]
    assert [e.actor for e in store.events(actor="sara@firm.com")] == ["sara@firm.com"]
    assert len(store.events(limit=1)) == 1


def test_unread_counts_exclude_your_own_activity(tmp_path: Path) -> None:
    store = make_store(tmp_path, actor="alan@firm.com")
    store.set_vote("acme", "yes")  # Alan's own action
    assert store.unread_count("alan@firm.com") == 0
    assert store.unread_count("sara@firm.com") == 1
    store.mark_read("sara@firm.com")
    assert store.unread_count("sara@firm.com") == 0
    store.set_vote("beta", "pass")
    assert store.unread_count("sara@firm.com") == 1


def test_notification_sweep_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("acme", status="longlisted")
    pending = store.unnotified_events()
    assert pending and pending[0].verb == "status_changed"
    store.mark_notified([e.id for e in pending])
    assert store.unnotified_events() == []


# --- comments -------------------------------------------------------------------


def test_comments_thread_with_mentions_and_soft_delete(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.add_comment("@Acme", "worth a call", mentions=["sara@firm.com"])
    store.add_comment("acme", "agreed", actor="sara@firm.com")
    thread = store.comments_for("acme")
    assert [c.body for c in thread] == ["worth a call", "agreed"]
    assert thread[0].mentions == ["sara@firm.com"]
    assert store.all_comment_counts() == {"acme": 2}
    # The mention rides on the event, which is what pings Slack.
    event = store.events(verbs=["comment_added"])[-1]
    assert event.payload["mentions"] == ["sara@firm.com"]
    store.delete_comment(first)
    assert [c.body for c in store.comments_for("acme")] == ["agreed"]
    assert len(store.comments_for("acme", include_deleted=True)) == 2


def test_empty_comment_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.add_comment("acme", "   ")


# --- memo versioning ------------------------------------------------------------


def test_regeneration_no_longer_destroys_a_human_edit(tmp_path: Path) -> None:
    """The failure this table exists to fix: an edited memo used to be
    overwritten by the next generation with no way back."""
    store = make_store(tmp_path)
    store.set_memo("acme", "# v1 draft", meta={"depth": "deep"},
                   kind="generated", actor="agent:memo")
    store.set_memo("acme", "# v1 draft, my edits", kind="edited")
    store.set_memo("acme", "# fresh generation", kind="generated",
                   actor="agent:memo")
    assert store.get_pipeline("acme")["brief"] == "# fresh generation"
    versions = store.memo_versions("acme")
    assert [v.version_no for v in versions] == [3, 2, 1]
    assert [v.kind for v in versions] == ["generated", "edited", "generated"]
    assert versions[1].author == "alan@firm.com"  # the human's edit is attributed
    # The edit is recoverable — and restoring appends rather than mutating.
    assert store.restore_memo_version("acme", 2) == 4
    assert store.get_pipeline("acme")["brief"] == "# v1 draft, my edits"
    assert store.events(verbs=["memo_edited"])[0].payload["restored_from"] == 2


def test_memo_generation_stamps_and_meta_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_memo("acme", "# memo", meta={"depth": "deep", "sources": 11},
                   kind="generated", actor="agent:memo")
    row = store.get_pipeline("acme")
    assert row["brief_at"] and row["brief_edited_at"] is None
    assert row["brief_meta"] == {"depth": "deep", "sources": 11}
    store.set_memo("acme", "# memo, edited", kind="edited")
    row = store.get_pipeline("acme")
    assert row["brief_edited_at"]  # an edit stamps its own time
    assert store.memo_versions("acme")[0].meta == {}
    assert store.restore_memo_version("acme", 99) is None  # unknown version


def test_memo_review_flow_pins_the_approved_version(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_memo("acme", "# memo", kind="generated", actor="agent:memo")
    store.set_memo_review("acme", "requested")
    assert store.get_pipeline("acme")["memo_review_status"] == "requested"
    store.set_memo_review("acme", "approved", version_no=1, actor="sara@firm.com")
    row = store.get_pipeline("acme")
    assert row["memo_review_status"] == "approved"
    assert row["memo_reviewed_by"] == "sara@firm.com"
    assert row["memo_approved_version"] == 1
    # A later edit leaves the pin behind, which is how the UI flags drift.
    store.set_memo("acme", "# memo, changed after approval", kind="edited")
    assert store.get_pipeline("acme")["memo_approved_version"] == 1
    assert store.memo_versions("acme")[0].version_no == 2
    with pytest.raises(ValueError):
        store.set_memo_review("acme", "rubber_stamped")


# --- assignment & notes concurrency ---------------------------------------------


def test_assignment_and_handoff_are_evented(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_assignment("acme", "sara@firm.com")
    assert store.get_pipeline("acme")["assignee"] == "sara@firm.com"
    store.set_assignment("acme", None)
    row = store.get_pipeline("acme")
    assert row["assignee"] == "" and row["assigned_at"] is None
    assert [e.verb for e in store.events(verbs=["assigned", "unassigned"])] == [
        "unassigned", "assigned",
    ]


def test_notes_conflict_is_detected_not_clobbered(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("acme", notes="first draft")
    stamp = store.get_pipeline("acme")["updated_at"]
    # Someone else saves in the meantime.
    store.set_pipeline("acme", notes="sara's version", actor="sara@firm.com")
    # The stale editor's save is refused rather than overwriting.
    assert store.set_pipeline("acme", notes="my stale version",
                              expect_updated_at=stamp) is False
    assert store.get_pipeline("acme")["notes"] == "sara's version"
    # Re-reading and saving against the current stamp succeeds.
    fresh = store.get_pipeline("acme")["updated_at"]
    assert store.set_pipeline("acme", notes="merged", expect_updated_at=fresh) is True
    assert store.get_pipeline("acme")["notes"] == "merged"


def test_users_and_presence(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.ensure_user("alan@firm.com", name="Alan")
    store.ensure_user("sara@firm.com", name="Sara")
    seen_before = store.get_user("sara@firm.com")["last_seen_at"]
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    store.db["users"].update("sara@firm.com", {"last_seen_at": stale})
    store.touch_user("sara@firm.com")
    assert store.get_user("sara@firm.com")["last_seen_at"] >= seen_before


# --- per-partner taste ----------------------------------------------------------


def _entry(handle: str, score: float, sector: str = "ai infra",
           company: str = "", fit: float = 0.7):
    from scout.models import Account, LedgerEntry, Lead, LLMVerdict, Signal

    return LedgerEntry(lead=Lead(
        account=Account(id=handle, handle=handle, name=f"{handle} person",
                        bio="building"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle=handle, is_founder=True, stage="launched",
                       sector=sector, business_model="b2b saas",
                       customer_type="b2b", company_name=company or None,
                       thesis_fit=fit, confidence=0.9),
        score=score,
    ))


def test_actor_stats_reads_your_votes_not_firm_status(tmp_path: Path) -> None:
    from scout.insights import actor_stats

    entries = [_entry(f"co{i}", 60.0 + i) for i in range(6)]
    votes = {
        "co0": [vote("co0", "alan@firm.com", "strong_yes"),
                vote("co0", "sara@firm.com", "pass")],
        "co1": [vote("co1", "alan@firm.com", "yes")],
        "co2": [vote("co2", "alan@firm.com", "pass")],
        "co3": [vote("co3", "alan@firm.com", "pass")],
        "co4": [vote("co4", "alan@firm.com", "unsure")],  # not a decision
        "co5": [vote("co5", "alan@firm.com", "yes")],
    }
    stats = actor_stats(entries, votes, "alan@firm.com")
    assert stats is not None
    assert stats.shortlisted == 3 and stats.passed == 2  # unsure excluded
    # Sara has only one vote — below MIN_DECISIONS, so no profile yet.
    assert actor_stats(entries, votes, "sara@firm.com") is None
    # The same startup can sit on opposite sides of two partners' profiles.
    assert stats.sector_counts["shortlisted"]["ai infra"] == 3


def test_model_disagreements_surface_both_directions() -> None:
    from scout.insights import model_disagreements

    entries = [
        _entry("hot", 92.0, company="HotCo"),      # model loved it
        _entry("mild", 75.0, company="MildCo"),    # model liked it
        _entry("cold", 22.0, company="ColdCo"),    # model dismissed it
        _entry("agreed", 88.0, company="AgreedCo"),
    ]
    votes = {
        "hot": [vote("hot", "alan@firm.com", "pass", rationale="team won't ship")],
        "mild": [vote("mild", "alan@firm.com", "pass")],
        "cold": [vote("cold", "alan@firm.com", "strong_yes", rationale="I know this founder")],
        "agreed": [vote("agreed", "alan@firm.com", "yes")],  # no disagreement
    }
    out = model_disagreements(entries, votes, "alan@firm.com")
    # Ordered by distance past the threshold: hot is 22 over the "liked"
    # line, cold 18 under the "cool" line, mild only 5 over.
    assert [d.handle for d in out] == ["hot", "cold", "mild"]
    assert out[0].kind == "model_liked_you_passed"
    assert out[0].name == "HotCo"
    assert out[0].rationale == "team won't ship"
    # The other direction — conviction the model missed — is what should
    # eventually move the weights.
    assert out[1].kind == "model_cool_you_liked"
    assert out[1].name == "ColdCo"
    assert all(d.handle != "agreed" for d in out)  # agreement teaches nothing


# --- single-user → multiplayer migration ----------------------------------------


def test_migrate_multiplayer_backfills_and_is_idempotent(tmp_path: Path) -> None:
    """The upgrade path for a database built while Scout was single-user."""
    solo = Store(tmp_path / "solo.db")  # no actor — as the old code ran
    solo.set_pipeline("acme", status="shortlisted", notes="great team")
    solo.set_pipeline("beta", status="passed")
    solo.set_pipeline("gamma", status="new")  # undecided → no vote
    solo.set_pipeline("acme", brief="# existing memo", brief_meta={"depth": "deep"})
    solo.set_override("acme", fit=0.9, note="conviction")
    solo.set_attrs("acme", {"vertical": "Fintech"})
    # Nothing is attributed yet.
    assert solo.get_pipeline("acme")["updated_by"] == ""

    counts = solo.migrate_multiplayer("alan@firm.com")
    assert counts["memo_versions"] == 1
    assert counts["votes"] == 2  # shortlisted + passed; "new" is not a decision
    assert counts["attributed"] > 0

    assert solo.get_pipeline("acme")["updated_by"] == "alan@firm.com"
    assert solo.all_overrides()["acme"]["actor"] == "alan@firm.com"
    assert solo.get_user("alan@firm.com")["role"] == "admin"

    # The pre-existing memo is now recoverable, and a regeneration can't eat it.
    versions = solo.memo_versions("acme")
    assert [v.version_no for v in versions] == [1]
    assert versions[0].body == "# existing memo"
    solo.set_memo("acme", "# regenerated", kind="generated", actor="agent:memo")
    solo.actor = "alan@firm.com"  # as the app binds it at login
    assert solo.restore_memo_version("acme", 1) == 3
    assert solo.get_pipeline("acme")["brief"] == "# existing memo"

    # Triage became votes, honestly labelled as imports.
    votes = {v.handle: v for v in solo.all_votes().get("acme", [])}
    assert votes["acme"].stance == "yes"
    assert "imported from triage" in votes["acme"].rationale
    assert solo.all_votes()["beta"][0].stance == "pass"
    assert "gamma" not in solo.all_votes()
    # One import event, not one per vote — the feed must not be flooded.
    assert len(solo.events(verbs=["votes_imported"])) == 1

    # Re-running changes nothing and adds nothing.
    again = solo.migrate_multiplayer("alan@firm.com")
    assert again == {"attributed": 0, "memo_versions": 0, "votes": 0}
    assert len(solo.memo_versions("acme")) == 3
    assert len(solo.events(verbs=["votes_imported"])) == 1


def test_migrate_never_overwrites_a_real_vote(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("acme", status="shortlisted")
    store.set_vote("acme", "pass", "changed my mind", actor="alan@firm.com")
    store.migrate_multiplayer("alan@firm.com")
    vote_row = store.votes_for("acme")[0]
    assert vote_row.stance == "pass"  # the real opinion wins over the import
    assert vote_row.rationale == "changed my mind"


def test_migrate_requires_an_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_store(tmp_path).migrate_multiplayer("  ")
