"""Store-level tests: search cache TTL + case-insensitive handle lookups."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout.models import Account, Lead, LLMVerdict, Signal, SitePage
from scout.store import Store


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


def make_account(handle: str, account_id: str = "1") -> Account:
    return Account(id=account_id, handle=handle, name="Some One", bio="building")


# --- search cache -------------------------------------------------------------


def test_record_search_then_cached_search_hits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_search('"ex-OpenAI" stealth', ["Alice", "bob"])
    assert store.cached_search('"ex-OpenAI" stealth', ttl_days=7) == ["Alice", "bob"]


def test_cached_search_miss_for_unknown_query(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.cached_search("never ran", ttl_days=7) is None


def test_cached_search_expires_after_ttl(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_search("old query", ["alice"])
    stale = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    store.db["searches"].upsert(
        {"query": "old query", "fetched_at": stale}, pk="query"
    )
    assert store.cached_search("old query", ttl_days=7) is None
    # A longer TTL still sees it.
    assert store.cached_search("old query", ttl_days=30) == ["alice"]


def test_record_search_overwrites_previous_entry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_search("q", ["one"])
    store.record_search("q", ["two", "three"])
    assert store.cached_search("q", ttl_days=7) == ["two", "three"]


# --- case-insensitive lookups --------------------------------------------------


def test_get_account_is_case_insensitive_and_preserves_display_case(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.upsert_account(make_account("CoolFounder"))
    for probe in ("coolfounder", "COOLFOUNDER", "CoolFounder"):
        found = store.get_account(probe)
        assert found is not None
        assert found.handle == "CoolFounder"  # display casing kept


def test_last_scored_and_recently_scored_case_insensitive(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    lead = Lead(
        account=make_account("MixedCase"),
        signals=[Signal(name="bio_intent", value=1.0)],
    )
    store.save_leads("run-1", [lead])
    assert store.last_scored_at("mixedcase") is not None
    assert store.recently_scored("MIXEDCASE", ttl_days=7) is True
    assert store.recently_scored("someoneelse", ttl_days=7) is False


def test_last_scored_map_matches_per_handle_lookups(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    signals = [Signal(name="bio_intent", value=1.0)]
    store.save_leads("run-1", [Lead(account=make_account("Alpha", "1"), signals=signals)])
    store.save_leads("run-2", [
        Lead(account=make_account("ALPHA", "1"), signals=signals),
        Lead(account=make_account("Beta", "2"), signals=signals),
    ])
    scored = store.last_scored_map()
    # Lowercase-keyed, casings merged, newest run wins.
    assert set(scored) == {"alpha", "beta"}
    last = store.last_scored_at("alpha")
    assert last is not None
    assert scored["alpha"] == (
        last if last.tzinfo else last.replace(tzinfo=timezone.utc)
    )


def test_last_scored_map_empty_without_leads(tmp_path: Path) -> None:
    assert make_store(tmp_path).last_scored_map() == {}


def test_secondary_indexes_created_once_tables_exist(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.save_leads(
        "run-1",
        [Lead(account=make_account("a"), signals=[Signal(name="bio_intent", value=1.0)])],
    )
    store.upsert_account(make_account("a"))
    # Tables are created lazily on first write; the next init indexes them.
    reopened = Store(tmp_path / "test.db")
    names = {
        r[0]
        for r in reopened.db.execute(
            "select name from sqlite_master where type = 'index'"
        ).fetchall()
    }
    assert {"idx_leads_handle", "idx_accounts_handle"} <= names


# --- account sources round-trip -------------------------------------------------


def test_upsert_accounts_batch_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    one = make_account("one", "1")
    one.sources = ["search"]
    two = make_account("two", "2")
    two.followed_by = ["eladgil"]
    store.upsert_accounts([one, two])
    loaded_one = store.get_account("one")
    loaded_two = store.get_account("two")
    assert loaded_one is not None and loaded_one.sources == ["search"]
    assert loaded_two is not None and loaded_two.followed_by == ["eladgil"]


def test_account_sources_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    account = make_account("multi")
    account.sources = ["search", "github"]
    store.upsert_account(account)
    loaded = store.get_account("multi")
    assert loaded is not None
    assert loaded.sources == ["search", "github"]


# --- verdict cache --------------------------------------------------------------


def make_verdict(handle: str = "alice") -> LLMVerdict:
    return LLMVerdict(
        handle=handle, account_type="founder", is_founder=True, stage="launched",
        thesis_fit=0.8, tags=["ai infra"], confidence=0.9,
        value_add_fit=0.7, value_add_levers={"global_expansion": 0.9},
        value_add_reason="expanding to the EU next quarter",
    )


def test_verdict_cache_hit_on_matching_fingerprint(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_verdict("Alice", "fp-1", make_verdict())
    cached = store.cached_verdict("alice", "fp-1", ttl_days=14)
    assert cached is not None
    assert cached.thesis_fit == 0.8
    assert cached.tags == ["ai infra"]
    assert cached.value_add_fit == 0.7
    assert cached.value_add_levers == {"global_expansion": 0.9}


def test_verdict_cache_misses_on_changed_inputs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_verdict("alice", "fp-1", make_verdict())
    assert store.cached_verdict("alice", "fp-CHANGED", ttl_days=14) is None
    assert store.cached_verdict("bob", "fp-1", ttl_days=14) is None
    assert store.cached_verdict("alice", "fp-1", ttl_days=0) is None  # ttl 0 = off


def test_verdict_cache_expires_after_ttl(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_verdict("alice", "fp-1", make_verdict())
    stale = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    store.db["llm_verdicts"].upsert({"handle": "alice", "created_at": stale}, pk="handle")
    assert store.cached_verdict("alice", "fp-1", ttl_days=14) is None
    assert store.cached_verdict("alice", "fp-1", ttl_days=30) is not None


def test_pipeline_counts_group_by_status_with_new_fallback(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("a", status="shortlisted")
    store.set_pipeline("b", status="passed")
    store.set_pipeline("c", status="passed")
    store.set_pipeline("d", notes="note only")  # no status → counted as "new"
    assert store.pipeline_counts() == {"shortlisted": 1, "passed": 2, "new": 1}


def test_pipeline_brief_persists_without_clobbering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("alice", status="shortlisted", notes="met at demo day")
    store.set_pipeline("alice", brief="**What they're building** — evals")
    row = store.get_pipeline("alice")
    assert row["status"] == "shortlisted"
    assert row["notes"] == "met at demo day"
    assert row["brief"].startswith("**What")


# --- lead ledger + runs ---------------------------------------------------------

T1 = "2026-07-01T00:00:00+00:00"
T2 = "2026-07-02T00:00:00+00:00"
T3 = "2026-07-03T00:00:00+00:00"


def seed_run(
    store: Store, run_id: str, created_at: str, entries: list[tuple[str, float]]
) -> None:
    """Insert lead rows with a controlled created_at (save_leads stamps now)."""
    rows = []
    for handle, score in entries:
        lead = Lead(
            account=Account(id=f"{run_id}-{handle}", handle=handle),
            score=score,
        )
        rows.append(
            {
                "run_id": run_id,
                "handle": handle,
                "rank": None,
                "score": score,
                "lead_json": lead.model_dump_json(),
                "created_at": created_at,
            }
        )
    store.db["leads"].upsert_all(rows, pk=("run_id", "handle"), alter=True)


def test_ledger_latest_per_handle_case_merge_delta_and_is_new(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_run(store, "r1", T1, [("alice", 10.0), ("bob", 30.0)])
    seed_run(store, "r2", T2, [("Alice", 20.0), ("carol", 15.0)])  # case differs

    ledger = store.load_lead_ledger()
    by = {e.lead.account.handle.lower(): e for e in ledger}

    assert len(ledger) == 3  # Alice/alice merged
    alice = by["alice"]
    assert alice.lead.score == 20.0  # latest appearance wins
    assert alice.prev_score == 10.0
    assert alice.score_delta == 10.0
    assert alice.times_seen == 2
    assert alice.first_seen_run == "r1"
    assert alice.is_new is False
    assert by["bob"].is_new is False  # first seen r1, newest run is r2
    assert by["bob"].prev_score is None
    assert by["carol"].is_new is True  # first appeared in the newest run
    scores = [e.lead.score for e in ledger]
    assert scores == sorted(scores, reverse=True)


def test_ledger_excludes_demo_includes_verify(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_run(store, "r1", T1, [("alice", 10.0)])
    seed_run(store, "demo-1", T2, [("demo_dan", 50.0)])
    seed_run(store, "verify-1", T3, [("alice", 25.0)])

    ledger = store.load_lead_ledger()
    handles = {e.lead.account.handle.lower() for e in ledger}
    assert handles == {"alice"}  # demo dropped, verify- kept
    assert ledger[0].lead.score == 25.0
    assert ledger[0].prev_score == 10.0  # verify re-score is a real appearance

    with_demo = store.load_lead_ledger(include_demo=True)
    assert {e.lead.account.handle.lower() for e in with_demo} == {"alice", "demo_dan"}


def test_ledger_strategy_filter(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_run(store, "r1", T1, [("alice", 10.0)])
    seed_run(store, "r2", T2, [("alice", 20.0), ("bob", 5.0)])
    store.record_run("r1", source="twscrape", strategy_hash="hash-A", thesis_statement="A")
    store.record_run("r2", source="twscrape", strategy_hash="hash-B", thesis_statement="B")

    only_a = store.load_lead_ledger(strategy_hash="hash-A")
    assert len(only_a) == 1
    assert only_a[0].lead.score == 10.0  # r2's alice filtered out
    assert only_a[0].prev_score is None
    assert only_a[0].is_new is True  # r1 is the newest run WITHIN the filter

    only_b = store.load_lead_ledger(strategy_hash="hash-B")
    assert {e.lead.account.handle for e in only_b} == {"alice", "bob"}


def test_ledger_thesis_filter_spans_every_version(tmp_path: Path) -> None:
    """Thesis identity, not tuning: both runs belong to one thesis even
    though a weight change gave them different versions."""
    store = make_store(tmp_path)
    seed_run(store, "r1", T1, [("alice", 10.0)])
    seed_run(store, "r2", T2, [("bob", 20.0)])
    store.record_run("r1", source="xapi", strategy_hash="v1", thesis_statement="A",
                     thesis_id="edge-ai", thesis_version="v1")
    store.record_run("r2", source="xapi", strategy_hash="v2", thesis_statement="A",
                     thesis_id="edge-ai", thesis_version="v2")

    both = store.load_lead_ledger(thesis_id="edge-ai")
    assert {e.lead.account.handle for e in both} == {"alice", "bob"}
    # …while the version filter still isolates one exact tuning.
    assert {e.lead.account.handle
            for e in store.load_lead_ledger(strategy_hash="v1")} == {"alice"}
    assert store.load_lead_ledger(thesis_id="other") == []


def test_stale_handles_flip_when_the_thesis_is_retuned(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_run(store, "r1", T1, [("alice", 10.0)])
    store.record_run("r1", source="xapi", strategy_hash="v1", thesis_statement="A",
                     thesis_id="edge-ai", thesis_version="v1")
    store.upsert_thesis("edge-ai", name="Edge AI", statement="A", version="v1")
    assert store.stale_handles("edge-ai") == []

    store.upsert_thesis("edge-ai", name="Edge AI", statement="A", version="v2")
    assert store.stale_handles("edge-ai") == ["alice"]


def test_verdict_history_survives_a_rescore(tmp_path: Path) -> None:
    """One row per handle, so the overwrite must archive first — otherwise
    'what did this score under the old thesis' is unanswerable."""
    store = make_store(tmp_path)
    store.record_verdict("acme", "fp1", LLMVerdict(handle="acme", thesis_fit=0.2),
                         thesis_id="edge-ai", thesis_version="v1")
    store.record_verdict("acme", "fp2", LLMVerdict(handle="acme", thesis_fit=0.75),
                         thesis_id="novel-arch", thesis_version="v1")

    assert store.cached_verdict("acme", "fp2", 30).thesis_fit == 0.75
    history = store.verdict_history("acme")
    assert [(h["thesis_id"], h["verdict"].thesis_fit) for h in history] == [
        ("edge-ai", 0.2)
    ]
    assert store.verdict_provenance("acme")["thesis_id"] == "novel-arch"


def test_backfill_groups_legacy_runs_by_statement_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """The real database had 7 strategy hashes for 4 theses — one thesis
    fragmented across every weight tweak it ever had. The statement is what
    stayed put, so grouping on it recovers the theses a human would name."""
    store = make_store(tmp_path)
    for run_id, hash_ in (("r1", "h1"), ("r2", "h2"), ("r3", "h3")):
        store.record_run(run_id, source="xapi", strategy_hash=hash_,
                         thesis_statement="Edge AI infrastructure")
    store.record_run("r4", source="xapi", strategy_hash="h4",
                     thesis_statement="Novel model architectures")

    assert store.backfill_thesis_ids() == 4
    assert {t["id"] for t in store.list_theses()} == {
        "edge-ai-infrastructure", "novel-model-architectures"
    }
    assert store.backfill_thesis_ids() == 0  # nothing left to fill


def test_list_strategies_groups_and_excludes_demo(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_run("r1", source="twscrape", strategy_hash="hash-A", thesis_statement="A")
    store.record_run("r2", source="twscrape", strategy_hash="hash-A", thesis_statement="A")
    store.record_run("r3", source="xapi", strategy_hash="hash-B", thesis_statement="B")
    store.record_run("demo-1", source="demo", strategy_hash="hash-D", thesis_statement="D")

    strategies = store.list_strategies()
    by_hash = {s["strategy_hash"]: s for s in strategies}
    assert set(by_hash) == {"hash-A", "hash-B"}  # demo excluded
    assert by_hash["hash-A"]["run_count"] == 2
    assert by_hash["hash-B"]["run_count"] == 1


def test_last_real_run_at(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.last_real_run_at() is None
    store.record_run("demo-1", source="demo", strategy_hash="h", thesis_statement="")
    assert store.last_real_run_at() is None  # demo doesn't count
    store.record_run("r1", source="twscrape", strategy_hash="h", thesis_statement="")
    assert store.last_real_run_at() is not None


def test_last_real_run_at_falls_back_to_legacy_leads(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_run(store, "r-legacy", T1, [("alice", 10.0)])  # no runs row
    assert store.last_real_run_at() is not None


# --- scan status ---------------------------------------------------------------


def test_scan_lifecycle(tmp_path: Path) -> None:
    import os

    store = make_store(tmp_path)
    assert store.current_scan() is None

    store.scan_start("run", os.getpid())  # our own pid — definitely alive
    store.scan_update("classifying", "42 candidates")
    scan = store.current_scan()
    assert scan is not None
    assert scan["status"] == "running"
    assert scan["phase"] == "classifying"
    assert scan["detail"] == "42 candidates"

    store.scan_finish("done")
    scan = store.current_scan()
    assert scan["status"] == "done"
    assert scan["finished_at"]
    store.scan_update("late", "ignored after finish")  # no-op once finished
    assert store.current_scan()["phase"] == "classifying"


def test_current_scan_flips_dead_process_to_failed(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.scan_start("run", pid=2 ** 22 + 1234567)  # far beyond macOS pid range
    scan = store.current_scan()
    assert scan["status"] == "failed"
    assert "exited" in scan["detail"]


# --- site cache -----------------------------------------------------------


def test_site_cache_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    page = SitePage(url="https://raindrop.ai/", final_url="https://www.raindrop.ai/",
                    status="ok", text="AI agent monitoring and observability.",
                    fetched_at=datetime.now(timezone.utc))
    store.record_site(page)
    cached = store.cached_site("https://raindrop.ai/", ttl_days=7)
    assert cached is not None
    assert cached.status == "ok"
    assert "observability" in cached.text
    assert cached.usable
    assert store.cached_site("https://other.ai/", ttl_days=7) is None


def test_site_cache_expires_after_ttl(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=8)
    store.record_site(SitePage(url="https://a.io/", status="ok",
                               text="x" * 300, fetched_at=old))
    assert store.cached_site("https://a.io/", ttl_days=7) is None
    assert store.cached_site("https://a.io/", ttl_days=30) is not None


def test_site_cache_failures_expire_after_one_day(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    stale = datetime.now(timezone.utc) - timedelta(days=2)
    store.record_site(SitePage(url="https://down.io/", status="error:timeout",
                               fetched_at=stale))
    # A 2-day-old failure is expired even under a 7-day TTL...
    assert store.cached_site("https://down.io/", ttl_days=7) is None
    # ...but a fresh failure IS served (negative caching within the day).
    store.record_site(SitePage(url="https://down.io/", status="error:timeout",
                               fetched_at=datetime.now(timezone.utc)))
    cached = store.cached_site("https://down.io/", ttl_days=7)
    assert cached is not None
    assert not cached.usable


def test_score_override_roundtrip_and_clear(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.set_override("Ada", quality={"traction": 0.9}, fit=0.5, note="saw the demo")
    row = store.all_overrides()["ada"]  # handle lowercased
    assert row["quality"] == {"traction": 0.9}
    assert row["sections"] == {}
    assert row["fit"] == 0.5
    assert row["score"] is None
    assert row["note"] == "saw the demo"
    # Replacing with a pinned score only
    store.set_override("ada", score=88.0)
    row = store.all_overrides()["ada"]
    assert row["score"] == 88.0 and row["quality"] == {} and row["fit"] is None
    store.clear_override("@Ada")
    assert store.all_overrides() == {}


def test_score_override_sections_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.set_override("Ada", sections={"market": 80.0, "technology": 55.0},
                       note="their traction is better than the model thinks")
    row = store.all_overrides()["ada"]
    assert row["sections"] == {"market": 80.0, "technology": 55.0}
    assert row["quality"] == {}
    # A section-only override still clears when everything empties out.
    store.set_override("ada", sections=None, quality=None, fit=None, score=None)
    assert store.all_overrides() == {}


def test_set_override_with_nothing_clears_the_row(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.set_override("ada", score=50.0)
    store.set_override("ada")  # everything None/empty -> row removed
    assert store.all_overrides() == {}
    store.clear_override("never_existed")  # no-op, no table -> must not raise


def test_pipeline_brief_edit_stamps(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.set_pipeline("ada", brief="v1 generated")
    row = store.get_pipeline("ada")
    assert row["brief_at"] and row.get("brief_edited_at") is None
    generated_at = row["brief_at"]
    store.set_pipeline("ada", brief="v2 hand-edited", brief_edited=True)
    row = store.get_pipeline("ada")
    assert row["brief"] == "v2 hand-edited"
    assert row["brief_at"] == generated_at  # generation stamp preserved
    assert row["brief_edited_at"]
    # A regeneration resets the edit stamp.
    store.set_pipeline("ada", brief="v3 regenerated")
    row = store.get_pipeline("ada")
    assert row["brief_edited_at"] is None


def test_pipeline_brief_meta_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    meta = {"depth": "deep", "sources": ["https://a.com", "https://b.com"],
            "searches": 5, "fetches": 3}
    store.set_pipeline("ada", brief="## Overview\nx", brief_meta=meta)
    row = store.get_pipeline("ada")
    assert row["brief_meta"] == meta
    # An edit leaves the generation meta untouched.
    store.set_pipeline("ada", brief="## Overview\nedited", brief_edited=True)
    assert store.get_pipeline("ada")["brief_meta"] == meta
    # Rows without meta decode to an empty dict, not a crash.
    store.set_pipeline("noone", status="longlisted")
    assert store.all_pipeline()["noone"]["brief_meta"] == {}


def test_startup_columns_crud_and_seed_only_once(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    defaults = [
        {"key": "vertical", "label": "Vertical", "type": "select",
         "options": ["Fintech", "Security"]},
        {"key": "priority", "label": "Priority", "type": "select",
         "options": ["High", "Low"]},
    ]
    store.ensure_default_columns(defaults)
    cols = store.list_columns()
    assert [c["key"] for c in cols] == ["vertical", "priority"]
    assert cols[0]["options"] == ["Fintech", "Security"]
    assert cols[0]["builtin"] is True
    # Deleting a builtin sticks — re-seeding must not resurrect it.
    store.delete_column("priority")
    store.ensure_default_columns(defaults)
    assert [c["key"] for c in store.list_columns()] == ["vertical"]
    # Custom columns append at the end; option edits keep position.
    store.save_column("warm_intro", "Warm intro", "checkbox")
    store.save_column("vertical", "Vertical", "select",
                      options=["Fintech", "Climate"], builtin=True, position=0)
    cols = store.list_columns()
    assert [c["key"] for c in cols] == ["vertical", "warm_intro"]
    assert cols[0]["options"] == ["Fintech", "Climate"]
    assert cols[1]["builtin"] is False
    # Judgment columns can be excluded from auto-categorization.
    store.save_column("conviction", "Conviction", "select",
                      options=["High", "Low"], ai_fill=False)
    by_key = {c["key"]: c for c in store.list_columns()}
    assert by_key["conviction"]["ai_fill"] is False
    assert by_key["vertical"]["ai_fill"] is True


def test_startup_attrs_merge_delete_and_types(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.set_attrs("@Ada", {"vertical": "Fintech",
                             "use_case": ["Payments", "Risk"],
                             "warm_intro": True, "round_size": 2.5})
    store.set_attrs("ada", {"vertical": "Security", "warm_intro": None})
    attrs = store.all_attrs()["ada"]  # lowercased, @-stripped
    assert attrs == {"vertical": "Security",
                     "use_case": ["Payments", "Risk"], "round_size": 2.5}
    assert store.all_attrs().get("nobody") is None
