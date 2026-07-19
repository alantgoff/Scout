"""Store-level tests: search cache TTL + case-insensitive handle lookups."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout.models import Account, Lead, LLMVerdict, Signal
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


# --- account sources round-trip -------------------------------------------------


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
    )


def test_verdict_cache_hit_on_matching_fingerprint(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_verdict("Alice", "fp-1", make_verdict())
    cached = store.cached_verdict("alice", "fp-1", ttl_days=14)
    assert cached is not None
    assert cached.thesis_fit == 0.8
    assert cached.tags == ["ai infra"]


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


def test_pipeline_brief_persists_without_clobbering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("alice", status="shortlisted", notes="met at demo day")
    store.set_pipeline("alice", brief="**What they're building** — evals")
    row = store.get_pipeline("alice")
    assert row["status"] == "shortlisted"
    assert row["notes"] == "met at demo day"
    assert row["brief"].startswith("**What")


def test_pipeline_tags_roundtrip_and_dedupe(tmp_path: Path) -> None:
    from scout.store import pipeline_tags

    store = make_store(tmp_path)
    store.set_pipeline("alice", tags=["Intro-Ready", "healthcare", " intro-ready "])
    assert pipeline_tags(store.get_pipeline("alice")) == ["Intro-Ready", "healthcare"]
    store.tag_lead("alice", "b2b")
    store.tag_lead("alice", "B2B")  # case-insensitive dedupe
    assert pipeline_tags(store.get_pipeline("alice")) == ["Intro-Ready", "healthcare", "b2b"]
    store.untag_lead("alice", "HEALTHCARE")
    assert pipeline_tags(store.get_pipeline("alice")) == ["Intro-Ready", "b2b"]
    # tags update never clobbers other fields (read-merge-write)
    store.set_pipeline("alice", status="shortlisted")
    store.set_pipeline("alice", tags=["solo"])
    assert store.get_pipeline("alice")["status"] == "shortlisted"


def test_pipeline_tags_tolerates_legacy_and_garbage() -> None:
    from scout.store import pipeline_tags

    assert pipeline_tags({}) == []
    assert pipeline_tags({"tags": None}) == []
    assert pipeline_tags({"tags": "a, b , a"}) == ["a", "b"]  # legacy comma string
    assert pipeline_tags({"tags": '["x", "", "x"]'}) == ["x"]


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


# --- memos + diligence usage ----------------------------------------------------


def make_memo(company_key: str = "tuvaai", fingerprint: str = "fp-1"):
    from scout.diligence.schema import DimensionFinding, Memo

    return Memo(
        company_key=company_key, company_name="Tuva AI", fingerprint=fingerprint,
        composite=71.5, memo_md="## Recommendation\nPursue.",
        findings=[DimensionFinding(key="data_flywheel", score=8.0, confidence=0.9)],
        cost_usd=5.12, created_at="2026-07-18T00:00:00+00:00",
    )


def test_memo_save_load_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.load_memo("tuvaai") is None
    store.save_memo(make_memo())
    loaded = store.load_memo("tuvaai")
    assert loaded is not None
    assert loaded.composite == 71.5
    assert loaded.findings[0].score == 8.0
    assert loaded.fingerprint == "fp-1"


def test_memo_fingerprint_cache_semantics(tmp_path: Path) -> None:
    """The memo cache is fingerprint-keyed like llm_verdicts: same key -> the
    caller compares fingerprints; a re-save replaces (one memo per company)."""
    store = make_store(tmp_path)
    store.save_memo(make_memo(fingerprint="fp-1"))
    assert store.load_memo("tuvaai").fingerprint == "fp-1"
    store.save_memo(make_memo(fingerprint="fp-2"))  # re-analysis replaces
    assert store.load_memo("tuvaai").fingerprint == "fp-2"
    assert len(store.list_memos()) == 1


def test_list_memos_newest_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    old = make_memo("oldco")
    old.created_at = "2026-01-01T00:00:00+00:00"
    store.save_memo(old)
    store.save_memo(make_memo("newco"))
    assert [m.company_key for m in store.list_memos()] == ["newco", "oldco"]


def test_diligence_usage_ledger_sums(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.diligence_spend_usd() == 0.0
    store.record_diligence_usage(
        company_key="tuvaai", stage="recon", model="claude-sonnet-4-6",
        input_tokens=1000, output_tokens=500, searches=2, est_cost_usd=0.30,
    )
    store.record_diligence_usage(
        company_key="tuvaai", stage="memo", model="claude-opus-4-8",
        input_tokens=2000, output_tokens=1500, searches=0, est_cost_usd=0.75,
    )
    store.record_diligence_usage(
        company_key="otherco", stage="recon", model="claude-sonnet-4-6",
        input_tokens=100, output_tokens=50, searches=1, est_cost_usd=0.05,
    )
    import pytest

    assert store.diligence_spend_usd() == pytest.approx(1.10)
    assert store.diligence_spend_usd("tuvaai") == pytest.approx(1.05)
    assert store.diligence_spend_usd("unknown") == 0.0
