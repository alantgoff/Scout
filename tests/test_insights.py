"""Insights + weight-proposal tests — pure functions, no network."""

from __future__ import annotations

import json

import pytest

from scout.agents import parse_weight_proposal
from scout.insights import MIN_DECISIONS, stats_prompt, triage_stats
from scout.models import Account, Lead, LedgerEntry, LLMVerdict, Signal


def make_entry(
    handle: str,
    sector: str = "ai infra",
    fit: float | None = 0.8,
    launch_pts: float = 0.0,
) -> LedgerEntry:
    signals = [Signal(name="bio_intent", value=1.0, weight=20.0)]
    if launch_pts:
        signals.append(Signal(name="launch_traction", value=1.0, weight=launch_pts))
    lead = Lead(
        account=Account(id=handle, handle=handle),
        signals=signals,
        llm=LLMVerdict(handle=handle, is_founder=True, stage="launched",
                       sector=sector, business_model="devtools",
                       thesis_fit=fit, confidence=0.9),
        score=50.0,
    )
    return LedgerEntry(lead=lead)


def pipeline_for(statuses: dict[str, str]) -> dict[str, dict]:
    return {h: {"handle": h, "status": s} for h, s in statuses.items()}


def test_triage_stats_below_threshold_returns_none() -> None:
    entries = [make_entry(f"h{i}") for i in range(MIN_DECISIONS - 1)]
    pipeline = pipeline_for({e.lead.account.handle: "passed" for e in entries})
    assert triage_stats(entries, pipeline) is None


def test_triage_stats_contrasts_groups_and_writes_findings() -> None:
    # 3 shortlisted ai-infra leads with strong launch traction + high fit,
    # 3 passed devtools leads with none — every contrast should fire.
    entries = (
        [make_entry(f"good{i}", sector="ai infra", fit=0.9, launch_pts=10.0)
         for i in range(3)]
        + [make_entry(f"bad{i}", sector="devtools", fit=0.3) for i in range(3)]
    )
    pipeline = pipeline_for(
        {f"good{i}": "shortlisted" for i in range(3)}
        | {f"bad{i}": "passed" for i in range(3)}
    )
    stats = triage_stats(entries, pipeline)
    assert stats is not None
    assert stats.shortlisted == 3 and stats.passed == 3
    assert stats.sector_counts["passed"] == {"devtools": 3}
    assert stats.fit_means["shortlisted"] == pytest.approx(0.9)
    assert stats.fit_means["passed"] == pytest.approx(0.3)
    assert stats.signal_means["shortlisted"]["launch_traction"] == 10.0
    assert stats.findings  # at least one plain-language contrast
    assert any("devtools" in f for f in stats.findings)
    # prompt rendering includes the numbers the agent needs
    prompt = stats_prompt(stats)
    assert "launch_traction" in prompt and "devtools" in prompt


def test_triage_stats_counts_deeper_stages_as_shortlisted() -> None:
    entries = [make_entry(f"h{i}") for i in range(6)]
    pipeline = pipeline_for({
        "h0": "won", "h1": "diligence", "h2": "meeting",
        "h3": "contacted", "h4": "passed", "h5": "passed",
    })
    stats = triage_stats(entries, pipeline)
    assert stats is not None
    assert stats.shortlisted == 4
    assert stats.passed == 2


CURRENT = {"bio_intent": 20.0, "launch_traction": 10.0, "github_evidence": 5.0}


def test_parse_weight_proposal_clamps_drops_and_backfills() -> None:
    text = json.dumps({
        "weights": {"bio_intent": 90, "launch_traction": -3, "made_up_signal": 25},
        "rationale": "because",
    })
    proposal = parse_weight_proposal(text, CURRENT)
    assert proposal.weights["bio_intent"] == 50.0  # clamped to the slider max
    assert proposal.weights["launch_traction"] == 0.0  # clamped at zero
    assert "made_up_signal" not in proposal.weights  # unknown name dropped
    assert proposal.weights["github_evidence"] == 5.0  # forgotten -> current kept


def test_parse_weight_proposal_handles_fences_and_rejects_junk() -> None:
    fenced = "```json\n" + json.dumps({"weights": {"bio_intent": 30}, "rationale": "r"}) + "\n```"
    assert parse_weight_proposal(fenced, CURRENT).weights["bio_intent"] == 30.0
    with pytest.raises(ValueError):
        parse_weight_proposal(json.dumps({"weights": {"unknown": 10}, "rationale": ""}), CURRENT)
    with pytest.raises(ValueError):
        parse_weight_proposal("[1, 2]", CURRENT)


# --- query yield ---------------------------------------------------------------


def test_query_yield_scores_and_flags_dead_queries() -> None:
    from scout.insights import performance_block, query_yield

    hits = ([{"query": "dead-q", "category": "launch", "handle": f"h{i}"}
             for i in range(12)]
            + [{"query": "gold-q", "category": "departure", "handle": "winner"},
               {"query": "gold-q", "category": "departure", "handle": "meh"},
               {"query": "young-q", "category": "hiring", "handle": "h1"}])
    ledger = {"winner", "meh"} | {f"h{i}" for i in range(6)}
    pipeline = {"winner": {"status": "shortlisted"}, "meh": {"status": "passed"}}

    yields = query_yield(hits, ledger, pipeline)
    by_query = {y.query: y for y in yields}
    assert yields[0].query == "gold-q"  # earners first
    assert (by_query["gold-q"].surfaced, by_query["gold-q"].triaged,
            by_query["gold-q"].passed) == (2, 1, 1)
    assert by_query["dead-q"].dead        # 12 surfaced, 0 triaged
    assert by_query["young-q"].unproven   # too small a sample to condemn
    assert not by_query["young-q"].dead

    block = performance_block(yields, [("Bpifrance", 2)])
    assert "'gold-q'" in block and "'dead-q'" in block
    assert "'young-q'" not in block       # no verdict on no evidence
    assert "Bpifrance (2 tracked companies)" in block
    # Nothing measured → empty, never a block of zeros.
    assert performance_block([], None) == ""


# --- source attribution ------------------------------------------------------------


def _src_entry(handle: str, source: str, sources: list[str] | None = None) -> LedgerEntry:
    return LedgerEntry(lead=Lead(account=Account(
        id=handle, handle=handle, source=source, sources=sources or [])))


def test_source_yield_counts_every_source_and_what_only_it_found() -> None:
    from scout.insights import source_yield

    ledger = [
        _src_entry("acme", "yc", ["yc", "search:launch"]),  # triaged, two sources
        _src_entry("beta", "sec", ["sec"]),                 # triaged, SEC only
        _src_entry("gamma", "rss", ["rss"]),                # passed
        _src_entry("delta", "search:hiring"),               # legacy row: no sources list
    ]
    pipeline = {"acme": {"status": "shortlisted"}, "beta": {"status": "longlisted"},
                "gamma": {"status": "passed"}}
    rows = {y.source: y for y in source_yield(ledger, pipeline)}

    # "search:launch" and "search:hiring" are one source: X search.
    assert rows["search"].label == "X search"
    assert (rows["search"].scored, rows["search"].triaged) == (2, 1)
    assert rows["search"].unique_triaged == 0 and rows["search"].redundant
    assert (rows["yc"].triaged, rows["yc"].unique_triaged) == (1, 0)
    assert (rows["sec"].triaged, rows["sec"].unique_triaged) == (1, 1)
    assert not rows["sec"].redundant
    assert (rows["rss"].passed, rows["rss"].triaged) == (1, 0)
    assert rows["sec"].hit_rate == 1.0 and rows["rss"].hit_rate == 0.0
    # Best earners first.
    assert [y.source for y in source_yield(ledger, pipeline)][0] == "sec"


def test_source_yield_flags_dead_sources_only_on_a_real_sample() -> None:
    from scout.insights import source_yield

    few = [_src_entry(f"h{i}", "github", ["github"]) for i in range(5)]
    many = [_src_entry(f"g{i}", "hn", ["hn"]) for i in range(25)]
    manual = [_src_entry(f"m{i}", "manual", ["manual"]) for i in range(25)]
    rows = {y.source: y for y in source_yield(few + many + manual, {})}
    assert not rows["github"].dead      # 5 scored is no evidence
    assert rows["hn"].dead              # 25 scored, none triaged
    assert not rows["manual"].dead      # a person adding companies is not a channel to prune


def test_demo_and_hindsight_rows_are_not_sources() -> None:
    from scout.insights import source_yield

    rows = source_yield([_src_entry("d", "demo", ["demo"]),
                         _src_entry("h", "hindsight")], {})
    assert rows == []


def test_performance_block_reports_source_yield_only_when_measured() -> None:
    from scout.insights import SourceYield, performance_block

    earner = SourceYield(source="sec", label="SEC Form D", scored=4, triaged=2,
                         passed=0, unique_triaged=2)
    dead = SourceYield(source="hn", label="Hacker News", scored=30, triaged=0,
                       passed=3, unique_triaged=0)
    quiet = SourceYield(source="yc", label="YC directory", scored=3, triaged=0,
                        passed=0, unique_triaged=0)
    block = performance_block([], None, sources=[earner, dead, quiet])
    assert "SEC Form D: 4 scored, 2 triaged (2 found by nothing else)" in block
    assert "Hacker News" in block and "nothing triaged" in block
    assert "YC directory" not in block  # unmeasured: silence, not a verdict
    assert performance_block([], None, sources=[quiet]) == ""
