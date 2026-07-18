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
