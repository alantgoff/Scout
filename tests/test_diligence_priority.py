"""Smart-surfacing ranking: boosts, exclusions (memoed / passed / off-track),
and the generated-from-data reason line. Pure — no store."""

from __future__ import annotations

from scout.diligence.priority import analysis_priority
from scout.models import Account, Lead, LedgerEntry, LLMVerdict, Signal


def entry(
    handle: str,
    company: str | None,
    score: float = 50.0,
    fit: float | None = 0.7,
    stage: str = "launched",
    account_type: str = "founder",
    is_new: bool = False,
    prev_score: float | None = None,
    signals: list[str] | None = None,
) -> LedgerEntry:
    lead = Lead(
        account=Account(id=handle, handle=handle),
        signals=[Signal(name=n, value=1.0) for n in (signals or [])],
        llm=LLMVerdict(
            handle=handle, account_type=account_type, is_founder=True,
            stage=stage, company_name=company, thesis_fit=fit, confidence=0.9,
        ),
        score=score,
    )
    return LedgerEntry(lead=lead, is_new=is_new, prev_score=prev_score)


def test_ranking_by_fit_and_score() -> None:
    entries = [
        entry("a", "AlphaCo", score=90.0, fit=0.9),
        entry("b", "BetaCo", score=40.0, fit=0.3),
    ]
    out = analysis_priority(entries)
    assert [s.company_key for s in out] == ["alphaco", "betaco"]
    assert out[0].priority > out[1].priority


def test_freshness_and_momentum_boosts() -> None:
    base = analysis_priority([entry("a", "AlphaCo")])[0].priority
    fresh = analysis_priority([entry("a", "AlphaCo", is_new=True)])[0].priority
    moving = analysis_priority(
        [entry("a", "AlphaCo", score=60.0, prev_score=50.0)]
    )[0].priority
    assert fresh > base
    assert moving > base


def test_signal_richness_boost_and_reason() -> None:
    plain = analysis_priority([entry("a", "AlphaCo")])[0]
    rich = analysis_priority(
        [entry("a", "AlphaCo", signals=["departure_signal", "launch_traction"])]
    )[0]
    assert rich.priority > plain.priority
    assert "70% fit" in rich.reason
    assert "departure signal" in rich.reason


def test_memoed_companies_excluded() -> None:
    entries = [entry("a", "AlphaCo"), entry("b", "BetaCo")]
    out = analysis_priority(entries, memoed={"alphaco"})
    assert [s.company_key for s in out] == ["betaco"]


def test_non_startup_track_excluded() -> None:
    entries = [
        entry("a", "AlphaCo", stage="stealth"),  # pre-launch
        entry("b", "BetaCo", account_type="other"),  # not a startup account
        entry("c", "GammaCo"),
    ]
    out = analysis_priority(entries)
    assert [s.company_key for s in out] == ["gammaco"]


def test_passed_leads_excluded() -> None:
    entries = [entry("a", "AlphaCo"), entry("b", "BetaCo")]
    out = analysis_priority(entries, pipeline={"a": {"status": "passed"}})
    assert [s.company_key for s in out] == ["betaco"]


def test_company_grouping_one_suggestion_per_startup() -> None:
    entries = [
        entry("founder", "AlphaCo", score=80.0),
        entry("companyacct", "AlphaCo", score=60.0, account_type="startup"),
    ]
    out = analysis_priority(entries)
    assert len(out) == 1
    assert out[0].handle == "founder"  # highest-scoring member is primary


def test_new_this_run_reason_text() -> None:
    out = analysis_priority([entry("a", "AlphaCo", is_new=True)])
    assert "new this run" in out[0].reason


def test_handle_key_fallback_without_company_name() -> None:
    out = analysis_priority([entry("solo_founder", None)])
    # The fallback key is the NORMALIZED handle (startup_key) so it matches
    # KG node ids and memo keys; the display name stays the raw handle.
    assert out[0].company_key == "solofounder"
    assert out[0].company_name == "solo_founder"
