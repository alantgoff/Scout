"""Unit tests for score.py aggregation (spec §7: encodes the thesis math)."""

from __future__ import annotations

import pytest

from scout.config import Thesis
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.score import score_breakdown, score_leads

# Spec §3 defaults: total weight 100, so contributions read as points.
DEFAULT_WEIGHTS: dict[str, float] = {
    "bio_intent": 25,
    "smart_money_follow": 25,
    "departure_signal": 20,
    "launch_traction": 15,
    "builder_evidence": 15,
}


def make_thesis(weights: dict[str, float] | None = None) -> Thesis:
    return Thesis(weights=DEFAULT_WEIGHTS if weights is None else weights)


def make_lead(
    handle: str,
    signals: list[Signal] | None = None,
    llm: LLMVerdict | None = None,
) -> Lead:
    account = Account(id=handle, handle=handle)
    return Lead(account=account, signals=signals or [], llm=llm)


def test_bio_intent_plus_departure_scores_45() -> None:
    lead = make_lead(
        "founder",
        signals=[
            Signal(name="bio_intent", value=1.0),
            Signal(name="departure_signal", value=1.0),
        ],
    )
    ranked = score_leads([lead], make_thesis())
    assert ranked[0].score == 45.0


def test_weights_filled_from_thesis() -> None:
    lead = make_lead("founder", signals=[Signal(name="bio_intent", value=1.0)])
    score_leads([lead], make_thesis())
    assert lead.signals[0].weight == 25.0


def test_llm_confidence_multiplies_score() -> None:
    signals = [
        Signal(name="bio_intent", value=1.0),
        Signal(name="departure_signal", value=1.0),
    ]
    without = make_lead("plain", signals=[s.model_copy() for s in signals])
    with_llm = make_lead(
        "vetted",
        signals=[s.model_copy() for s in signals],
        llm=LLMVerdict(handle="vetted", is_founder=True, confidence=0.5,
                       grounding="website"),
    )
    score_leads([without, with_llm], make_thesis())
    assert without.score == 45.0
    assert with_llm.score == 22.5


def test_fractional_signal_contributes_proportionally() -> None:
    lead = make_lead("shipper", signals=[Signal(name="launch_traction", value=0.5)])
    score_leads([lead], make_thesis())
    assert lead.score == 7.5


def test_ranking_desc_with_alphabetical_tiebreak() -> None:
    top = make_lead("zed", signals=[Signal(name="bio_intent", value=1.0)])
    tie_b = make_lead("bravo", signals=[Signal(name="launch_traction", value=1.0)])
    tie_a = make_lead("alpha", signals=[Signal(name="builder_evidence", value=1.0)])
    ranked = score_leads([tie_b, top, tie_a], make_thesis())

    assert [lead.account.handle for lead in ranked] == ["zed", "alpha", "bravo"]
    assert [lead.rank for lead in ranked] == [1, 2, 3]
    assert ranked[0].score == 25.0
    assert ranked[1].score == ranked[2].score == 15.0


def test_unknown_signal_gets_zero_weight() -> None:
    lead = make_lead(
        "mystery",
        signals=[
            Signal(name="not_in_thesis", value=1.0),
            Signal(name="bio_intent", value=1.0),
        ],
    )
    score_leads([lead], make_thesis())
    assert lead.signals[0].weight == 0.0
    assert lead.score == 25.0  # only bio_intent counts


def test_empty_weights_scores_zero_without_zerodivision() -> None:
    lead = make_lead("anyone", signals=[Signal(name="bio_intent", value=1.0)])
    ranked = score_leads([lead], make_thesis(weights={}))
    assert ranked[0].score == 0.0
    assert ranked[0].rank == 1


def test_not_founder_verdict_slashes_score() -> None:
    thesis = make_thesis()
    founder = make_lead(
        "founder",
        signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="founder", is_founder=True, confidence=0.8,
                       grounding="website"),
    )
    corp = make_lead(
        "corp",
        signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="corp", is_founder=False, confidence=0.8,
                       grounding="website"),
    )
    ranked = score_leads([corp, founder], thesis)
    assert ranked[0].account.handle == "founder"
    assert ranked[1].score < ranked[0].score * 0.5


def test_stage_mismatch_halves_score() -> None:
    from scout.config import Thesis
    from scout.models import Account, Lead, LLMVerdict, Signal

    thesis = Thesis(weights={"bio_intent": 100.0}, target_stages=["launched"])
    on = Lead(account=Account(id="a", handle="a"), signals=[Signal(name="bio_intent", value=1.0)],
              llm=LLMVerdict(handle="a", is_founder=True, stage="launched", confidence=1.0,
                             grounding="website"))
    off = Lead(account=Account(id="b", handle="b"), signals=[Signal(name="bio_intent", value=1.0)],
               llm=LLMVerdict(handle="b", is_founder=True, stage="idea", confidence=1.0,
                              grounding="website"))
    ranked = score_leads([off, on], thesis)
    by = {x.account.handle: x for x in ranked}
    assert by["a"].score == 100.0
    assert by["b"].score == 50.0  # off-target stage × 0.5


def test_thesis_fit_scales_score() -> None:
    from scout.config import Thesis
    from scout.models import Account, Lead, LLMVerdict, Signal

    thesis = Thesis(weights={"bio_intent": 100.0}, target_stages=["launched"])

    def lead(handle: str, fit: float | None) -> Lead:
        return Lead(
            account=Account(id=handle, handle=handle),
            signals=[Signal(name="bio_intent", value=1.0)],
            llm=LLMVerdict(handle=handle, is_founder=True, stage="launched", grounding="website",
                           confidence=1.0, thesis_fit=fit),
        )

    by = {
        x.account.handle: x
        for x in score_leads([lead("a", 1.0), lead("b", 0.0), lead("c", None)], thesis)
    }
    assert by["a"].score == 100.0  # perfect fit keeps the score
    assert by["b"].score == 50.0   # fit 0.0 with default weight 0.5 halves it
    assert by["c"].score == 100.0  # legacy verdict without fit -> no fit step


def test_thesis_fit_weight_zero_disables_fit() -> None:
    from scout.config import SignalParams, Thesis
    from scout.models import Account, Lead, LLMVerdict, Signal

    thesis = Thesis(
        weights={"bio_intent": 100.0},
        signal_params=SignalParams(thesis_fit_weight=0.0),
    )
    lead = Lead(
        account=Account(id="a", handle="a"),
        signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="a", is_founder=True, confidence=1.0, thesis_fit=0.0,
                       grounding="website"),
    )
    assert score_leads([lead], thesis)[0].score == 100.0


def test_value_add_fit_informational_at_default_weight() -> None:
    """Default value_add_weight is 0 — the dimension never moves the score."""
    from scout.score import score_breakdown

    thesis = Thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a",
        signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="a", is_founder=True, confidence=1.0, value_add_fit=0.0,
                       grounding="website"),
    )
    assert score_leads([lead], thesis)[0].score == 100.0
    assert not any("value-add" in desc for desc, _ in score_breakdown(lead, thesis))


def test_value_add_weight_scales_score() -> None:
    from scout.config import SignalParams

    thesis = Thesis(
        weights={"bio_intent": 100.0},
        signal_params=SignalParams(value_add_weight=0.5),
    )

    def lead(handle: str, fit: float | None) -> Lead:
        return make_lead(
            handle,
            signals=[Signal(name="bio_intent", value=1.0)],
            llm=LLMVerdict(handle=handle, is_founder=True, confidence=1.0, grounding="website",
                           value_add_fit=fit),
        )

    by = {
        x.account.handle: x
        for x in score_leads([lead("a", 1.0), lead("b", 0.0), lead("c", None)], thesis)
    }
    assert by["a"].score == 100.0  # firm's value-add fully applies — score kept
    assert by["b"].score == 50.0   # nothing the firm offers helps — halved at w=0.5
    assert by["c"].score == 100.0  # legacy verdict without value_add_fit → no step


def test_signal_params_drive_traction() -> None:
    from datetime import datetime, timezone
    from scout.config import SignalParams, Thesis
    from scout.models import Account, Tweet
    from scout.signals.heuristics import run_heuristics

    account = Account(id="a", handle="a", followers=100)
    tweet = Tweet(id="t", account_id="a", text="launching today", created_at=datetime.now(timezone.utc),
                  likes=4, retweets=0, replies=0)  # ratio 0.04
    strict = Thesis(launch_phrases=["launching"], signal_params=SignalParams(traction_floor=0.05))
    loose = Thesis(launch_phrases=["launching"], signal_params=SignalParams(traction_floor=0.01))
    strict_sig = next(s for s in run_heuristics(account, [tweet], strict)[0] if s.name == "launch_traction")
    loose_sig = next(s for s in run_heuristics(account, [tweet], loose)[0] if s.name == "launch_traction")
    assert strict_sig.value == 0.0  # 0.04 below the 0.05 floor
    assert loose_sig.value > 0.0    # above the lowered floor


# --- grounding penalty ---------------------------------------------------------


def _grounding_lead(**verdict_overrides) -> Lead:
    verdict = LLMVerdict(
        handle="x", account_type="founder", is_founder=True, stage="launched",
        thesis_fit=None, confidence=1.0, **verdict_overrides,
    )
    return Lead(
        account=Account(id="1", handle="x"),
        signals=[Signal(name="bio_intent", value=1.0)],
        llm=verdict,
    )


def _has_penalty_step(lead: Lead, thesis: Thesis) -> bool:
    return any("product unverified" in desc
               for desc, _ in score_breakdown(lead, thesis))


def test_ungrounded_penalty_fires_for_none_and_bio() -> None:
    thesis = Thesis(weights={"bio_intent": 20.0})
    assert _has_penalty_step(_grounding_lead(grounding="none"), thesis)
    assert _has_penalty_step(_grounding_lead(grounding="bio"), thesis)


def test_grounded_and_audited_leads_exempt() -> None:
    thesis = Thesis(weights={"bio_intent": 20.0})
    assert not _has_penalty_step(_grounding_lead(grounding="website"), thesis)
    assert not _has_penalty_step(_grounding_lead(grounding="tweets"), thesis)
    # Audit outcome outranks grounding in both directions:
    assert not _has_penalty_step(
        _grounding_lead(grounding="bio", verification="confirmed"), thesis)
    assert not _has_penalty_step(
        _grounding_lead(grounding="none", verification="corrected"), thesis)
    assert _has_penalty_step(
        _grounding_lead(grounding="website", verification="unverifiable"), thesis)


def test_ungrounded_multiplier_math_and_knob() -> None:
    thesis = Thesis(weights={"bio_intent": 20.0})
    grounded = _grounding_lead(grounding="website")
    ungrounded = _grounding_lead(grounding="none")
    base = score_breakdown(grounded, thesis)[-1][1]
    penalized = score_breakdown(ungrounded, thesis)[-1][1]
    assert penalized == pytest.approx(base * 0.6)
    thesis.signal_params.ungrounded_multiplier = 1.0  # knob off
    assert score_breakdown(ungrounded, thesis)[-1][1] == pytest.approx(base)
