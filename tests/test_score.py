"""Unit tests for score.py — the quality/fit/signals blend + multiplier chain."""

from __future__ import annotations

import pytest

from scout.config import SignalParams, Thesis
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.score import quality_score, score_breakdown, score_leads

# Spec §3 defaults: total weight 100, so contributions read as points.
DEFAULT_WEIGHTS: dict[str, float] = {
    "bio_intent": 25,
    "smart_money_follow": 25,
    "departure_signal": 20,
    "launch_traction": 15,
    "builder_evidence": 15,
}


def make_thesis(weights: dict[str, float] | None = None, **kw) -> Thesis:
    return Thesis(weights=DEFAULT_WEIGHTS if weights is None else weights, **kw)


def make_lead(
    handle: str,
    signals: list[Signal] | None = None,
    llm: LLMVerdict | None = None,
) -> Lead:
    account = Account(id=handle, handle=handle)
    return Lead(account=account, signals=signals or [], llm=llm)


def final(lead: Lead, thesis: Thesis) -> float:
    return score_breakdown(lead, thesis)[-1][1]


# --- signals component (unchanged semantics) ----------------------------------


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
    assert lead.score == 25.0


def test_empty_weights_scores_zero_without_zerodivision() -> None:
    lead = make_lead("anyone", signals=[Signal(name="bio_intent", value=1.0)])
    ranked = score_leads([lead], make_thesis(weights={}))
    assert ranked[0].score == 0.0
    assert ranked[0].rank == 1


def test_no_verdict_is_signals_only() -> None:
    """Demo / heuristics-only leads: one signals step, nothing else."""
    lead = make_lead("plain", signals=[Signal(name="bio_intent", value=1.0)])
    steps = score_breakdown(lead, make_thesis())
    assert len(steps) == 1
    assert steps[0][1] == 25.0


# --- quality_score -------------------------------------------------------------


def make_quality_verdict(**overrides) -> LLMVerdict:
    base = dict(
        handle="q", account_type="founder", is_founder=True, stage="launched",
        grounding="website", confidence=1.0,
        customer_type="b2b",
        quality={"team": 0.8, "tech_product": 0.6, "traction": 0.4},
        quality_reasons={"team": "prior exit", "tech_product": "live product",
                         "traction": "3 logos on site"},
    )
    base.update(overrides)
    return LLMVerdict(**base)


def test_quality_score_renormalizes_present_dims() -> None:
    thesis = make_thesis()  # default quality_weights: team 20, tech 20, traction 20...
    verdict = make_quality_verdict()
    # (0.8×20 + 0.6×20 + 0.4×20) / 60 = 0.6 → 60
    assert quality_score(verdict, thesis) == pytest.approx(60.0)


def test_quality_score_ignores_unknown_keys_and_clamps() -> None:
    thesis = make_thesis()
    verdict = make_quality_verdict(quality={"team": 1.7, "moat_madeup": 0.9})
    # moat_madeup has no weight → ignored; team clamps to 1.0 → 100
    assert quality_score(verdict, thesis) == pytest.approx(100.0)


def test_quality_score_none_when_no_dims_or_no_verdict() -> None:
    thesis = make_thesis()
    assert quality_score(None, thesis) is None
    assert quality_score(make_quality_verdict(quality={}), thesis) is None
    zero_thesis = make_thesis(quality_weights={"team": 0.0})
    assert quality_score(make_quality_verdict(), zero_thesis) is None


# --- the blend -------------------------------------------------------------------


def test_blend_all_components_45_35_20() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],  # S = 100
        llm=make_quality_verdict(thesis_fit=0.5),  # Q = 60, F = 50
    )
    # (0.45×60 + 0.35×50 + 0.20×100) / 1.0 = 27 + 17.5 + 20 = 64.5
    assert final(lead, thesis) == pytest.approx(64.5)


def test_blend_renormalizes_missing_quality() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(quality={}, quality_reasons={}, thesis_fit=1.0),
    )
    # (0.35×100 + 0.20×100) / 0.55 = 100
    assert final(lead, thesis) == pytest.approx(100.0)
    lead2 = make_lead(
        "b", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(quality={}, quality_reasons={}, thesis_fit=0.0),
    )
    # (0.35×0 + 0.20×100) / 0.55 ≈ 36.36
    assert final(lead2, thesis) == pytest.approx(100 * 0.20 / 0.55)


def test_blend_renormalizes_missing_fit() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(thesis_fit=None),  # Q=60, S=100, no F
    )
    # (0.45×60 + 0.20×100) / 0.65 ≈ 72.3
    assert final(lead, thesis) == pytest.approx((0.45 * 60 + 0.20 * 100) / 0.65)


def test_blend_verdict_without_quality_or_fit_stays_signals() -> None:
    """Legacy cached verdicts (no quality, no fit) keep the signals base."""
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="a", is_founder=True, confidence=1.0,
                       grounding="website"),
    )
    assert final(lead, thesis) == pytest.approx(100.0)
    assert not any("blend" in d for d, _ in score_breakdown(lead, thesis))


def test_blend_weight_knobs() -> None:
    thesis = make_thesis(
        weights={"bio_intent": 100.0},
        signal_params=SignalParams(score_weight_quality=1.0,
                                   score_weight_fit=0.0,
                                   score_weight_signals=0.0),
    )
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(thesis_fit=0.1),
    )
    assert final(lead, thesis) == pytest.approx(60.0)  # quality only


def test_blend_all_weights_zero_falls_back_to_signals() -> None:
    thesis = make_thesis(
        weights={"bio_intent": 100.0},
        signal_params=SignalParams(score_weight_quality=0.0,
                                   score_weight_fit=0.0,
                                   score_weight_signals=0.0),
    )
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(thesis_fit=0.5),
    )
    steps = score_breakdown(lead, thesis)
    assert any("blend weights all 0" in d for d, _ in steps)
    assert final(lead, thesis) == pytest.approx(100.0)


def test_fit_is_a_component_not_a_multiplier() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(thesis_fit=0.0),
    )
    steps = [d for d, _ in score_breakdown(lead, thesis)]
    assert any(d.startswith("thesis fit:") for d in steps)
    assert not any("(thesis fit" in d and d.startswith("×") for d in steps)


# --- multiplier chain after the blend -------------------------------------------


def test_confidence_multiplies_blended_base() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_quality_verdict(thesis_fit=0.5, confidence=0.5),
    )
    assert final(lead, thesis) == pytest.approx(64.5 * 0.5)


def test_not_founder_verdict_slashes_score() -> None:
    thesis = make_thesis()
    founder = make_lead(
        "founder", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="founder", is_founder=True, confidence=0.8,
                       grounding="website"),
    )
    corp = make_lead(
        "corp", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="corp", is_founder=False, confidence=0.8,
                       grounding="website"),
    )
    ranked = score_leads([corp, founder], thesis)
    assert ranked[0].account.handle == "founder"
    assert ranked[1].score < ranked[0].score * 0.5


def test_stage_mismatch_halves_score() -> None:
    thesis = Thesis(weights={"bio_intent": 100.0}, target_stages=["launched"])
    on = make_lead("a", signals=[Signal(name="bio_intent", value=1.0)],
                   llm=LLMVerdict(handle="a", is_founder=True, stage="launched",
                                  confidence=1.0, grounding="website"))
    off = make_lead("b", signals=[Signal(name="bio_intent", value=1.0)],
                    llm=LLMVerdict(handle="b", is_founder=True, stage="idea",
                                   confidence=1.0, grounding="website"))
    ranked = score_leads([off, on], thesis)
    by = {x.account.handle: x for x in ranked}
    assert by["a"].score == 100.0
    assert by["b"].score == 50.0


def test_value_add_fit_informational_at_default_weight() -> None:
    thesis = Thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(handle="a", is_founder=True, confidence=1.0,
                       value_add_fit=0.0, grounding="website"),
    )
    assert score_leads([lead], thesis)[0].score == 100.0
    assert not any("value-add" in d for d, _ in score_breakdown(lead, thesis))


def test_value_add_weight_scales_score() -> None:
    thesis = Thesis(
        weights={"bio_intent": 100.0},
        signal_params=SignalParams(value_add_weight=0.5),
    )

    def lead(handle: str, fit: float | None) -> Lead:
        return make_lead(
            handle, signals=[Signal(name="bio_intent", value=1.0)],
            llm=LLMVerdict(handle=handle, is_founder=True, confidence=1.0,
                           grounding="website", value_add_fit=fit),
        )

    by = {
        x.account.handle: x
        for x in score_leads([lead("a", 1.0), lead("b", 0.0), lead("c", None)], thesis)
    }
    assert by["a"].score == 100.0  # perfect value-add fit keeps the score
    assert by["b"].score == 50.0   # 0.0 with weight 0.5 halves it
    assert by["c"].score == 100.0  # absent → no step


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


# --- the pedigree guard, end to end ---------------------------------------------


def test_stealth_pedigree_founder_sinks_despite_team_score() -> None:
    """The Raindrop guard: a pedigree-only stealth lead (team evidence but no
    product) must land in single digits even with a strong team dim."""
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "stealthy", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(
            handle="stealthy", account_type="founder", is_founder=True,
            stage="stealth", customer_type=None, grounding="none",
            quality={"team": 0.9}, quality_reasons={"team": "ex-Apple staff eng"},
            thesis_fit=0.2, confidence=0.3,
        ),
    )
    # blend: (0.45×90 + 0.35×20 + 0.20×100)/1.0 = 67.5 → ×0.3 conf → ×0.6
    # ungrounded ≈ 12.2 — an order of magnitude below a verified fit lead.
    assert final(lead, thesis) == pytest.approx(67.5 * 0.3 * 0.6)
    assert final(lead, thesis) < 15
