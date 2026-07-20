"""Unit tests for score.py — the quality/fit/signals blend + multiplier chain."""

from __future__ import annotations

import pytest

from scout.config import SignalParams, Thesis
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.score import (
    apply_override,
    company_quality,
    quality_score,
    score_breakdown,
    score_leads,
    scorecard_score,
)

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


# --- manual overrides -----------------------------------------------------------


def _override_lead() -> Lead:
    return make_lead(
        "ada", signals=[Signal(name="bio_intent", value=1.0)],
        llm=LLMVerdict(
            handle="ada", account_type="founder", is_founder=True,
            stage="launched", grounding="website", thesis_fit=0.4,
            confidence=1.0, quality={"traction": 0.2},
            quality_reasons={"traction": "few users"},
        ),
    )


def test_apply_override_none_is_noop() -> None:
    from scout.score import apply_override

    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = _override_lead()
    base = score_breakdown(lead, thesis)[-1][1]
    lead.score = round(base, 1)
    assert apply_override(lead, None, thesis) is False
    assert lead.score == round(base, 1)


def test_apply_override_dims_and_fit_recompute_through_the_math() -> None:
    from scout.score import apply_override

    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = _override_lead()
    base = score_breakdown(lead, thesis)[-1][1]
    assert apply_override(
        lead,
        {"quality": {"traction": 1.0, "team": 0.8}, "fit": 0.9,
         "note": "insider info"},
        thesis,
    ) is True
    # Overridden values live IN the verdict, with a manual citation…
    assert lead.llm.quality["team"] == 0.8
    assert lead.llm.thesis_fit == 0.9
    assert lead.llm.quality_reasons["traction"] == "manual — insider info"
    assert lead.llm.fit_reason == "manual — insider info"
    # …and the score was recomputed through the normal breakdown.
    assert lead.score == pytest.approx(score_breakdown(lead, thesis)[-1][1], abs=0.06)
    assert lead.score > base


def test_apply_override_pinned_score_wins_and_is_clamped() -> None:
    from scout.score import apply_override

    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = _override_lead()
    assert apply_override(lead, {"quality": {}, "score": 150.0}, thesis) is True
    assert lead.score == 100.0


def test_apply_override_without_verdict_only_pins() -> None:
    from scout.score import apply_override

    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead("bare", signals=[Signal(name="bio_intent", value=1.0)])
    # dim/fit overrides have no verdict to live on -> nothing applies…
    assert apply_override(lead, {"quality": {"team": 0.9}, "fit": 0.8}, thesis) is False
    # …but a pinned score still does.
    assert apply_override(lead, {"quality": {"team": 0.9}, "score": 55.0}, thesis) is True
    assert lead.score == 55.0


def test_score_breakdown_manual_step_is_final() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = _override_lead()
    steps = score_breakdown(lead, thesis, manual_score=42.0)
    assert steps[-1] == ("manual score override (set by you)", 42.0)
    # Without the argument the step is absent.
    assert "manual" not in score_breakdown(lead, thesis)[-1][0]


# --- the readiness scorecard ----------------------------------------------------


def make_scorecard_verdict(**overrides) -> LLMVerdict:
    base = dict(
        handle="sc", account_type="founder", is_founder=True, stage="launched",
        grounding="website", confidence=1.0, customer_type="b2b",
        # One criterion in each B2B section, so every section is present.
        scorecard={
            "prev_founder_experience": 3, "problem_urgency": 2,
            "tech_moat_ip": 3, "product_maturity": 1, "commercial_traction": 2,
        },
        scorecard_reasons={"prev_founder_experience": "sold DevCo"},
    )
    base.update(overrides)
    return LLMVerdict(**base)


def test_scorecard_single_criterion_maps_1_to_3_onto_0_to_100() -> None:
    thesis = make_thesis()
    for value, expected in [(1, 0.0), (2, 50.0), (3, 100.0)]:
        verdict = make_scorecard_verdict(
            scorecard={"prev_founder_experience": value})
        result, sections = scorecard_score(verdict, thesis)
        # Only one section present, so the total equals that section's score.
        assert result.total == pytest.approx(expected)
        founders = next(s for s in sections if s.key == "founders_captable")
        assert founders.score == pytest.approx(expected)
        assert founders.n_present == 1


def test_scorecard_all_threes_is_100_all_ones_is_0() -> None:
    from scout import rubric

    thesis = make_thesis()
    all_keys = [c.key for c in rubric.B2B_RUBRIC.criteria]
    hi = make_scorecard_verdict(scorecard={k: 3 for k in all_keys})
    lo = make_scorecard_verdict(scorecard={k: 1 for k in all_keys})
    assert scorecard_score(hi, thesis)[0].total == pytest.approx(100.0)
    assert scorecard_score(lo, thesis)[0].total == pytest.approx(0.0)


def test_scorecard_within_section_renormalizes_over_present_criteria() -> None:
    thesis = make_thesis()
    # founders: prev_founder_experience (w .20, score 3) + cap_table_risk
    # (w .10, score 1). avg = (3×.2 + 1×.1)/(.3) = 0.7/0.3 ≈ 2.333 → 66.7
    verdict = make_scorecard_verdict(
        scorecard={"prev_founder_experience": 3, "cap_table_risk": 1})
    _result, sections = scorecard_score(verdict, thesis)
    founders = next(s for s in sections if s.key == "founders_captable")
    assert founders.score == pytest.approx(100 * ((0.7 / 0.3) - 1) / 2)
    assert founders.n_present == 2 and founders.n_total == 6


def test_scorecard_renormalizes_over_present_sections() -> None:
    thesis = make_thesis()
    # Only two sections present: founders (weight 18, score 100) and market
    # (weight 23, score 50). total = (18×100 + 23×50)/(41).
    verdict = make_scorecard_verdict(
        scorecard={"prev_founder_experience": 3, "problem_urgency": 2})
    result, _sections = scorecard_score(verdict, thesis)
    assert result.total == pytest.approx((18 * 100 + 23 * 50) / 41)
    assert result.n_present_sections == 2
    assert result.n_sections == 5


def test_scorecard_thesis_section_weight_override() -> None:
    thesis = make_thesis()
    thesis.scorecard_weights["b2b"]["market"] = 0.0  # zero out market
    verdict = make_scorecard_verdict(
        scorecard={"prev_founder_experience": 3, "problem_urgency": 1})
    result, _ = scorecard_score(verdict, thesis)
    # market's weight is 0 → only founders (score 100) counts.
    assert result.total == pytest.approx(100.0)
    assert result.n_present_sections == 1


def test_scorecard_ignores_unknown_and_cross_rubric_keys() -> None:
    thesis = make_thesis()
    verdict = make_scorecard_verdict(
        customer_type="b2b",
        scorecard={"prev_founder_experience": 3,
                   "made_up_key": 3,        # not in any rubric
                   "retention_mechanics": 1})  # a B2C-only key
    result, sections = scorecard_score(verdict, thesis)
    founders = next(s for s in sections if s.key == "founders_captable")
    assert founders.n_present == 1  # only prev_founder_experience counted
    assert result.total == pytest.approx(100.0)


def test_scorecard_routing_b2c_uses_consumer_rubric() -> None:
    thesis = make_thesis()
    verdict = make_scorecard_verdict(
        customer_type="b2c",
        scorecard={"user_growth": 3, "retention_mechanics": 3})
    result, _ = scorecard_score(verdict, thesis)
    assert result.rubric_key == "b2c"
    assert result.rubric_label == "Consumer readiness"


def test_scorecard_none_customer_type_routes_to_b2b() -> None:
    thesis = make_thesis()
    verdict = make_scorecard_verdict(
        customer_type=None, scorecard={"prev_founder_experience": 3})
    result, _ = scorecard_score(verdict, thesis)
    assert result.rubric_key == "b2b"


def test_scorecard_band_assignment() -> None:
    thesis = make_thesis()
    # all-3 founders → 100 → strong
    strong = make_scorecard_verdict(scorecard={"prev_founder_experience": 3})
    assert scorecard_score(strong, thesis)[0].band == "strong"
    # avg 2 → 50 → weak
    weak = make_scorecard_verdict(scorecard={"prev_founder_experience": 2})
    assert scorecard_score(weak, thesis)[0].band == "weak"


def test_scorecard_none_for_empty_and_legacy_verdicts() -> None:
    thesis = make_thesis()
    assert scorecard_score(None, thesis) is None
    assert scorecard_score(make_scorecard_verdict(scorecard={}), thesis) is None
    legacy = make_quality_verdict()  # flat quality dims, no scorecard
    assert scorecard_score(legacy, thesis) is None


def test_company_quality_dispatch_scorecard_beats_legacy() -> None:
    thesis = make_thesis()
    # A verdict carrying BOTH shapes: scorecard wins.
    verdict = make_scorecard_verdict(
        scorecard={"prev_founder_experience": 3},   # → 100
        quality={"team": 0.2}, quality_reasons={"team": "x"})
    assert company_quality(verdict, thesis) == pytest.approx(100.0)
    # Legacy-only verdict falls back to quality_score.
    legacy = make_quality_verdict()
    assert company_quality(legacy, thesis) == pytest.approx(quality_score(legacy, thesis))
    assert company_quality(None, thesis) is None


def test_blend_uses_scorecard_as_quality_component() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],  # S = 100
        llm=make_scorecard_verdict(
            scorecard={"prev_founder_experience": 3},  # Q = 100
            thesis_fit=0.5),  # F = 50
    )
    # (0.45×100 + 0.35×50 + 0.20×100) / 1.0 = 45 + 17.5 + 20 = 82.5
    assert final(lead, thesis) == pytest.approx(82.5)


def test_score_breakdown_shows_scorecard_step() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "a", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_scorecard_verdict(scorecard={"prev_founder_experience": 3}),
    )
    descs = [d for d, _ in score_breakdown(lead, thesis)]
    assert any("scorecard:" in d and "Enterprise readiness" in d for d in descs)
    assert not any("dims evidence-backed" in d for d in descs)


def test_apply_override_sections_recompute_and_flag_manual() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    lead = make_lead(
        "ada", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_scorecard_verdict(scorecard={"prev_founder_experience": 1}),  # Q≈0
    )
    base = score_breakdown(lead, thesis)[-1][1]
    assert apply_override(lead, {"sections": {"founders_captable": 90}}, thesis) is True
    assert lead.llm.scorecard_manual["founders_captable"] == 90.0
    _result, sections = scorecard_score(lead.llm, thesis)
    founders = next(s for s in sections if s.key == "founders_captable")
    assert founders.manual is True and founders.score == pytest.approx(90.0)
    assert lead.score > base


def test_apply_override_section_can_add_absent_section() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    # A verdict with only founders scored; market has no evidence.
    lead = make_lead(
        "ada", signals=[Signal(name="bio_intent", value=1.0)],
        llm=make_scorecard_verdict(scorecard={"prev_founder_experience": 2}),
    )
    apply_override(lead, {"sections": {"market": 80}}, thesis)
    _result, sections = scorecard_score(lead.llm, thesis)
    market = next(s for s in sections if s.key == "market")
    assert market.manual is True and market.score == pytest.approx(80.0)


def test_apply_override_legacy_quality_inert_on_scorecard_verdict() -> None:
    thesis = make_thesis(weights={"bio_intent": 100.0})
    verdict = make_scorecard_verdict(scorecard={"prev_founder_experience": 3})
    lead = make_lead("ada", signals=[Signal(name="bio_intent", value=1.0)],
                     llm=verdict)
    q_before = company_quality(verdict, thesis)
    # Legacy dim override does NOT touch a scorecard verdict's quality…
    apply_override(lead, {"quality": {"team": 0.1}}, thesis)
    assert company_quality(lead.llm, thesis) == pytest.approx(q_before)
    # …but a section override does.
    apply_override(lead, {"sections": {"founders_captable": 10}}, thesis)
    assert company_quality(lead.llm, thesis) < q_before
