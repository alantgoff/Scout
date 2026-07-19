"""Composite math: weighted mean, None-score exclusion, the thin-wrapper
cap rule, stepped output. Pure functions — no store, no network."""

from __future__ import annotations

import pytest

from scout.diligence.composite import (
    COMMODITY_CAP,
    DEFAULT_WEIGHTS,
    diligence_breakdown,
    hard_flags,
)
from scout.diligence.schema import DIMENSION_KEYS, DimensionFinding


def finding(key: str, score: float | None, payload: dict | None = None) -> DimensionFinding:
    return DimensionFinding(key=key, score=score, confidence=0.8, payload=payload or {})


def test_weighted_mean_math() -> None:
    findings = [
        finding("data_flywheel", 8.0),  # weight 20
        finding("deployment_model", 4.0),  # weight 8
    ]
    steps = diligence_breakdown(findings)
    expected = 10.0 * (8.0 * 20 + 4.0 * 8) / (20 + 8)
    assert steps[-1][1] == pytest.approx(expected)
    assert "2/2 scored" in steps[0][0]


def test_none_scores_excluded_from_numerator_and_denominator() -> None:
    findings = [
        finding("data_flywheel", 8.0),
        finding("market_size", None),  # failed dimension — no drag on the score
    ]
    steps = diligence_breakdown(findings)
    assert steps[-1][1] == pytest.approx(80.0)  # only the scored dimension counts
    assert "1/2 scored" in steps[0][0]


def test_all_none_returns_empty() -> None:
    assert diligence_breakdown([finding("data_flywheel", None)]) == []
    assert diligence_breakdown([]) == []


def test_custom_weights_merge_over_defaults() -> None:
    findings = [finding("data_flywheel", 10.0), finding("competition", 0.0)]
    # Crank competition weight so it dominates.
    steps = diligence_breakdown(findings, {"competition": 80.0})
    expected = 10.0 * (10.0 * 20 + 0.0 * 80) / (20 + 80)
    assert steps[-1][1] == pytest.approx(expected)


def test_zero_weight_removes_dimension() -> None:
    findings = [finding("data_flywheel", 2.0), finding("competition", 10.0)]
    steps = diligence_breakdown(findings, {"competition": 0.0})
    assert steps[-1][1] == pytest.approx(20.0)  # competition ignored entirely


def test_default_weights_cover_all_dimensions() -> None:
    assert set(DEFAULT_WEIGHTS) == set(DIMENSION_KEYS)


# --- thin-wrapper hard-flag rule ---------------------------------------------


def thin_wrapper_findings(flywheel_score: float) -> list[DimensionFinding]:
    return [
        finding("wrapper_vs_proprietary", 9.0, {"classification": "thin_wrapper"}),
        finding("data_flywheel", flywheel_score),
        finding("founder_pedigree", 10.0),
        finding("market_size", 10.0),
    ]


def test_thin_wrapper_weak_flywheel_caps_composite() -> None:
    findings = thin_wrapper_findings(flywheel_score=2.0)
    assert hard_flags(findings) == ["commodity_risk"]
    steps = diligence_breakdown(findings)
    assert steps[-1][1] == COMMODITY_CAP
    assert "commodity" in steps[-1][0]
    assert len(steps) == 2  # weighted mean step + cap step


def test_no_cap_when_flywheel_strong() -> None:
    findings = thin_wrapper_findings(flywheel_score=7.0)
    assert hard_flags(findings) == []
    steps = diligence_breakdown(findings)
    assert len(steps) == 1


def test_no_cap_for_non_wrapper_classification() -> None:
    findings = [
        finding("wrapper_vs_proprietary", 9.0, {"classification": "proprietary_model"}),
        finding("data_flywheel", 1.0),
        finding("founder_pedigree", 10.0),
    ]
    assert hard_flags(findings) == []


def test_flag_applies_even_when_composite_already_low() -> None:
    findings = [
        finding("wrapper_vs_proprietary", 1.0, {"classification": "thin_wrapper"}),
        finding("data_flywheel", 1.0),
    ]
    assert hard_flags(findings) == ["commodity_risk"]
    steps = diligence_breakdown(findings)
    assert len(steps) == 1  # already under the cap — no cap step
    assert steps[-1][1] < COMMODITY_CAP
