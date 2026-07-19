"""Diligence Score composite — pure weighted math, mirrors scout.score.

`diligence_breakdown` is the single source of truth for the composite the
same way `score_breakdown` is for the screening score: the pipeline takes
its last value, the UI renders every step. 0-10 dimension scores ->
weighted mean -> × 10 -> 0-100. Dimensions with score=None (failed /
insufficient data) are excluded from BOTH numerator and denominator — a
gap lowers coverage, never the score itself.

No I/O and no network — fully unit-testable.
"""

from __future__ import annotations

from scout.diligence.schema import DimensionFinding

# Defaults emphasize the thesis dimensions; thesis.yaml `diligence_weights`
# overrides per key (weights are relative, like the screening weights).
DEFAULT_WEIGHTS: dict[str, float] = {
    "data_flywheel": 20.0,
    "defensibility_vs_labs": 18.0,
    "founder_pedigree": 15.0,
    "technical_differentiation": 12.0,
    "wrapper_vs_proprietary": 10.0,
    "market_size": 10.0,
    "competition": 9.0,
    "deployment_model": 8.0,
}

# Hard-flag rule: a thin wrapper with no data story is a commodity — the
# composite is capped no matter how well the other dimensions read.
COMMODITY_CAP = 40.0
COMMODITY_FLYWHEEL_FLOOR = 4.0


def _find(findings: list[DimensionFinding], key: str) -> DimensionFinding | None:
    return next((f for f in findings if f.key == key), None)


def hard_flags(findings: list[DimensionFinding]) -> list[str]:
    """Structural red flags computed from the findings (currently one rule:
    thin_wrapper + weak data flywheel -> commodity_risk)."""
    wrapper = _find(findings, "wrapper_vs_proprietary")
    flywheel = _find(findings, "data_flywheel")
    if (
        wrapper is not None
        and wrapper.payload.get("classification") == "thin_wrapper"
        and flywheel is not None
        and flywheel.score is not None
        and flywheel.score < COMMODITY_FLYWHEEL_FLOOR
    ):
        return ["commodity_risk"]
    return []


def diligence_breakdown(
    findings: list[DimensionFinding], weights: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """The composite calculation as (step description, running value) pairs.

    Empty list when no dimension produced a score (composite is then None).
    Partial thesis weights merge over the defaults; a key explicitly set to
    0 removes that dimension from the composite.
    """
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    scored = [
        f for f in findings
        if f.score is not None and merged.get(f.key, 0.0) > 0
    ]
    if not scored:
        return []
    numerator = sum(f.score * merged[f.key] for f in scored)  # type: ignore[operator]
    denominator = sum(merged[f.key] for f in scored)
    composite = 10.0 * numerator / denominator
    steps = [
        (
            f"weighted mean of {len(scored)}/{len(findings)} scored dimensions × 10",
            composite,
        )
    ]
    if hard_flags(findings) and composite > COMMODITY_CAP:
        steps.append(
            (
                f"capped at {COMMODITY_CAP:g} (thin wrapper + weak data "
                "flywheel — commodity risk)",
                COMMODITY_CAP,
            )
        )
    return steps
