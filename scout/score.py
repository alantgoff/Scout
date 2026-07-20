"""Three-component lead score → 0–100: company QUALITY + thesis FIT +
X-signal momentum, blended, then a chain of trust multipliers.

The QUALITY component is the readiness scorecard (scout.rubric): criteria
scored 1–3 by the classifier roll up through sub-weighted section averages
into a 0–100 total with an interpretation band. Pre-scorecard cached
verdicts fall back to the legacy flat quality rubric — `company_quality`
is the dispatch point.

Manual overrides (`apply_override`) layer the investor's own scoring on top
at load time: adjusted scorecard sections / thesis fit re-enter the same
math, and a pinned final score wins outright.

Pure: no I/O, no network. Covered by unit tests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scout import rubric as rubric_mod
from scout.config import Thesis
from scout.models import Lead, LLMVerdict


class CriterionScore(BaseModel):
    """One rubric criterion as scored for a verdict (UI/memo rendering)."""

    key: str
    label: str
    purpose: str = ""  # what it measures (tooltip text)
    weight: float  # sub-weight within the section
    value: float | None = None  # clamped 1..3; None = no evidence (excluded)
    reason: str = ""


class SectionScore(BaseModel):
    """One rubric section: weighted average of its PRESENT criteria mapped
    to 0..100 via (avg − 1) / 2 × 100; None when nothing was scored. A
    manual override (verdict.scorecard_manual) replaces the computed score
    and can add a section that had no evidence."""

    key: str
    label: str
    weight: float  # effective section weight (thesis.scorecard_weights)
    score: float | None = None
    manual: bool = False
    n_present: int = 0
    n_total: int = 0
    criteria: list[CriterionScore] = Field(default_factory=list)


class ScorecardResult(BaseModel):
    """The scorecard rollup: 0–100 total renormalized over present
    sections, plus the interpretation band (rubric.band_for)."""

    rubric_key: str
    rubric_label: str
    total: float
    band: str
    n_present_sections: int
    n_sections: int


def scorecard_score(
    verdict: LLMVerdict | None, thesis: Thesis
) -> tuple[ScorecardResult, list[SectionScore]] | None:
    """Deterministic scorecard rollup for a verdict, or None when the
    verdict carries no scorecard at all (legacy cache / product unknown).

    customer_type routes the rubric (rubric.rubric_for). Per section:
    avg = Σ(clamp(value, 1, 3) × sub-weight) / Σ(sub-weights of PRESENT
    criteria), score = (avg − 1) / 2 × 100. Criterion keys outside the
    routed rubric are ignored (classifier-invented, or orphans from the
    other rubric after a customer_type correction). Section weights come
    from thesis.scorecard_weights (code default when a key is missing);
    the total renormalizes over present sections so thin evidence isn't
    punished — coverage is surfaced via n_present/n_total instead."""
    if verdict is None or (not verdict.scorecard and not verdict.scorecard_manual):
        return None
    rubric = rubric_mod.rubric_for(verdict.customer_type)
    weights = thesis.scorecard_weights.get(rubric.key, {})
    sections: list[SectionScore] = []
    for section in rubric.sections:
        criteria: list[CriterionScore] = []
        raw = 0.0
        denom = 0.0
        for criterion in section.criteria:
            value = verdict.scorecard.get(criterion.key)
            if value is not None:
                value = min(max(float(value), 1.0), 3.0)
                raw += value * criterion.weight
                denom += criterion.weight
            criteria.append(CriterionScore(
                key=criterion.key, label=criterion.label,
                purpose=criterion.purpose, weight=criterion.weight, value=value,
                reason=verdict.scorecard_reasons.get(criterion.key, ""),
            ))
        score = 100.0 * (raw / denom - 1.0) / 2.0 if denom > 0 else None
        manual = section.key in verdict.scorecard_manual
        if manual:
            score = min(max(float(verdict.scorecard_manual[section.key]), 0.0), 100.0)
        sections.append(SectionScore(
            key=section.key, label=section.label,
            weight=weights.get(section.key, section.weight),
            score=score, manual=manual,
            n_present=sum(1 for c in criteria if c.value is not None),
            n_total=len(section.criteria), criteria=criteria,
        ))
    present = [s for s in sections if s.score is not None and s.weight > 0]
    wsum = sum(s.weight for s in present)
    if not present or wsum <= 0:
        return None
    total = sum(s.score * s.weight for s in present) / wsum
    result = ScorecardResult(
        rubric_key=rubric.key, rubric_label=rubric.label,
        total=total, band=rubric_mod.band_for(total),
        n_present_sections=len(present), n_sections=len(sections),
    )
    return result, sections


def company_quality(verdict: LLMVerdict | None, thesis: Thesis) -> float | None:
    """THE quality component everywhere (blend, sorts, Q columns, CSV):
    scorecard total when the verdict carries one, legacy quality_score()
    for pre-scorecard cached verdicts, None when nothing is scorable."""
    scorecard = scorecard_score(verdict, thesis)
    if scorecard is not None:
        return scorecard[0].total
    return quality_score(verdict, thesis)


def quality_score(verdict: LLMVerdict | None, thesis: Thesis) -> float | None:
    """LEGACY (v7) flat quality rubric: 100 × Σ(dim × weight) / Σ(weights
    of PRESENT dims). Only reached for pre-scorecard cached verdicts —
    new verdicts score via scorecard_score. Renormalizes over the
    dimensions the verdict actually evidences — a startup with 3 scorable
    dims isn't punished for the other 3 being unknowable. Dim values
    clamped to 0..1; keys the classifier invents (not in
    thesis.quality_weights) are ignored. None when nothing is scorable
    (no verdict / no dims / all weights 0)."""
    if verdict is None or not verdict.quality:
        return None
    raw = 0.0
    denom = 0.0
    for key, value in verdict.quality.items():
        weight = thesis.quality_weights.get(key, 0.0)
        if weight <= 0:
            continue
        raw += min(max(value, 0.0), 1.0) * weight
        denom += weight
    if denom <= 0:
        return None
    return 100.0 * raw / denom


def score_components(lead: Lead, thesis: Thesis) -> dict:
    """The blend inputs, each 0–100 or None when absent:
    {'signals', 'quality', 'fit', 'signals_raw', 'signals_denom'}, plus
    'scorecard' — the (ScorecardResult, [SectionScore]) detail behind the
    quality component, or None for legacy/heuristics-only leads."""
    denom = sum(thesis.weights.values())
    for signal in lead.signals:
        signal.weight = thesis.weights.get(signal.name, 0.0)
    raw = sum(s.value * s.weight for s in lead.signals)
    verdict = lead.llm
    fit = None
    if verdict is not None and verdict.thesis_fit is not None:
        fit = 100.0 * min(max(verdict.thesis_fit, 0.0), 1.0)
    scorecard = scorecard_score(verdict, thesis)
    return {
        "signals": 100.0 * raw / denom if denom else None,
        "quality": scorecard[0].total if scorecard else quality_score(verdict, thesis),
        "fit": fit,
        "scorecard": scorecard,
        "signals_raw": raw,
        "signals_denom": denom,
    }


def score_leads(leads: list[Lead], thesis: Thesis) -> list[Lead]:
    """Fill signal weights from the thesis, compute scores, rank descending.

    The full calculation (mirrored by `score_breakdown` for the UI):
      signals = 100 × Σ(value_i × weight_i) / Σ(all thesis weights)
      quality = scorecard rollup (scorecard_score: 1–3 criteria → sub-
                weighted section averages → 0–100 over present sections);
                legacy verdicts: quality_score over the flat v7 dims
      fit     = 100 × llm.thesis_fit
      base    = Σ(score_weight_c × component_c) / Σ(weights of PRESENT
                components)   — the 45/35/20 blend, renormalized
      × confidence            when an LLM verdict is attached
      × 0.2                   when the verdict says not-a-founder
      × stage multiplier      when the verdict's stage is off-target
      × value-add multiplier  (1-w) + w × value_add_fit — off by default
      × ungrounded multiplier when the product claim never traced to
                              evidence (audit "unverifiable", or unaudited
                              with grounding none/bio)
    A lead with no verdict scores on signals alone (demo & heuristics-only).
    """
    for lead in leads:
        steps = score_breakdown(lead, thesis)
        lead.score = round(steps[-1][1], 1)

    ranked = sorted(leads, key=lambda lead: (-lead.score, lead.account.handle))
    for i, lead in enumerate(ranked, start=1):
        lead.rank = i
    return ranked


def apply_override(lead: Lead, override: dict | None, thesis: Thesis) -> bool:
    """Layer one manual-score row (store.all_overrides) onto a loaded lead,
    in place. Returns True when anything applied.

    Adjusted scorecard sections (section key → 0..100) land in
    verdict.scorecard_manual and replace the computed section scores inside
    the normal rollup; legacy quality-dim overrides still apply, but only
    to pre-scorecard verdicts (on a scorecard verdict the flat dims aren't
    read, so they'd be inert). Thesis fit is written into the verdict (with
    a "manual" reason) and the score is recomputed through the normal math,
    so bars, chips, and the step-by-step breakdown all reflect the
    investor's numbers. A pinned `score` then wins outright. Section/dim/fit
    overrides need a verdict to live on; without one only the pinned score
    applies."""
    if not override:
        return False
    applied = False
    verdict = lead.llm
    note = (override.get("note") or "").strip()
    reason = f"manual — {note}" if note else "manual override"
    if verdict is not None:
        for key, value in (override.get("sections") or {}).items():
            verdict.scorecard_manual[key] = min(max(float(value), 0.0), 100.0)
            applied = True
        if not verdict.scorecard:
            for key, value in (override.get("quality") or {}).items():
                verdict.quality[key] = min(max(float(value), 0.0), 1.0)
                verdict.quality_reasons[key] = reason
                applied = True
        if override.get("fit") is not None:
            verdict.thesis_fit = min(max(float(override["fit"]), 0.0), 1.0)
            verdict.fit_reason = reason
            applied = True
    if applied:
        lead.score = round(score_breakdown(lead, thesis)[-1][1], 1)
    if override.get("score") is not None:
        lead.score = round(min(max(float(override["score"]), 0.0), 100.0), 1)
        applied = True
    return applied


def score_breakdown(
    lead: Lead, thesis: Thesis, manual_score: float | None = None
) -> list[tuple[str, float]]:
    """The score calculation as (step description, running score) pairs.

    Single source of truth for the math — score_leads applies it; the UI
    renders it so every sub-score is inspectable. The last step's value IS
    the final score. Component lines carry the component's own value; the
    blend line restores the running-score semantic. Also fills Signal.weight
    as a side effect (idempotent). `manual_score` (a pinned override) appends
    a final step so the displayed math ends at the displayed number."""
    def _finish(steps: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if manual_score is not None:
            steps.append(("manual score override (set by you)",
                          min(max(float(manual_score), 0.0), 100.0)))
        return steps

    parts = score_components(lead, thesis)
    signals_score = parts["signals"] if parts["signals"] is not None else 0.0
    steps: list[tuple[str, float]] = [
        (f"signals: 100 × {parts['signals_raw']:g} / {parts['signals_denom']:g}",
         signals_score),
    ]
    if lead.llm is None:
        # Demo / heuristics-only: signals are the whole story.
        return _finish(steps)
    verdict = lead.llm
    params = thesis.signal_params

    if parts["quality"] is not None:
        if parts["scorecard"] is not None:
            result, _sections = parts["scorecard"]
            steps.append(
                (f"scorecard: {result.total:.0f} ({result.rubric_label} rubric, "
                 f"{result.n_present_sections}/{result.n_sections} sections "
                 "evidence-backed)",
                 parts["quality"])
            )
        else:
            # Legacy (v7) flat-rubric verdict from the pre-scorecard cache.
            n_dims = sum(
                1 for k in verdict.quality if thesis.quality_weights.get(k, 0) > 0
            )
            n_total = sum(1 for w in thesis.quality_weights.values() if w > 0)
            steps.append(
                (f"quality: {parts['quality']:.0f} ({n_dims}/{n_total} dims "
                 "evidence-backed)",
                 parts["quality"])
            )
    if parts["fit"] is not None:
        steps.append((f"thesis fit: 100 × {verdict.thesis_fit:.2f}", parts["fit"]))

    # The blend — weighted mean over the components this lead actually has.
    weighted = [
        (params.score_weight_quality, parts["quality"], "Q"),
        (params.score_weight_fit, parts["fit"], "F"),
        (params.score_weight_signals, parts["signals"], "S"),
    ]
    present = [(w, value, tag) for w, value, tag in weighted if value is not None]
    wsum = sum(w for w, _v, _t in present)
    if parts["quality"] is None and parts["fit"] is None:
        # Verdict carries no blendable dimensions (legacy cache) — signals
        # remain the base, exactly like the pre-blend behavior.
        score = signals_score
    elif wsum <= 0:
        score = signals_score
        steps.append(("blend weights all 0 — signals only", score))
    else:
        score = sum(w * v for w, v, _t in present) / wsum
        terms = " + ".join(f"{w:g}×{tag} {v:.0f}" for w, v, tag in present)
        steps.append((f"blend: ({terms}) / {wsum:g}", score))

    score *= verdict.confidence
    steps.append((f"× Claude confidence {verdict.confidence:.2f}", score))
    if not verdict.is_founder:
        score *= 0.2
        steps.append(("× 0.2 (classified not-a-founder)", score))
    if (
        verdict.stage
        and thesis.target_stages
        and verdict.stage not in thesis.target_stages
    ):
        multiplier = params.stage_mismatch_multiplier
        score *= multiplier
        steps.append(
            (
                f"× {multiplier:g} (stage '{verdict.stage}' outside "
                f"target {', '.join(thesis.target_stages)})",
                score,
            )
        )
    # Value-add fit only enters the math when opted into (weight > 0) —
    # at the default 0 it stays a purely informational dimension.
    if verdict.value_add_fit is not None and params.value_add_weight > 0:
        w = params.value_add_weight
        fit = min(max(verdict.value_add_fit, 0.0), 1.0)
        multiplier = (1.0 - w) + w * fit
        score *= multiplier
        steps.append(
            (f"× {multiplier:.2f} (value-add fit {fit:.2f}, weight {w:g})", score)
        )
    # Grounding penalty: a product claim that never traced to evidence
    # must not float on confidence alone (the Raindrop failure mode).
    # Audited leads answer for themselves ("unverifiable" is penalized,
    # confirmed/corrected are exempt); unaudited leads are penalized only
    # when their own grounding is none/bio.
    ungrounded = (
        verdict.verification == "unverifiable"
        if verdict.verification is not None
        else verdict.grounding in (None, "none", "bio")
    )
    if ungrounded:
        multiplier = params.ungrounded_multiplier
        score *= multiplier
        basis = (
            "audit: unverifiable"
            if verdict.verification == "unverifiable"
            else f"grounding: {verdict.grounding or 'none'}"
        )
        steps.append(
            (f"× {multiplier:g} (product unverified — {basis})", score)
        )
    return _finish(steps)
