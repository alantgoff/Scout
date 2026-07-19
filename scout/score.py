"""Weighted signal aggregation → 0–100 lead score (spec §4).

Pure: no I/O, no network. Covered by a unit test.
"""

from __future__ import annotations

from scout.config import Thesis
from scout.models import Lead


def score_leads(leads: list[Lead], thesis: Thesis) -> list[Lead]:
    """Fill signal weights from the thesis, compute scores, rank descending.

    The full calculation (mirrored by `score_breakdown` for the UI):
      base       = 100 × Σ(value_i × weight_i) / Σ(all thesis weights)
      × confidence        when an LLM verdict is attached
      × 0.2               when the verdict says not-a-founder (account_type
                          "other") — a corporate account or commentator must
                          never outrank a verified founder
      × stage multiplier  when the verdict's stage is outside
                          thesis.target_stages (signal_params.stage_mismatch_multiplier)
      × fit multiplier    when the verdict carries thesis_fit (0..1):
                          (1 - w) + w × fit, w = signal_params.thesis_fit_weight —
                          a perfect fit keeps the score, an off-thesis lead is
                          scaled down but never to zero on fit alone
      × value-add multiplier  same shape for value_add_fit (would the firm's
                          value-add accelerate this startup?), w =
                          signal_params.value_add_weight — default 0, so the
                          dimension is informational unless opted into
    """
    for lead in leads:
        steps = score_breakdown(lead, thesis)
        lead.score = round(steps[-1][1], 1)

    ranked = sorted(leads, key=lambda lead: (-lead.score, lead.account.handle))
    for i, lead in enumerate(ranked, start=1):
        lead.rank = i
    return ranked


def score_breakdown(lead: Lead, thesis: Thesis) -> list[tuple[str, float]]:
    """The score calculation as (step description, running score) pairs.

    Single source of truth for the math — score_leads applies it; the UI
    renders it so every sub-score is inspectable. Also fills Signal.weight
    as a side effect (idempotent).
    """
    denom = sum(thesis.weights.values())
    for signal in lead.signals:
        signal.weight = thesis.weights.get(signal.name, 0.0)
    raw = sum(s.value * s.weight for s in lead.signals)
    score = 100 * raw / denom if denom else 0.0
    steps = [
        (f"signals: 100 × {raw:g} / {denom:g}", score),
    ]
    if lead.llm is not None:
        score *= lead.llm.confidence
        steps.append((f"× Claude confidence {lead.llm.confidence:.2f}", score))
        if not lead.llm.is_founder:
            score *= 0.2
            steps.append(("× 0.2 (classified not-a-founder)", score))
        if (
            lead.llm.stage
            and thesis.target_stages
            and lead.llm.stage not in thesis.target_stages
        ):
            multiplier = thesis.signal_params.stage_mismatch_multiplier
            score *= multiplier
            steps.append(
                (
                    f"× {multiplier:g} (stage '{lead.llm.stage}' outside "
                    f"target {', '.join(thesis.target_stages)})",
                    score,
                )
            )
        if lead.llm.thesis_fit is not None:
            w = thesis.signal_params.thesis_fit_weight
            fit = min(max(lead.llm.thesis_fit, 0.0), 1.0)
            multiplier = (1.0 - w) + w * fit
            score *= multiplier
            steps.append(
                (f"× {multiplier:.2f} (thesis fit {fit:.2f}, weight {w:g})", score)
            )
        # Value-add fit only enters the math when opted into (weight > 0) —
        # at the default 0 it stays a purely informational dimension.
        if lead.llm.value_add_fit is not None and thesis.signal_params.value_add_weight > 0:
            w = thesis.signal_params.value_add_weight
            fit = min(max(lead.llm.value_add_fit, 0.0), 1.0)
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
            lead.llm.verification == "unverifiable"
            if lead.llm.verification is not None
            else lead.llm.grounding in (None, "none", "bio")
        )
        if ungrounded:
            multiplier = thesis.signal_params.ungrounded_multiplier
            score *= multiplier
            basis = (
                "audit: unverifiable"
                if lead.llm.verification == "unverifiable"
                else f"grounding: {lead.llm.grounding or 'none'}"
            )
            steps.append(
                (f"× {multiplier:g} (product unverified — {basis})", score)
            )
    return steps
