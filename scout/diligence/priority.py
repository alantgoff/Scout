"""Smart surfacing — which startups deserve a deep analysis next.

Pure and $0: computed entirely from data scout already has (the ledger, the
pipeline table, existing memos). Analysis stays manual-trigger by user
decision; this module only ranks the queue:

  priority = thesis_fit × score_percentile × freshness_boost (new: 1.3)
             × momentum_boost (score_delta > 5: 1.2)
             × signal_richness (departure / traction / smart-money hits)

Companies with a fresh memo are excluded (pass the fresh keys in `memoed`;
the caller decides freshness via the memo fingerprint), as are non-startup-
track leads and passed leads.
"""

from __future__ import annotations

from pydantic import BaseModel

from scout.companies import group_by_company, startup_key
from scout.models import Lead, LedgerEntry

LAUNCHED_STAGES = {"launched", "scaling"}
FRESHNESS_BOOST = 1.3
MOMENTUM_BOOST = 1.2
MOMENTUM_MIN_DELTA = 5.0
RICHNESS_PER_HIT = 0.1  # 1 + 0.1 per rich signal, capped
RICHNESS_CAP = 1.3
DEFAULT_FIT = 0.4  # verdicts without thesis_fit rank, but never top

RICH_SIGNALS = {
    "departure_signal": "departure signal",
    "launch_traction": "launch traction",
    "smart_money_follow": "smart-money follows",
    "smart_money_convergence": "smart-money convergence",
    "bio_change": "fresh intent language",
}


class Suggestion(BaseModel):
    """One "worth a deep look" candidate for the UI shelf / CLI."""

    company_key: str
    company_name: str
    handle: str  # primary account — what `scout analyze` takes
    priority: float
    reason: str  # generated-from-data one-liner ("82% fit · new this run · …")


def _is_startup(lead: Lead) -> bool:
    verdict = lead.llm
    return (
        verdict is not None
        and verdict.stage in LAUNCHED_STAGES
        and (verdict.account_type or "other") != "other"
    )


def analysis_priority(
    entries: list[LedgerEntry],
    memoed: set[str] | None = None,
    pipeline: dict[str, dict] | None = None,
) -> list[Suggestion]:
    """Rank tracked startups by how much a deep analysis would be worth.

    `memoed` = company keys whose memo is still fresh (fingerprint match) —
    excluded. `pipeline` rows exclude passed leads. Pure — unit-tested.
    """
    memoed = memoed or set()
    pipeline = pipeline or {}
    grouped = group_by_company([(e.lead, e) for e in entries])

    candidates: list[tuple[str, Lead, LedgerEntry | None]] = []
    for primary, entry, _secondaries in grouped:
        if not _is_startup(primary):
            continue
        status = pipeline.get(primary.account.handle.lower(), {}).get("status") or "new"
        if status == "passed":
            continue
        key = startup_key(primary)  # the canonical memo/KG identity
        if key in memoed:
            continue
        candidates.append((key, primary, entry))
    if not candidates:
        return []

    scores_desc = sorted((lead.score for _, lead, _ in candidates), reverse=True)

    def percentile(score: float) -> float:
        # 1.0 for the strongest lead in the candidate pool, linearly down.
        if len(scores_desc) == 1:
            return 1.0
        rank = scores_desc.index(score)
        return 1.0 - rank / len(scores_desc)

    suggestions: list[Suggestion] = []
    for key, lead, entry in candidates:
        verdict = lead.llm
        fit = (
            verdict.thesis_fit
            if verdict is not None and verdict.thesis_fit is not None
            else DEFAULT_FIT
        )
        priority = fit * percentile(lead.score)
        reasons: list[str] = []
        if verdict is not None and verdict.thesis_fit is not None:
            reasons.append(f"{verdict.thesis_fit:.0%} fit")
        if entry is not None and entry.is_new:
            priority *= FRESHNESS_BOOST
            reasons.append("new this run")
        delta = entry.score_delta if entry is not None else None
        if delta is not None and delta > MOMENTUM_MIN_DELTA:
            priority *= MOMENTUM_BOOST
            reasons.append(f"score ▲{delta:.0f}")
        rich_hits = [
            RICH_SIGNALS[s.name]
            for s in lead.signals
            if s.value > 0 and s.name in RICH_SIGNALS
        ]
        priority *= min(1.0 + RICHNESS_PER_HIT * len(rich_hits), RICHNESS_CAP)
        reasons += rich_hits[:2]
        suggestions.append(
            Suggestion(
                company_key=key,
                company_name=(verdict.company_name if verdict else "")
                or lead.account.name
                or lead.account.handle,
                handle=lead.account.handle,
                priority=round(priority, 4),
                reason=" · ".join(reasons[:3]) or "tracked startup",
            )
        )
    suggestions.sort(key=lambda s: (-s.priority, s.company_key))
    return suggestions
