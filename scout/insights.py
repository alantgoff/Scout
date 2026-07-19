"""Triage insights — what your shortlist/pass decisions say about the config.

Pure functions over ledger entries + the pipeline table: contrast the leads
the investor shortlisted (or moved further) against the ones they passed on,
per signal, sector, stage, business model, and thesis fit. `findings` renders
the 2–4 most actionable contrasts as plain sentences; `stats_prompt` formats
the whole thing for the weight-suggestion agent (scout.agents.suggest_weights).

No I/O and no network — fully unit-testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scout.models import Lead, LedgerEntry

MIN_DECISIONS = 5

# Any of these statuses counts as a positive ("shortlisted") decision.
POSITIVE_STATUSES = {"longlisted", "shortlisted", "contacted", "meeting", "diligence", "won"}


class TriageStats(BaseModel):
    """Shortlisted-vs-passed contrast across every scoring dimension."""

    shortlisted: int
    passed: int
    sector_counts: dict[str, dict[str, int]]  # group -> sector -> count
    stage_counts: dict[str, dict[str, int]]
    model_counts: dict[str, dict[str, int]]
    signal_means: dict[str, dict[str, float]]  # group -> signal -> mean points
    fit_means: dict[str, float | None]
    findings: list[str] = Field(default_factory=list)

    @property
    def decisions(self) -> int:
        return self.shortlisted + self.passed


def _count_by(leads: list[Lead], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lead in leads:
        value = getattr(lead.llm, attr, None) if lead.llm else None
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _signal_means(leads: list[Lead]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for lead in leads:
        for signal in lead.signals:
            sums[signal.name] = sums.get(signal.name, 0.0) + signal.contribution
            counts[signal.name] = counts.get(signal.name, 0) + 1
    return {name: round(sums[name] / counts[name], 2) for name in sums}


def _fit_mean(leads: list[Lead]) -> float | None:
    fits = [x.llm.thesis_fit for x in leads if x.llm and x.llm.thesis_fit is not None]
    return round(sum(fits) / len(fits), 3) if fits else None


def triage_stats(
    entries: list[LedgerEntry], pipeline: dict[str, dict]
) -> TriageStats | None:
    """Contrast shortlisted vs passed leads; None below MIN_DECISIONS."""
    shortlisted: list[Lead] = []
    passed: list[Lead] = []
    for entry in entries:
        status = pipeline.get(entry.lead.account.handle.lower(), {}).get("status") or "new"
        if status in POSITIVE_STATUSES:
            shortlisted.append(entry.lead)
        elif status == "passed":
            passed.append(entry.lead)
    if len(shortlisted) + len(passed) < MIN_DECISIONS:
        return None

    groups = {"shortlisted": shortlisted, "passed": passed}
    stats = TriageStats(
        shortlisted=len(shortlisted),
        passed=len(passed),
        sector_counts={g: _count_by(ls, "sector") for g, ls in groups.items()},
        stage_counts={g: _count_by(ls, "stage") for g, ls in groups.items()},
        model_counts={g: _count_by(ls, "business_model") for g, ls in groups.items()},
        signal_means={g: _signal_means(ls) for g, ls in groups.items()},
        fit_means={g: _fit_mean(ls) for g, ls in groups.items()},
    )
    stats.findings = _findings(stats)
    return stats


def _findings(stats: TriageStats) -> list[str]:
    out: list[str] = []

    passed_sectors = stats.sector_counts.get("passed", {})
    if stats.passed >= 3 and passed_sectors:
        top, count = max(passed_sectors.items(), key=lambda kv: kv[1])
        if count / stats.passed >= 0.5:
            out.append(
                f"{count} of {stats.passed} passes are {top} — consider a "
                "disqualifier or a narrower query bank for that space."
            )

    short_sectors = stats.sector_counts.get("shortlisted", {})
    if stats.shortlisted >= 3 and short_sectors:
        top, count = max(short_sectors.items(), key=lambda kv: kv[1])
        if count / stats.shortlisted >= 0.6:
            out.append(
                f"{count} of {stats.shortlisted} shortlists are {top} — "
                "the thesis is converging there."
            )

    fit_short = stats.fit_means.get("shortlisted")
    fit_passed = stats.fit_means.get("passed")
    if fit_short is not None and fit_passed is not None and abs(fit_short - fit_passed) >= 0.1:
        if fit_short > fit_passed:
            out.append(
                f"Average thesis fit: {fit_short:.0%} shortlisted vs "
                f"{fit_passed:.0%} passed — the fit score is tracking your taste."
            )
        else:
            out.append(
                f"Average thesis fit: {fit_short:.0%} shortlisted vs "
                f"{fit_passed:.0%} passed — inverted; the thesis statement may "
                "not describe what you actually pick."
            )

    means_short = stats.signal_means.get("shortlisted", {})
    means_passed = stats.signal_means.get("passed", {})
    diffs = [
        (name, means_short.get(name, 0.0) - means_passed.get(name, 0.0))
        for name in set(means_short) | set(means_passed)
    ]
    if diffs:
        name, diff = max(diffs, key=lambda kv: abs(kv[1]))
        if abs(diff) >= 2:
            if diff > 0:
                out.append(
                    f"{name} averages {diff:.1f} pts higher on shortlists — "
                    "that weight is earning its keep."
                )
            else:
                out.append(
                    f"{name} averages {abs(diff):.1f} pts higher on passes — "
                    "consider lowering its weight."
                )

    return out[:4]


def stats_prompt(stats: TriageStats) -> str:
    """Compact text rendering of the stats for the weight-suggestion agent."""
    lines = [
        f"Decisions: {stats.shortlisted} shortlisted (or further), {stats.passed} passed.",
        f"Mean signal points — shortlisted: {stats.signal_means.get('shortlisted', {})}",
        f"Mean signal points — passed: {stats.signal_means.get('passed', {})}",
        f"Sectors — shortlisted: {stats.sector_counts.get('shortlisted', {})}",
        f"Sectors — passed: {stats.sector_counts.get('passed', {})}",
        f"Stages — shortlisted: {stats.stage_counts.get('shortlisted', {})}",
        f"Stages — passed: {stats.stage_counts.get('passed', {})}",
        f"Business models — shortlisted: {stats.model_counts.get('shortlisted', {})}",
        f"Business models — passed: {stats.model_counts.get('passed', {})}",
        f"Mean thesis fit — shortlisted: {stats.fit_means.get('shortlisted')}, "
        f"passed: {stats.fit_means.get('passed')}",
    ]
    if stats.findings:
        lines.append("Notable contrasts: " + " | ".join(stats.findings))
    return "\n".join(lines)
