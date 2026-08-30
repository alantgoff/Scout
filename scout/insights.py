"""Triage insights — what your shortlist/pass decisions say about the config.

Pure functions over ledger entries + the pipeline table: contrast the leads
the investor shortlisted (or moved further) against the ones they passed on,
per signal, sector, stage, business model, and thesis fit. `findings` renders
the 2–4 most actionable contrasts as plain sentences; `stats_prompt` formats
the whole thing for the weight-suggestion agent (scout.agents.suggest_weights).

`actor_stats` runs the same contrast over ONE partner's votes instead of the
firm's pipeline status — the per-partner taste profile — and
`model_disagreements` surfaces where a partner and the model parted ways,
which is the most interesting thing either of them produces.

No I/O and no network — fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from scout.models import Lead, LedgerEntry, Vote
from scout.status import POSITIVE_STATUSES  # noqa: F401 — re-exported for callers

MIN_DECISIONS = 5
# Score bands for human-vs-model disagreement. A pass on a lead the model
# ranked highly (or conviction on one it ranked low) is the signal worth
# reading — the rest is agreement, which teaches nothing.
MODEL_LIKED = 70.0
MODEL_COOL = 40.0


class TriageStats(BaseModel):
    """Shortlisted-vs-passed contrast across every scoring dimension."""

    shortlisted: int
    passed: int
    sector_counts: dict[str, dict[str, int]]  # group -> sector -> count
    stage_counts: dict[str, dict[str, int]]
    model_counts: dict[str, dict[str, int]]
    ctype_counts: dict[str, dict[str, int]] = Field(default_factory=dict)  # b2b vs b2c
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
        ctype_counts={g: _count_by(ls, "customer_type") for g, ls in groups.items()},
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
        f"Customer types — shortlisted: {stats.ctype_counts.get('shortlisted', {})}",
        f"Customer types — passed: {stats.ctype_counts.get('passed', {})}",
        f"Mean thesis fit — shortlisted: {stats.fit_means.get('shortlisted')}, "
        f"passed: {stats.fit_means.get('passed')}",
    ]
    if stats.findings:
        lines.append("Notable contrasts: " + " | ".join(stats.findings))
    return "\n".join(lines)


# ------------------------------------------------------------- per-partner taste


class ModelDisagreement(BaseModel):
    """One place a partner and the model parted ways."""

    handle: str
    name: str
    score: float
    stance: str
    rationale: str = ""
    kind: str  # "model_liked_you_passed" | "model_cool_you_liked"


def actor_stats(
    entries: list[LedgerEntry],
    votes_by_handle: dict[str, list[Vote]],
    actor: str,
) -> TriageStats | None:
    """One partner's taste profile: the leads THEY voted yes on contrasted
    with the ones THEY passed, in the same shape triage_stats produces.

    Deliberately reads votes rather than pipeline status: status is the
    firm's shared state (whoever moved it last), while a vote is a named
    person's own judgment — the only honest basis for "your taste". Unsure
    votes are excluded; they are the absence of a decision.
    """
    from scout.collab import STANCES

    liked: list[Lead] = []
    passed: list[Lead] = []
    for entry in entries:
        handle = entry.lead.account.handle.lower()
        vote = next(
            (v for v in votes_by_handle.get(handle, []) if v.actor == actor), None
        )
        if vote is None:
            continue
        value = STANCES.get(vote.stance, 0)
        if value > 0:
            liked.append(entry.lead)
        elif value < 0:
            passed.append(entry.lead)
    if len(liked) + len(passed) < MIN_DECISIONS:
        return None

    groups = {"shortlisted": liked, "passed": passed}
    stats = TriageStats(
        shortlisted=len(liked),
        passed=len(passed),
        sector_counts={g: _count_by(ls, "sector") for g, ls in groups.items()},
        stage_counts={g: _count_by(ls, "stage") for g, ls in groups.items()},
        model_counts={g: _count_by(ls, "business_model") for g, ls in groups.items()},
        ctype_counts={g: _count_by(ls, "customer_type") for g, ls in groups.items()},
        signal_means={g: _signal_means(ls) for g, ls in groups.items()},
        fit_means={g: _fit_mean(ls) for g, ls in groups.items()},
    )
    stats.findings = _findings(stats)
    return stats


def model_disagreements(
    entries: list[LedgerEntry],
    votes_by_handle: dict[str, list[Vote]],
    actor: str,
    limit: int = 10,
) -> list[ModelDisagreement]:
    """Where this partner and the model disagreed, strongest first.

    Two directions, both worth a look: startups the model ranked highly that
    the partner passed on (is the thesis wrong, or was the model fooled?),
    and startups the model ranked low that the partner backed (what does the
    partner see that the scoring does not?). The second list is the one that
    should eventually change the weights.
    """
    from scout.collab import STANCES

    out: list[ModelDisagreement] = []
    for entry in entries:
        lead = entry.lead
        handle = lead.account.handle.lower()
        vote = next(
            (v for v in votes_by_handle.get(handle, []) if v.actor == actor), None
        )
        if vote is None:
            continue
        value = STANCES.get(vote.stance, 0)
        name = (lead.llm.company_name if lead.llm else "") or lead.account.name \
            or f"@{lead.account.handle}"
        if value < 0 and lead.score >= MODEL_LIKED:
            kind = "model_liked_you_passed"
        elif value > 0 and lead.score < MODEL_COOL:
            kind = "model_cool_you_liked"
        else:
            continue
        out.append(ModelDisagreement(
            handle=handle, name=name, score=lead.score,
            stance=vote.stance, rationale=vote.rationale, kind=kind,
        ))
    # Widest gaps first: a pass on a 90 outranks a pass on a 71.
    out.sort(key=lambda d: -abs(d.score - (
        MODEL_LIKED if d.kind == "model_liked_you_passed" else MODEL_COOL
    )))
    return out[:limit]


# ------------------------------------------------------------- query yield
# The discovery loop's report card: which search queries actually produce
# companies the firm triages, and which just burn the sourcing time budget.
# Pure functions over store.query_hits() + the ledger + pipeline — the same
# contrast philosophy as triage_stats, pointed at the query bank.


@dataclass
class QueryYield:
    """One query's lifetime scoreboard."""

    query: str
    category: str
    surfaced: int  # distinct accounts this query ever hit
    scored: int  # of those, how many made it into the ledger
    triaged: int  # … and were longlisted or further (POSITIVE_STATUSES)
    passed: int

    @property
    def dead(self) -> bool:
        """Surfaced a real sample, triaged nothing — the pruning candidate.
        The floor keeps a query that has only ever hit 3 accounts from being
        condemned on no evidence."""
        return self.surfaced >= 10 and self.triaged == 0

    @property
    def unproven(self) -> bool:
        return self.surfaced < 10 and self.triaged == 0


def query_yield(
    hits: list[dict],
    ledger_handles: set[str],
    pipeline: dict[str, dict],
) -> list[QueryYield]:
    """Scoreboard rows, best earners first (triaged desc, then surfaced).

    `hits` = store.query_hits(); `ledger_handles` = lowercased handles that
    exist in the lead ledger; `pipeline` = store.all_pipeline().
    """
    by_query: dict[tuple[str, str], set[str]] = {}
    for hit in hits:
        by_query.setdefault((hit["query"], hit.get("category") or ""),
                            set()).add(hit["handle"])
    rows = []
    for (query, category), handles in by_query.items():
        scored = {h for h in handles if h in ledger_handles}
        statuses = [(pipeline.get(h) or {}).get("status") or "new"
                    for h in scored]
        rows.append(QueryYield(
            query=query, category=category,
            surfaced=len(handles), scored=len(scored),
            triaged=sum(1 for s in statuses if s in POSITIVE_STATUSES),
            passed=sum(1 for s in statuses if s == "passed"),
        ))
    rows.sort(key=lambda r: (-r.triaged, -r.surfaced, r.query))
    return rows


def performance_block_for(store) -> str:
    """The strategy agent's briefing, straight from a Store — the one-call
    wrapper the CLI and UI share. Empty string when nothing is measured."""
    from scout.graph import watchlist_candidates

    hits = store.query_hits()
    edges = store.all_graph_edges()
    if not hits and not edges:
        return ""
    ledger_handles = {
        e.lead.account.handle.lower() for e in store.load_lead_ledger()
    }
    yields = query_yield(hits, ledger_handles, store.all_pipeline())
    try:
        from scout.config import load_seeds

        watchers = load_seeds().watchers
    except Exception:  # seeds file missing/unreadable — suggestions still work
        watchers = []
    return performance_block(yields, watchlist_candidates(edges, watchers))


def performance_block(
    yields: list[QueryYield],
    watchlist_candidates: list[tuple[str, int]] | None = None,
) -> str:
    """The strategy agent's performance briefing — measured yield per query
    plus graph-derived watchlist leads, compact enough to sit in the prompt.
    Empty string when there is nothing measured yet (a block of zeros would
    teach the agent that everything is dead)."""
    lines: list[str] = []
    earners = [y for y in yields if y.triaged > 0]
    dead = [y for y in yields if y.dead]
    if earners:
        lines.append("Queries that produced triaged companies (keep this shape):")
        lines += [f"- [{y.category}] {y.query!r}: {y.surfaced} surfaced, "
                  f"{y.triaged} triaged" for y in earners[:10]]
    if dead:
        lines.append("Queries that surfaced plenty but produced NOTHING triaged "
                     "(drop or replace these):")
        lines += [f"- [{y.category}] {y.query!r}: {y.surfaced} surfaced, 0 triaged"
                  for y in dead[:10]]
    if watchlist_candidates:
        lines.append(
            "Investors/labs connected to 2+ companies already in the database "
            "(if any have a known X account, they belong on the watchlist):")
        lines += [f"- {name} ({n} tracked companies)"
                  for name, n in watchlist_candidates[:8]]
    return "\n".join(lines)
