"""Forward measurement — did Scout catch them at the time?

The backtest (hindsight.py) rewinds the EVIDENCE to a past date but scores it
with TODAY's thesis, written by people who already know who raised. Its
honest claim is "today's thesis would have ranked these highly". Whether
Scout, configured as it was then, surfaced a company BEFORE it raised is a
question only forward measurement answers (AGENTS §11, limitation 1). This
module answers it from data the system already keeps:

- the FIRST score Scout gave each company (store.first_sightings) — frozen:
  saved under that run's thesis version and never rewritten;
- raises recorded since — SEC Form D filings matched to a tracked company
  (dated by first sale, the sworn date of the round) and rounds the tracked
  refresh detected (dated by detection).

Every guard below exists because the naive version produces a confident,
wrong number:

- MATURED companies only. A company first seen last month that has not
  raised is undecided, not a miss: counting it as a negative deflates the
  hit rate, and counting its fast-raising peers as positives while it waits
  inflates it. Only companies first seen at least `horizon_days` ago enter
  the rates; younger raises are listed as early hits, outside them.
- A raise dated at or before first sight is not a catch. Those companies are
  excluded ("already raised"), not scored as hits.
- Form D first. The refresh only re-researches companies the firm already
  triaged, and triage follows the score — so the rounds it detects inflate
  the hit rate of whatever scored well. Form D matching covers every known
  company. Both variants are computable; the gap between them is that bias.
- Below MIN_SAMPLES_PER_GROUP raises (or non-raises) the rates are withheld,
  not shown with a caveat — with how many more are needed and the smallest
  effect the current sample could detect.
- A zero hit rate is bounded by the rule of three, never reported as zero.

Pure — rows in, a report out — except `inputs_from_store`, the one place
the three callers (cli.py `forward`, the UI's Thesis → Evidence → Forward,
notify's digest) read the store, so they cannot disagree about the inputs.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Literal

from pydantic import BaseModel, Field

from scout.models import FUNDING_STAGE_LABELS
from scout.signal_eval import MIN_SAMPLES_PER_GROUP, auc_ci, min_detectable_auc

HORIZONS = (90, 180, 365)
DEFAULT_HORIZON = 180
DEFAULT_THRESHOLD = 60.0
# The share of evaluable startups, by first-sight score, whose hit rate is
# compared with the base rate ("the top fifth raised at 3× the rate").
TOP_SHARE = 0.2
SOURCES = ("form_d", "detected")
SOURCE_LABELS = {"form_d": "Form D", "detected": "Detected round"}

Status = Literal["raised", "not_raised", "early_hit", "watching", "already_raised"]
STATUS_LABELS = {
    "raised": "Raised",
    "not_raised": "No raise",
    "early_hit": "Raised (early)",
    "watching": "Too soon to tell",
    "already_raised": "Already raised",
}


class Sighting(BaseModel):
    """A company at first sight: the score Scout gave it then."""

    handle: str
    name: str
    first_seen: datetime
    score: float
    run_id: str = ""
    thesis_id: str = ""
    thesis_version: str = ""


class RaiseEvent(BaseModel):
    """One recorded raise for one tracked company."""

    handle: str
    date: datetime
    source: Literal["form_d", "detected"]
    detail: str = ""


class ForwardRow(BaseModel):
    handle: str
    name: str
    first_seen: datetime
    score: float
    thesis_id: str = ""
    thesis_version: str = ""
    status: Status
    raise_date: datetime | None = None
    raise_source: str = ""
    raise_detail: str = ""
    days_ahead: int | None = None  # first sight → raise; negative = before

    @property
    def evaluable(self) -> bool:
        return self.status in ("raised", "not_raised")

    @property
    def hit(self) -> bool:
        return self.status in ("raised", "early_hit")


class VersionRow(BaseModel):
    """How one thesis version's first sightings have fared."""

    thesis_id: str
    thesis_version: str
    sighted: int
    evaluable: int
    raised: int
    first_seen: datetime


class ForwardReport(BaseModel):
    generated_at: datetime
    horizon_days: int
    threshold: float
    sources: list[str]
    rows: list[ForwardRow]
    n_sighted: int
    n_evaluable: int
    n_raised: int  # evaluable AND raised within the horizon
    n_already: int
    n_watching: int
    n_early_hits: int
    base_rate: float | None = None
    auc: float | None = None
    auc_low: float | None = None
    auc_high: float | None = None
    # Whether the AUC's interval clears 0.5 — a point estimate above a coin
    # flip with an interval that includes one is not evidence of skill.
    beats_chance: bool | None = None
    min_detectable_auc: float = 1.0
    top_n: int = 0
    top_hits: int = 0
    top_hit_rate: float | None = None
    # Rule of three: with no hits among n, the true rate could still be ~3/n.
    top_hit_rate_bound: float | None = None
    lift: float | None = None
    at_threshold: int = 0
    at_threshold_hits: int = 0
    precision_at_threshold: float | None = None
    recall_at_threshold: float | None = None
    median_days_ahead: float | None = None
    underpowered: bool = True
    needs: str = ""
    by_version: list[VersionRow] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def hits(self) -> list[ForwardRow]:
        """Every company that raised after Scout first saw it — matured or
        not — the most days ahead first."""
        return sorted((r for r in self.rows if r.hit),
                      key=lambda r: (-(r.days_ahead or 0), r.name.lower()))

    def underrated(self) -> list[ForwardRow]:
        """Raised within the horizon but first scored below the threshold —
        the misses worth reading when tuning the thesis."""
        return sorted((r for r in self.rows if r.hit and r.score < self.threshold),
                      key=lambda r: (r.score, r.name.lower()))


# --------------------------------------------------------------- inputs


def to_utc(value) -> datetime | None:
    """ISO datetime or YYYY-MM-DD → aware UTC datetime; None if unreadable."""
    if not value:
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _money(value) -> str:
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return ""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M".replace(".0M", "M")
    if amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount}" if amount else ""


def sightings_from_rows(rows: list[dict]) -> list[Sighting]:
    """store.first_sightings() rows → Sightings, named the startup-first way
    (companies.display_name). An unparseable lead keeps its handle."""
    from scout.companies import display_name
    from scout.models import Lead

    out: list[Sighting] = []
    for row in rows:
        first_seen = to_utc(row.get("created_at"))
        if first_seen is None:
            continue
        handle = (row.get("handle") or "").lstrip("@").lower()
        try:
            name = display_name(Lead.model_validate_json(row.get("lead_json") or ""))
        except ValueError:
            name = f"@{handle}"
        out.append(Sighting(
            handle=handle, name=name, first_seen=first_seen,
            score=float(row.get("score") or 0.0), run_id=row.get("run_id") or "",
            thesis_id=row.get("thesis_id") or "",
            thesis_version=row.get("thesis_version") or "",
        ))
    return out


def raises_from(filings: list[dict], outcomes: list[dict]) -> list[RaiseEvent]:
    """Matched Form D filings (store.matched_filings) and refresh-detected
    rounds (store.auto_outcomes) → dated raise events, keyed on handle.

    Joined on handle, never on a name slug: "Acme" from a refresh and "Acme
    Inc" from a filing are one company and must not count twice. Amendments
    restate an earlier filing rather than a new round, so they are skipped.
    A filing is dated by its first sale — the sworn date of the round — and
    falls back to the filing date when the first sale is unstated.
    """
    events: list[RaiseEvent] = []
    for filing in filings:
        handle = (filing.get("matched_handle") or "").lower()
        if not handle or filing.get("is_amendment"):
            continue
        date = to_utc(filing.get("first_sale")) or to_utc(filing.get("filed_at"))
        if date is None:
            continue
        sold = _money(filing.get("amount_sold"))
        offered = _money(filing.get("amount_offered"))
        detail = "Form D" + (f" · {sold} sold" if sold else
                             f" · {offered} offered" if offered else "")
        events.append(RaiseEvent(handle=handle, date=date, source="form_d", detail=detail))
    for outcome in outcomes:
        handle = (outcome.get("handle") or "").lower()
        date = to_utc(outcome.get("detected_at"))
        if not handle or date is None:
            continue
        stage = outcome.get("round_stage") or ""
        detail = (FUNDING_STAGE_LABELS.get(stage) or stage.replace("_", " ").capitalize()
                  or "Round") + (f" · {outcome['amount']}" if outcome.get("amount") else "")
        events.append(RaiseEvent(handle=handle, date=date, source="detected", detail=detail))
    return events


# --------------------------------------------------------------- the report


def _classify(sighting: Sighting, events: list[RaiseEvent], *, now: datetime,
              horizon: timedelta) -> ForwardRow:
    """One company's fate, from its first sighting and its raise events."""
    events = sorted(events, key=lambda e: e.date)
    deadline = sighting.first_seen + horizon
    matured = deadline <= now
    within = [e for e in events if sighting.first_seen < e.date <= deadline]
    before = [e for e in events if e.date <= sighting.first_seen]
    if within:
        event, status = within[0], ("raised" if matured else "early_hit")
    elif before:
        event, status = before[-1], "already_raised"
    else:
        event, status = None, ("not_raised" if matured else "watching")
    return ForwardRow(
        handle=sighting.handle, name=sighting.name, first_seen=sighting.first_seen,
        score=sighting.score, thesis_id=sighting.thesis_id,
        thesis_version=sighting.thesis_version, status=status,
        raise_date=event.date if event else None,
        raise_source=SOURCE_LABELS.get(event.source, event.source) if event else "",
        raise_detail=event.detail if event else "",
        days_ahead=(event.date - sighting.first_seen).days if event else None,
    )


def _needs(n_pos: int, n_neg: int, horizon_days: int) -> str:
    parts = []
    if n_pos < MIN_SAMPLES_PER_GROUP:
        more = MIN_SAMPLES_PER_GROUP - n_pos
        parts.append(f"{more} more raise{'s' if more != 1 else ''}")
    if n_neg < MIN_SAMPLES_PER_GROUP:
        more = MIN_SAMPLES_PER_GROUP - n_neg
        parts.append(f"{more} more startup{'s' if more != 1 else ''} that didn't raise")
    if not parts:
        return ""
    return ("Needs " + " and ".join(parts) + f" among startups first seen at least "
            f"{horizon_days} days ago.")


def build_forward(
    sightings: list[Sighting],
    raises: list[RaiseEvent],
    *,
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
    sources: tuple[str, ...] | list[str] = ("form_d",),
) -> ForwardReport:
    """The forward scorecard: how Scout's first-sight scores relate to the
    raises that followed, under the guards in the module docstring."""
    now = now or datetime.now(timezone.utc)
    horizon = timedelta(days=horizon_days)
    wanted = [s for s in SOURCES if s in set(sources)]
    by_handle: dict[str, list[RaiseEvent]] = {}
    for event in raises:
        if event.source in wanted:
            by_handle.setdefault(event.handle.lower(), []).append(event)

    rows = [_classify(s, by_handle.get(s.handle.lower(), []), now=now, horizon=horizon)
            for s in sightings]
    evaluable = [r for r in rows if r.evaluable]
    positives = [r.score for r in evaluable if r.status == "raised"]
    negatives = [r.score for r in evaluable if r.status == "not_raised"]
    n_pos, n_neg = len(positives), len(negatives)

    report = ForwardReport(
        generated_at=now, horizon_days=horizon_days, threshold=threshold,
        sources=wanted, rows=rows, n_sighted=len(rows), n_evaluable=len(evaluable),
        n_raised=n_pos,
        n_already=sum(1 for r in rows if r.status == "already_raised"),
        n_watching=sum(1 for r in rows if r.status == "watching"),
        n_early_hits=sum(1 for r in rows if r.status == "early_hit"),
        base_rate=(n_pos / len(evaluable)) if evaluable else None,
        min_detectable_auc=min_detectable_auc(n_pos, n_neg),
        underpowered=n_pos < MIN_SAMPLES_PER_GROUP or n_neg < MIN_SAMPLES_PER_GROUP,
        needs=_needs(n_pos, n_neg, horizon_days),
        by_version=_by_version(rows),
        limitations=forward_limitations(horizon_days, wanted),
    )
    if report.underpowered:
        # Withheld, not caveated: a caveat under a number still leaves the
        # number on the slide.
        return report

    report.auc, report.auc_low, report.auc_high = auc_ci(positives, negatives)
    report.beats_chance = report.auc_low > 0.5
    ranked = sorted(evaluable, key=lambda r: (-r.score, r.handle))
    report.top_n = max(1, math.ceil(len(evaluable) * TOP_SHARE))
    report.top_hits = sum(1 for r in ranked[:report.top_n] if r.status == "raised")
    report.top_hit_rate = report.top_hits / report.top_n
    if report.top_hits == 0:
        report.top_hit_rate_bound = min(1.0, 3 / report.top_n)
    if report.base_rate:
        report.lift = round(report.top_hit_rate / report.base_rate, 2)
    above = [r for r in evaluable if r.score >= threshold]
    report.at_threshold = len(above)
    report.at_threshold_hits = sum(1 for r in above if r.status == "raised")
    if above:
        report.precision_at_threshold = report.at_threshold_hits / len(above)
    report.recall_at_threshold = report.at_threshold_hits / n_pos
    ahead = [r.days_ahead for r in rows if r.hit and r.days_ahead is not None]
    if len(ahead) >= MIN_SAMPLES_PER_GROUP:
        report.median_days_ahead = float(median(ahead))
    return report


def _by_version(rows: list[ForwardRow]) -> list[VersionRow]:
    groups: dict[tuple[str, str], list[ForwardRow]] = {}
    for row in rows:
        groups.setdefault((row.thesis_id, row.thesis_version), []).append(row)
    out = [
        VersionRow(
            thesis_id=thesis_id, thesis_version=version, sighted=len(members),
            evaluable=sum(1 for r in members if r.evaluable),
            raised=sum(1 for r in members if r.status == "raised"),
            first_seen=min(r.first_seen for r in members),
        )
        for (thesis_id, version), members in groups.items()
    ]
    return sorted(out, key=lambda v: v.first_seen)


def forward_limitations(horizon_days: int, sources: list[str] | tuple[str, ...]) -> list[str]:
    """What this measurement does not show — rendered into every view of it,
    the same contract as hindsight.default_limitations."""
    lines = [
        f"Only startups first seen at least {horizon_days} days ago count as hits "
        "or misses. Younger ones are undecided: their raises are listed as early "
        "hits but kept out of every rate.",
        "A raise counts only if Scout recorded it. Form D covers US raises that "
        "file one (most priced rounds do; many SAFEs and non-US rounds don't), "
        "and matches only a company whose legal name equals the tracked name — "
        "so every rate here is a floor, not an estimate.",
        "A Form D is matched only to a company Scout already knew on the day it "
        "was filed; a startup found after its filing never gets one.",
        "The score is the one Scout gave a company the first time it saw it, "
        "under that day's thesis and run settings. A company first scored on a "
        "thin profile is judged on that thin profile.",
        "Companies that had already raised when Scout first saw them are left "
        "out rather than counted as catches.",
    ]
    if "detected" in sources:
        lines.append(
            "Detected rounds come from the tracked refresh, which only "
            "re-researches startups the firm already longlisted or beyond — and "
            "triage follows the score, so including them inflates the hit rate "
            "of high scorers. They are dated when Scout noticed them, so a round "
            "from before first sight can count as a catch.")
    lines.append(
        "This measures whether a high first score came before a raise. It does "
        "not measure whether the firm acted on it, or whether a raise is the "
        "outcome your fund cares about.")
    return lines


# --------------------------------------------------------------- rendering


def render_markdown(report: ForwardReport) -> str:
    """The shareable version — headline, hits, misses, and every limitation."""
    sources = " + ".join(SOURCE_LABELS.get(s, s) for s in report.sources)
    lines = [
        "# Forward scorecard",
        "",
        f"Generated {report.generated_at:%d %B %Y} · horizon {report.horizon_days} "
        f"days · raises from {sources} · threshold {report.threshold:.0f}",
        "",
        f"- Startups sighted: {report.n_sighted}",
        f"- First seen ≥ {report.horizon_days} days ago (evaluable): {report.n_evaluable}",
        f"- Raised within {report.horizon_days} days of first sight: {report.n_raised}",
        f"- Raised early (younger than the horizon): {report.n_early_hits}",
        f"- Already raised when first seen (excluded): {report.n_already}",
        "",
    ]
    if report.underpowered:
        lines += [
            "**Rates withheld — too few outcomes.** " + report.needs,
            f"With this sample, no AUC below {report.min_detectable_auc:.2f} "
            "could be told apart from chance.",
            "",
        ]
    else:
        lines += [
            f"- Base rate: {report.base_rate:.1%}",
            f"- AUC of the first-sight score: {report.auc:.2f} "
            f"(95% CI {report.auc_low:.2f}–{report.auc_high:.2f})"
            + ("" if report.beats_chance else
               " — the interval includes 0.5: not yet distinguishable from chance"),
            f"- Top {TOP_SHARE:.0%} by score ({report.top_n}): {report.top_hits} raised"
            + (f" — {report.lift:.1f}× the base rate" if report.lift is not None else ""),
            f"- At score ≥ {report.threshold:.0f}: {report.at_threshold_hits} of "
            f"{report.at_threshold} raised; recall {report.recall_at_threshold:.0%}",
        ]
        if report.median_days_ahead is not None:
            lines.append(f"- Median days from first sight to raise: "
                         f"{report.median_days_ahead:.0f}")
        lines.append("")
    hits = report.hits()
    if hits:
        lines += ["## Caught before the raise", "",
                  "| Startup | First seen | Score then | Raise | Days ahead |",
                  "|---|---|---|---|---|"]
        lines += [
            f"| {r.name} | {r.first_seen:%Y-%m-%d} | {r.score:.0f} | "
            f"{r.raise_detail or r.raise_source} ({r.raise_date:%Y-%m-%d}) | {r.days_ahead} |"
            for r in hits
        ]
        lines.append("")
    misses = report.underrated()
    if misses:
        lines += ["## Raised, but scored below the threshold", ""]
        lines += [f"- {r.name} — scored {r.score:.0f} on {r.first_seen:%Y-%m-%d}; "
                  f"{r.raise_detail or r.raise_source} {r.days_ahead} days later" for r in misses]
        lines.append("")
    lines += ["## What this does and does not show", ""]
    lines += [f"- {line}" for line in report.limitations]
    return "\n".join(lines) + "\n"


def caught_note(first_seen: datetime | None, score: float | None,
                raise_date: datetime | None) -> str:
    """The digest's line on a raise alert: how far ahead Scout was. Empty
    when Scout saw the company only after (or on) the raise date."""
    if first_seen is None or raise_date is None or raise_date <= first_seen:
        return ""
    days = (raise_date - first_seen).days
    if days < 1:
        return ""
    return (f"Scout first saw it {days} day{'s' if days != 1 else ''} earlier"
            + (f" (score {score:.0f} then)" if score is not None else ""))


def inputs_from_store(store) -> tuple[list[Sighting], list[RaiseEvent]]:
    """(sightings, raises) as the store holds them today. The only impure
    function here: the CLI, the UI and the digest all read through it."""
    return (sightings_from_rows(store.first_sightings()),
            raises_from(store.matched_filings(), store.auto_outcomes()))
