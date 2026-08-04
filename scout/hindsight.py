"""Hindsight: would Scout have found this company before it raised?

The question every investor asks about a sourcing tool, and the one no
amount of architecture answers. This module answers it empirically —
reconstruct what the public evidence looked like on a date in the past,
score it with the SAME scorer production uses, and check where known
outcomes ranked.

Three commitments make the result worth trusting, and each is enforced in
code rather than promised in prose:

1. **Point-in-time discipline.** Only evidence created before the cutoff is
   admissible. A GitHub repo's star count is reconstructed from starring
   timestamps, not read as it stands today; HN posts are filtered by
   `created_at_i`. Anything that cannot be rewound is excluded rather than
   approximated.

2. **A control group.** Recall alone is meaningless — a scorer that flags
   everyone recalls everything. Outcomes are scored alongside companies
   from the same sources and window that did NOT go on to raise, and the
   headline metric is the SEPARATION between the two (AUC), not the hit
   rate.

3. **Named limitations.** X history is not reconstructible without paid
   full-archive access, so the backtest runs without X signals that
   production has. Results are therefore a LOWER bound on production
   recall. The language model also knows what happened after the cutoff to
   companies famous enough to be in its training data; `blinded` mode
   redacts names to measure how much that recognition is doing.

Everything in the "metrics" section is pure and unit-tested; only the
reconstruction section touches the network.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

import httpx
from pydantic import BaseModel, Field

from scout.models import Account

_HN_API = "https://hn.algolia.com/api/v1"
_GH_API = "https://api.github.com"
_TIMEOUT_S = 20
# GitHub caps starring-timestamp pagination; beyond this we know the star
# count only as "at least N", which is recorded honestly rather than guessed.
_STAR_PAGES_MAX = 10
_STARS_PER_PAGE = 100


class Outcome(BaseModel):
    """A company that went on to raise — the thing we hope Scout caught.

    Identifiers are whatever you have. More is better, but one is enough:
    the reconstruction gathers evidence from every identifier supplied and
    scores the union.
    """

    company: str
    round_date: datetime
    round_stage: str = ""
    amount: str = ""
    github_users: list[str] = Field(default_factory=list)
    github_repos: list[str] = Field(default_factory=list)  # "owner/name"
    hn_users: list[str] = Field(default_factory=list)
    x_handles: list[str] = Field(default_factory=list)
    domain: str = ""
    note: str = ""

    @property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.company.lower()).strip("-")


class Evidence(BaseModel):
    """Public evidence for one company as it stood at the cutoff."""

    key: str
    company: str
    as_of: datetime
    hn_posts: list[dict] = Field(default_factory=list)
    github_repos: list[dict] = Field(default_factory=list)
    bio: str = ""
    website: str = ""
    followers: int = 0
    notes: list[str] = Field(default_factory=list)
    fetch_errors: list[str] = Field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return bool(self.hn_posts or self.github_repos or self.bio)

    @property
    def unchecked(self) -> bool:
        """True when we could not look, as opposed to looked and found
        nothing. A zero score means very different things in each case."""
        return bool(self.fetch_errors) and not self.found_any

    @property
    def total_stars(self) -> int:
        return sum(int(r.get("stars_at_cutoff") or 0) for r in self.github_repos)

    @property
    def hn_points(self) -> int:
        return sum(int(p.get("points") or 0) for p in self.hn_posts)

    def summary(self) -> str:
        """One line describing what was actually visible at the cutoff."""
        bits = []
        if self.github_repos:
            bits.append(f"{len(self.github_repos)} repo(s), {self.total_stars} stars")
        if self.hn_posts:
            bits.append(f"{len(self.hn_posts)} HN post(s), {self.hn_points} points")
        return " · ".join(bits) or "no public evidence found"


class Verdict(BaseModel):
    """How one company fared in the backtest."""

    key: str
    company: str
    surfaced: bool
    score: float
    rank: int | None = None
    lead_time_days: int | None = None
    round_date: datetime | None = None
    round_stage: str = ""
    evidence: str = ""
    blinded_score: float | None = None
    is_control: bool = False
    # Normalized 0..1 signal values, kept per company so predictive power
    # can be attributed to individual signals rather than only to the
    # composite. Weight-free on purpose: scoring signals through the current
    # weights would measure our own assumptions reflected back at us.
    signal_values: dict[str, float] = Field(default_factory=dict)

    @property
    def lead_time_months(self) -> float | None:
        if self.lead_time_days is None:
            return None
        return round(self.lead_time_days / 30.44, 1)


class BacktestReport(BaseModel):
    cutoff: datetime
    thesis_id: str
    thesis_statement: str = ""
    threshold: float = 60.0
    verdicts: list[Verdict] = Field(default_factory=list)
    controls: list[Verdict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    blinded: bool = False
    unreachable: list[str] = Field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """False when sources were unreachable — the numbers are then a
        floor on a floor, and the report says so rather than implying the
        scorer did badly."""
        return not self.unreachable

    @property
    def outcomes(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.is_control]

    def metrics(self) -> "Metrics":
        return compute_metrics(
            [v.score for v in self.outcomes],
            [v.score for v in self.controls],
            [v.lead_time_days for v in self.outcomes if v.surfaced],
            threshold=self.threshold,
        )

    def observations(self) -> list:
        """Per-company signal values, for attributing predictive power to
        individual signals rather than only to the composite."""
        from scout.signal_eval import Observation

        return [
            Observation(key=v.key, raised=not v.is_control,
                        signal_values=v.signal_values, composite=v.score)
            for v in self.outcomes + self.controls
        ]

    def evaluate_signals(self, weights: dict[str, float] | None = None):
        from scout.signal_eval import evaluate_signals as _evaluate

        return _evaluate(self.observations(), weights)


class Metrics(BaseModel):
    """The numbers an investor should judge the system on."""

    n_outcomes: int
    n_controls: int
    recall: float                # share of outcomes at or above threshold
    auc: float                   # P(random outcome outranks random control)
    precision_at_n: float | None = None  # reviewing the top n_outcomes overall
    median_lead_days: float | None = None
    mean_outcome_score: float = 0.0
    mean_control_score: float = 0.0


# ----------------------------------------------------------------- metrics
# Pure. This is where the credibility of the whole exercise lives, so it is
# separated from any I/O and tested directly.


def auc_score(positives: list[float], negatives: list[float]) -> float:
    """Probability a random outcome outscores a random control.

    The Mann-Whitney U statistic, computed exactly rather than approximated
    — the sample sizes here are tens, not millions. Ties count as half,
    which is the standard convention and matters because a scorer that
    gives everything the same score should land at 0.5 (no discrimination),
    not 1.0.

    0.5 means the scoring is worthless; 1.0 means every outcome outranks
    every control.
    """
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 3)


def compute_metrics(
    outcome_scores: list[float],
    control_scores: list[float],
    lead_times: list[int | None] | None = None,
    threshold: float = 60.0,
) -> Metrics:
    """Recall, separation, and precision over a pooled review list."""
    n_out, n_ctl = len(outcome_scores), len(control_scores)
    recall = (
        sum(1 for s in outcome_scores if s >= threshold) / n_out if n_out else 0.0
    )

    # Precision proxy: if an investor reviewed the top n_outcomes of the
    # pooled list, how many would be real? This is the honest way to state
    # "is the ranking useful" without inventing a base rate.
    precision = None
    if n_out and n_ctl:
        pooled = [(s, True) for s in outcome_scores] + [
            (s, False) for s in control_scores
        ]
        pooled.sort(key=lambda pair: -pair[0])
        top = pooled[:n_out]
        precision = round(sum(1 for _, is_outcome in top if is_outcome) / n_out, 3)

    clean_leads = [d for d in (lead_times or []) if d is not None]
    return Metrics(
        n_outcomes=n_out,
        n_controls=n_ctl,
        recall=round(recall, 3),
        auc=auc_score(outcome_scores, control_scores),
        precision_at_n=precision,
        median_lead_days=round(median(clean_leads), 1) if clean_leads else None,
        mean_outcome_score=round(
            sum(outcome_scores) / n_out, 1) if n_out else 0.0,
        mean_control_score=round(
            sum(control_scores) / n_ctl, 1) if n_ctl else 0.0,
    )


def lead_time_days(round_date: datetime, cutoff: datetime) -> int:
    """How far ahead of the round the cutoff sits. Negative means the round
    had already happened, which disqualifies the company as an outcome."""
    if round_date.tzinfo is None:
        round_date = round_date.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return (round_date - cutoff).days


def eligible(outcome: Outcome, cutoff: datetime) -> bool:
    """An outcome only tests the system if its round came AFTER the cutoff.

    Scoring a company whose round was already public is not hindsight, it
    is reading the answer — so those are excluded rather than counted.
    """
    return lead_time_days(outcome.round_date, cutoff) > 0


# ---------------------------------------------------- point-in-time sources


async def _get_json(client: httpx.AsyncClient, url: str, **params) -> Any:
    resp = await client.get(url, params=params or None, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


async def hn_evidence(
    client: httpx.AsyncClient, terms: list[str], cutoff: datetime, limit: int = 20
) -> tuple[list[dict], list[str]]:
    """HN posts created strictly before the cutoff, plus any fetch failures.

    Algolia's `created_at_i` filter does the rewinding for us, and the
    archive is complete and free — which is what makes HN the most honest
    leg of the backtest.

    Failures are RETURNED, not swallowed. In a backtest "we found nothing"
    counts against the system, so silently reporting a network error as an
    empty result would understate the scorer and nobody would know. The two
    outcomes have to stay distinguishable all the way into the report.
    """
    before = int(cutoff.timestamp())
    seen: dict[str, dict] = {}
    errors: list[str] = []
    for term in terms:
        if not term:
            continue
        try:
            data = await _get_json(
                client, f"{_HN_API}/search",
                query=term,
                numericFilters=f"created_at_i<{before}",
                hitsPerPage=limit,
            )
        except Exception as exc:  # noqa: BLE001 — one dead query must not kill the run
            errors.append(f"HN search {term!r} failed: {type(exc).__name__}")
            continue
        for hit in data.get("hits", []):
            object_id = str(hit.get("objectID") or "")
            if not object_id or object_id in seen:
                continue
            created = hit.get("created_at_i")
            if created and int(created) >= before:
                continue  # belt and braces: never admit post-cutoff evidence
            seen[object_id] = {
                "title": hit.get("title") or hit.get("story_title") or "",
                "points": hit.get("points") or 0,
                "num_comments": hit.get("num_comments") or 0,
                "author": hit.get("author") or "",
                "url": hit.get("url") or "",
                "created_at": hit.get("created_at") or "",
            }
    return list(seen.values()), errors


async def github_repo_at(
    client: httpx.AsyncClient, full_name: str, cutoff: datetime, token: str = ""
) -> tuple[dict | None, str]:
    """A repo as it stood at the cutoff, and any error encountered.

    Star count is rebuilt by counting starring timestamps before the cutoff
    — GitHub does not expose historical counts, but it does expose WHEN each
    star was given, which is the same information. Beyond the pagination cap
    the count is recorded as a floor and flagged, rather than silently
    reported as exact.

    Returns (None, "") for a repo that legitimately did not exist yet, and
    (None, reason) when the lookup failed. Collapsing those two into one
    None is how a rate-limited backtest quietly turns into a bad-looking
    one.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        repo = await _get_json(client, f"{_GH_API}/repos/{full_name}")
    except Exception as exc:  # noqa: BLE001
        return None, f"GitHub {full_name} unreachable: {type(exc).__name__}"
    created_raw = repo.get("created_at") or ""
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    except ValueError:
        return None, f"GitHub {full_name} has no usable creation date"
    if created >= cutoff:
        return None, ""  # did not exist at the cutoff — a real exclusion

    stars, exact = 0, True
    star_headers = {**headers, "Accept": "application/vnd.github.star+json"}
    for page in range(1, _STAR_PAGES_MAX + 1):
        try:
            resp = await client.get(
                f"{_GH_API}/repos/{full_name}/stargazers",
                params={"per_page": _STARS_PER_PAGE, "page": page},
                headers=star_headers, timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception:  # noqa: BLE001
            exact = False
            break
        if not batch:
            break
        for entry in batch:
            starred = entry.get("starred_at") if isinstance(entry, dict) else None
            if not starred:
                continue
            try:
                when = datetime.fromisoformat(str(starred).replace("Z", "+00:00"))
            except ValueError:
                continue
            if when < cutoff:
                stars += 1
        if len(batch) < _STARS_PER_PAGE:
            break
        if page == _STAR_PAGES_MAX:
            exact = False

    return {
        "full_name": full_name,
        "created_at": created_raw,
        "description": repo.get("description") or "",
        "topics": repo.get("topics") or [],
        "language": repo.get("language") or "",
        "homepage": repo.get("homepage") or "",
        "stars_at_cutoff": stars,
        "stars_exact": exact,
        "stars_now": repo.get("stargazers_count") or 0,
    }, ""


async def github_user_repos(
    client: httpx.AsyncClient, user: str, cutoff: datetime, token: str = "",
    limit: int = 5,
) -> tuple[list[str], str]:
    """The user's repos that existed before the cutoff, most-starred first.

    Sorted by CURRENT stars only to choose which repos to examine — the
    counts that enter the score are always rebuilt as of the cutoff.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        repos = await _get_json(
            client, f"{_GH_API}/users/{user}/repos",
            sort="pushed", per_page=100,
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"GitHub user {user!r} unreachable: {type(exc).__name__}"
    eligible_repos = []
    for repo in repos if isinstance(repos, list) else []:
        created_raw = repo.get("created_at") or ""
        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created < cutoff and not repo.get("fork"):
            eligible_repos.append(
                (repo.get("stargazers_count") or 0, repo.get("full_name") or "")
            )
    eligible_repos.sort(reverse=True)
    return [name for _stars, name in eligible_repos[:limit] if name], ""


async def reconstruct(
    outcome: Outcome,
    cutoff: datetime,
    *,
    github_token: str = "",
    client: httpx.AsyncClient | None = None,
) -> Evidence:
    """Gather everything publicly visible about one company at the cutoff."""
    owns_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    notes: list[str] = []
    try:
        search_terms = [outcome.company] + outcome.hn_users + (
            [outcome.domain] if outcome.domain else []
        )
        hn_posts, hn_errors = await hn_evidence(client, search_terms, cutoff)
        errors: list[str] = list(hn_errors)

        repo_names = list(outcome.github_repos)
        for user in outcome.github_users:
            found, error = await github_user_repos(
                client, user, cutoff, github_token)
            repo_names.extend(found)
            if error:
                errors.append(error)
        repos: list[dict] = []
        for name in dict.fromkeys(repo_names):
            repo, error = await github_repo_at(
                client, name, cutoff, github_token)
            if error:
                errors.append(error)
            if repo:
                repos.append(repo)
                if not repo["stars_exact"]:
                    notes.append(
                        f"{name}: star count is a floor (pagination cap reached)"
                    )
        if outcome.x_handles:
            notes.append(
                "X history not reconstructible without full-archive access — "
                "scored without X signals, so this is a lower bound"
            )

        bio = " ".join(
            [r["description"] for r in repos if r["description"]][:2]
            + [p["title"] for p in hn_posts[:2]]
        ).strip()
        website = outcome.domain or next(
            (r["homepage"] for r in repos if r.get("homepage")), ""
        )
        return Evidence(
            key=outcome.key, company=outcome.company, as_of=cutoff,
            hn_posts=hn_posts, github_repos=repos, bio=bio[:600],
            website=website, notes=notes, fetch_errors=errors,
        )
    finally:
        if owns_client:
            await client.aclose()


def _redact(text: str, evidence: Evidence) -> str:
    """Remove IDENTIFYING tokens while leaving the substance intact.

    The distinction matters more than it looks. Blanking every token in a
    repo's full name would delete words like "runtime" or "inference" —
    exactly the evidence the blinded score is supposed to judge — and would
    make the blinded run look artificially weak, which is the opposite of
    the control's purpose. So: the company name, its slug forms, and the
    GitHub owner go; a repo's own name goes only if it embeds the company
    name.
    """
    slug = re.sub(r"[^a-z0-9]+", "", evidence.company.lower())
    tokens = {evidence.company, evidence.key, slug}
    for repo in evidence.github_repos:
        full_name = str(repo.get("full_name") or "")
        owner, _, short = full_name.partition("/")
        if owner:
            tokens.add(owner)
        # Only identifying if it carries the company name inside it.
        if short and slug and slug in re.sub(r"[^a-z0-9]+", "", short.lower()):
            tokens.add(short)
    for token in sorted(filter(None, tokens), key=len, reverse=True):
        text = re.sub(re.escape(token), "[redacted]", text, flags=re.I)
    return text


def evidence_to_account(evidence: Evidence, blinded: bool = False) -> Account:
    """Turn reconstructed evidence into the Account the real scorer takes.

    Blinded mode strips the company and repo names, leaving only the
    substance (what it does, how much traction). Comparing blinded to
    unblinded scores measures how much of the result is the model
    RECOGNISING a company it read about in training, rather than judging
    the evidence — the single biggest threat to a backtest's validity.
    """
    bio = evidence.bio
    handle = evidence.key
    name = evidence.company
    if blinded:
        bio = _redact(bio, evidence)
        handle, name = f"anon-{abs(hash(evidence.key)) % 10**8}", "[redacted]"
    return Account(
        id=handle,
        handle=handle,
        name=name,
        bio=bio,
        website="" if blinded else evidence.website,
        followers=evidence.followers,
        source="hindsight",
        created_at=evidence.as_of,
    )


# ------------------------------------------------------------------ report


def render_report(report: BacktestReport, signal_evaluation=None) -> str:
    """Markdown an investor can read without needing the code explained."""
    metrics = report.metrics()
    surfaced = [v for v in report.outcomes if v.surfaced]
    lines = [
        f"# Hindsight backtest — {report.cutoff:%d %B %Y}",
        "",
        f"**Thesis:** {report.thesis_statement or report.thesis_id}",
        "",
        "The question: scoring only evidence that was public on "
        f"{report.cutoff:%d %B %Y}, would Scout have surfaced companies that "
        "went on to raise afterwards?",
        "",
    ]
    if not report.trustworthy:
        lines += [
            f"> **These numbers are not usable.** {len(report.unreachable)} "
            "company/companies could not be checked because the sources were "
            "unreachable, and an unchecked company scores zero — so the "
            "results below understate the scorer by an unknown margin. "
            "Re-run with working network access before drawing any "
            "conclusion.",
            "",
        ]
    lines += [
        "## Results",
        "",
        f"- **{len(surfaced)} of {metrics.n_outcomes}** outcomes scored at or "
        f"above {report.threshold:.0f} (recall **{metrics.recall:.0%}**)",
        f"- **AUC {metrics.auc:.2f}** — the chance a company that raised "
        "outranks one that did not. 0.5 is a coin flip.",
    ]
    if metrics.precision_at_n is not None:
        lines.append(
            f"- Reviewing the top {metrics.n_outcomes} of all "
            f"{metrics.n_outcomes + metrics.n_controls} scored companies would "
            f"have surfaced **{metrics.precision_at_n:.0%}** real outcomes"
        )
    if metrics.median_lead_days is not None:
        lines.append(
            f"- Median lead time **{metrics.median_lead_days / 30.44:.1f} months** "
            "before the round was announced"
        )
    lines += [
        f"- Mean score: **{metrics.mean_outcome_score}** outcomes vs "
        f"**{metrics.mean_control_score}** controls",
        "",
        "## Companies",
        "",
        "| Company | Score | Surfaced | Lead time | Round | Evidence at cutoff |",
        "| --- | ---: | :---: | ---: | --- | --- |",
    ]
    for verdict in sorted(report.outcomes, key=lambda v: -v.score):
        lead = (f"{verdict.lead_time_months} mo"
                if verdict.lead_time_months is not None else "—")
        mark = "yes" if verdict.surfaced else "no"
        round_label = " ".join(
            x for x in [verdict.round_stage,
                        verdict.round_date.strftime("%b %Y")
                        if verdict.round_date else ""] if x
        )
        lines.append(
            f"| {verdict.company} | {verdict.score:.0f} | {mark} | {lead} | "
            f"{round_label} | {verdict.evidence} |"
        )

    if report.blinded:
        lines += ["", "## Blinded control", "",
                  "Scores with company and repo names redacted, to separate "
                  "judgment of the evidence from recognition of a company the "
                  "model may have read about during training."]
        for verdict in sorted(report.outcomes, key=lambda v: -v.score):
            if verdict.blinded_score is not None:
                delta = verdict.score - verdict.blinded_score
                lines.append(
                    f"- {verdict.company}: {verdict.score:.0f} named vs "
                    f"{verdict.blinded_score:.0f} blinded ({delta:+.0f})"
                )

    if signal_evaluation is not None:
        lines += render_signal_section(signal_evaluation)

    lines += ["", "## What this does and does not show", ""]
    for limitation in report.limitations or default_limitations():
        lines.append(f"- {limitation}")
    return "\n".join(lines)


def render_signal_section(evaluation) -> list[str]:
    """Which individual signals carried information — the part that turns a
    validation report into an instrument you can tune against."""
    lines = ["", "## Which signals actually predicted this", ""]
    if evaluation.underpowered:
        lines += [f"_{note}_" for note in evaluation.notes]
        return lines

    lines += [
        "| Signal | AUC | 95% CI | Coverage | Adds beyond the rest | Verdict |",
        "| --- | ---: | :---: | ---: | ---: | --- |",
    ]
    for finding in evaluation.ranked:
        marginal = (f"{finding.marginal_auc:+.3f}"
                    if finding.marginal_auc is not None else "—")
        lines.append(
            f"| {finding.name} | {finding.auc:.2f} | "
            f"{finding.ci_low:.2f}–{finding.ci_high:.2f} | "
            f"{finding.coverage:.0%} | {marginal} | {finding.verdict} |"
        )
    lines += [""] + [f"- {note}" for note in evaluation.notes]
    return lines


def default_limitations() -> list[str]:
    """Stated up front, because a backtest that hides these is marketing."""
    return [
        "X/Twitter history cannot be reconstructed without paid full-archive "
        "access, so these runs score without the X signals production uses. "
        "Every number here is therefore a lower bound: real recall should be "
        "HIGHER in production, not lower.",
        "The language model has training knowledge of well-known companies, "
        "which can inflate scores for names it recognises. The blinded column "
        "measures that effect directly rather than assuming it away.",
        "GitHub star counts are rebuilt from starring timestamps; repos with "
        "more than "
        f"{_STAR_PAGES_MAX * _STARS_PER_PAGE:,} stars report a floor, not an "
        "exact figure.",
        "Controls are companies visible in the same sources and window that "
        "did not go on to raise. They are a sample, not the full negative "
        "universe, so precision is indicative rather than absolute.",
        "A backtest measures the scorer, not the sourcing. It cannot show "
        "whether discovery would have found a company in the first place — "
        "only how it would have ranked once seen.",
    ]


def build_verdict(
    outcome: Outcome,
    evidence: Evidence,
    score: float,
    cutoff: datetime,
    threshold: float,
    blinded_score: float | None = None,
) -> Verdict:
    return Verdict(
        key=outcome.key,
        company=outcome.company,
        surfaced=score >= threshold,
        score=round(score, 1),
        lead_time_days=lead_time_days(outcome.round_date, cutoff),
        round_date=outcome.round_date,
        round_stage=outcome.round_stage,
        evidence=evidence.summary(),
        blinded_score=round(blinded_score, 1) if blinded_score is not None else None,
    )


def rank_verdicts(report: BacktestReport) -> None:
    """Assign each outcome its rank in the pooled list, in place.

    Rank is against controls too — "3rd of 40 scored" is the number that
    means something, where "scored 82" alone does not.
    """
    pooled = sorted(report.verdicts + report.controls, key=lambda v: -v.score)
    positions = {id(v): i + 1 for i, v in enumerate(pooled)}
    for verdict in report.verdicts:
        verdict.rank = positions.get(id(verdict))


# ------------------------------------------------------------------ runner


def score_evidence(
    evidence_list: list[Evidence],
    thesis,
    settings,
    store=None,
    blinded: bool = False,
) -> dict[str, tuple[float, dict[str, float]]]:
    """Score reconstructed evidence with the SAME pipeline production uses.

    Returns {company key: (score, {signal name: normalized value})} — the
    per-signal values are what makes signal attribution possible later.

    Deliberately not a reimplementation: heuristics → classifier → weighted
    score, exactly as a live run does. A backtest against a parallel scoring
    path would measure the parallel path, which is worth nothing.

    Tweets are empty throughout — that is the X limitation made concrete
    rather than hidden, and it means signals keyed on tweets contribute
    zero, biasing every score DOWNWARD.
    """
    from scout.models import Lead
    from scout.score import score_leads
    from scout.signals.heuristics import run_heuristics
    from scout.signals.llm import classify

    accounts = [evidence_to_account(e, blinded=blinded) for e in evidence_list]
    by_handle = {
        account.handle.lower(): evidence
        for account, evidence in zip(accounts, evidence_list)
    }
    candidates = [(account, []) for account in accounts]
    verdicts = classify(candidates, thesis, settings, store=None)

    leads: list[Lead] = []
    for account in accounts:
        signals, disqualified = run_heuristics(account, [], thesis)
        if disqualified:
            continue
        leads.append(Lead(
            account=account,
            signals=signals,
            llm=verdicts.get(account.handle.lower()),
        ))
    scored = score_leads(leads, thesis)
    return {
        by_handle[lead.account.handle.lower()].key: (
            lead.score,
            {signal.name: signal.value for signal in lead.signals},
        )
        for lead in scored
        if lead.account.handle.lower() in by_handle
    }


async def gather_evidence(
    outcomes: list[Outcome],
    cutoff: datetime,
    *,
    github_token: str = "",
    concurrency: int = 4,
    on_progress=None,
) -> list[Evidence]:
    """Reconstruct every company concurrently, politely."""
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def one(outcome: Outcome) -> Evidence:
            nonlocal done
            async with semaphore:
                evidence = await reconstruct(
                    outcome, cutoff, github_token=github_token, client=client
                )
            done += 1
            if on_progress:
                on_progress(done, len(outcomes), outcome.company)
            return evidence

        return list(await asyncio.gather(*(one(o) for o in outcomes)))


def run_backtest(
    outcomes: list[Outcome],
    controls: list[Outcome],
    cutoff: datetime,
    thesis,
    settings,
    *,
    threshold: float = 60.0,
    blinded: bool = False,
    on_progress=None,
) -> BacktestReport:
    """The whole exercise: reconstruct, score, compare, report.

    Outcomes whose round predates the cutoff are dropped rather than
    counted — scoring a company whose raise was already public is reading
    the answer, not predicting it.
    """
    usable = [o for o in outcomes if eligible(o, cutoff)]
    skipped = len(outcomes) - len(usable)
    github_token = getattr(settings, "github_token", "") or ""

    all_targets = usable + controls
    evidence = asyncio.run(gather_evidence(
        all_targets, cutoff, github_token=github_token, on_progress=on_progress,
    ))
    found = [e for e in evidence if e.found_any]
    scores = score_evidence(found, thesis, settings, blinded=False)
    blinded_scores = (
        score_evidence(found, thesis, settings, blinded=True) if blinded else {}
    )

    by_key = {e.key: e for e in evidence}
    report = BacktestReport(
        cutoff=cutoff,
        thesis_id=getattr(thesis, "id", "") or "",
        thesis_statement=getattr(thesis, "thesis", "") or "",
        threshold=threshold,
        blinded=blinded,
    )
    for outcome in usable:
        ev = by_key.get(outcome.key) or Evidence(
            key=outcome.key, company=outcome.company, as_of=cutoff)
        score, signal_values = scores.get(outcome.key, (0.0, {}))
        blinded_pair = blinded_scores.get(outcome.key) if blinded else None
        verdict = build_verdict(
            outcome, ev, score, cutoff, threshold,
            blinded_pair[0] if blinded_pair else None,
        )
        verdict.signal_values = signal_values
        report.verdicts.append(verdict)
    for control in controls:
        ev = by_key.get(control.key) or Evidence(
            key=control.key, company=control.company, as_of=cutoff)
        score, signal_values = scores.get(control.key, (0.0, {}))
        verdict = build_verdict(control, ev, score, cutoff, threshold)
        verdict.is_control = True
        verdict.signal_values = signal_values
        report.controls.append(verdict)

    report.limitations = default_limitations()
    if skipped:
        report.limitations.insert(0, (
            f"{skipped} supplied company/companies raised on or before the "
            "cutoff and were excluded — their outcome was already public."
        ))
    # "Looked and found nothing" and "could not look" both produce a zero,
    # and they mean opposite things. Separating them is what stops a
    # rate-limited or firewalled run from silently reading as a bad result.
    unchecked = [e for e in evidence if e.unchecked]
    missing = [e.company for e in evidence if not e.found_any and not e.unchecked]
    if missing:
        report.limitations.append(
            f"No public evidence was found at the cutoff for: "
            f"{', '.join(missing[:8])}"
            + (" …" if len(missing) > 8 else "")
            + ". These score zero, which counts AGAINST the system here."
        )
    if unchecked:
        report.unreachable = [e.company for e in unchecked]
        sample = "; ".join(unchecked[0].fetch_errors[:2])
        report.limitations.insert(0, (
            f"⚠ {len(unchecked)} company/companies could not be checked at all "
            f"({', '.join(report.unreachable[:6])}"
            + (" …" if len(unchecked) > 6 else "")
            + f") — the sources were unreachable, e.g. {sample}. They score "
            "zero here, so THIS RUN UNDERSTATES the scorer. Re-run with "
            "network access (and a GitHub token, which raises the rate limit "
            "from 60 to 5,000 requests an hour) before showing it to anyone."
        ))
    rank_verdicts(report)
    return report


def load_outcomes(path) -> tuple[list[Outcome], list[Outcome]]:
    """Read the outcomes file: (outcomes, controls).

    YAML, because a human curates this by hand — usually the firm's own
    'ones that got away', which is the most persuasive possible input.
    """
    import yaml

    raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
    if isinstance(raw, list):  # bare list = all outcomes, no controls
        raw = {"outcomes": raw}
    outcomes = [Outcome.model_validate(item) for item in raw.get("outcomes", [])]
    controls = [Outcome.model_validate(item) for item in raw.get("controls", [])]
    return outcomes, controls


# ------------------------------------------------------- longitudinal sweep


def sweep_cutoffs(
    latest: datetime, n_windows: int = 3, months_apart: int = 6
) -> list[datetime]:
    """Evenly spaced cutoffs, oldest first.

    Spacing matters more than count. Windows a month apart share almost all
    their evidence, so their results are not independent observations and a
    "trend" across them is mostly the same measurement repeated. Six months
    is roughly the shortest gap at which a signal's power can actually have
    moved.
    """
    return [
        latest - timedelta(days=int(months_apart * 30.44 * i))
        for i in range(n_windows - 1, -1, -1)
    ]


def run_sweep(
    outcomes: list[Outcome],
    controls: list[Outcome],
    cutoffs: list[datetime],
    thesis,
    settings,
    *,
    threshold: float = 60.0,
    on_progress=None,
    on_window=None,
) -> list[tuple[datetime, BacktestReport]]:
    """Backtest at several points in the past, oldest first.

    This is what answers "is this signal wearing out?". A signal like
    stealth language in a bio worked when few founders used it and stopped
    working once everyone did; measured at one cutoff that is invisible,
    and across three it is obvious.

    Each window drops outcomes whose round predates it, so later windows
    naturally have fewer companies — which is exactly why the per-signal
    power check refuses to report on an underpowered window rather than
    letting a shrinking sample masquerade as a declining signal.
    """
    results: list[tuple[datetime, BacktestReport]] = []
    for cutoff in sorted(cutoffs):
        report = run_backtest(
            outcomes, controls, cutoff, thesis, settings,
            threshold=threshold, blinded=False, on_progress=on_progress,
        )
        results.append((cutoff, report))
        if on_window:
            on_window(cutoff, report)
    return results


def build_signal_trends(
    sweep: list[tuple[datetime, BacktestReport]],
    weights: dict[str, float] | None = None,
):
    """Per-signal power across the sweep's windows."""
    from scout.signal_eval import build_trends

    return build_trends([
        (cutoff.date().isoformat(), report.evaluate_signals(weights))
        for cutoff, report in sweep
    ])


def render_trend_section(trends) -> list[str]:
    """How each signal's power has moved — including the ones going stale."""
    if not trends:
        return []
    lines = ["", "## Signal power over time", "",
             "Each signal measured at successive cutoffs. A falling number "
             "is only called a decay when the confidence intervals separate "
             "— at these sample sizes two AUCs can look far apart and be "
             "perfectly consistent with no change at all.", ""]
    decayed = [t for t in trends if t.decayed]
    if decayed:
        lines += [
            "**Losing power:** " + ", ".join(t.name for t in decayed),
            "",
        ]
    lines += ["| Signal | " + " | ".join(
        point[0] for point in trends[0].points) + " | Reading |",
        "| --- |" + " ---: |" * len(trends[0].points) + " --- |"]
    for trend in trends:
        cells = " | ".join(f"{point[1]:.2f}" for point in trend.points)
        lines.append(f"| {trend.name} | {cells} | {trend.summary} |")
    return lines
