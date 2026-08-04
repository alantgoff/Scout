"""The backtest: metrics, point-in-time discipline, and report honesty.

The metrics are pure and get tested hard — a backtest whose arithmetic is
wrong is worse than no backtest, because it produces a confident number.
The network reconstruction is tested against recorded-shape fakes, with
particular attention to the one property that matters: no evidence created
after the cutoff may ever enter a score.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scout import hindsight as hs

UTC = timezone.utc
CUTOFF = datetime(2025, 2, 1, tzinfo=UTC)


def outcome(company: str, months_after: int = 8, **kw) -> hs.Outcome:
    return hs.Outcome(
        company=company,
        round_date=CUTOFF + timedelta(days=int(months_after * 30.44)),
        **kw,
    )


# --- metrics --------------------------------------------------------------------


def test_auc_is_a_coin_flip_when_scores_are_identical() -> None:
    """The property that stops a flat scorer looking perfect: ties are half
    a win, so scoring everything the same lands at 0.5, not 1.0."""
    assert hs.auc_score([70.0, 70.0], [70.0, 70.0]) == 0.5


def test_auc_spans_worthless_to_perfect() -> None:
    assert hs.auc_score([90.0, 80.0], [20.0, 10.0]) == 1.0   # perfect separation
    assert hs.auc_score([10.0, 20.0], [80.0, 90.0]) == 0.0   # perfectly inverted
    # One outcome beaten by one control out of four pairings.
    assert hs.auc_score([90.0, 30.0], [40.0, 20.0]) == 0.75


def test_auc_without_a_control_group_is_undefined_not_flattering() -> None:
    """No controls must not silently produce a great-looking number."""
    assert hs.auc_score([90.0, 95.0], []) == 0.0
    assert hs.auc_score([], [10.0]) == 0.0


def test_recall_counts_scores_at_or_above_the_threshold() -> None:
    metrics = hs.compute_metrics([80.0, 60.0, 20.0], [10.0], threshold=60.0)
    assert metrics.recall == pytest.approx(2 / 3, abs=0.001)
    assert metrics.n_outcomes == 3 and metrics.n_controls == 1


def test_precision_at_n_measures_a_realistic_review_list() -> None:
    """If an investor reviewed the top N of everything scored, how many
    would be real? Three outcomes, three controls, ranking imperfect."""
    metrics = hs.compute_metrics(
        [90.0, 85.0, 30.0],           # one outcome ranks below a control
        [70.0, 20.0, 10.0],
        threshold=60.0,
    )
    # Top 3 pooled = 90, 85, 70 → two outcomes, one control.
    assert metrics.precision_at_n == pytest.approx(2 / 3, abs=0.001)
    assert metrics.mean_outcome_score == pytest.approx(68.3, abs=0.1)
    assert metrics.mean_control_score == pytest.approx(33.3, abs=0.1)


def test_lead_time_is_reported_as_a_median_not_a_mean() -> None:
    """One company caught three years early must not drag the headline."""
    metrics = hs.compute_metrics(
        [90.0, 90.0, 90.0], [10.0], lead_times=[30, 60, 1000], threshold=60.0)
    assert metrics.median_lead_days == 60.0


def test_metrics_survive_empty_inputs() -> None:
    metrics = hs.compute_metrics([], [], threshold=60.0)
    assert metrics.recall == 0.0 and metrics.auc == 0.0
    assert metrics.precision_at_n is None and metrics.median_lead_days is None


# --- point-in-time discipline ---------------------------------------------------


def test_a_company_that_already_raised_is_not_hindsight() -> None:
    """Scoring a company whose round was public is reading the answer."""
    already = hs.Outcome(company="Old News",
                         round_date=CUTOFF - timedelta(days=30))
    later = outcome("Still Ahead")
    assert hs.eligible(already, CUTOFF) is False
    assert hs.eligible(later, CUTOFF) is True
    # Same-day is also excluded — the round was announced by then.
    same_day = hs.Outcome(company="Same Day", round_date=CUTOFF)
    assert hs.eligible(same_day, CUTOFF) is False


def test_lead_time_is_measured_from_the_cutoff_to_the_round() -> None:
    assert hs.lead_time_days(CUTOFF + timedelta(days=90), CUTOFF) == 90
    assert hs.lead_time_days(CUTOFF - timedelta(days=10), CUTOFF) == -10
    # Naive datetimes are treated as UTC rather than raising.
    assert hs.lead_time_days(datetime(2025, 5, 2), CUTOFF) == 90


@pytest.mark.anyio
async def test_hn_evidence_never_admits_a_post_from_after_the_cutoff() -> None:
    """The core invariant. The API filter is asked for, and the result is
    re-checked — a source that ignored the filter must not leak through."""
    before = int(CUTOFF.timestamp())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"hits": [
            {"objectID": "1", "title": "Show HN: our runtime", "points": 120,
             "created_at_i": before - 86_400, "created_at": "2025-01-31"},
            # The API should never return this; if it does, we drop it.
            {"objectID": "2", "title": "Show HN: after the cutoff",
             "points": 400, "created_at_i": before + 86_400},
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        posts, errors = await hs.hn_evidence(client, ["example"], CUTOFF)

    assert captured["numericFilters"] == f"created_at_i<{before}"
    assert [p["title"] for p in posts] == ["Show HN: our runtime"]
    assert errors == []


@pytest.mark.anyio
async def test_github_stars_are_rebuilt_from_starring_timestamps() -> None:
    """Today's star count is not evidence about the past. Only stars given
    before the cutoff may count."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stargazers"):
            if request.url.params.get("page") == "1":
                return httpx.Response(200, json=[
                    {"starred_at": "2024-06-01T00:00:00Z"},
                    {"starred_at": "2024-12-01T00:00:00Z"},
                    {"starred_at": "2025-06-01T00:00:00Z"},  # after the cutoff
                ])
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={
            "created_at": "2024-01-15T00:00:00Z",
            "description": "fast inference runtime",
            "topics": ["llm"], "language": "Rust",
            "homepage": "https://example.dev", "stargazers_count": 5000,
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repo, error = await hs.github_repo_at(client, "ex/runtime", CUTOFF)

    assert repo is not None and error == ""
    assert repo["stars_at_cutoff"] == 2      # not 3, and certainly not 5000
    assert repo["stars_now"] == 5000         # kept for context, never scored
    assert repo["stars_exact"] is True


@pytest.mark.anyio
async def test_a_repo_created_after_the_cutoff_is_inadmissible() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "created_at": "2025-06-01T00:00:00Z", "stargazers_count": 900})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repo, error = await hs.github_repo_at(client, "ex/new", CUTOFF)
    # Did not exist yet: a legitimate exclusion, NOT a failure.
    assert repo is None and error == ""


@pytest.mark.anyio
async def test_star_pagination_cap_reports_a_floor_rather_than_guessing() -> None:
    """A very popular repo cannot be counted exactly, and says so."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stargazers"):
            return httpx.Response(200, json=[
                {"starred_at": "2024-06-01T00:00:00Z"}] * 100)
        return httpx.Response(200, json={
            "created_at": "2023-01-15T00:00:00Z", "stargazers_count": 99_000})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repo, _error = await hs.github_repo_at(client, "ex/famous", CUTOFF)

    assert repo["stars_exact"] is False
    assert repo["stars_at_cutoff"] == hs._STAR_PAGES_MAX * 100


# --- blinding -------------------------------------------------------------------


def test_blinding_removes_the_name_the_model_might_recognise() -> None:
    """The biggest threat to a backtest is the model knowing the answer."""
    evidence = hs.Evidence(
        key="acme-labs", company="Acme Labs", as_of=CUTOFF,
        bio="Acme Labs builds a fast inference runtime",
        website="https://acme.dev",
        github_repos=[{"full_name": "acmelabs/runtime", "stars_at_cutoff": 40}],
    )
    named = hs.evidence_to_account(evidence, blinded=False)
    blinded = hs.evidence_to_account(evidence, blinded=True)

    assert "Acme Labs" in named.bio and named.name == "Acme Labs"
    assert "Acme" not in blinded.bio
    assert "runtime" in blinded.bio          # the substance survives
    assert blinded.name == "[redacted]"
    assert blinded.website == ""             # the domain would give it away
    assert blinded.handle != named.handle


# --- report -------------------------------------------------------------------


def _report(blinded: bool = False) -> hs.BacktestReport:
    report = hs.BacktestReport(
        cutoff=CUTOFF, thesis_id="ai-infra",
        thesis_statement="AI infrastructure", threshold=60.0, blinded=blinded,
    )
    report.verdicts = [
        hs.Verdict(key="hot", company="HotCo", surfaced=True, score=88.0,
                   lead_time_days=240, round_date=CUTOFF + timedelta(days=240),
                   round_stage="Series A", evidence="2 repos, 900 stars",
                   blinded_score=71.0 if blinded else None),
        hs.Verdict(key="missed", company="MissedCo", surfaced=False, score=31.0,
                   lead_time_days=120, round_date=CUTOFF + timedelta(days=120),
                   round_stage="Seed", evidence="no public evidence found",
                   blinded_score=29.0 if blinded else None),
    ]
    report.controls = [
        hs.Verdict(key="c1", company="QuietCo", surfaced=False, score=22.0,
                   is_control=True),
        hs.Verdict(key="c2", company="StillQuiet", surfaced=False, score=40.0,
                   is_control=True),
    ]
    report.limitations = hs.default_limitations()
    return report


def test_report_states_recall_separation_and_lead_time() -> None:
    report = _report()
    metrics = report.metrics()
    assert metrics.n_outcomes == 2 and metrics.n_controls == 2
    assert metrics.recall == 0.5
    assert metrics.auc == 0.75  # HotCo beats both; MissedCo beats neither
    text = hs.render_report(report)
    assert "recall **50%**" in text
    assert "AUC 0.75" in text
    assert "HotCo" in text and "MissedCo" in text
    assert "0.5 is a coin flip" in text


def test_report_always_carries_its_limitations() -> None:
    """A backtest that hides these is marketing, so they are not optional."""
    text = hs.render_report(_report())
    assert "What this does and does not show" in text
    assert "lower bound" in text.lower()
    assert "training" in text.lower()          # the leakage caveat
    assert "measures the scorer, not the sourcing" in text


def test_blinded_report_shows_the_recognition_gap() -> None:
    text = hs.render_report(_report(blinded=True))
    assert "Blinded control" in text
    assert "88 named vs 71 blinded (+17)" in text


def test_ranks_are_assigned_against_the_controls() -> None:
    """"3rd of 40" means something; "scored 82" alone does not."""
    report = _report()
    hs.rank_verdicts(report)
    ranks = {v.company: v.rank for v in report.outcomes}
    assert ranks["HotCo"] == 1     # 88 beats every control
    assert ranks["MissedCo"] == 3  # behind StillQuiet (40), ahead of QuietCo (22)


def test_evidence_summary_is_honest_about_finding_nothing() -> None:
    empty = hs.Evidence(key="x", company="X", as_of=CUTOFF)
    assert empty.found_any is False
    assert empty.summary() == "no public evidence found"
    rich = hs.Evidence(
        key="y", company="Y", as_of=CUTOFF,
        github_repos=[{"full_name": "y/z", "stars_at_cutoff": 300}],
        hn_posts=[{"title": "Show HN", "points": 150}],
    )
    assert rich.summary() == "1 repo(s), 300 stars · 1 HN post(s), 150 points"


# --- inputs ---------------------------------------------------------------------


def test_load_outcomes_reads_both_groups(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.yaml"
    path.write_text(
        "outcomes:\n"
        "  - company: HotCo\n"
        "    round_date: 2025-09-15\n"
        "    round_stage: Series A\n"
        "    github_users: [hotco]\n"
        "controls:\n"
        "  - company: QuietCo\n"
        "    round_date: 2099-01-01\n"
    )
    outcomes, controls = hs.load_outcomes(path)
    assert [o.company for o in outcomes] == ["HotCo"]
    assert [c.company for c in controls] == ["QuietCo"]
    assert outcomes[0].github_users == ["hotco"]
    assert outcomes[0].key == "hotco"


def test_a_bare_list_is_treated_as_outcomes_with_no_controls(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.yaml"
    path.write_text("- company: HotCo\n  round_date: 2025-09-15\n")
    outcomes, controls = hs.load_outcomes(path)
    assert len(outcomes) == 1 and controls == []


def test_company_key_is_stable_and_filesystem_safe() -> None:
    assert hs.Outcome(company="Acme Labs, Inc.",
                      round_date=CUTOFF).key == "acme-labs-inc"


# --- "could not look" must never masquerade as "found nothing" ------------------


@pytest.mark.anyio
async def test_a_failed_fetch_is_reported_not_swallowed() -> None:
    """A network failure that reads as 'no evidence' would silently
    understate the scorer, and nobody would know to re-run."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        posts, errors = await hs.hn_evidence(client, ["example"], CUTOFF)
        repo, repo_error = await hs.github_repo_at(client, "ex/x", CUTOFF)

    assert posts == [] and errors and "failed" in errors[0]
    assert repo is None and "unreachable" in repo_error


def test_evidence_distinguishes_unchecked_from_genuinely_absent() -> None:
    absent = hs.Evidence(key="a", company="A", as_of=CUTOFF)
    unchecked = hs.Evidence(key="b", company="B", as_of=CUTOFF,
                            fetch_errors=["HN search 'B' failed: ConnectError"])
    assert absent.found_any is False and absent.unchecked is False
    assert unchecked.found_any is False and unchecked.unchecked is True


def test_report_refuses_to_look_credible_when_sources_were_unreachable() -> None:
    """The failure mode this guards against: showing a partner a backtest
    that says 20% recall when really the network was down."""
    report = _report()
    report.unreachable = ["MissedCo"]
    assert report.trustworthy is False
    text = hs.render_report(report)
    assert "These numbers are not usable" in text
    assert "understate" in text
    # And a clean run carries no such warning.
    assert "not usable" not in hs.render_report(_report())
