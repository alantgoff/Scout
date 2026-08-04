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


def test_report_names_every_way_it_could_mislead() -> None:
    """The limitations are the product here. Each of these is a distinct
    fallacy a competent reader would otherwise find first."""
    text = hs.render_report(_report())
    assert "What this does and does not show" in text
    lowered = text.lower()
    # The deepest one: the thesis encodes knowledge of these outcomes.
    assert "thesis is not frozen" in lowered
    # Both sides of the selection problem.
    assert "chosen from memory" in lowered
    assert "hand-picked" in lowered
    assert "raised quietly" in lowered        # mislabelled controls
    # The number most likely to be misread.
    assert "not production precision" in lowered
    # Leakage, bounded in both directions rather than assumed away.
    assert "overstates leakage" in lowered
    # Scope of the claim, and the longitudinal caveat.
    assert "measures the scorer" in lowered
    assert "not independent observations" in lowered
    assert "lower bound" in lowered


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


# --- base rates: the correction that stops the headline being misread ----------


def test_precision_collapses_at_realistic_base_rates() -> None:
    """The single most dangerous misreading. A scorer catching 80% of
    outcomes while flagging 10% of controls looks like 84% precision on a
    balanced backtest pool, and is 14% in a funnel where 2% raise."""
    balanced = hs.precision_at_base_rate(0.8, 0.1, 0.40)
    realistic = hs.precision_at_base_rate(0.8, 0.1, 0.02)
    assert balanced == pytest.approx(0.842, abs=0.005)
    assert realistic == pytest.approx(0.140, abs=0.005)
    assert balanced > realistic * 5   # off by an order of magnitude


def test_lift_survives_the_base_rate_problem() -> None:
    """Precision falls with the base rate but lift does not — which is why
    lift is the honest headline for a sourcing tool."""
    assert hs.lift_at_base_rate(0.8, 0.1, 0.02) == pytest.approx(7.0, abs=0.1)
    assert hs.lift_at_base_rate(0.8, 0.1, 0.01) == pytest.approx(7.5, abs=0.1)
    # A worthless scorer flags outcomes and controls alike: no lift at all.
    assert hs.lift_at_base_rate(0.5, 0.5, 0.02) == pytest.approx(1.0, abs=0.01)


def test_a_scorer_that_flags_nothing_has_no_precision_not_perfect_precision() -> None:
    assert hs.precision_at_base_rate(0.0, 0.0, 0.02) == 0.0
    # Degenerate base rates cannot produce a number.
    assert hs.precision_at_base_rate(0.8, 0.1, 0.0) == 0.0
    assert hs.precision_at_base_rate(0.8, 0.1, 1.0) == 0.0


def test_metrics_expose_the_rates_that_actually_transfer() -> None:
    """Precision depends on how many controls happened to be picked; TPR
    and FPR do not, which is what makes them portable to production."""
    metrics = hs.compute_metrics([80.0, 70.0, 40.0], [70.0, 20.0, 10.0, 5.0],
                                 threshold=60.0)
    assert metrics.true_positive_rate == pytest.approx(2 / 3, abs=0.01)
    assert metrics.false_positive_rate == pytest.approx(1 / 4, abs=0.01)


def test_report_restates_performance_at_production_prevalence() -> None:
    text = hs.render_report(_report())
    assert "What this means in production" in text
    assert "Lift vs random" in text
    assert "Companies read per find" in text


# --- fairness: the most likely way this gets rigged by accident ----------------


def _fairness_report(outcome_evidence: list[str],
                     control_evidence: list[str]) -> hs.BacktestReport:
    report = hs.BacktestReport(cutoff=CUTOFF, thesis_id="t", threshold=60.0)
    for i, evidence in enumerate(outcome_evidence):
        report.verdicts.append(hs.Verdict(
            key=f"w{i}", company=f"W{i}", surfaced=True, score=80.0,
            evidence=evidence))
    for i, evidence in enumerate(control_evidence):
        control = hs.Verdict(key=f"c{i}", company=f"C{i}", surfaced=False,
                             score=20.0, evidence=evidence)
        control.is_control = True
        report.controls.append(control)
    return report


def test_lopsided_evidence_coverage_is_called_out() -> None:
    """If every company that raised has a GitHub org and most controls have
    nothing, the backtest is measuring public footprint, not judgment —
    and it would otherwise show a beautiful, meaningless AUC."""
    rigged = _fairness_report(
        ["2 repo(s), 400 stars"] * 5,
        ["no public evidence found"] * 4 + ["1 repo(s), 10 stars"],
    )
    check = hs.evidence_symmetry(rigged)
    assert check.unfair is True
    assert check.outcome_evidence_rate == 1.0
    assert check.control_evidence_rate == 0.2
    assert "PUBLIC FOOTPRINT" in check.message
    assert "not comparable" in hs.render_report(rigged)


def test_comparable_groups_pass_the_fairness_check() -> None:
    fair = _fairness_report(
        ["2 repo(s), 400 stars"] * 5,
        ["1 repo(s), 30 stars"] * 4 + ["no public evidence found"],
    )
    check = hs.evidence_symmetry(fair)
    assert check.unfair is False
    assert "not simply a public-footprint artefact" in check.message


def test_no_controls_means_no_separation_can_be_claimed() -> None:
    report = _fairness_report(["2 repo(s), 400 stars"] * 3, [])
    check = hs.evidence_symmetry(report)
    assert "nothing to compare against" in check.message
    assert check.unfair is False  # not rigged — just uninformative


def test_weights_derived_from_a_backtest_are_flagged_as_circular() -> None:
    """Tune weights on these companies, then measure them on the same
    companies, and the score improves for no reason. Nothing else in the
    module would notice, so the report has to say it."""
    clean = _report()
    assert clean.circular is False
    assert "CIRCULAR" not in hs.render_report(clean)

    circular = _report()
    circular.weights_from_backtest = 3
    assert circular.circular is True
    text = hs.render_report(circular)
    assert "CIRCULAR" in text
    assert "its own answer key" in text
    # And it leads the limitations rather than being buried among them.
    limitations = text.split("## What this does and does not show")[1]
    assert limitations.strip().startswith("- CIRCULAR")


def test_zero_observed_false_positives_does_not_mean_a_zero_rate() -> None:
    """The overconfidence trap in the base-rate projection, and the same
    class of error as a bootstrap collapsing under perfect separation:
    feeding a literal FPR of 0 into Bayes yields '100% precision, 100x
    lift' from a handful of controls."""
    metrics = hs.compute_metrics([90.0] * 12, [10.0] * 18, threshold=60.0)
    assert metrics.false_positive_rate == 0.0    # observed, and true
    tpr, fpr, bounded = hs.bounded_rates(metrics)
    assert bounded is True
    assert fpr == pytest.approx(3 / 18, abs=0.001)   # rule of three
    assert tpr == pytest.approx(1 - 3 / 12, abs=0.001)

    rows = {row.base_rate: row for row in hs.realistic_performance(metrics)}
    assert rows[0.02].precision < 0.15   # not 100%
    assert rows[0.02].lift < 10          # not 100x


def test_the_report_explains_why_the_projection_differs_from_the_headline() -> None:
    """A reader who spots that 0% false positives did not become 100%
    precision deserves the reason, not a silent adjustment."""
    report = hs.BacktestReport(cutoff=CUTOFF, thesis_id="t", threshold=60.0)
    report.verdicts = [hs.Verdict(key=f"w{i}", company=f"W{i}", surfaced=True,
                                  score=90.0) for i in range(12)]
    for i in range(18):
        control = hs.Verdict(key=f"c{i}", company=f"C{i}", surfaced=False,
                             score=10.0)
        control.is_control = True
        report.controls.append(control)
    text = hs.render_report(report)
    assert "rule-of-three" in text
    assert "cannot be extrapolated" in text


def test_ordinary_rates_pass_through_unbounded() -> None:
    """The correction only fires at the boundaries; a normal run is
    projected from exactly what was measured."""
    metrics = hs.compute_metrics([90.0, 90.0, 40.0], [70.0, 10.0, 10.0, 10.0],
                                 threshold=60.0)
    tpr, fpr, bounded = hs.bounded_rates(metrics)
    assert bounded is False
    assert (tpr, fpr) == (metrics.true_positive_rate,
                          metrics.false_positive_rate)
