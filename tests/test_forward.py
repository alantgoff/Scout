"""Forward scorecard (scout/forward.py): the guards that keep a young,
sparse outcome record from producing a confident wrong number."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scout.forward import (
    RaiseEvent,
    Sighting,
    build_forward,
    caught_note,
    forward_limitations,
    raises_from,
    render_markdown,
    sightings_from_rows,
)
from scout.models import Account, Lead, LLMVerdict

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def sighting(handle: str, days_ago: int, score: float, version: str = "v1") -> Sighting:
    return Sighting(handle=handle, name=handle.title(), score=score,
                    first_seen=NOW - timedelta(days=days_ago),
                    thesis_id="t", thesis_version=version)


def raised(handle: str, days_ago: int, source: str = "form_d") -> RaiseEvent:
    return RaiseEvent(handle=handle, date=NOW - timedelta(days=days_ago), source=source)


def test_rows_are_classified_by_maturity_and_timing():
    sightings = [
        sighting("hit", 300, 80),        # raised 100 days after first sight
        sighting("late", 300, 70),       # raised 250 days after — outside 180
        sighting("quiet", 300, 40),      # matured, never raised
        sighting("young", 30, 90),       # too soon to tell
        sighting("fast", 30, 85),        # young, but already raised after sight
        sighting("before", 300, 75),     # raised BEFORE Scout saw it
    ]
    raises = [raised("hit", 200), raised("late", 50), raised("fast", 10),
              raised("before", 310)]
    report = build_forward(sightings, raises, now=NOW, horizon_days=180)
    status = {r.handle: r.status for r in report.rows}
    assert status == {
        "hit": "raised", "late": "not_raised", "quiet": "not_raised",
        "young": "watching", "fast": "early_hit", "before": "already_raised",
    }
    assert report.n_evaluable == 3 and report.n_raised == 1
    assert report.n_early_hits == 1 and report.n_watching == 1 and report.n_already == 1
    hit = next(r for r in report.rows if r.handle == "hit")
    assert hit.days_ahead == 100 and hit.raise_source == "Form D"
    # Early hits are reported as catches, but stay out of every rate.
    assert {r.handle for r in report.hits()} == {"hit", "fast"}
    assert report.base_rate == 1 / 3


def test_rates_are_withheld_below_the_minimum_sample():
    sightings = [sighting(f"co{i}", 400, 50 + i) for i in range(12)]
    raises = [raised("co11", 300), raised("co10", 300)]
    report = build_forward(sightings, raises, now=NOW)
    assert report.underpowered
    assert report.auc is None and report.lift is None and report.top_hit_rate is None
    assert report.precision_at_threshold is None and report.median_days_ahead is None
    assert "3 more raises" in report.needs
    # The smallest effect this sample could detect is stated, not hidden.
    assert 0.5 < report.min_detectable_auc <= 1.0
    assert "Rates withheld" in render_markdown(report)


def test_a_well_powered_sample_reports_auc_lift_and_lead_time():
    # 20 matured startups; the six best-scored raised, 60–160 days on.
    sightings = [sighting(f"co{i:02d}", 400, 30 + 3 * i) for i in range(20)]
    raises = [raised(f"co{i:02d}", 400 - (60 + 20 * (19 - i))) for i in range(14, 20)]
    report = build_forward(sightings, raises, now=NOW, threshold=70)
    assert not report.underpowered
    assert report.n_raised == 6 and report.n_evaluable == 20
    assert report.auc == 1.0                       # perfect separation…
    assert report.auc_low is not None and report.auc_low < 1.0  # …but not certainty
    assert report.beats_chance is True
    assert report.top_n == 4 and report.top_hits == 4
    assert report.lift == round(1.0 / (6 / 20), 2)
    assert report.median_days_ahead is not None and report.median_days_ahead > 0
    # score ≥ 70: co14 (72) … co19 (87) — all six raised.
    assert report.at_threshold == 6 and report.precision_at_threshold == 1.0
    assert report.recall_at_threshold == 1.0


def test_a_top_fifth_with_no_hits_is_bounded_not_zero():
    sightings = [sighting(f"co{i:02d}", 400, 90 - i) for i in range(25)]
    # The five LOWEST scorers raised; the top fifth (5 companies) did not.
    raises = [raised(f"co{i:02d}", 300) for i in range(20, 25)]
    report = build_forward(sightings, raises, now=NOW, threshold=75)
    assert report.top_hits == 0 and report.top_hit_rate == 0.0
    assert report.top_hit_rate_bound == 3 / 5
    assert report.auc == 0.0 and report.beats_chance is False
    assert {r.handle for r in report.underrated()} == {f"co{i:02d}" for i in range(20, 25)}


def test_detected_rounds_count_only_when_asked():
    sightings = [sighting("a", 300, 80), sighting("b", 300, 60)]
    raises = [raised("a", 200, source="detected")]
    form_d_only = build_forward(sightings, raises, now=NOW)
    assert form_d_only.n_raised == 0 and form_d_only.sources == ["form_d"]
    both = build_forward(sightings, raises, now=NOW, sources=("form_d", "detected"))
    assert both.n_raised == 1
    assert any("inflates the hit rate" in line for line in both.limitations)
    assert not any("inflates the hit rate" in line for line in form_d_only.limitations)


def test_raise_events_join_on_handle_and_skip_amendments():
    filings = [
        {"matched_handle": "AcmeAI", "first_sale": "2026-03-02", "filed_at": "2026-03-10",
         "amount_sold": 4_000_000, "is_amendment": 0},
        {"matched_handle": "acmeai", "first_sale": "", "filed_at": "2026-05-01",
         "amount_sold": 0, "amount_offered": 2_500_000, "is_amendment": 1},
        {"matched_handle": "", "first_sale": "2026-03-02", "is_amendment": 0},
    ]
    outcomes = [{"handle": "acmeai", "round_stage": "series_a", "amount": "$12M",
                 "detected_at": "2026-06-01T09:00:00+00:00"}]
    events = raises_from(filings, outcomes)
    assert [(e.handle, e.source) for e in events] == [("acmeai", "form_d"),
                                                      ("acmeai", "detected")]
    assert events[0].date == datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert events[0].detail == "Form D · $4M sold"
    assert events[1].detail == "Series A · $12M"


def test_sightings_are_named_like_the_startup():
    lead = Lead(account=Account(id="1", handle="AdaLin", name="Ada Lin", source="search"),
                llm=LLMVerdict(handle="AdaLin", company_name="Lumen Labs",
                               account_type="founder", is_founder=True))
    rows = [
        {"handle": "AdaLin", "created_at": "2026-01-05T10:00:00+00:00", "score": 71.5,
         "lead_json": lead.model_dump_json(), "run_id": "r1", "thesis_id": "t",
         "thesis_version": "abc"},
        {"handle": "Broken", "created_at": "2026-01-06T10:00:00", "score": 10,
         "lead_json": "not json"},
        {"handle": "NoDate", "created_at": "", "score": 10, "lead_json": ""},
    ]
    out = sightings_from_rows(rows)
    assert [(s.handle, s.name) for s in out] == [("adalin", "Lumen Labs"),
                                                  ("broken", "@broken")]
    assert out[1].first_seen.tzinfo is not None


def test_the_digest_note_only_credits_a_real_head_start():
    seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert caught_note(seen, 71.2, seen + timedelta(days=94)) == (
        "Scout first saw it 94 days earlier (score 71 then)")
    assert caught_note(seen, 71.2, seen - timedelta(days=3)) == ""
    assert caught_note(None, None, seen) == ""


def test_versions_and_limitations_are_part_of_every_report():
    sightings = [sighting("a", 300, 80, "v1"), sighting("b", 100, 60, "v2")]
    report = build_forward(sightings, [raised("a", 250)], now=NOW, horizon_days=90)
    assert [(v.thesis_version, v.sighted, v.evaluable, v.raised)
            for v in report.by_version] == [("v1", 1, 1, 1), ("v2", 1, 1, 0)]
    assert report.limitations == forward_limitations(90, ["form_d"])
    assert "What this does and does not show" in render_markdown(report)
