"""Presentation helpers (scout/present.py) — pure, so they test without
Streamlit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scout.present import relative_time, run_label

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def test_relative_time_reads_both_directions():
    assert relative_time(NOW - timedelta(seconds=30), NOW) == "just now"
    assert relative_time(NOW + timedelta(seconds=30), NOW) == "just now"
    assert relative_time(NOW - timedelta(minutes=5), NOW) == "5m ago"
    assert relative_time(NOW - timedelta(hours=3), NOW) == "3h ago"
    assert relative_time(NOW - timedelta(days=4), NOW) == "4d ago"
    # The bug this fixes: tomorrow's 06:00 schedule read "just now".
    assert relative_time(NOW + timedelta(hours=18), NOW) == "in 18h"
    assert relative_time(NOW + timedelta(minutes=20), NOW) == "in 20m"
    assert relative_time(NOW + timedelta(days=3), NOW) == "in 3d"


def test_relative_time_accepts_iso_strings_and_naive_times():
    assert relative_time((NOW - timedelta(hours=2)).isoformat(), NOW) == "2h ago"
    naive = (NOW + timedelta(hours=5)).replace(tzinfo=None).isoformat()
    assert relative_time(naive, NOW) == "in 5h"
    assert relative_time("", NOW) == "" and relative_time(None, NOW) == ""
    assert relative_time("not a date", NOW) == ""


def test_run_label_never_says_run_running():
    assert run_label("run") == "Run"
    assert run_label("reclassify") == "Rescore"
    assert run_label("source preview") == "Preview"
    assert run_label(None) == "Run"
