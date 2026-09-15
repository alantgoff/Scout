"""Star velocity — the launch moment, read from daily snapshots.

The GitHub source records each discovery repo's stars every run at no cost;
the signal is the difference over the window. What these tests pin: a repo
seen once has no velocity (no invented baseline), the baseline is the
snapshot at the window's start when history reaches that far and the
oldest inside it when not, enrichment lands the delta on the account, the
delta is never persisted as if it were a fact about the account, and the
hindsight backtest counts the same window from stargazer timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout.cli import _enrich_accounts
from scout.config import Settings, Thesis
from scout.hindsight import count_stars
from scout.models import Account
from scout.store import Store

REPO = "https://github.com/acme/agents"
NOW = datetime.now(timezone.utc)


def _snapshot(store: Store, repo: str, stars: int, days_ago: float) -> None:
    store.db["repo_stars"].insert({
        "repo_url": repo, "stars": stars,
        "seen_at": (NOW - timedelta(days=days_ago)).isoformat(),
    })


def test_one_snapshot_is_no_velocity_and_the_window_start_is_the_baseline(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    store.record_repo_stars([(REPO + "/", 40), ("", 5)])  # trailing slash normalized, blanks dropped
    assert store.star_deltas(7) == {}  # first sighting: nothing to compare against
    _snapshot(store, REPO, 10, days_ago=9)   # before the window: the baseline
    _snapshot(store, REPO, 25, days_ago=3)   # inside the window: ignored as baseline
    assert store.star_deltas(7) == {REPO: 30}  # 40 now − 10 at the window start


def test_a_short_history_uses_its_oldest_snapshot_honestly(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    _snapshot(store, REPO, 100, days_ago=2)
    _snapshot(store, REPO, 160, days_ago=0)
    other = "https://github.com/x/y"
    _snapshot(store, other, 7, days_ago=1)  # alone → absent, not 0
    assert store.star_deltas(7) == {REPO: 60}
    assert store.star_deltas(1) == {REPO: 60}  # window shorter than history still works


def test_enrichment_lands_the_delta_on_the_account_and_never_persists_it(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    _snapshot(store, REPO, 10, days_ago=8)
    _snapshot(store, REPO, 130, days_ago=0)
    seen = Account(id="gh-acme", handle="acme", github_repo=REPO, github_stars=130)
    fresh = Account(id="gh-new", handle="newco", github_repo="https://github.com/new/repo")
    none = Account(id="x", handle="nobody")
    _enrich_accounts([seen, fresh, none], store, Thesis(), Settings())
    assert seen.star_velocity == 120
    assert fresh.star_velocity == 0 and none.star_velocity == 0

    store.upsert_account(seen)
    stored = store.get_account("acme")
    assert stored.github_stars == 130  # a fact from the source: kept
    assert stored.star_velocity == 0   # enrichment: recomputed per run, never stored
    assert "star_velocity" not in store.db["accounts"].columns_dict


def test_hindsight_counts_the_window_before_the_cutoff() -> None:
    cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)
    times = [cutoff - timedelta(days=d) for d in (30, 8, 6, 3, 0.5)] + [cutoff + timedelta(days=1)]
    assert count_stars(times, cutoff, 7) == (5, 3)   # 5 before; 3 inside the last 7 days
    assert count_stars(times, cutoff, 1) == (5, 1)
    assert count_stars([], cutoff) == (0, 0)
