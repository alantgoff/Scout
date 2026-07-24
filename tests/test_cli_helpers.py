"""Orchestration-layer tests: candidate ranking + the parallel tweet fetch.

No network — the fetch tests drive a fake in-memory adapter.
"""

from __future__ import annotations

import asyncio

from scout.cli import _fetch_tweets, _rank_candidates
from scout.config import Seeds, Thesis
from scout.ingest.base import SourceAdapter
from scout.ingest.xapi_src import BudgetExceededError
from scout.models import Account, Lead, Signal, Tweet


# --- _rank_candidates -----------------------------------------------------------


def make_lead(handle: str, *signals: tuple[str, float]) -> Lead:
    return Lead(
        account=Account(id=handle, handle=handle),
        signals=[Signal(name=n, value=v) for n, v in signals],
    )


def test_rank_candidates_orders_by_pre_score_and_caps() -> None:
    thesis = Thesis(weights={"bio_intent": 20.0, "departure_signal": 20.0})
    strong = make_lead("strong", ("bio_intent", 1.0), ("departure_signal", 1.0))
    medium = make_lead("medium", ("bio_intent", 1.0))
    silent = make_lead("silent")  # no signals hit — never sent to Claude

    ranked, skipped = _rank_candidates([silent, medium, strong], thesis, cap=10)
    assert [x.account.handle for x in ranked] == ["strong", "medium"]
    assert skipped == 0

    ranked, skipped = _rank_candidates([silent, medium, strong], thesis, cap=1)
    assert [x.account.handle for x in ranked] == ["strong"]
    assert skipped == 1


def test_thesis_identity_survives_the_tuning_that_changes_its_version() -> None:
    """The exact bug: `strategy_fingerprint` hashes the whole config, so every
    weight tweak minted a new 'strategy' and one thesis fragmented across
    several — the live database had 7 hashes for 4 theses. Identity must hold
    still while the version moves."""
    from scout.config import Seeds, ensure_thesis_id, thesis_version

    thesis = Thesis(name="Novel Architectures", thesis="Novel model architectures",
                    weights={"bio_intent": 20.0})
    seeds = Seeds()
    id_before, version_before = ensure_thesis_id(thesis), thesis_version(thesis, seeds)

    thesis.weights["bio_intent"] = 25.0  # the kind of tuning that fragmented it
    assert ensure_thesis_id(thesis) == id_before
    assert thesis_version(thesis, seeds) != version_before


def test_naming_a_thesis_does_not_mark_its_work_stale() -> None:
    """Identity is not tuning. Hashing id/name meant that giving a running
    thesis a proper name flagged all 602 of its startups stale — a rename
    changes nothing about how anything scores."""
    from scout.config import Seeds, thesis_version

    thesis = Thesis(thesis="Edge AI", weights={"bio_intent": 20.0})
    seeds = Seeds()
    before = thesis_version(thesis, seeds)

    thesis.id, thesis.name = "edge-ai", "Edge AI Infrastructure"
    assert thesis_version(thesis, seeds) == before

    thesis.weights["bio_intent"] = 25.0
    assert thesis_version(thesis, seeds) != before


def test_ensure_thesis_id_falls_back_through_name_then_statement() -> None:
    """Statement fallback is also the backfill rule for runs recorded before
    identity existed."""
    from scout.config import ensure_thesis_id

    assert ensure_thesis_id(Thesis(id="edge-ai")) == "edge-ai"
    assert ensure_thesis_id(Thesis(name="Edge AI!")) == "edge-ai"
    assert ensure_thesis_id(
        Thesis(thesis="Identify technical founders building hardware")
    ) == "identify-technical-founders-building-hardware"
    assert ensure_thesis_id(Thesis()) == "untitled-thesis"


def test_rank_candidates_keeps_paid_signalless_search_leads() -> None:
    """A company account fires none of the founder-shaped signals, and the
    search that found it was already paid for. @BiggerMax19 had no bio, no
    followers and no signal, and classified at thesis fit 0.70."""
    thesis = Thesis(weights={"bio_intent": 20.0})
    strong = make_lead("strong", ("bio_intent", 1.0))
    company = make_lead("company")
    company.account.source = "search"
    scraped = make_lead("scraped")
    scraped.account.source = "github"  # bulk discovery, free to skip

    ranked, skipped = _rank_candidates([scraped, company, strong], thesis, cap=10)
    assert [x.account.handle for x in ranked] == ["strong", "company"]
    assert skipped == 0

    # The cap still protects spend, and signal-bearing leads still win it.
    ranked, skipped = _rank_candidates([scraped, company, strong], thesis, cap=1)
    assert [x.account.handle for x in ranked] == ["strong"]
    assert skipped == 1


# --- _fetch_tweets ----------------------------------------------------------------


class FakeAdapter(SourceAdapter):
    name = "fake"

    def __init__(self, *, parallel_safe: bool, fail: set[str] | None = None,
                 budget_fail: str | None = None) -> None:
        self.parallel_safe = parallel_safe
        self.fail = fail or set()
        self.budget_fail = budget_fail
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def fetch_accounts(self, seeds: Seeds, *, max_accounts: int) -> list[Account]:
        return []

    async def fetch_account(self, handle: str) -> Account | None:
        return None

    async def fetch_tweets(self, account: Account, *, limit: int = 20) -> list[Tweet]:
        self.calls.append(account.handle)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if account.handle == self.budget_fail:
            raise BudgetExceededError(4.5, 1.0, 5.0)
        if account.handle in self.fail:
            raise ValueError("boom")
        return [Tweet(id=f"t-{account.handle}", account_id=account.id, text="hi")]


def accounts(*handles: str) -> list[Account]:
    return [Account(id=h, handle=h) for h in handles]


def test_parallel_fetch_fans_out_and_tolerates_failures() -> None:
    adapter = FakeAdapter(parallel_safe=True, fail={"b"})
    result = asyncio.run(
        _fetch_tweets(adapter, accounts("a", "b", "c", "d"), limit=5, concurrency=4)
    )
    assert adapter.max_in_flight > 1  # actually ran concurrently
    assert len(result["a"]) == 1 and len(result["d"]) == 1
    assert result["b"] == []  # one failing account degrades, not aborts


def test_fetch_accounts_respects_sourcing_time_budget(tmp_path, monkeypatch) -> None:
    """An expired budget means NO network calls — every leg checks the
    deadline before touching X (regression for the 36-minute hang: twscrape
    sleeps through rate-limit windows unless bounded)."""
    from scout.config import Settings
    from scout.ingest.twscrape_src import TwscrapeSource
    from scout.store import Store

    monkeypatch.chdir(tmp_path)  # twscrape's account pool writes ./accounts.db
    cookies = tmp_path / "c.json"
    cookies.write_text('{"auth_token": "a", "ct0": "b"}')
    adapter = TwscrapeSource(Settings(tw_cookies=cookies, _env_file=None),
                             Store(tmp_path / "t.db"))
    seeds = Seeds(searches=["should never run"], bio_searches=["nor this"],
                  watchlist=["nobody"])

    accounts = asyncio.run(
        adapter.fetch_accounts(seeds, max_accounts=10, time_budget_s=0.0)
    )
    assert accounts == []


def test_parallel_fetch_budget_falls_back_to_cached_tweets(tmp_path) -> None:
    """Expired tweet budget = zero network calls; accounts with tweets cached
    from earlier runs keep them instead of scoring bio-only."""
    from scout.store import Store

    store = Store(tmp_path / "t.db")
    store.upsert_tweets([Tweet(id="c1", account_id="a", text="cached hi")])
    adapter = FakeAdapter(parallel_safe=True)

    result = asyncio.run(
        _fetch_tweets(adapter, accounts("a", "b"), limit=5, concurrency=4,
                      time_budget_s=0.0, store=store)
    )
    assert adapter.calls == []  # no network at all
    assert [t.text for t in result["a"]] == ["cached hi"]
    assert result["b"] == []


def test_paid_adapter_stays_sequential_and_stops_on_budget() -> None:
    adapter = FakeAdapter(parallel_safe=False, budget_fail="b")
    result = asyncio.run(
        _fetch_tweets(adapter, accounts("a", "b", "c"), limit=5, concurrency=8)
    )
    assert adapter.max_in_flight == 1  # never parallel, regardless of concurrency arg
    assert adapter.calls == ["a", "b"]  # stopped at the budget error —
    assert "c" not in result  # remaining fetches skipped, run continues
    assert len(result["a"]) == 1


# --- reclassify (empty-store guard; no network) ---------------------------------


def test_reclassify_exits_cleanly_on_empty_store(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from scout.cli import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "empty.db"))
    result = CliRunner().invoke(app, ["reclassify"])
    assert result.exit_code == 1
    assert "No leads yet" in result.output
