"""The daily spend envelope — what makes an unattended daily scan safe.

Three layers, tested separately:
  pricing math (config.llm_cost_usd, pure) → the ledger (store.llm_usage +
  spend_today_usd) → the gates at the spend sites (classify, verify,
  refresh) that read the ledger and stop paid work when the envelope is
  spent. The invariant across all of them: the cheap work (heuristics,
  caches) always runs; only NEW paid calls stop; and everything that did
  spend is on the ledger — including the calls that failed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as _NS

from typer.testing import CliRunner

from scout.config import Settings, Thesis, llm_cost_usd
from scout.models import Account, Lead, LLMVerdict
from scout.store import Store

runner = CliRunner()


# --- pricing math -------------------------------------------------------------


def test_cost_covers_tokens_cache_and_searches() -> None:
    # sonnet-4-6: $3 in / $15 out. 1M in = $3; 1M out = $15;
    # 1M cache-read = $0.30; 1M cache-write = $3.75; 10 searches = $0.10.
    cost = llm_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000,
                        cache_read_tokens=1_000_000,
                        cache_write_tokens=1_000_000, searches=10)
    assert abs(cost - (3 + 15 + 0.30 + 3.75 + 0.10)) < 1e-9


def test_unknown_model_prices_at_opus_tier_not_zero() -> None:
    """Overcounting an unknown model throttles the scan early; undercounting
    blows the envelope. Only one of those is safe."""
    assert llm_cost_usd("claude-next-99", 1_000_000, 0) == 5.0


def test_dated_snapshot_prices_like_its_family() -> None:
    assert llm_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0


# --- the ledger ---------------------------------------------------------------


def test_spend_today_sums_both_ledgers_from_utc_midnight(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.record_llm_usage("classify", "claude-sonnet-4-6",
                           input_tokens=100, output_tokens=50, cost_usd=0.40)
    store.record_llm_usage("research", "claude-sonnet-4-6",
                           searches=5, cost_usd=0.10)
    store.record_xapi_usage("search/recent", 10, 0, 0.05)
    assert abs(store.spend_today_usd() - 0.55) < 1e-9
    assert abs(store.llm_spend_usd() - 0.50) < 1e-9

    # Yesterday's rows are not today's spend — the envelope refills at
    # midnight UTC, when the daily schedule fires.
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    store.db["llm_usage"].insert({"agent": "classify", "model": "m",
                                  "cost_usd": 99.0, "at": yesterday})
    assert abs(store.spend_today_usd() - 0.55) < 1e-9
    assert abs(store.llm_spend_usd() - 99.5) < 1e-9


def test_budget_left_and_disabled_cap(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.record_llm_usage("memo", "m", cost_usd=0.75)
    assert abs(store.daily_budget_left_usd(1.0) - 0.25) < 1e-9
    assert store.daily_budget_left_usd(0.0) == float("inf")


# --- the classify gate --------------------------------------------------------


def _spent_store(tmp_path: Path, spent: float) -> Store:
    store = Store(tmp_path / "t.db")
    if spent:
        store.record_llm_usage("classify", "m", cost_usd=spent)
    return store


def _candidate(handle: str = "someone"):
    return (Account(id=handle, handle=handle, bio="building something"), [])


def test_classify_serves_cache_but_spends_nothing_when_cap_reached(
    tmp_path: Path, monkeypatch
) -> None:
    from scout.signals import llm as llm_mod

    store = _spent_store(tmp_path, spent=2.0)  # cap is 1.0 → exhausted
    settings = Settings(anthropic_api_key="k", daily_spend_cap_usd=1.0,
                        _env_file=None)
    thesis = Thesis()

    # A cached verdict for one candidate; the other would need a paid call.
    cached_account, _ = _candidate("cachedco")
    fingerprint = llm_mod._fingerprint(
        cached_account, [], thesis, settings, None,
        system_prompt=llm_mod._system_prompt(thesis))
    store.record_verdict("cachedco", fingerprint,
                         LLMVerdict(handle="cachedco", one_line_summary="x"))

    def boom(*_a, **_kw):
        raise AssertionError("a paid Claude call went out past the cap")

    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", boom)
    results = llm_mod.classify(
        [(cached_account, []), _candidate("freshco")],
        thesis, settings, store=store,
    )
    assert set(results) == {"cachedco"}  # cache served, fresh skipped


def test_classify_proceeds_and_ledgers_under_the_cap(
    tmp_path: Path, monkeypatch
) -> None:
    from scout.signals import llm as llm_mod

    store = _spent_store(tmp_path, spent=0.0)
    settings = Settings(anthropic_api_key="k", daily_spend_cap_usd=1.0,
                        llm_concurrency=1, _env_file=None)

    verdict_json = json.dumps([{"handle": "freshco",
                                "one_line_summary": "builds things"}])

    class _FakeMessages:
        def create(self, **_kw):
            return _NS(
                content=[_NS(type="text", text=verdict_json)],
                usage=_NS(input_tokens=1000, output_tokens=500,
                          cache_read_input_tokens=200,
                          cache_creation_input_tokens=100),
            )

    monkeypatch.setattr(llm_mod.anthropic, "Anthropic",
                        lambda **_kw: _NS(messages=_FakeMessages()))
    results = llm_mod.classify([_candidate("freshco")], Thesis(), settings,
                               store=store)
    assert "freshco" in results
    rows = list(store.db["llm_usage"].rows_where("agent = 'classify'"))
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 1000
    assert rows[0]["cache_read_tokens"] == 200
    assert rows[0]["cost_usd"] > 0
    assert store.spend_today_usd() > 0


def test_verify_skips_entirely_when_cap_reached(tmp_path: Path, monkeypatch) -> None:
    from scout.signals import llm as llm_mod

    store = _spent_store(tmp_path, spent=2.0)
    settings = Settings(anthropic_api_key="k", daily_spend_cap_usd=1.0,
                        _env_file=None)
    lead = Lead(account=Account(id="a", handle="a"),
                llm=LLMVerdict(handle="a"))

    def boom(*_a, **_kw):
        raise AssertionError("audit call went out past the cap")

    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", boom)
    llm_mod.verify_leads([lead], {}, {}, Thesis(), settings, store=store)
    assert lead.llm.verification is None  # untouched, not audited


# --- research/memo metering ---------------------------------------------------


def test_research_stream_usage_lands_on_the_ledger(tmp_path: Path, monkeypatch) -> None:
    from scout import agents
    from tests.test_agents import _FakeClient

    store = Store(tmp_path / "t.db")
    final = _NS(
        stop_reason="end_turn",
        content=[_NS(type="text", text=json.dumps({
            "is_company": True, "company_name": "X", "sources": []}),
            citations=None)],
        usage=_NS(input_tokens=3000, output_tokens=800,
                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    fake = _FakeClient([([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    _profile, meta = agents.research_company(
        "https://x.example/", Settings(anthropic_api_key="k", _env_file=None),
        store=store,
    )
    rows = list(store.db["llm_usage"].rows_where("agent = 'research'"))
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 3000
    assert meta["cost_usd"] > 0
