"""`scout doctor` — readiness, judged against one outcome: will the daily
scan produce classified leads.

The checks are only useful if their severity is right. A doctor that fails
on an optional feature trains people to ignore it; one that passes a
missing Anthropic key lets a firm run a month of keyword-only scoring. So
these tests pin the grading, and that every non-ok line carries its fix.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from scout import doctor as dr
from scout import worker
from scout.cli import app
from scout.config import Seeds, Settings, Thesis
from scout.store import Store

runner = CliRunner()
REPO = Path(__file__).resolve().parent.parent
LAUNCHED = Thesis(name="t", target_stages=["launched"])  # github, hn, rss, sec, yc


def _settings(tmp_path: Path, **kw) -> Settings:
    base = {"db_path": tmp_path / "t.db", "anthropic_api_key": "k", "tw_cookies": None,
            "github_token": None, "sec_user_agent": "Acme Ventures ops@acme.vc"}
    return Settings(**{**base, **kw})


def _by_name(checks: list[dr.Check]) -> dict[str, dr.Check]:
    return {c.name: c for c in checks}


def _config(tmp_path: Path, *, settings: Settings | None = None,
            thesis: Thesis | None = LAUNCHED, seeds: Seeds | None = None,
            **kw) -> dict[str, dr.Check]:
    settings = settings or _settings(tmp_path)
    store = Store(settings.db_path)
    return _by_name(dr.config_checks(
        settings, thesis, seeds or Seeds(watchlist=["myinvestor"], github_topics=["llm"]),
        store, **kw))


# --- grading ---------------------------------------------------------------------


def test_a_missing_anthropic_key_blocks_and_says_how_to_fix_it(tmp_path: Path) -> None:
    checks = _config(tmp_path, settings=_settings(tmp_path, anthropic_api_key=None))
    key = checks["Anthropic key"]
    assert key.status == "fail"
    assert "ANTHROPIC_API_KEY" in key.fix
    ready, summary = dr.verdict(list(checks.values()))
    assert not ready and summary.startswith("Not ready")


def test_no_x_cookies_is_a_warning_because_runs_fall_back(tmp_path: Path) -> None:
    check = _config(tmp_path)["X cookies (TW_COOKIES)"]
    assert check.status == "warn" and "free sources" in check.detail


def test_cookie_files_are_checked_for_the_two_cookies_x_needs(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"ct0": "a", "auth_token": "b"}))
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps([{"name": "ct0", "value": "a"}]))
    assert _config(tmp_path, settings=_settings(tmp_path, tw_cookies=good))[
        "X cookies (TW_COOKIES)"].status == "ok"
    check = _config(tmp_path, settings=_settings(tmp_path, tw_cookies=partial))[
        "X cookies (TW_COOKIES)"]
    assert check.status == "warn" and "auth_token" in check.detail
    check = _config(tmp_path, settings=_settings(tmp_path, tw_cookies=tmp_path))[
        "X cookies (TW_COOKIES)"]
    assert check.status == "warn" and "not a file" in check.detail


def test_the_example_watchlist_is_flagged_and_a_real_one_is_not(tmp_path: Path) -> None:
    example = Seeds(watchlist=["eladgil", "@garrytan", "Karpathy"], github_topics=["x"])
    assert _config(tmp_path, seeds=example)["Investor watchlist"].status == "warn"
    mixed = Seeds(watchlist=["eladgil", "our_partner"], github_topics=["x"])
    assert _config(tmp_path, seeds=mixed)["Investor watchlist"].status == "ok"
    assert _config(tmp_path, seeds=Seeds(github_topics=["x"]))["Investor watchlist"].status == "warn"


def test_source_specific_config_is_only_checked_when_the_source_is_active(tmp_path: Path) -> None:
    anonymous = _settings(tmp_path, sec_user_agent="scout/0.1")
    checks = _config(tmp_path, settings=anonymous, seeds=Seeds(watchlist=["me"]))
    assert checks["SEC User-Agent"].status == "warn"
    assert "@" in checks["SEC User-Agent"].fix
    assert checks["GitHub topics"].status == "warn"
    assert checks["RSS feeds"].status == "info"
    # An idea-stage thesis runs arXiv only: none of those apply.
    idea = _config(tmp_path, settings=anonymous, thesis=Thesis(target_stages=["idea"]))
    assert not {"SEC User-Agent", "GitHub topics", "RSS feeds"} & set(idea)


def test_a_thesis_that_does_not_load_blocks(tmp_path: Path) -> None:
    checks = _config(tmp_path, thesis=None, thesis_error="thesis.yaml did not load")
    assert checks["Thesis"].status == "fail"
    assert "thesis.yaml did not load" in checks["Thesis"].detail


def test_schedules_and_worker_liveness(tmp_path: Path) -> None:
    checks = _config(tmp_path)
    assert checks["Schedules"].status == "warn" and "--bootstrap" in checks["Schedules"].fix
    assert checks["Worker"].status == "warn" and checks["Worker"].detail == "never run"

    store = Store(tmp_path / "t.db")
    worker.bootstrap_schedules(store)
    store.set_setting("worker_last_seen", datetime.now(timezone.utc).isoformat())
    checks = _config(tmp_path)
    assert checks["Schedules"].status == "ok" and "Daily sourcing run" in checks["Schedules"].detail
    assert checks["Worker"].status == "ok"


def test_optional_features_never_block(tmp_path: Path) -> None:
    checks = _config(tmp_path)
    for name in ("Slack digest", "Phone app", "X API token (paid)"):
        assert checks[name].status in ("info", "ok"), name


# --- network (stubbed probe) ------------------------------------------------------


def _probe(answers: dict[str, tuple[int | None, str]]):
    calls: list[tuple[str, dict]] = []

    def probe(url: str, headers: dict) -> tuple[int | None, str]:
        calls.append((url, headers))
        for fragment, answer in answers.items():
            if fragment in url:
                return answer
        return 200, ""
    return probe, calls


FEED = ('<?xml version="1.0"?><rss version="2.0"><channel><title>F</title>'
        '<item><title>A</title><link>https://a.com/</link></item></channel></rss>')


def test_network_probes_use_the_real_credentials_and_grade_answers(tmp_path: Path) -> None:
    settings = _settings(tmp_path, github_token="ghp_x", sec_user_agent="Acme ops@acme.vc")
    seeds = Seeds(rss_feeds=["https://good.example/feed", "https://page.example/"])
    probe, calls = _probe({
        "api.anthropic.com": (401, "invalid x-api-key"),
        "www.sec.gov": (403, "Undeclared Automated Tool"),
        "good.example": (200, FEED),
        "page.example": (200, "<html>not a feed</html>"),
        "hn.algolia.com": (None, "ConnectError: timed out"),
    })
    checks = _by_name(dr.network_checks(settings, LAUNCHED, seeds, probe))

    assert checks["Anthropic API"].status == "fail"
    assert checks["SEC EDGAR"].status == "warn" and "SEC_USER_AGENT" in checks["SEC EDGAR"].fix
    assert checks["RSS https://good.example/feed"].status == "ok"
    assert checks["RSS https://page.example/"].status == "warn"
    assert checks["Hacker News"].status == "warn"
    assert checks["GitHub"].status == "ok" and checks["YC directory"].status == "ok"
    # The probes carry what a real run would send.
    sent = {url.split("/")[2]: headers for url, headers in calls}
    assert sent["www.sec.gov"]["User-Agent"] == "Acme ops@acme.vc"
    assert sent["api.github.com"]["Authorization"] == "Bearer ghp_x"
    assert sent["api.anthropic.com"]["x-api-key"] == "k"
    assert "Discovery" not in checks  # some sources work


def test_every_discovery_host_down_and_no_x_blocks(tmp_path: Path) -> None:
    probe, _ = _probe({"": (None, "ProxyError: 403")})
    checks = _by_name(dr.network_checks(_settings(tmp_path), LAUNCHED, Seeds(), probe))
    assert checks["Discovery"].status == "fail"
    # …but with X connected, the run still has something to read.
    cookies = tmp_path / "c.json"
    cookies.write_text("{}")
    checks = _by_name(dr.network_checks(_settings(tmp_path, tw_cookies=cookies),
                                        LAUNCHED, Seeds(), probe))
    assert "Discovery" not in checks


def test_no_key_means_no_anthropic_probe(tmp_path: Path) -> None:
    probe, calls = _probe({})
    dr.network_checks(_settings(tmp_path, anthropic_api_key=None), LAUNCHED, Seeds(), probe)
    assert not any("anthropic" in url for url, _ in calls)


def test_github_rate_limit_is_named_as_such(tmp_path: Path) -> None:
    probe, _ = _probe({"api.github.com": (403, "API rate limit exceeded for 1.2.3.4")})
    check = _by_name(dr.network_checks(_settings(tmp_path), LAUNCHED, Seeds(), probe))["GitHub"]
    assert check.status == "warn" and "GITHUB_TOKEN" in check.fix


def test_verdict_wording() -> None:
    ok = dr.Check(area="a", name="x", status="ok", detail="")
    warn = dr.Check(area="a", name="y", status="warn", detail="")
    fail = dr.Check(area="a", name="z", status="fail", detail="")
    assert dr.verdict([ok]) == (True, "Ready — the daily scan will run with everything configured.")
    assert dr.verdict([ok, warn])[0] is True
    assert dr.verdict([ok, warn, fail]) == (False, "Not ready — 1 blocking issue, 1 warning.")


# --- the command --------------------------------------------------------------------


def _doctor(tmp_path: Path, **env: str):
    return runner.invoke(
        app, ["doctor", "--offline", "--thesis", str(REPO / "thesis.yaml"),
              "--seeds", str(REPO / "seeds.yaml")],
        env={"DB_PATH": str(tmp_path / "t.db"), "TW_COOKIES": "", **env},
    )


def test_doctor_exits_1_when_blocked_and_0_when_only_degraded(tmp_path: Path) -> None:
    blocked = _doctor(tmp_path, ANTHROPIC_API_KEY="")
    assert blocked.exit_code == 1
    assert "Not ready" in blocked.output and "ANTHROPIC_API_KEY" in blocked.output
    degraded = _doctor(tmp_path, ANTHROPIC_API_KEY="k")
    assert degraded.exit_code == 0, degraded.output
    assert "Ready, degraded" in degraded.output
    assert "network not checked" in degraded.output
