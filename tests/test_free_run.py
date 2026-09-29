"""`scout run` without X — the free sources still source.

The scheduled 06:00 run is `run --source twscrape`. It used to build the X
adapter before anything else, so on a machine without TW_COOKIES it exited
at once and GitHub, HN, RSS, SEC and YC never ran either — every morning.
These tests pin the fallback: missing or broken cookies degrade to the free
sources with a loud line and truthful run provenance, while an EXPLICIT
choice of the paid API still fails rather than quietly downgrading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scout import cli
from scout.config import Settings
from scout.models import Account
from scout.store import Store

runner = CliRunner()
REPO = Path(__file__).resolve().parent.parent
SITE = "https://lawco.ai/"


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"DB_PATH": str(tmp_path / "t.db"), "OUT_DIR": str(tmp_path / "out"),
            "TW_COOKIES": "", "ANTHROPIC_API_KEY": "", "X_BEARER_TOKEN": "",
            "VERIFY_TOP_N": "0", **extra}


def _stub_discovery(monkeypatch) -> list[list[str]]:
    """Discovery returns one YC company; classify records who reached it."""
    company = Account(id="yc:lawco", handle="lawco", name="LawCo",
                      bio="Contract review for law firms — YC Summer 2026",
                      website=SITE, profile_url=SITE, source="yc", sources=["yc"])
    monkeypatch.setattr(cli, "_run_discovery",
                        lambda names, seeds, thesis, settings, store: ([company], []))
    monkeypatch.setattr(cli, "_fetch_candidate_sites", lambda *a, **kw: {})
    classified: list[list[str]] = []
    monkeypatch.setattr(cli, "classify", lambda candidates, *a, **kw: (
        classified.append([acct.handle for acct, _ in candidates]) or {}))
    return classified


def _run(tmp_path: Path, *args: str, **env: str):
    return runner.invoke(
        cli.app,
        ["run", *args, "--thesis", str(REPO / "thesis.yaml"),
         "--seeds", str(REPO / "seeds.yaml")],
        env=_env(tmp_path, **env),
    )


def test_run_without_cookies_falls_back_to_the_free_sources(tmp_path: Path, monkeypatch) -> None:
    classified = _stub_discovery(monkeypatch)
    result = _run(tmp_path)  # the scheduled form: --source twscrape by default
    assert result.exit_code == 0, result.output
    assert "X is not connected" in result.output
    assert "free sources only" in result.output
    # The company reached the classifier and was saved as a lead.
    assert classified == [["lawco"]]
    store = Store(tmp_path / "t.db")
    assert [e.lead.account.handle for e in store.load_lead_ledger()] == ["lawco"]
    # Provenance tells the truth: this run had no X in it.
    assert [r["source"] for r in store.db["runs"].rows] == ["free"]


def test_broken_cookies_fall_back_too(tmp_path: Path, monkeypatch) -> None:
    """A directory where the cookies file should be used to crash with
    IsADirectoryError — not a 'not connected' case, so no fallback."""
    _stub_discovery(monkeypatch)
    result = _run(tmp_path, TW_COOKIES=str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "X is not connected" in result.output


def test_explicit_free_source_needs_no_warning(tmp_path: Path, monkeypatch) -> None:
    classified = _stub_discovery(monkeypatch)
    result = _run(tmp_path, "--source", "free")
    assert result.exit_code == 0, result.output
    assert "X is not connected" not in result.output
    assert classified == [["lawco"]]


def test_asking_for_the_paid_api_without_a_token_still_fails(tmp_path: Path, monkeypatch) -> None:
    """Choosing xapi is a decision to spend; a missing token must stop the
    run, never quietly run something else."""
    classified = _stub_discovery(monkeypatch)
    result = _run(tmp_path, "--source", "xapi")
    assert result.exit_code == 1
    assert "Cannot run" in result.output
    assert classified == []


def test_inspect_rejects_the_free_source_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["inspect", "@nobody", "--source", "free"],
                           env=_env(tmp_path))
    assert result.exit_code == 1
    assert "free sources cannot fetch X accounts" in result.output
    assert "Traceback" not in result.output


# --- blank path settings -------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_cookies_line_means_unset(value: str, monkeypatch, tmp_path: Path) -> None:
    """`TW_COOKIES=` in .env arrives as "", and Path("") is the current
    directory — which exists, so it used to be read as a cookies file."""
    monkeypatch.setenv("TW_COOKIES", value)
    monkeypatch.setenv("DB_PATH", " ")
    monkeypatch.setenv("OUT_DIR", "")
    settings = Settings()
    assert settings.tw_cookies is None
    assert settings.db_path == Settings.model_fields["db_path"].default
    assert settings.out_dir == Path("out")


def test_a_directory_is_not_a_cookies_file(tmp_path: Path) -> None:
    from scout.ingest.twscrape_src import TwscrapeSource

    settings = Settings(tw_cookies=tmp_path)
    with pytest.raises(RuntimeError, match="needs X session cookies"):
        TwscrapeSource(settings, Store(tmp_path / "t.db"))


def test_an_x_failure_mid_run_still_leaves_the_free_sources(tmp_path: Path, monkeypatch) -> None:
    """Cookies present but expired, or X down: the fetch raises after the
    adapter was built. The free legs must still run and save."""
    classified = _stub_discovery(monkeypatch)

    class Broken:
        name = "twscrape"
        parallel_safe = True

        async def fetch_accounts(self, *a, **kw):
            raise ConnectionError("x.com unreachable")

    monkeypatch.setattr(cli, "_build_adapter", lambda *a, **kw: Broken())
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "X discovery failed" in result.output
    assert classified == [["lawco"]]


def test_the_spend_cap_still_stops_a_paid_run(tmp_path: Path, monkeypatch) -> None:
    from scout.ingest.xapi_src import BudgetExceededError

    classified = _stub_discovery(monkeypatch)

    class Capped:
        name = "xapi"
        parallel_safe = False

        async def fetch_accounts(self, *a, **kw):
            raise BudgetExceededError(19.99, 0.30, 20.0)

    monkeypatch.setattr(cli, "_build_adapter", lambda *a, **kw: Capped())
    result = _run(tmp_path, "--source", "xapi")
    assert result.exit_code == 1
    assert "spend cap reached" in result.output
    assert classified == []
