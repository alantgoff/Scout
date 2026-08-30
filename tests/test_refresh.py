"""`scout refresh` — the rotating watch on tracked companies.

The rotation is the cost model: a fixed daily allowance (default 3) sweeps
the whole tracked list oldest-first, so spend stays flat as the list grows
and every company still gets re-researched on a cycle. The command reuses
the `scout add` research stack (research_company → apply_research), so what
these tests pin is the rotation, the budget stop, and that a status change
reaches the activity feed — the digest's alert source.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from scout.agents import CompanyProfile
from scout.cli import app
from scout.models import Account, Lead, LLMVerdict
from scout.store import Store

runner = CliRunner()


def _track(store: Store, handle: str, *, status: str = "longlisted",
           refreshed_days_ago: int | None = None,
           website: str | None = "https://example.com/") -> None:
    store.upsert_account(Account(id=handle, handle=handle, website=website))
    store.save_leads(
        f"seed-{handle}",
        [Lead(account=store.get_account(handle),
              llm=LLMVerdict(handle=handle, account_type="startup",
                             company_name=handle.title(), company_url=website,
                             grounding="website"))],
    )
    store.set_pipeline(handle, status=status)
    if refreshed_days_ago is not None:
        row = store.get_pipeline(handle)
        row["researched_at"] = (
            datetime.now(timezone.utc) - timedelta(days=refreshed_days_ago)
        ).isoformat()
        store.db["pipeline"].upsert(row, pk="handle", alter=True)


# --- the rotation (store.refresh_queue) ---------------------------------------


def test_queue_is_tracked_only_oldest_first_never_recent(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    _track(store, "neverdone")                          # no stamp → first
    _track(store, "stale", refreshed_days_ago=30)
    _track(store, "fresh", refreshed_days_ago=2)        # inside the 7-day floor
    _track(store, "untracked", status="new")            # not the firm's problem
    _track(store, "dismissed", status="passed")

    assert store.refresh_queue(min_age_days=7, limit=10) == ["neverdone", "stale"]
    assert store.refresh_queue(min_age_days=7, limit=1) == ["neverdone"]
    assert store.refresh_queue(min_age_days=7, limit=0) == []


def test_mark_refreshed_moves_a_company_to_the_back(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    _track(store, "a", refreshed_days_ago=30)
    _track(store, "b", refreshed_days_ago=20)
    assert store.refresh_queue(7, 10) == ["a", "b"]
    store.mark_refreshed("a")
    assert store.refresh_queue(7, 10) == ["b"]  # a is fresh now


# --- the command --------------------------------------------------------------


def _stub_research(monkeypatch, profiles: dict[str, CompanyProfile]):
    """research_company answers per-domain from `profiles`; records which
    domains were actually researched."""
    from scout import agents

    calls: list[str] = []

    def fake(domain, settings, *, site_text="", on_event=None, store=None):
        calls.append(domain)
        profile = profiles.get(domain, CompanyProfile())
        if store is not None:
            store.record_llm_usage("research", settings.claude_model,
                                   input_tokens=1000, cost_usd=0.05)
        return profile, {"researched": True, "searches": 1, "fetches": 1,
                         "sources": [], "cost_usd": 0.05}

    monkeypatch.setattr(agents, "research_company", fake)
    return calls


def _refresh(tmp_path: Path, *args: str, cap: str = "1.0"):
    return runner.invoke(
        app, ["refresh", *args, "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(tmp_path / "t.db"), "ANTHROPIC_API_KEY": "k",
             "DAILY_SPEND_CAP_USD": cap},
    )


def test_refresh_takes_the_oldest_stamps_them_and_flags_status_changes(
    tmp_path: Path, monkeypatch
) -> None:
    store = Store(tmp_path / "t.db")
    _track(store, "acquiredco", refreshed_days_ago=30,
           website="https://acquired.example/")
    _track(store, "steadyco", refreshed_days_ago=20,
           website="https://steady.example/")
    calls = _stub_research(monkeypatch, {
        "https://acquired.example/": CompanyProfile(
            company_status="acquired",
            company_status_note="BigCo, Aug 2026",
            company_status_evidence="press release",
            sources=["https://press.example/a"],
        ),
    })

    result = _refresh(tmp_path, "--limit", "2")
    assert result.exit_code == 0, result.output
    assert calls == ["https://acquired.example/", "https://steady.example/"]

    store = Store(tmp_path / "t.db")
    # Both stamped — the rotation moved on.
    assert store.refresh_queue(7, 10) == []
    # The finding survived to the stored verdict…
    lead = store.latest_lead("acquiredco")
    assert lead is not None and lead.llm.company_status == "acquired"
    # …and to the activity spine, where the digest reads alerts from.
    events = store.events(verbs=["company_status_changed"])
    assert len(events) == 1
    assert events[0].handle == "acquiredco"
    assert events[0].payload["new"] == "acquired"
    # The unchanged company raised no alert.
    assert Store(tmp_path / "t.db").latest_lead("steadyco").llm.company_status \
        != "acquired"


def test_refresh_stops_mid_list_when_the_envelope_empties(
    tmp_path: Path, monkeypatch
) -> None:
    """The gate checks BEFORE each call, so a day can overshoot by at most
    one call — the documented trade. Here the first research ($0.05 stub)
    empties the envelope ($0.02 left), so the second company must wait for
    tomorrow — and stays at the front of the queue, never stamped."""
    store = Store(tmp_path / "t.db")
    _track(store, "first", refreshed_days_ago=30, website="https://f.example/")
    _track(store, "second", refreshed_days_ago=20, website="https://s.example/")
    store.record_llm_usage("classify", "m", cost_usd=0.98)  # 0.02 left of 1.00
    calls = _stub_research(monkeypatch, {})

    result = _refresh(tmp_path, "--limit", "2")
    assert result.exit_code == 0, result.output
    assert calls == ["https://f.example/"]
    assert "cap reached" in result.output
    assert Store(tmp_path / "t.db").refresh_queue(7, 10) == ["second"]


def test_refresh_skips_and_stamps_companies_it_cannot_research(
    tmp_path: Path, monkeypatch
) -> None:
    """No website on file → nothing to research; stamping anyway keeps the
    queue from serving the same unresearchable company every single day."""
    store = Store(tmp_path / "t.db")
    _track(store, "nosite", refreshed_days_ago=30, website=None)
    calls = _stub_research(monkeypatch, {})
    result = _refresh(tmp_path)
    assert result.exit_code == 0, result.output
    assert calls == []
    assert Store(tmp_path / "t.db").refresh_queue(7, 10) == []


def test_refresh_without_a_key_does_nothing(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["refresh", "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(tmp_path / "t.db"), "ANTHROPIC_API_KEY": ""},
    )
    assert result.exit_code == 0
    assert "needs the research agent" in result.output.replace("\n", " ")
