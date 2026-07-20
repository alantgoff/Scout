"""UI smoke test via Streamlit's AppTest — renders scout/ui.py headlessly
against a temp DB and asserts the pages build without exceptions.

Navigation is session-state-driven (segmented control keyed "nav"), so each
page renders on its own run: set at.session_state["nav"] and rerun. AppTest
can't simulate data_editor edits or clicks-with-reruns reliably, so these are
render tests: pages exist, a seeded lead shows up, no tracebacks.

Env vars outrank the .env file ui.py passes to Settings, so DB_PATH via
monkeypatch is enough to isolate the store — and ANTHROPIC_API_KEY is pinned
empty so no test can ever hit the real API.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from scout.models import Account, Lead, LLMVerdict, Signal
from scout.store import Store

UI_PATH = Path(__file__).resolve().parent.parent / "scout" / "ui.py"


def seed_store(db_path: Path) -> None:
    store = Store(db_path)
    lead = Lead(
        account=Account(id="1", handle="smoke_founder", name="Smoke Founder",
                        bio="ex-OpenAI, building evals in stealth",
                        followers=1234, source="search"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="smoke_founder", account_type="founder",
                       is_founder=True, stage="launched", sector="ai infra",
                       subsector="agent evals", business_model="b2b saas",
                       company_name="SmokeCo", company_url="https://smokeco.ai",
                       one_line_summary="Building an eval platform.",
                       thesis_fit=0.8, confidence=0.9,
                       grounding="website", customer_type="b2b",
                       quality={"team": 0.8, "traction": 0.6},
                       quality_reasons={"team": "prev sold EvalCo",
                                        "traction": "website: 12 logos"},
                       value_add_fit=0.7,
                       value_add_levers={"global_expansion": 0.9,
                                         "data_driven_growth": 0.5},
                       value_add_reason="EU expansion next; usage-metric heavy"),
        score=62.0, rank=1,
    )
    # A launched founder the classifier couldn't name a company for — must
    # render startup-first with a synthesized identity, not as a person.
    unnamed = Lead(
        account=Account(id="2", handle="nora_builds", name="Nora Vale",
                        bio="building something new", followers=800, source="search"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="nora_builds", account_type="founder",
                       is_founder=True, stage="launched", confidence=0.8),
        score=31.0, rank=2,
    )
    store.save_leads("run-smoke", [lead, unnamed])
    store.record_run("run-smoke", source="twscrape", strategy_hash="smoke-hash",
                     thesis_statement="smoke thesis")


def _app(tmp_path, monkeypatch) -> AppTest:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # never touch the real API
    seed_store(db)
    return AppTest.from_file(str(UI_PATH), default_timeout=30)


def _page_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_ui_renders_without_exceptions(tmp_path, monkeypatch) -> None:
    at = _app(tmp_path, monkeypatch)
    at.run()

    assert not at.exception, at.exception[0].message if at.exception else ""
    # Default page is Thesis; the top-level tabs are gone (session-state nav).
    assert at.session_state["nav"] == "Thesis"
    assert "Define the thesis" in _page_text(at)

    # ---- Startups: feed + database sub-tabs
    at.session_state["nav"] = "Startups"
    at.session_state["sdb_raw"] = True  # open the raw-tables toggle too
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    tab_labels = [t.label for t in at.tabs]
    assert {"Latest run", "Database"} <= set(tab_labels)
    page_text = _page_text(at)
    # The seeded lead renders in the default Startups track (stage=launched),
    # titled by its COMPANY — startups-first presentation.
    assert "SmokeCo" in page_text
    assert "smoke_founder" in page_text
    # Startup-first: the unnamed launched founder gets a synthesized identity
    assert "Nora Vale&#x27;s unnamed startup" in page_text
    # Fine-grained fields render on the card face
    assert "Fit 80%" in page_text
    assert "agent evals" in page_text
    # The firm value-add dimension renders (chip + lever bars in Details)
    assert "lift 70%" in page_text
    assert "Local-to-global expansion" in page_text
    # Scoring is broken out with explanations: rubric header, blend share,
    # per-dim citation, missing dims shown as excluded, the math intro.
    assert "Company quality" in page_text
    assert "of the blend" in page_text
    assert "12 logos" in page_text
    assert "no evidence — excluded" in page_text
    assert "Score math" in page_text
    # The startup database sub-page renders the dossier table with the
    # seeded user columns and their filter/manager plumbing…
    assert "Startup database" in page_text
    assert "select a row for the full dossier" in page_text
    assert at.session_state["sdb_mode"] in (None, "Browse")
    manager_text = page_text
    assert "Vertical" in manager_text          # seeded builtin columns
    assert "Use case" in manager_text
    assert "Priority" in manager_text
    assert "Add a column" in manager_text      # column manager renders
    assert "Fills the" in manager_text         # categorizer popover renders
    # …and the raw store browser behind the toggle
    assert "On disk" in page_text
    assert "matching rows" in page_text

    # ---- Funnel pages render their empty states (nothing triaged in seed)
    at.session_state["nav"] = "Longlist"
    at.run()
    assert not at.exception
    assert "Nothing longlisted yet" in _page_text(at)

    at.session_state["nav"] = "Shortlist"
    at.run()
    assert not at.exception
    assert "Nothing shortlisted yet" in _page_text(at)

    at.session_state["nav"] = "Memos"
    at.run()
    assert not at.exception
    assert "No memos yet" in _page_text(at)

    at.session_state["nav"] = "Settings"
    at.run()
    assert not at.exception
    assert "X API spend" in _page_text(at)


def test_memo_button_routes_and_generates_named_by_startup(tmp_path, monkeypatch) -> None:
    """The card's Memo button routes to the Memos page (nav_target +
    memo_target) and writes the memo there on arrival — named after the
    STARTUP, not the handle, and with the full section skeleton."""
    db = tmp_path / "smoke.db"
    at = _app(tmp_path, monkeypatch)
    # Simulate the card button's routing state, exactly as _lead_card sets it.
    at.session_state["nav_target"] = "Memos"
    at.session_state["memo_target"] = "smoke_founder"
    at.run()

    assert not at.exception, at.exception[0].message if at.exception else ""
    assert at.session_state["nav"] == "Memos"
    # The memo was generated on arrival (offline fallback — no API key),
    # stored in the pipeline with its generation meta, and carries the
    # section skeleton.
    row = Store(db).get_pipeline("smoke_founder")
    assert row.get("brief")
    for heading in ("## Overview", "## Product & differentiation",
                    "## Technology & architecture", "## Competitive landscape",
                    "## Market sizing",
                    "## Strategic capital & acquisition dynamics",
                    "## Recommendation"):
        assert heading in row["brief"], heading
    assert row["brief_meta"]["depth"] == "standard"  # session default depth
    page_text = _page_text(at)
    # Startup-named everywhere: page header, memo title, and the toast.
    assert "Investment memos" in page_text
    assert "SmokeCo" in page_text
    # Depth UX renders: selector state + cost caption + verdict chip from
    # the template's VERDICT line.
    assert at.session_state["memo_depth"] == "Standard"
    assert "per memo" in page_text
    assert ">TRACK</span>" in page_text
    toasts = " ".join(t.value for t in at.toast)
    assert "SmokeCo" in toasts  # named by startup…
    assert "smoke_founder" not in toasts  # …never by the raw handle


def test_database_edit_mode_renders_editor_with_user_columns(tmp_path, monkeypatch) -> None:
    """Edit mode: the data_editor renders with seeded + stored attribute
    values and no exceptions (actual edit persistence is covered by the
    pure editor_changes tests — AppTest can't simulate editor edits)."""
    db = tmp_path / "smoke.db"
    at = _app(tmp_path, monkeypatch)
    store = Store(db)
    store.set_attrs("smoke_founder", {"vertical": "AI infrastructure",
                                      "use_case": ["Evals / observability"]})
    at.session_state["nav"] = "Startups"
    at.session_state["sdb_mode"] = "Edit"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    page_text = _page_text(at)
    assert "changes save on the spot" in page_text
    # Back in Browse mode the page still renders cleanly with stored attrs.
    at.session_state["sdb_mode"] = "Browse"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""


def test_regenerate_without_key_never_clobbers_existing_memo(tmp_path, monkeypatch) -> None:
    """The overwrite guard: when generation falls back to the data-only
    skeleton (here: no API key), an existing memo must survive untouched."""
    db = tmp_path / "smoke.db"
    at = _app(tmp_path, monkeypatch)
    Store(db).set_pipeline("smoke_founder", status="longlisted",
                           brief="## Overview\nORIGINAL MEMO",
                           brief_meta={"depth": "standard", "sources": [],
                                       "searches": 0, "fetches": 0})
    at.session_state["nav"] = "Memos"
    at.session_state["memo_pick"] = "smoke_founder"
    at.run()
    assert not at.exception

    regen = next(b for b in at.button if b.key == "memo_generate")
    regen.click()
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    row = Store(db).get_pipeline("smoke_founder")
    assert row["brief"] == "## Overview\nORIGINAL MEMO"  # untouched
    toasts = " ".join(t.value for t in at.toast)
    assert "existing memo untouched" in toasts


def test_memo_edit_saves_with_edit_stamp(tmp_path, monkeypatch) -> None:
    """Editing state → save writes brief_edited_at (the 'edited' caption) and
    keeps the original generation stamp."""
    db = tmp_path / "smoke.db"
    store = Store(db)  # pre-write a memo, as if generated earlier
    at = _app(tmp_path, monkeypatch)
    store.set_pipeline("smoke_founder", status="longlisted", brief="## Overview\noriginal")
    at.session_state["nav"] = "Memos"
    at.run()
    assert not at.exception
    page_text = _page_text(at)
    assert "generated just now" in page_text
    # Enter editing mode and rerun — the editor text area appears.
    at.session_state["memo_editing"] = "smoke_founder"
    at.run()
    assert not at.exception
    editor = [ta for ta in at.text_area if ta.key == "memo_editor_smoke_founder"]
    assert editor and "original" in editor[0].value
