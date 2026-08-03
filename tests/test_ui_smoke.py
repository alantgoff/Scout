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

from scout.agents import MEMO_SECTIONS
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
                       scorecard={"prev_founder_experience": 3,
                                  "commercial_traction": 2,
                                  "tech_moat_ip": 3},
                       scorecard_reasons={"prev_founder_experience": "prev sold EvalCo",
                                          "commercial_traction": "website: 12 logos"},
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
    return store


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
    # Feed and Database are a rail-level View switch, not st.tabs: only ONE
    # body renders per run (st.tabs rendered both server-side and leaked the
    # feed's sidebar controls onto the Database view). Feed is the default.
    assert at.session_state["startups_view"] == "Feed"
    page_text = _page_text(at)
    # The seeded lead renders in the default Startups track (stage=launched),
    # titled by its COMPANY — startups-first presentation.
    assert "SmokeCo" in page_text
    assert "smoke_founder" in page_text
    # Startup-first: the unnamed launched founder gets a synthesized identity
    assert "Nora Vale&#x27;s unnamed startup" in page_text
    # The detail pane carries what the dense row has no room for: thesis fit
    # as its own readout, the three scoring dimensions, and the sector line.
    assert "Thesis fit" in page_text
    for dimension in ("Quality", "Fit", "Signal"):
        assert f'dpane-dlab">{dimension}<' in page_text
    assert "agent evals" in page_text
    # The firm value-add dimension renders (chip + lever bars in Details)
    assert "lift 70%" in page_text
    assert "Local-to-global expansion" in page_text
    # Scoring is broken out with explanations: scorecard header + band,
    # blend share, per-criterion citation, excluded sections, the math intro.
    assert "Scorecard" in page_text
    assert "Enterprise readiness" in page_text
    assert "of the blend" in page_text
    assert "12 logos" in page_text
    assert "no evidence — excluded" in page_text
    assert "Score math" in page_text
    # ---- Database is a separate body, not a sibling tab: it only renders
    # when the rail switch selects it (st.tabs used to render both at once).
    at.session_state["startups_view"] = "Database"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    page_text = _page_text(at)
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


def test_longlist_and_shortlist_render_cards_with_leads(tmp_path, monkeypatch) -> None:
    """The funnel pages call _lead_card, which reaches module-level helpers
    (e.g. _attr_display) that must be bound before ANY nav block runs — not
    just the Startups page. The base smoke test only hits the empty states,
    so this seeds a longlisted + shortlisted lead and renders both pages."""
    at = _app(tmp_path, monkeypatch)
    at.run()
    store = Store(tmp_path / "smoke.db")
    store.set_pipeline("smoke_founder", status="longlisted")
    store.set_pipeline("nora_builds", status="shortlisted")

    at.session_state["nav"] = "Longlist"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert "SmokeCo" in _page_text(at)  # the card rendered, not the empty state

    at.session_state["nav"] = "Shortlist"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""


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
    # The full section contract, including the four a partner expects and the
    # memo used to omit: Why now, Team, Traction, Deal terms, Risks.
    for section in MEMO_SECTIONS:
        assert f"## {section}" in row["brief"], section
    # Deep by default: it is the only depth that verifies funding figures and
    # researches founders, and an unverified number is what sinks a memo.
    assert row["brief_meta"]["depth"] == "deep"
    page_text = _page_text(at)
    # Startup-named everywhere: page header, memo title, and the toast.
    assert "Investment memos" in page_text
    assert "SmokeCo" in page_text
    # Depth UX renders: selector state + cost caption + verdict chip from
    # the template's VERDICT line.
    assert at.session_state["memo_depth"] == "Deep research"
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
    # Must select the Database view explicitly: the rail switch renders one
    # body per run, so the editor is not on screen while the feed is showing.
    at.session_state["startups_view"] = "Database"
    at.session_state["sdb_mode"] = "Edit"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    page_text = _page_text(at)
    assert "changes save on the spot" in page_text
    # Back in Browse mode the page still renders cleanly with stored attrs.
    at.session_state["sdb_mode"] = "Browse"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""


def test_score_names_the_thesis_it_was_measured_against(tmp_path, monkeypatch) -> None:
    """A score is meaningless without the thesis behind it, and stale when the
    thesis has been retuned since. Both must be visible, not inferred."""
    from scout.config import ensure_thesis_id, load_thesis, thesis_version
    from scout.config import load_seeds

    db = tmp_path / "smoke.db"
    project = Path(__file__).resolve().parents[1]
    active = load_thesis(project / "thesis.yaml")
    thesis_id = ensure_thesis_id(active)
    current = thesis_version(active, load_seeds(project / "seeds.yaml"))

    at = _app(tmp_path, monkeypatch)
    store = Store(db)
    store.upsert_thesis(thesis_id, name=active.name or thesis_id,
                        statement=active.thesis, version=current)
    # Scored under the version that is current → named, not flagged.
    store.record_verdict("smoke_founder", "fp", LLMVerdict(handle="smoke_founder",
                         thesis_fit=0.8, confidence=0.9),
                         thesis_id=thesis_id, thesis_version=current)
    at.session_state["nav"] = "Startups"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert "Scored against" in _page_text(at)
    assert "thesis has changed since" not in _page_text(at)

    # A lead scored under an EARLIER tuning of the same thesis is stale — the
    # live thesis.yaml is the source of truth for "what the thesis is now",
    # so staleness is the verdict's version disagreeing with it.
    at = _app(tmp_path, monkeypatch)
    store = Store(db)
    store.upsert_thesis(thesis_id, name=active.name or thesis_id,
                        statement=active.thesis, version=current)
    store.record_verdict("smoke_founder", "fp", LLMVerdict(handle="smoke_founder",
                         thesis_fit=0.8, confidence=0.9),
                         thesis_id=thesis_id, thesis_version="an-earlier-version")
    at.session_state["nav"] = "Startups"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert "thesis has changed since" in _page_text(at)


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


# --- identity & multiplayer gating ----------------------------------------------


def test_dev_identity_provisions_first_admin(tmp_path, monkeypatch) -> None:
    """Without an [auth] block the app resolves a dev identity; the first
    user ever provisioned is the admin."""
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    user = Store(tmp_path / "smoke.db").get_user("local@scout")
    assert user is not None and user["role"] == "admin"


def test_dev_identity_env_override_and_actor_stamp(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCOUT_DEV_USER", "sara@firm.com")
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    store = Store(tmp_path / "smoke.db")
    assert store.get_user("sara@firm.com") is not None
    # A write made through the app's store is attributed to the signed-in user.
    # (Simulated at the store level: the UI binds store.actor = ACTOR.)
    store.actor = "sara@firm.com"
    store.set_pipeline("smoke_founder", status="longlisted")
    assert store.get_pipeline("smoke_founder")["updated_by"] == "sara@firm.com"


def test_oidc_gate_blocks_until_signed_in(tmp_path, monkeypatch) -> None:
    """With [auth] configured but nobody logged in, the app renders only the
    sign-in gate — no nav, no data."""
    at = _app(tmp_path, monkeypatch)
    at.secrets["auth"] = {
        "client_id": "x", "client_secret": "y", "cookie_secret": "z",
        "redirect_uri": "https://scout.test/oauth2callback",
    }
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert any("Sign in with Google" in b.label for b in at.button)
    assert "nav" not in at.session_state  # the page stopped at the gate


def test_allowlist_blocks_unlisted_user(tmp_path, monkeypatch) -> None:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SCOUT_DEV_USER", "outsider@elsewhere.com")
    seed_store(db)
    gate = Store(db)
    gate.set_setting("allowed_email_domain", "firm.com")
    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert any("not on this workspace" in e.value for e in at.error)
    assert "nav" not in at.session_state
    # And a firm-domain user gets in.
    monkeypatch.setenv("SCOUT_DEV_USER", "sara@firm.com")
    at2 = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at2.run()
    assert not at2.exception
    assert at2.session_state["nav"] == "Thesis"


# --- collaboration surfaces -----------------------------------------------------


def test_vote_from_the_detail_pane_persists_and_shows_the_split(
    tmp_path, monkeypatch
) -> None:
    """Two partners, opposite stances: both survive, and the startup reads
    as contested rather than one overwriting the other."""
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SCOUT_DEV_USER", "alan@firm.com")
    seed_store(db)

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.session_state["nav"] = "Startups"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    # The stance buttons render in the detail pane for the selected lead.
    vote_buttons = [b for b in at.button if b.key.startswith("feeddet_vote_")]
    assert {b.key.split("_")[2] for b in vote_buttons} >= {"strong", "yes", "unsure", "pass"}
    next(b for b in at.button if b.key == "feeddet_vote_strong_yes_smoke_founder").click()
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""

    store = Store(db)
    assert store.votes_for("smoke_founder")[0].stance == "strong_yes"
    assert store.votes_for("smoke_founder")[0].actor == "alan@firm.com"

    # The partner in another timezone disagrees.
    store.actor = "sara@firm.com"
    store.ensure_user("sara@firm.com", name="Sara Lin")
    store.set_vote("smoke_founder", "pass", "crowded space")

    monkeypatch.setenv("SCOUT_DEV_USER", "sara@firm.com")
    at2 = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at2.session_state["nav"] = "Startups"
    at2.run()
    assert not at2.exception, at2.exception[0].message if at2.exception else ""
    page = _page_text(at2)
    assert "stance-badge" in page  # both partners' initials render
    assert "Split" in page  # and the disagreement is called out
    # Both votes survived — neither partner clobbered the other.
    assert len(Store(db).votes_for("smoke_founder")) == 2


def test_activity_page_renders_and_clears_the_unread_badge(tmp_path, monkeypatch) -> None:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SCOUT_DEV_USER", "alan@firm.com")
    seed_store(db)
    # Something happened while Alan was asleep.
    other = Store(db, actor="sara@firm.com")
    other.ensure_user("sara@firm.com", name="Sara Lin")
    other.set_vote("smoke_founder", "pass", "not for us")
    other.add_comment("smoke_founder", "passing — thin moat")
    assert other.unread_count("alan@firm.com") == 2

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.session_state["nav"] = "Activity"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    page = _page_text(at)
    assert "Sara" in page
    assert "voted pass" in page and "not for us" in page
    assert "commented" in page
    # Visiting the page advances the read cursor.
    assert Store(db).unread_count("alan@firm.com") == 0


def test_comment_with_mention_is_stored_and_pinged(tmp_path, monkeypatch) -> None:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("SCOUT_DEV_USER", "alan@firm.com")
    seed_store(db)
    setup = Store(db)
    setup.ensure_user("alan@firm.com", name="Alan Goff")
    setup.ensure_user("sara@firm.com", name="Sara Lin")
    setup.set_setting("slack_webhook_url", "https://hooks.slack.test/x")
    setup.set_setting("app_base_url", "https://scout.test")

    sent: list[dict] = []
    monkeypatch.setattr("scout.notify._post",
                        lambda url, payload: sent.append(payload))

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.session_state["nav"] = "Startups"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    box = next(t for t in at.text_area if t.key == "feeddet_cmt_smoke_founder")
    box.set_value("@sara worth a look — ex-OpenAI team").run()
    next(b for b in at.button if b.key == "feeddet_cmtbtn_smoke_founder").click().run()
    assert not at.exception, at.exception[0].message if at.exception else ""

    comments = Store(db).comments_for("smoke_founder")
    assert len(comments) == 1
    assert comments[0].mentions == ["sara@firm.com"]
    # And the mention pinged Slack with a deep link back to the startup.
    assert sent and "Sara" in sent[0]["text"]
    assert "https://scout.test/?s=smoke_founder" in sent[0]["text"]


def test_memo_regeneration_keeps_the_edit_restorable(tmp_path, monkeypatch) -> None:
    """The failure memo versioning exists to fix, end to end through the UI's
    store calls."""
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    seed_store(db)
    store = Store(db, actor="alan@firm.com")
    store.set_memo("smoke_founder", "# generated", kind="generated", actor="agent:memo")
    store.set_memo("smoke_founder", "# my careful edits", kind="edited")
    store.set_memo("smoke_founder", "# regenerated", kind="generated", actor="agent:memo")

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.session_state["nav"] = "Memos"
    at.session_state["memo_target"] = "smoke_founder"
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert any(b.key.startswith("memo_restore_") for b in at.button)
    next(b for b in at.button if b.key == "memo_restore_2").click().run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert Store(db).get_pipeline("smoke_founder")["brief"] == "# my careful edits"
