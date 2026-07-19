"""UI smoke test via Streamlit's AppTest — renders scout/ui.py headlessly
against a temp DB and asserts the page builds without exceptions.

AppTest can't simulate data_editor edits or clicks-with-reruns reliably, so
this is a render test: tabs exist, a seeded lead shows up, no tracebacks.
Env vars outrank the .env file ui.py passes to Settings, so DB_PATH via
monkeypatch is enough to isolate the store.
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


def test_ui_renders_without_exceptions(tmp_path, monkeypatch) -> None:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    seed_store(db)

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.run()

    assert not at.exception, at.exception[0].message if at.exception else ""
    # All six surfaces render, plus the two Startups sub-pages (nested tabs
    # flatten into the element tree inside Startups)
    tab_labels = [t.label for t in at.tabs]
    top_level = ["Thesis", "Startups", "Longlist", "Shortlist", "Memo", "Settings"]
    assert [l for l in tab_labels if l in top_level] == top_level
    assert {"Latest run", "Database"} <= set(tab_labels)
    # The seeded lead renders in the default Startups track (stage=launched),
    # titled by its COMPANY — startups-first presentation.
    page_text = " ".join(m.value for m in at.markdown)
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
    # v7 quality rubric renders: lens chip, quality heading, dim bar + citation
    assert "B2B" in page_text
    assert "evidence-backed" in page_text
    assert "Traction" in page_text
    assert "12 logos" in page_text
    # The Database sub-page renders the store browser (tiles + row counts)
    assert "On disk" in page_text
    assert "matching rows" in page_text
    # Funnel pages render their empty states (nothing triaged in the seed)
    assert "Nothing longlisted yet" in page_text
    assert "Nothing shortlisted yet" in page_text
    assert "No startups to brief yet" in page_text
