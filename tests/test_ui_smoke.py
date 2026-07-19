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
    from scout.diligence.schema import DimensionFinding, Memo, ReconReport

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
                       thesis_fit=0.8, confidence=0.9),
        score=62.0, rank=1,
    )
    store.save_leads("run-smoke", [lead])
    store.record_run("run-smoke", source="twscrape", strategy_hash="smoke-hash",
                     thesis_statement="smoke thesis")
    # A second startup WITHOUT a memo keeps the "Worth a deep look" shelf alive.
    other = Lead(
        account=Account(id="2", handle="other_founder", name="Other Founder",
                        bio="building agents", followers=500, source="search"),
        signals=[Signal(name="launch_traction", value=1.0, weight=10.0)],
        llm=LLMVerdict(handle="other_founder", account_type="founder",
                       is_founder=True, stage="launched", sector="agents",
                       company_name="OtherCo", thesis_fit=0.7, confidence=0.9),
        score=55.0, rank=2,
    )
    store.save_leads("run-smoke", [lead, other])
    # Seed a deep-analysis memo for SmokeCo -> the Diligence section renders.
    store.save_memo(Memo(
        company_key="smokeco", company_name="SmokeCo", fingerprint="stale-fp",
        composite=72.0, flags=["commodity_risk"],
        findings=[
            DimensionFinding(key="data_flywheel", score=8.0, confidence=0.9,
                             summary="production data loop",
                             unknowns=["data rights unverified"]),
            DimensionFinding(key="market_size", score=None, confidence=0.0,
                             summary="Analysis unavailable: no sources",
                             flags=["analysis_failed"]),
        ],
        recon=ReconReport(company_name="SmokeCo", website="https://smokeco.ai"),
        memo_md="## Recommendation\nPursue.\n\n### Sources\n[1] https://smokeco.ai",
        cost_usd=5.25, created_at="2026-07-18T00:00:00+00:00",
    ))
    store.kg_link_competitors("SmokeCo", "RivalCo")


def test_ui_renders_without_exceptions(tmp_path, monkeypatch) -> None:
    db = tmp_path / "smoke.db"
    monkeypatch.setenv("DB_PATH", str(db))
    seed_store(db)

    at = AppTest.from_file(str(UI_PATH), default_timeout=30)
    at.run()

    assert not at.exception, at.exception[0].message if at.exception else ""
    # All four surfaces render
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Leads", "Pipeline", "Sourcing", "Settings"]
    # The seeded lead renders in the default Startups track (stage=launched),
    # titled by its COMPANY — startups-first presentation.
    page_text = " ".join(m.value for m in at.markdown)
    assert "SmokeCo" in page_text
    assert "smoke_founder" in page_text
    # Fine-grained fields render on the card face
    assert "Fit 80%" in page_text
    assert "agent evals" in page_text
    # Diligence surfaces: composite chip + scorecard render from the seeded
    # memo; the un-memoed startup keeps the "Worth a deep look" shelf alive.
    assert "Diligence 72" in page_text
    assert "Diligence Score 72" in page_text
    assert "Data flywheel" in page_text
    assert "commodity_risk" in page_text
    assert "Worth a deep look" in page_text
    assert "OtherCo" in page_text
    # Landscape chips from the knowledge graph (user-linked competitor).
    assert "RivalCo" in page_text
