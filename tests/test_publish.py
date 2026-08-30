"""Digest publisher tests — offline render against a seeded temp store."""

from __future__ import annotations

from pathlib import Path

from scout.config import Thesis
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.publish import build_digest, _icon_png
from scout.store import Store


def seeded_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "t.db")
    startup = Lead(
        account=Account(id="1", handle="ada_infra", name="Ada Lin",
                        bio="ex-OpenAI", website="https://evalhq.ai"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="ada_infra", account_type="founder", is_founder=True,
                       stage="launched", sector="ai infra", subsector="agent evals",
                       business_model="b2b saas", company_name="EvalHQ",
                       company_url="https://evalhq.ai",
                       one_line_summary="Eval platform for agents.",
                       thesis_fit=0.8, confidence=0.9),
        score=46.0,
    )
    stealth = Lead(
        account=Account(id="2", handle="stealth_sam", name="Sam O"),
        signals=[Signal(name="departure_signal", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="stealth_sam", account_type="founder", is_founder=True,
                       stage="stealth", confidence=0.8),
        score=20.0,
    )
    store.save_leads("r1", [startup, stealth])
    store.record_run("r1", source="twscrape", strategy_hash="h", thesis_statement="t")
    store.set_pipeline("ada_infra", status="shortlisted",
                       brief="**What they're building** — evals")
    return store


def test_build_digest_renders_tracks_and_no_secrets(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    out = build_digest(store, Thesis(thesis="Vertical agents"), tmp_path / "docs")

    page = out.read_text(encoding="utf-8")
    assert "EvalHQ" in page  # startup titled by company
    assert "@ada_infra" in page
    assert "stealth_sam" in page  # pre-launch watch section
    assert "stealth startup" in page  # unnamed founder → synthesized identity
    assert "Shortlisted" in page  # pipeline status chip
    assert "What they&#x27;re building" in page or "Research brief" in page
    assert 'name="robots" content="noindex"' in page
    assert "apple-mobile-web-app-capable" in page
    # never anything key-shaped or env-flavored
    assert "ANTHROPIC" not in page and "BEARER" not in page and ".env" not in page

    icon = (tmp_path / "docs" / "icon.png").read_bytes()
    assert icon.startswith(b"\x89PNG\r\n\x1a\n")


def test_icon_png_is_wellformed() -> None:
    data = _icon_png(8)
    assert data.startswith(b"\x89PNG") and data.endswith(
        b"IEND" + (0xAE426082).to_bytes(4, "big")
    )


def test_digest_search_index_is_lowercased(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    page = build_digest(store, Thesis(), tmp_path / "docs").read_text(encoding="utf-8")
    assert 'data-search="' in page
    start = page.index('data-search="') + len('data-search="')
    blob = page[start:page.index('"', start)]
    assert blob == blob.lower()


# --- the four-view app ---------------------------------------------------------


def app_store(tmp_path: Path) -> Store:
    """A store exercising every view: a funded/acquired startup with
    connections, a mover, private notes and votes that must never publish."""
    store = Store(tmp_path / "app.db")
    pollen = Lead(
        account=Account(id="1", handle="pollenrobotics", name="Pollen Robotics",
                        website="https://pollen-robotics.com"),
        llm=LLMVerdict(
            handle="pollenrobotics", account_type="startup", stage="launched",
            company_name="Pollen Robotics", company_url="https://pollen-robotics.com",
            one_line_summary="Open-source robots.", thesis_fit=0.6, confidence=0.9,
            funding_stage="seed", funding_amount="€2.5M",
            funding_investors=["Bpifrance"], funding_evidence="techcrunch",
            hq="Bordeaux, France", founded_year=2016,
            founders=["Matthieu Lapeyre — co-founder"],
            company_status="acquired",
            company_status_note="Hugging Face, April 2025",
            company_status_evidence="techcrunch 2025-04-14",
        ),
        score=55.0,
    )
    sibling = Lead(
        account=Account(id="2", handle="acmebot", name="Acme Robotics"),
        llm=LLMVerdict(handle="acmebot", account_type="startup", stage="launched",
                       company_name="Acme Robotics", funding_investors=["Bpifrance"],
                       funding_evidence="press", one_line_summary="Robots too.",
                       confidence=0.8),
        score=48.0,
    )
    store.save_leads("r0", [sibling.model_copy(update={"score": 30.0})])
    store.save_leads("r1", [pollen, sibling])  # sibling +18 → a mover
    store.record_run("r1", source="twscrape", strategy_hash="h", thesis_statement="t")
    store.set_pipeline("pollenrobotics", status="shortlisted",
                       notes="PRIVATE-NOTE-do-not-publish")
    store.set_pipeline("acmebot", status="longlisted")
    store.rebuild_graph(store.load_lead_ledger())
    return store


def test_app_views_render_with_data_attributes_and_tabs(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    # Tabs + hash routing.
    for view in ("view-startups", "view-funnel", "view-graph", "view-alerts"):
        assert view in page
    assert 'data-view="funnel"' in page
    # Sort/filter attributes stamped on cards.
    assert 'data-score="55.0"' in page
    assert 'data-status="shortlisted"' in page
    assert 'data-roundrank="2"' in page  # seed
    # Toolbar controls exist.
    for control in ('id="sort"', 'id="fstatus"', 'id="fstage"', 'id="fround"'):
        assert control in page


def test_researched_facts_and_warning_reach_the_page(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "Bordeaux, France" in page and "founded 2016" in page
    assert "Matthieu Lapeyre" in page
    assert "Seed · €2.5M" in page          # the round chip
    assert "Backed by Bpifrance" in page
    assert "⚠ Acquired — Hugging Face, April 2025" in page
    # The connections cross-link: Bpifrance backs both companies.
    assert "also backs" in page


def test_funnel_and_alerts_views(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    # Funnel groups in FUNNEL_STAGES order: Longlisted before Shortlisted.
    assert page.index(">Longlisted") < page.index(">Shortlisted")
    assert 'data-target="card-pollenrobotics"' in page
    # Alerts: the acquisition with its source, and the mover.
    assert "No longer independent" in page
    assert "techcrunch 2025-04-14" in page
    assert "Moving up" in page and "▲ 18" in page


def test_graph_view_embeds_the_canvas(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "Knowledge graph" in page
    assert "const NODES =" in page  # the canvas fragment landed
    assert "Bpifrance" in page


def test_adversarial_labels_cannot_break_out_of_the_graph_payload(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    evil = Lead(
        account=Account(id="1", handle="evilco", name="EvilCo"),
        llm=LLMVerdict(handle="evilco", account_type="startup", stage="launched",
                       company_name="EvilCo",
                       funding_investors=['</script><script>alert(1)',
                                          "Honest Capital"],
                       funding_evidence="press", confidence=0.8),
        score=10.0,
    )
    honest = Lead(
        account=Account(id="2", handle="other", name="OtherCo"),
        llm=LLMVerdict(handle="other", account_type="startup", stage="launched",
                       company_name="OtherCo",
                       funding_investors=["Honest Capital"],
                       funding_evidence="press", confidence=0.8),
        score=9.0,
    )
    store.save_leads("r1", [evil, honest])
    store.rebuild_graph(store.load_lead_ledger())
    page = build_digest(store, Thesis(thesis="t"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "</script><script>alert" not in page


def test_pwa_files_written_and_registered(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    out = build_digest(store, Thesis(thesis="t"), tmp_path / "docs")
    docs = out.parent
    assert (docs / "manifest.webmanifest").exists()
    sw = (docs / "sw.js").read_text(encoding="utf-8")
    assert "caches" in sw and "index.html" in sw
    page = out.read_text(encoding="utf-8")
    assert 'serviceWorker' in page and 'rel="manifest"' in page
    assert 'name="robots" content="noindex"' in page  # still private-ish


def test_private_judgments_never_publish(tmp_path: Path) -> None:
    """Notes are the firm's private judgment; spend is firm-internal.
    Neither may reach a public page."""
    store = app_store(tmp_path)
    store.record_llm_usage("classify", "m", cost_usd=7.77)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "PRIVATE-NOTE-do-not-publish" not in page
    assert "7.77" not in page and "spend" not in page.lower()
