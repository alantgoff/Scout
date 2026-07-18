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
    assert "To reach out" in page  # pipeline status chip
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
