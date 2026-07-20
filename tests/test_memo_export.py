"""Memo export tests — PDF bytes + override-aware pipeline rows."""

from __future__ import annotations

from pathlib import Path

from scout.config import Thesis
from scout.export import memo_pdf_bytes, pipeline_rows
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.store import Store


def test_memo_pdf_bytes_renders_markdown_to_pdf() -> None:
    data = memo_pdf_bytes(
        "## Overview\nEvalHQ builds evals.\n\n## Recommendation\n**PURSUE**",
        "EvalHQ — Investment memo",
        "Scout · score 72/100 · internal",
    )
    assert data[:5] == b"%PDF-"
    assert len(data) > 1000


def test_pipeline_rows_apply_manual_overrides_with_thesis(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    lead = Lead(
        account=Account(id="1", handle="ada", name="Ada Lin"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="ada", account_type="founder", is_founder=True,
                       stage="launched", company_name="EvalHQ",
                       thesis_fit=0.4, confidence=1.0, grounding="website"),
        score=40.0, rank=1,
    )
    store.save_leads("run-1", [lead])
    store.set_pipeline("ada", status="shortlisted")
    store.set_override("ada", score=91.0, note="conviction")

    thesis = Thesis(weights={"bio_intent": 20.0})
    with_overrides = pipeline_rows(store, thesis)
    assert with_overrides[0]["score"] == 91.0
    # Without a thesis the raw stored score is exported (legacy callers).
    assert pipeline_rows(store)[0]["score"] == 40.0
