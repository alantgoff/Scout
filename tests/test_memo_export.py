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


def test_write_csv_carries_scorecard_columns(tmp_path: Path) -> None:
    import csv

    from scout.export import write_csv

    lead = Lead(
        account=Account(id="1", handle="ada", name="Ada Lin"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(
            handle="ada", account_type="founder", is_founder=True,
            stage="launched", company_name="EvalHQ", customer_type="b2b",
            thesis_fit=0.4, confidence=1.0, grounding="website",
            scorecard={"prev_founder_experience": 3, "commercial_traction": 2},
            scorecard_reasons={"prev_founder_experience": "sold DevCo"},
        ),
        score=72.0, rank=1,
    )
    thesis = Thesis(weights={"bio_intent": 20.0})
    path = write_csv([lead], tmp_path, thesis)
    row = next(iter(csv.DictReader(path.open(encoding="utf-8"))))
    assert row["scorecard_rubric"] == "b2b"
    assert row["scorecard_band"] in ("strong", "promising", "weak")
    assert "founders_captable=" in row["scorecard_sections"]
    assert "prev_founder_experience=3" in row["scorecard"]
    assert "sold DevCo" in row["scorecard_reasons"]
    # quality_score column now reflects the scorecard total (company_quality).
    assert row["quality_score"] != ""


def test_write_csv_legacy_verdict_leaves_scorecard_blank(tmp_path: Path) -> None:
    import csv

    from scout.export import write_csv

    lead = Lead(
        account=Account(id="1", handle="old", name="Old Founder"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="old", is_founder=True, confidence=1.0,
                       grounding="website", quality={"team": 0.8},
                       quality_reasons={"team": "prev exit"}),
        score=50.0, rank=1,
    )
    thesis = Thesis(weights={"bio_intent": 20.0})
    path = write_csv([lead], tmp_path, thesis)
    row = next(iter(csv.DictReader(path.open(encoding="utf-8"))))
    assert row["scorecard"] == "" and row["scorecard_band"] == ""
    assert "team=0.80" in row["quality"]  # legacy column still populated
    assert row["quality_score"] != ""    # dispatches to legacy quality_score


def test_pipeline_rows_and_csv_carry_attr_columns(tmp_path: Path) -> None:
    from scout.export import write_pipeline_csv

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
    store.save_column("vertical", "Vertical", "select", options=["Fintech"])
    store.save_column("use_case", "Use case", "multiselect", options=["A", "B"])
    store.save_column("warm_intro", "Warm intro", "checkbox")
    store.set_attrs("ada", {"vertical": "Fintech", "use_case": ["A", "B"],
                            "warm_intro": True})

    rows = pipeline_rows(store)
    assert rows[0]["Vertical"] == "Fintech"
    assert rows[0]["Use case"] == "A;B"
    assert rows[0]["Warm intro"] == "yes"
    path = write_pipeline_csv(rows, tmp_path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "Vertical" in header and "Use case" in header and "Warm intro" in header
