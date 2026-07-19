"""Parse-helper tests: fenced/malformed JSON, unknown-field tolerance,
clamping, evidence-source normalization, payload validation per dimension."""

from __future__ import annotations

import json

import pytest

from scout.diligence.schema import (
    CrossExam,
    DimensionFinding,
    apply_cross_exam,
    extract_json,
    gap_finding,
    parse_cross_exam,
    parse_finding,
    parse_recon,
)


def finding_payload(key: str = "data_flywheel", **overrides) -> dict:
    data = {
        "key": key,
        "score": 7.5,
        "confidence": 0.8,
        "summary": "solid",
        "detail_md": "## steps",
        "evidence": [{"claim": "ships evals", "url": "https://x.ai/docs", "source": "web"}],
        "unknowns": ["revenue unknown"],
        "flags": [],
        "payload": {},
    }
    data.update(overrides)
    return data


# --- extract_json / fences ----------------------------------------------------


def test_parse_finding_plain_json() -> None:
    finding = parse_finding(json.dumps(finding_payload()), "data_flywheel")
    assert finding.score == 7.5
    assert finding.evidence[0].url == "https://x.ai/docs"
    assert finding.unknowns == ["revenue unknown"]


def test_parse_finding_fenced_json() -> None:
    text = "```json\n" + json.dumps(finding_payload()) + "\n```"
    assert parse_finding(text, "data_flywheel").score == 7.5


def test_parse_finding_with_narration_around_json() -> None:
    # Research agents narrate; the outermost {...} span is still parseable.
    text = "Based on my research:\n" + json.dumps(finding_payload()) + "\nDone."
    assert parse_finding(text, "data_flywheel").summary == "solid"


def test_parse_finding_malformed_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_finding("not json at all", "data_flywheel")


def test_parse_finding_array_raises() -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_finding("[1, 2]", "data_flywheel")


def test_extract_json_prefers_direct_parse() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


# --- sanitization -------------------------------------------------------------


def test_key_is_forced_from_caller() -> None:
    text = json.dumps(finding_payload(key="competition"))
    assert parse_finding(text, "market_size").key == "market_size"


def test_unknown_fields_tolerated() -> None:
    data = finding_payload()
    data["totally_new_field"] = "surprise"
    assert parse_finding(json.dumps(data), "data_flywheel").score == 7.5


def test_score_and_confidence_clamped() -> None:
    finding = parse_finding(
        json.dumps(finding_payload(score=85, confidence=1.7)), "data_flywheel"
    )
    assert finding.score == 10.0
    assert finding.confidence == 1.0
    assert parse_finding(
        json.dumps(finding_payload(score=None)), "data_flywheel"
    ).score is None


def test_evidence_source_normalization() -> None:
    data = finding_payload(
        evidence=[
            {"claim": "a", "url": "https://a.com", "source": "website"},  # bad tag
            {"claim": "b", "url": "", "source": "web"},  # no URL claiming web
            {"claim": "c", "url": "", "source": "x-data"},  # legit x-data
        ]
    )
    finding = parse_finding(json.dumps(data), "data_flywheel")
    assert finding.evidence[0].source == "web"
    assert finding.evidence[1].source == "assumption"  # law 1 enforced
    assert finding.evidence[2].source == "x-data"


# --- payload validation -------------------------------------------------------


def test_market_size_payload_validated() -> None:
    payload = {
        "top_down": {
            "low": 500, "base": 1200, "high": 3000, "year": 2026,
            "cagr": "24%", "sources": ["https://gartner.com/x"],
        },
        "bottom_up": {
            "buyer_count": 40000, "acv_low": 20000, "acv_high": 80000,
            "derived_range": "$0.8B-$3.2B", "assumptions": ["all hospitals buy"],
        },
        "reconciliation": "bottom-up trusted more",
    }
    finding = parse_finding(
        json.dumps(finding_payload("market_size", payload=payload)), "market_size"
    )
    assert finding.payload["top_down"]["base"] == 1200
    assert finding.payload["bottom_up"]["assumptions"] == ["all hospitals buy"]
    assert "payload_invalid" not in finding.flags


def test_market_size_payload_invalid_kept_and_flagged() -> None:
    finding = parse_finding(
        json.dumps(finding_payload("market_size", payload={"top_down": "big"})),
        "market_size",
    )
    assert finding.flags == ["payload_invalid"]
    assert finding.payload == {"top_down": "big"}  # raw kept for the reader


def test_off_vocabulary_classification_flagged() -> None:
    finding = parse_finding(
        json.dumps(
            finding_payload(
                "wrapper_vs_proprietary",
                payload={"classification": "mostly a wrapper", "model_dependency": "",
                         "evidence_trail": []},
            )
        ),
        "wrapper_vs_proprietary",
    )
    assert finding.payload["classification"] == "mostly_a_wrapper"
    assert "unrecognized_classification" in finding.flags


def test_prompt_registries_cover_all_dimensions() -> None:
    """The dimension key set is declared in several structures — a missing
    rubric would crash a whole analysis, so guard them all here."""
    from scout.diligence.rubrics import RUBRICS
    from scout.diligence.schema import DIMENSION_KEYS, DIMENSION_LABELS, PAYLOAD_MODELS

    assert set(RUBRICS) == set(DIMENSION_KEYS)
    assert set(DIMENSION_LABELS) == set(DIMENSION_KEYS)
    assert set(PAYLOAD_MODELS) <= set(DIMENSION_KEYS)


def test_wrapper_classification_normalized() -> None:
    finding = parse_finding(
        json.dumps(
            finding_payload(
                "wrapper_vs_proprietary",
                payload={"classification": "Thin Wrapper", "model_dependency": "gpt",
                         "evidence_trail": []},
            )
        ),
        "wrapper_vs_proprietary",
    )
    assert finding.payload["classification"] == "thin_wrapper"


def test_competition_payload_validated() -> None:
    finding = parse_finding(
        json.dumps(
            finding_payload(
                "competition",
                payload={"competitors": [{"name": "RivalCo", "url": "https://rival.co"}],
                         "positioning_axis": "vertical", "whitespace": "on-prem"},
            )
        ),
        "competition",
    )
    assert finding.payload["competitors"][0]["name"] == "RivalCo"
    assert finding.payload["competitors"][0]["stage"] == ""  # defaulted


# --- recon / cross-exam / gaps ------------------------------------------------


def test_parse_recon() -> None:
    recon = parse_recon(
        json.dumps(
            {
                "company_name": "Tuva AI",
                "website": "https://tuva.ai",
                "product_summary": "health data models",
                "founders": [{"name": "A", "role": "CEO", "background_hints": "ex-Google"}],
                "funding_events": ["seed $4M 2025"],
                "github_org": "tuva-health",
                "key_sources": [{"url": "https://tuva.ai/docs", "title": "Docs", "what": "product"}],
            }
        )
    )
    assert recon.company_name == "Tuva AI"
    assert recon.founders[0].role == "CEO"
    assert recon.key_sources[0].url == "https://tuva.ai/docs"


def test_parse_recon_malformed_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_recon("nope")


def test_parse_cross_exam_clamps_and_drops_unknown_keys() -> None:
    exam = parse_cross_exam(
        json.dumps(
            {
                "downgrades": {"market_size": 1.4, "not_a_dimension": 0.1,
                               "data_flywheel": -0.2},
                "contradictions": ["TAM math off"],
            }
        )
    )
    assert exam.downgrades == {"market_size": 1.0, "data_flywheel": 0.0}
    assert exam.contradictions == ["TAM math off"]


def test_apply_cross_exam_only_downgrades() -> None:
    findings = [
        DimensionFinding(key="market_size", confidence=0.9),
        DimensionFinding(key="data_flywheel", confidence=0.3),
    ]
    apply_cross_exam(
        findings,
        CrossExam(downgrades={"market_size": 0.4, "data_flywheel": 0.8}),
    )
    assert findings[0].confidence == 0.4  # lowered
    assert "cross_exam_downgrade" in findings[0].flags
    assert findings[1].confidence == 0.3  # never raised
    assert findings[1].flags == []


def test_gap_finding_shape() -> None:
    gap = gap_finding("market_size", "timed out")
    assert gap.score is None
    assert gap.confidence == 0.0
    assert gap.flags == ["analysis_failed"]
    assert gap.unknowns == ["timed out"]
    skipped = gap_finding("competition", "cap reached", flag="skipped_budget_cap")
    assert skipped.flags == ["skipped_budget_cap"]
