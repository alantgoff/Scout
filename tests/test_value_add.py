"""The firm value-add dimension: prompt rendering + verdict parsing.

Pure functions only — no network, no Anthropic client.
"""

from __future__ import annotations

import json

from scout.config import Thesis, ValueAddLever
from scout.signals.llm import _parse_verdicts, _system_prompt


def test_default_prompt_includes_firm_and_levers() -> None:
    thesis = Thesis(thesis="vertical agents")
    prompt = _system_prompt(thesis)
    assert "Headline" in prompt  # default firm
    # Every default lever key is offered to the classifier.
    for lever in thesis.firm_value_add:
        assert lever.key in prompt
        assert lever.label in prompt
    assert "value_add_fit" in prompt
    assert "value_add_levers" in prompt


def test_custom_levers_replace_defaults_in_prompt() -> None:
    thesis = Thesis(
        firm_name="TestFund",
        firm_value_add=[
            ValueAddLever(key="gpu_credits", label="GPU credits",
                          description="Discounted training compute.")
        ],
    )
    prompt = _system_prompt(thesis)
    assert "TestFund" in prompt
    assert "gpu_credits — GPU credits: Discounted training compute." in prompt
    assert "global_expansion" not in prompt


def test_custom_prompt_without_new_placeholders_still_renders() -> None:
    thesis = Thesis(llm_prompt="Screen for: {thesis} in {sectors} at {stages}.",
                    thesis="agents", sectors=["ai"], target_stages=["launched"])
    prompt = _system_prompt(thesis)
    assert prompt.startswith("Screen for: agents in ai at launched.")
    # The grounding rules are appended to EVERY prompt — even fully custom
    # ones. They are the anti-speculation contract and not editable away.
    assert "EVIDENCE RULES" in prompt
    assert "past employers" in prompt


def test_parse_verdict_with_value_add_fields() -> None:
    payload = [{
        "handle": "@ada_infra", "account_type": "founder", "is_founder": True,
        "stage": "launched", "company_name": "EvalHQ", "company_url": None,
        "sector": "ai infra", "subsector": "agent evals", "business_model": "b2b saas",
        "thesis_fit": 0.9, "fit_reason": "textbook",
        "value_add_fit": 0.8,
        "value_add_levers": {"global_expansion": 0.9, "follow_on_capital": 0.4},
        "value_add_reason": "expanding into the EU; usage-metric heavy",
        "tags": ["evals"], "one_line_summary": "evals", "why_interesting": "yes",
        "confidence": 0.9,
    }]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.handle == "ada_infra"
    assert verdict.value_add_fit == 0.8
    assert verdict.value_add_levers["global_expansion"] == 0.9
    assert verdict.value_add_reason.startswith("expanding")


def test_parse_verdict_without_value_add_fields_still_validates() -> None:
    payload = [{"handle": "x", "account_type": "other", "is_founder": False,
                "confidence": 0.5}]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.value_add_fit is None
    assert verdict.value_add_levers == {}
