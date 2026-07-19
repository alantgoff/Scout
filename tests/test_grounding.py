"""Grounded classification: evidence dossiers, fingerprint sensitivity,
verdict parsing with the v6 fields, and the verification-pass pure logic.

House style — pure functions on fixtures; no Anthropic client, no network.
"""

from __future__ import annotations

import json

from scout.config import Settings, Thesis
from scout.models import Account, LLMVerdict, SitePage, Tweet
from scout.signals.llm import (
    _fingerprint,
    _format_account,
    _parse_verdicts,
    _system_prompt,
)


def make_account(**overrides) -> Account:
    base = dict(
        id="1", handle="benhylak", name="ben hylak",
        bio="cto @raindrop_ai, prev @apple design, @spacex eng",
        website="http://raindrop.ai/careers", followers=51529,
    )
    base.update(overrides)
    return Account(**base)


def make_tweets(n: int = 3) -> list[Tweet]:
    return [Tweet(id=str(i), account_id="1", text=f"tweet number {i}")
            for i in range(1, n + 1)]


OK_SITE = SitePage(url="https://raindrop.ai/", final_url="https://raindrop.ai/",
                   status="ok",
                   text="Raindrop — AI Agent Monitoring & Observability. "
                        "Trace every run. Auto-fix silent failures.")
DEAD_SITE = SitePage(url="https://raindrop.ai/", status="error:timeout")


# --- dossier rendering ------------------------------------------------------


def test_format_account_includes_website_text() -> None:
    text = _format_account(make_account(), make_tweets(), OK_SITE)
    assert "website text (from raindrop.ai, extracted" in text
    assert "AI Agent Monitoring" in text
    # Site text renders BEFORE the tweets — strongest evidence first.
    assert text.index("AI Agent Monitoring") < text.index("tweet number 1")


def test_format_account_marks_unreachable_sites() -> None:
    text = _format_account(make_account(), make_tweets(), DEAD_SITE)
    assert "unreachable (error:timeout)" in text
    assert "UNVERIFIED" in text


def test_format_account_without_site_says_none_or_url_only() -> None:
    text = _format_account(make_account(website=None), make_tweets(), None)
    assert "(none listed)" in text
    # A listed-but-unfetched site renders the URL only, no text block.
    text2 = _format_account(make_account(), make_tweets(), None)
    assert "raindrop.ai/careers" in text2
    assert "website text" not in text2


def test_format_account_truncates_site_text() -> None:
    long_site = SitePage(url="https://a.io/", status="ok", text="x" * 10_000)
    text = _format_account(make_account(), make_tweets(), long_site,
                           site_text_chars=500)
    assert "x" * 500 in text
    assert "x" * 501 not in text


def test_format_account_thin_site_flagged_weak() -> None:
    thin = SitePage(url="https://a.io/", status="thin", text="Acme. Coming soon.")
    text = _format_account(make_account(), make_tweets(), thin)
    assert "thin — JS-heavy site, weak evidence" in text


# --- fingerprint sensitivity -------------------------------------------------


def test_fingerprint_changes_with_site_text_and_website() -> None:
    settings = Settings(anthropic_api_key="k", _env_file=None)
    thesis = Thesis(thesis="edge ai")
    account, tweets = make_account(), make_tweets()
    base = _fingerprint(account, tweets, thesis, settings, None)
    with_site = _fingerprint(account, tweets, thesis, settings, OK_SITE)
    changed_site = _fingerprint(
        account, tweets, thesis, settings,
        OK_SITE.model_copy(update={"text": "totally different copy"}),
    )
    assert base != with_site != changed_site
    # Stable when nothing changed.
    assert with_site == _fingerprint(account, tweets, thesis, settings, OK_SITE)
    # The website URL alone (even unfetched) is part of the identity.
    assert base != _fingerprint(make_account(website="https://other.io"),
                                tweets, thesis, settings, None)
    # A failed fetch fingerprints by status, not text.
    assert _fingerprint(account, tweets, thesis, settings, DEAD_SITE) != base


# --- verdict parsing (v6 fields + legacy payloads) ---------------------------


def test_parse_verdicts_with_grounding_fields() -> None:
    payload = [{
        "handle": "@benhylak", "account_type": "founder", "is_founder": True,
        "stage": "launched", "company_name": "Raindrop", "company_url": "https://raindrop.ai",
        "product_summary": "AI agent monitoring (website: 'Trace every run')",
        "grounding": "website", "sector": "developer tools",
        "subsector": "ai observability", "business_model": "b2b saas",
        "thesis_fit": 0.2, "fit_reason": "observability software, not edge infra",
        "tags": ["observability"], "one_line_summary": "Monitors AI agents.",
        "why_interesting": "strong team", "confidence": 0.9,
    }]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.grounding == "website"
    assert verdict.product_summary is not None and "monitoring" in verdict.product_summary
    assert verdict.verification is None  # not audited yet


def test_parse_verdicts_unknown_escape_nulls() -> None:
    payload = [{
        "handle": "quiet", "account_type": "founder", "is_founder": True,
        "stage": "stealth", "company_name": None, "company_url": None,
        "product_summary": None, "grounding": "none", "sector": None,
        "subsector": None, "business_model": None, "thesis_fit": 0.15,
        "fit_reason": "product unknown", "tags": [],
        "one_line_summary": "Stealth; founder ex-Apple; product not yet public.",
        "why_interesting": "watch for launch", "confidence": 0.25,
    }]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.sector is None
    assert verdict.grounding == "none"
    assert verdict.confidence == 0.25


def test_parse_verdicts_legacy_payload_still_validates() -> None:
    # A cached v5 verdict has none of the v6 fields.
    payload = [{"handle": "old", "account_type": "founder", "is_founder": True,
                "stage": "launched", "sector": "ai infra", "confidence": 0.8}]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.grounding is None
    assert verdict.product_summary is None
    assert verdict.verification is None


# --- prompt contract ----------------------------------------------------------


def test_default_prompt_carries_grounding_contract() -> None:
    prompt = _system_prompt(Thesis(thesis="edge ai"))
    assert "EVIDENCE RULES" in prompt
    assert "WEBSITE TEXT > pinned tweet" in prompt
    assert "past employers" in prompt
    assert "UNKNOWN ESCAPE" in prompt
    assert "product_summary" in prompt
    assert '"grounding"' in prompt


# --- verification pass (pure pieces) ------------------------------------------


def make_verdict(**overrides) -> LLMVerdict:
    base = dict(
        handle="benhylak", account_type="founder", is_founder=True,
        stage="launched", sector="ai infra", subsector="edge AI hardware",
        business_model="hardware", company_name="Raindrop AI",
        company_url="https://raindrop.ai", thesis_fit=0.7,
        grounding="bio", confidence=0.55,
    )
    base.update(overrides)
    return LLMVerdict(**base)


def test_parse_verification_all_outcomes() -> None:
    from scout.signals.llm import parse_verification

    confirmed = parse_verification(
        '{"handle": "a", "verification": "confirmed", "corrections": {}, "note": "checks out"}'
    )
    assert confirmed.verification == "confirmed"

    corrected = parse_verification(
        '{"handle": "benhylak", "verification": "corrected", '
        '"corrections": {"sector": "developer tools", "business_model": "b2b saas", '
        '"thesis_fit": 0.2}, "note": "website says observability, not hardware"}'
    )
    assert corrected.corrections["business_model"] == "b2b saas"

    unverifiable = parse_verification(
        '```json\n{"handle": "q", "verification": "unverifiable", '
        '"corrections": {"confidence": 0.2}, "note": "no product evidence"}\n```'
    )
    assert unverifiable.verification == "unverifiable"

    import pytest
    with pytest.raises(ValueError):
        parse_verification('{"handle": "x", "verification": "maybe"}')
    with pytest.raises(Exception):
        parse_verification("not json at all")


def test_apply_verification_whitelists_and_caps() -> None:
    from scout.signals.llm import VerificationResult, apply_verification

    verdict = make_verdict()
    result = VerificationResult(
        handle="benhylak", verification="corrected",
        corrections={
            "sector": "developer tools", "subsector": "ai observability",
            "business_model": "b2b saas", "thesis_fit": 0.2,
            "grounding": "website",
            "handle": "HACKED",           # not whitelisted — ignored
            "company_name": "Other Co",   # not whitelisted — ignored
        },
        note="website contradicts the hardware claim",
    )
    fixed = apply_verification(verdict, result)
    assert fixed.business_model == "b2b saas"
    assert fixed.thesis_fit == 0.2
    assert fixed.grounding == "website"
    assert fixed.handle == "benhylak"          # whitelist held
    assert fixed.company_name == "Raindrop AI"  # whitelist held
    assert fixed.verification == "corrected"
    assert "contradicts" in fixed.verification_note
    # Original verdict untouched (pure function).
    assert verdict.business_model == "hardware"
    assert verdict.verification is None


def test_apply_verification_unverifiable_caps_confidence() -> None:
    from scout.signals.llm import VerificationResult, apply_verification

    fixed = apply_verification(
        make_verdict(confidence=0.9),
        VerificationResult(handle="benhylak", verification="unverifiable",
                           corrections={}, note="cannot establish product"),
    )
    assert fixed.confidence <= 0.3
    assert fixed.verification == "unverifiable"
