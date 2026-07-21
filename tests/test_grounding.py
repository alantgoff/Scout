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
            "company_name": "Other Co",   # whitelisted (demote non-companies)
            "handle": "HACKED",           # not whitelisted — ignored
            "why_interesting": "HACKED",  # not whitelisted — ignored
        },
        note="website contradicts the hardware claim",
    )
    fixed = apply_verification(verdict, result)
    assert fixed.business_model == "b2b saas"
    assert fixed.thesis_fit == 0.2
    assert fixed.grounding == "website"
    assert fixed.company_name == "Other Co"    # whitelisted, applied
    assert fixed.handle == "benhylak"          # whitelist held
    assert fixed.why_interesting == ""         # whitelist held (default)
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


# --- v7: quality rubric + customer_type ---------------------------------------


def test_parse_verdicts_with_quality_fields() -> None:
    payload = [{
        "handle": "acme", "account_type": "founder", "is_founder": True,
        "stage": "launched", "grounding": "website", "customer_type": "B2B",
        "quality": {"team": 0.7, "traction": 0.5, "moat_invented": 0.9},
        "quality_reasons": {"team": "prev sold DevCo to Datadog",
                            "traction": "website: 12 customer logos"},
        "sector": "devtools", "thesis_fit": 0.4, "confidence": 0.8,
    }]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.customer_type == "b2b"  # normalized from "B2B"
    assert verdict.quality["traction"] == 0.5
    assert "logos" in verdict.quality_reasons["traction"]


def test_customer_type_normalization() -> None:
    for raw, expected in [("B2C", "b2c"), ("consumer", "b2c"),
                          ("Enterprise", "b2b"), ("b2b/b2c", "mixed"),
                          ("weird-vertical", None), (None, None), (7, None)]:
        verdict = LLMVerdict(handle="x", customer_type=raw)
        assert verdict.customer_type == expected, raw


def test_legacy_v6_payload_has_empty_quality() -> None:
    payload = [{"handle": "old", "account_type": "founder", "is_founder": True,
                "grounding": "website", "confidence": 0.8}]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.quality == {}
    assert verdict.quality_reasons == {}
    assert verdict.customer_type is None


def test_prompt_carries_both_scorecards_and_routing() -> None:
    prompt = _system_prompt(Thesis(thesis="edge ai"))
    assert "SCORECARD" in prompt
    assert "customer_type" in prompt
    assert "ENTERPRISE READINESS SCORECARD" in prompt
    assert "CONSUMER READINESS SCORECARD" in prompt
    # Routing: b2c → consumer; everything else (incl. unknown) → enterprise.
    assert '"b2c" -> the CONSUMER scorecard' in prompt
    assert "OMIT the key" in prompt
    # Every criterion of both rubrics is rendered from the registry.
    from scout import rubric
    for rb in rubric.RUBRICS.values():
        for criterion in rb.criteria:
            assert f"- {criterion.key}:" in prompt, criterion.key


def test_custom_prompt_still_gets_scorecard_evidence_rules() -> None:
    thesis = Thesis(llm_prompt="Custom: {thesis}", thesis="agents")
    prompt = _system_prompt(thesis)
    assert "SCORECARD: score a criterion only" in prompt  # EVIDENCE_RULES survive


def test_custom_prompt_scorecard_placeholder_substitutes() -> None:
    thesis = Thesis(llm_prompt="Custom: {thesis}\n{scorecard}", thesis="agents")
    prompt = _system_prompt(thesis)
    assert "ENTERPRISE READINESS SCORECARD" in prompt
    assert "{scorecard}" not in prompt


def test_prompt_defines_stages_and_oss_is_not_a_company() -> None:
    prompt = _system_prompt(Thesis(thesis="ai infra"))
    # Stage definitions exist (not just the bare enum) so launched isn't
    # under-called to stealth.
    assert "a product real users can access RIGHT NOW" in prompt
    assert "choose launched" in prompt.lower()
    # OSS-repo-is-not-a-company guidance, in the decision list AND the
    # non-editable evidence rules (so custom prompts keep it).
    assert "open-source library, framework" in prompt
    assert "A GitHub repo by itself is build evidence" in prompt
    custom = _system_prompt(Thesis(llm_prompt="Custom: {thesis}", thesis="x"))
    assert "AN OPEN-SOURCE REPO IS NOT A COMPANY" in custom
    assert "STAGE from the PRODUCT" in custom


def test_audit_can_demote_oss_repo_to_non_company() -> None:
    from scout.signals.llm import VerificationResult, apply_verification

    # A repo the classifier over-called as a launched startup.
    verdict = make_verdict(account_type="startup", is_founder=True,
                           stage="launched", company_name="cool-lib",
                           company_url="https://github.com/x/cool-lib")
    result = VerificationResult(
        handle="benhylak", verification="corrected",
        corrections={"account_type": "other", "is_founder": False,
                     "company_name": None, "company_url": None,
                     "stage": "idea"},
        note="just an OSS library, no company",
    )
    fixed = apply_verification(verdict, result)
    assert fixed.account_type == "other"
    assert fixed.is_founder is False
    assert fixed.company_name is None
    assert fixed.company_url is None


def test_dossier_includes_watchlist_follow_lines() -> None:
    account = make_account()
    account.followed_by = ["vc_ann", "vc_bob"]
    account.recent_followed_by = ["vc_ann"]
    text = _format_account(account, make_tweets(), None)
    assert "followed by watchlist investors: vc_ann, vc_bob" in text
    assert "newly followed this window by: vc_ann" in text


def test_apply_verification_replaces_quality_wholesale() -> None:
    from scout.signals.llm import VerificationResult, apply_verification

    verdict = make_verdict()
    verdict = verdict.model_copy(update={
        "customer_type": "b2b",
        "quality": {"team": 0.9, "traction": 0.8, "market": 0.7},
        "quality_reasons": {"team": "x", "traction": "y", "market": "z"},
    })
    result = VerificationResult(
        handle="benhylak", verification="corrected",
        corrections={
            "quality": {"team": 0.9},  # traction/market had no evidence
            "quality_reasons": {"team": "prev @apple design lead (concrete)"},
            "customer_type": "b2c",
        },
        note="traction and market scores were uncited",
    )
    fixed = apply_verification(verdict, result)
    assert fixed.quality == {"team": 0.9}  # wholesale replace
    assert "traction" not in fixed.quality_reasons
    assert fixed.customer_type == "b2c"


def test_verify_user_claim_includes_quality() -> None:
    from scout.models import Lead
    from scout.signals.llm import _verify_user

    lead = Lead(account=make_account(),
                llm=make_verdict().model_copy(update={
                    "customer_type": "b2b", "quality": {"team": 0.5}}))
    claim_text = _verify_user(lead, make_tweets(), None, 2000)
    assert '"quality"' in claim_text
    assert '"customer_type"' in claim_text


# --- v8: readiness scorecard ---------------------------------------------------


def test_parse_verdicts_with_scorecard_fields() -> None:
    payload = [{
        "handle": "acme", "account_type": "founder", "is_founder": True,
        "stage": "launched", "grounding": "website", "customer_type": "b2b",
        "scorecard": {"prev_founder_experience": 3, "commercial_traction": 2},
        "scorecard_reasons": {"prev_founder_experience": "sold DevCo to Datadog",
                              "commercial_traction": "website: 3 named pilots"},
        "sector": "devtools", "thesis_fit": 0.4, "confidence": 0.8,
    }]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.scorecard["commercial_traction"] == 2
    assert "pilots" in verdict.scorecard_reasons["commercial_traction"]
    assert verdict.quality == {}  # legacy fields stay empty on new verdicts


def test_legacy_v7_payload_has_empty_scorecard() -> None:
    payload = [{"handle": "old", "is_founder": True, "confidence": 0.8,
                "quality": {"team": 0.7}}]
    verdict = _parse_verdicts(json.dumps(payload))[0]
    assert verdict.scorecard == {}
    assert verdict.scorecard_reasons == {}
    assert verdict.scorecard_manual == {}


def test_apply_verification_replaces_scorecard_wholesale() -> None:
    from scout.signals.llm import VerificationResult, apply_verification

    verdict = make_verdict().model_copy(update={
        "customer_type": "b2b",
        "scorecard": {"prev_founder_experience": 3.0, "commercial_traction": 3.0},
        "scorecard_reasons": {"prev_founder_experience": "x",
                              "commercial_traction": "y"},
    })
    result = VerificationResult(
        handle="benhylak", verification="corrected",
        corrections={
            "scorecard": {"prev_founder_experience": 3},
            "scorecard_reasons": {"prev_founder_experience": "prev exit (site)"},
        },
        note="traction score was uncited",
    )
    fixed = apply_verification(verdict, result)
    assert fixed.scorecard == {"prev_founder_experience": 3.0}  # wholesale
    assert "commercial_traction" not in fixed.scorecard_reasons


def test_apply_verification_cross_rubric_correction_rescores_cleanly() -> None:
    from scout.config import Thesis as ThesisModel
    from scout.score import scorecard_score
    from scout.signals.llm import VerificationResult, apply_verification

    verdict = make_verdict().model_copy(update={
        "customer_type": "b2b",
        "scorecard": {"commercial_traction": 3.0},
    })
    result = VerificationResult(
        handle="benhylak", verification="corrected",
        corrections={"customer_type": "b2c",
                     "scorecard": {"user_growth": 3},
                     "scorecard_reasons": {"user_growth": "app store #4"}},
        note="sells to consumers",
    )
    fixed = apply_verification(verdict, result)
    scorecard = scorecard_score(fixed, ThesisModel())
    assert scorecard is not None
    assert scorecard[0].rubric_key == "b2c"  # re-routed and scoreable


def test_verify_user_claim_includes_scorecard() -> None:
    from scout.models import Lead
    from scout.signals.llm import _verify_user

    lead = Lead(account=make_account(),
                llm=make_verdict().model_copy(update={
                    "customer_type": "b2b",
                    "scorecard": {"prev_founder_experience": 3.0}}))
    claim_text = _verify_user(lead, make_tweets(), None, 2000)
    assert '"scorecard"' in claim_text
    assert '"prev_founder_experience"' in claim_text
