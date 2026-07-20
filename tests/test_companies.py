"""Startup identity + grouping tests — the accounts→startups fold (scout/companies.py)."""

from __future__ import annotations

from scout.companies import company_key, group_by_company, startup_identity
from scout.models import Account, Lead, LLMVerdict, Signal


def make_lead(handle: str, company: str | None, score: float,
              account_type: str = "founder") -> Lead:
    return Lead(
        account=Account(id=handle, handle=handle),
        llm=LLMVerdict(handle=handle, account_type=account_type, is_founder=True,
                       stage="launched", company_name=company, confidence=0.9),
        score=score,
    )


def test_company_key_normalizes_variants() -> None:
    assert company_key(make_lead("a", "EvalHQ", 1)) == "evalhq"
    assert company_key(make_lead("b", "eval hq", 1)) == "evalhq"
    assert company_key(make_lead("c", "Eval-HQ.ai", 1)) == "evalhq"  # TLD stripped
    assert company_key(make_lead("d", None, 1)) is None
    assert company_key(make_lead("e", "X", 1)) is None  # single char = noise
    assert company_key(Lead(account=Account(id="f", handle="f"))) is None  # no verdict


def test_startup_identity_prefers_real_company_name() -> None:
    name, synthesized = startup_identity(make_lead("a", "EvalHQ", 40.0))
    assert name == "EvalHQ" and synthesized is False


def test_startup_identity_synthesizes_stealth_name_from_founder() -> None:
    lead = Lead(
        account=Account(id="s", handle="stealth_sam", name="Sam Okafor"),
        llm=LLMVerdict(handle="stealth_sam", account_type="founder",
                       is_founder=True, stage="stealth", confidence=0.8),
    )
    assert startup_identity(lead) == ("Sam Okafor's stealth startup", True)


def test_startup_identity_launched_without_name_is_unnamed() -> None:
    lead = make_lead("n", None, 30.0)  # stage=launched, no company_name
    lead.account.name = "Nora Vale"
    assert startup_identity(lead) == ("Nora Vale's unnamed startup", True)


def test_startup_identity_possessive_of_s_ending_name() -> None:
    lead = Lead(
        account=Account(id="j", handle="james_b", name="James"),
        llm=LLMVerdict(handle="james_b", account_type="founder",
                       is_founder=True, stage="idea", confidence=0.8),
    )
    assert startup_identity(lead) == ("James' stealth startup", True)


def test_startup_identity_none_for_non_founders() -> None:
    corp = make_lead("bigco", None, 10.0, account_type="other")
    assert startup_identity(corp) is None
    # Unclassified with no founder-evidence signals: stays an account.
    quiet = Lead(account=Account(id="q", handle="quiet_quinn", name="Quinn"))
    assert startup_identity(quiet) is None


def test_startup_identity_heuristics_only_uses_founder_signals() -> None:
    lead = Lead(
        account=Account(id="a", handle="ada_infra", name="Ada Lin"),
        signals=[Signal(name="departure_signal", value=1.0)],
    )
    assert startup_identity(lead) == ("Ada Lin's stealth startup", True)


def test_startup_identity_falls_back_to_handle_when_nameless() -> None:
    lead = Lead(
        account=Account(id="x", handle="builder_x"),
        signals=[Signal(name="bio_intent", value=1.0)],
    )
    assert startup_identity(lead) == ("@builder_x's stealth startup", True)


def test_group_by_company_folds_and_keeps_highest_score_primary() -> None:
    founder = make_lead("ada_infra", "EvalHQ", 40.0)
    company = make_lead("evalhq", "EvalHQ", 25.0, account_type="startup")
    other = make_lead("kite_ci", "KiteCI", 30.0)
    solo = make_lead("no_co", None, 20.0)

    grouped = group_by_company([(founder, None), (other, None), (company, None), (solo, None)])

    assert len(grouped) == 3  # founder+company folded
    primary, _entry, secondary = grouped[0]
    assert primary.account.handle == "ada_infra"  # higher score wins primary
    assert [x.account.handle for x in secondary] == ["evalhq"]
    # order preserved: EvalHQ group (first seen), then KiteCI, then the solo
    assert [g[0].account.handle for g in grouped] == ["ada_infra", "kite_ci", "no_co"]
    assert grouped[1][2] == [] and grouped[2][2] == []


def test_display_name_prefers_company_then_identity_then_person() -> None:
    from scout.companies import display_name

    named = Lead(account=Account(id="1", handle="ada", name="Ada Lin"),
                 llm=LLMVerdict(handle="ada", account_type="founder",
                                is_founder=True, company_name="EvalHQ"))
    assert display_name(named) == "EvalHQ"
    stealth = Lead(account=Account(id="2", handle="nora", name="Nora Vale"),
                   llm=LLMVerdict(handle="nora", account_type="founder",
                                  is_founder=True, stage="stealth"))
    assert display_name(stealth) == "Nora Vale's stealth startup"
    other = Lead(account=Account(id="3", handle="vcpundit", name="VC Pundit"),
                 llm=LLMVerdict(handle="vcpundit", account_type="other"))
    assert display_name(other) == "VC Pundit"
    bare = Lead(account=Account(id="4", handle="ghost"))
    assert display_name(bare) == "@ghost"
