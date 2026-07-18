"""Company grouping tests — the accounts→startups fold (scout/companies.py)."""

from __future__ import annotations

from scout.companies import company_key, group_by_company
from scout.models import Account, Lead, LLMVerdict


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
