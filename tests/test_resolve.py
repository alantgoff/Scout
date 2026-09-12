"""`scout resolve` — the signals discovery could not key to a company.

A funding headline on a publisher's site is recorded as an unlinked lead
because keying it to techcrunch.com would be a lie. These tests pin what
the resolver may spend and what it may write: the free move (the article's
own links) comes before the paid one; every headline is attempted ONCE
whatever the outcome; the budget gate is check-before-call; a headline about
a company already tracked lands on that row and captures the round it
cites; and people in the hiring thread cost nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from scout.agents import CompanyProfile, parse_location
from scout.cli import app
from scout.models import Account, Lead, LLMVerdict, UnlinkedLead
from scout.store import Store
from scout.web import candidate_company_links, pick_company_domain

runner = CliRunner()
SITE = "https://acme-robotics.com/"
ARTICLE = "https://techcrunch.com/2026/09/09/acme-raises"
ARTICLE_HTML = f"""<html><body>
<a href="https://techcrunch.com/tag/robotics">Robotics</a>
<a href="https://twitter.com/techcrunch">Follow us</a>
<a href="https://adnetwork.example/x">ad</a><a href="https://adnetwork.example/y">ad</a>
<a href="https://adnetwork.example/z">ad</a>
<p>Acme Robotics, whose <a href="{SITE}">website</a> went live today...</p>
</body></html>"""


def _unlinked(store: Store, headline: str, url: str = ARTICLE, source: str = "rss",
              ref: str | None = None, bio: str | None = None) -> None:
    store.upsert_unlinked_leads([UnlinkedLead(
        source=source, ref=ref or url, name=headline,
        bio=bio if bio is not None else f"{headline} — via TechCrunch",
        url=url, found_at=datetime.now(timezone.utc),
    )])


def _resolutions(store: Store) -> dict[str, str | None]:
    return {r["ref"]: r.get("resolution") for r in store.db["unlinked_leads"].rows}


def _stubs(monkeypatch, *, html: str | None, located=None,
           profile: CompanyProfile | None = None, research_cost: float = 0.05):
    """No network, no key spend: the page fetch, the site crawl, the locate
    call, the research call and the classifier are all replaced. `located`
    is what locate_company answers — or an exception class when the paid
    step must not run at all."""
    from scout import agents, cli, web

    calls: dict[str, int] = {"locate": 0, "research": 0}

    async def no_crawl(*_a, **_kw):
        return []

    def fake_research(domain, settings, *, store=None, **_kw):
        calls["research"] += 1
        if store is not None:
            store.record_llm_usage("research", "m", cost_usd=research_cost)
        return (profile or CompanyProfile(company_name="Acme Robotics", website=SITE,
                                          one_line_summary="agentic warehouse robots"),
                {"researched": True, "searches": 1, "fetches": 1, "sources": []})

    def fake_locate(headline, context, settings, *, store=None, **_kw):
        calls["locate"] += 1
        if isinstance(located, type) and issubclass(located, BaseException):
            raise located("the paid step must not run here")
        if store is not None:
            store.record_llm_usage("resolve", "m", cost_usd=0.01)
        return located

    monkeypatch.setattr(web, "fetch_page_html", lambda *_a, **_kw: html)
    monkeypatch.setattr(web, "fetch_site_bundle", no_crawl)
    monkeypatch.setattr(agents, "research_company", fake_research)
    monkeypatch.setattr(agents, "locate_company", fake_locate)
    monkeypatch.setattr(cli, "classify", lambda *a, **kw: {})
    return calls


def _resolve(tmp_path: Path, *args: str):
    return runner.invoke(
        app, ["resolve", *args, "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(tmp_path / "t.db"), "ANTHROPIC_API_KEY": "k"},
    )


# --- the pure helpers ---------------------------------------------------------


def test_article_links_shortlist_companies_and_the_name_picks_one() -> None:
    candidates = candidate_company_links(ARTICLE_HTML, ARTICLE)
    # Own host, socials dropped; most-linked first — the ad network leads.
    assert candidates == ["adnetwork.example", "acme-robotics.com"]
    assert pick_company_domain(candidates, "Acme Robotics raises $2M pre-seed") == "acme-robotics.com"
    assert pick_company_domain(candidates, "Show HN: Acme – agentic robots") == "acme-robotics.com"
    # No name match → None, never "the most linked one".
    assert pick_company_domain(candidates, "Zeta Systems lands seed round") is None
    assert pick_company_domain(candidates, "Raises $2M") is None
    assert pick_company_domain([], "Acme") is None


def test_parse_location_runs_every_field_through_the_source_gates() -> None:
    assert parse_location('{"website": "https://acme.io/", "x_handle": "@acme_io", "note": ""}') \
        == ("acme.io", "acme_io", "")
    # A publisher or code host is never a company key, whatever the model says.
    assert parse_location('{"website": "https://techcrunch.com/x", "x_handle": null}')[0] is None
    assert parse_location('{"website": "https://github.com/acme"}')[0] is None
    # Junk handles are refused; fenced JSON is fine.
    assert parse_location('```json\n{"website": "acme.io", "x_handle": "https://x.com/"}\n```') \
        == ("acme.io", None, "")
    assert parse_location("not json") == (None, None, "unparseable answer")
    assert parse_location("[1, 2]") == (None, None, "unparseable answer")


# --- the command --------------------------------------------------------------


def test_article_link_bridges_for_free_and_is_never_retried(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "t.db")
    _unlinked(store, "Acme Robotics raises $2M pre-seed")
    calls = _stubs(monkeypatch, html=ARTICLE_HTML, located=AssertionError)

    result = _resolve(tmp_path)
    assert result.exit_code == 0, result.output
    assert calls == {"locate": 0, "research": 1}  # the free move was enough

    store = Store(tmp_path / "t.db")
    lead = store.latest_lead("acme-robotics")
    assert lead is not None
    assert lead.account.source == "rss"  # the feed surfaced it, not a person
    assert lead.account.profile_url == SITE  # never a fabricated x.com/<slug>
    assert _resolutions(store) == {ARTICLE: "bridged:@acme-robotics"}
    assert store.unresolved_leads() == []
    events = store.events(since=datetime(2000, 1, 1, tzinfo=timezone.utc), limit=20)
    assert any(e.verb == "lead_resolved" and e.handle == "acme-robotics" for e in events)

    again = _resolve(tmp_path)
    assert "Nothing to resolve" in again.output


def test_paid_lookup_runs_only_when_the_article_gives_nothing(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "t.db")
    _unlinked(store, "Beta Robots raises $3M seed")
    calls = _stubs(monkeypatch, html=None, located=("beta.io", "betarobots", {"note": ""}),
                   profile=CompanyProfile(company_name="Beta Robots", website="https://beta.io/",
                                          one_line_summary="robots"))

    result = _resolve(tmp_path)
    assert result.exit_code == 0, result.output
    assert calls == {"locate": 1, "research": 1}
    store = Store(tmp_path / "t.db")
    lead = store.latest_lead("betarobots")
    assert lead is not None
    assert lead.account.url == "https://x.com/betarobots"  # a real handle, so x.com is right
    assert _resolutions(store) == {ARTICLE: "bridged:@betarobots"}


def test_no_company_is_stamped_and_not_paid_for_again(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "t.db")
    _unlinked(store, "Big Fund closes $500M for climate")
    calls = _stubs(monkeypatch, html=None, located=(None, None, {"note": "a fund, not a startup"}))

    result = _resolve(tmp_path)
    assert result.exit_code == 0, result.output
    assert calls == {"locate": 1, "research": 0}
    assert "a fund, not a startup" in result.output
    store = Store(tmp_path / "t.db")
    assert _resolutions(store) == {ARTICLE: "no_company"}
    assert store.load_lead_ledger() == []
    assert store.unresolved_leads() == []
    assert calls == {"locate": 1, "research": 0}
    _resolve(tmp_path)
    assert calls == {"locate": 1, "research": 0}  # stamped, so never asked twice


def test_budget_stop_is_check_before_call(tmp_path: Path, monkeypatch) -> None:
    """The gate looks before each lead, not after: with $0.02 left the first
    lead is worked (and overspends by one call — the documented trade), and
    the second waits for tomorrow, unstamped, at the front of the queue."""
    store = Store(tmp_path / "t.db")
    _unlinked(store, "Acme Robotics raises $2M pre-seed", url=ARTICLE)
    _unlinked(store, "Acme Robotics ships v2", url="https://techcrunch.com/2026/09/08/acme-v2")
    store.record_llm_usage("classify", "m", cost_usd=0.98)  # 0.02 left of 1.00
    calls = _stubs(monkeypatch, html=ARTICLE_HTML, located=AssertionError)

    result = _resolve(tmp_path, "--limit", "2")
    assert result.exit_code == 0, result.output
    assert calls["research"] == 1
    assert "cap reached" in result.output
    store = Store(tmp_path / "t.db")
    resolutions = _resolutions(store)
    assert sum(1 for v in resolutions.values() if v) == 1
    assert len(store.unresolved_leads()) == 1


def test_headline_about_a_tracked_company_lands_on_its_row_and_captures_the_round(
    tmp_path: Path, monkeypatch
) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE,
                                 bio="agentic robots", source="search", sources=["search"]))
    store.save_leads("run-1", [Lead(
        account=store.get_account("acmerobots"),
        llm=LLMVerdict(handle="acmerobots", account_type="startup", company_name="Acme",
                       company_url=SITE, grounding="website"),
    )])
    _unlinked(store, "Acme Robotics raises $2M seed")
    _stubs(monkeypatch, html=ARTICLE_HTML, located=AssertionError,
           profile=CompanyProfile(company_name="Acme Robotics", website=SITE,
                                  funding_stage="seed", funding_amount="$2M",
                                  funding_evidence="TechCrunch, 2026-09-09"))

    result = _resolve(tmp_path)
    assert result.exit_code == 0, result.output
    assert "already tracked as @acmerobots" in result.output
    store = Store(tmp_path / "t.db")
    assert store.latest_lead("acme-robotics") is None  # no second Acme
    lead = store.latest_lead("acmerobots")
    assert lead.llm.funding_stage == "seed"
    assert lead.llm.funding_evidence == "TechCrunch, 2026-09-09"
    outcomes = store.auto_outcomes()
    assert [(o["handle"], o["round_stage"]) for o in outcomes] == [("acmerobots", "seed")]
    assert "raised" in result.output
    assert _resolutions(store) == {ARTICLE: "bridged:@acmerobots"}


def test_people_and_github_logins_are_never_paid_for(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "t.db")
    # A hiring-thread comment: HN item URL, not a Show HN.
    _unlinked(store, "jobseeker", url="https://news.ycombinator.com/item?id=1", source="hn",
              ref="jobseeker", bio="Ex-DeepMind, looking for an agentic robotics role")
    # A GitHub owner with no site is a person too — not even listed.
    _unlinked(store, "octodev", url="https://github.com/octodev/repo", source="github",
              ref="octodev", bio="")
    calls = _stubs(monkeypatch, html=None, located=AssertionError)

    result = _resolve(tmp_path)
    assert result.exit_code == 0, result.output
    assert calls == {"locate": 0, "research": 0}
    store = Store(tmp_path / "t.db")
    assert _resolutions(store)["jobseeker"] == "skipped:person"
    assert _resolutions(store)["octodev"] is None  # out of scope, untouched
    assert store.unresolved_leads() == []
    assert store.load_lead_ledger() == []
