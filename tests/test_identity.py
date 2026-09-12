"""Identity across sources — one company, one handle, whichever source saw it.

`handle` is the key in every table, and one company reaches the database
under different keys: its X handle from search, a slug invented from its
domain by RSS / HN / `scout add`, a GitHub org. Left alone that is two
leads, split votes, and a source_corroboration signal that never fires for
exactly the multi-source hits it rewards. These tests pin the three rules
that keep it one row: a domain can find its handle, a rename moves EVERY
table (and the JSON inside the lead), and the pipeline reconciles before
anything is scored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from scout.cli import _reconcile_identities, app
from scout.ingest.github_src import profile_to_site
from scout.models import Account, Lead, LLMVerdict
from scout.store import Store
from scout.web import company_domain, registrable_domain

runner = CliRunner()
SITE = "https://acme-robotics.com/"
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _lead(handle: str, *, website: str | None = SITE, company_url: str | None = None,
          profile_url: str | None = None) -> Lead:
    account = Account(id=f"t:{handle}", handle=handle, website=website,
                      profile_url=profile_url, bio="agentic robots")
    return Lead(account=account, llm=LLMVerdict(
        handle=handle, account_type="startup", company_name="Acme",
        company_url=company_url or website, grounding="website"))


# --- the shared gate ----------------------------------------------------------


def test_registrable_domain_and_the_shared_company_gate() -> None:
    assert registrable_domain("https://www.acme.io/launch") == "acme.io"
    assert registrable_domain("acme.io") == "acme.io"
    assert registrable_domain("https://github.com/acme") is None  # code host, never a company key
    assert registrable_domain(None) is None
    assert company_domain("https://acme.io/", "feed.example") == "acme.io"
    assert company_domain("https://techcrunch.com/2026/acme", "feed.example") is None
    assert company_domain("https://feed.example/post", "feed.example") is None
    assert company_domain("https://www.producthunt.com/posts/acme", "") is None


def test_github_owner_site_bridges_only_real_company_sites() -> None:
    assert profile_to_site({"blog": "acme.io"}) == "https://acme.io/"
    # Root only (the /about page is not the company), host kept as written.
    assert profile_to_site({"blog": "https://www.acme.io/about"}) == "https://www.acme.io/"
    assert profile_to_site({"blog": "https://twitter.com/acme"}) is None
    assert profile_to_site({"blog": "https://medium.com/@acme"}) is None
    assert profile_to_site({"blog": ""}) is None
    assert profile_to_site({}) is None


# --- a domain finds its handle ------------------------------------------------


def test_handle_for_domain_reads_account_and_verdict_claims(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="x:AcmeRobots", handle="AcmeRobots",
                                 website="https://www.acme-robotics.com/"))
    assert store.handle_for_domain("acme-robotics.com") == "AcmeRobots"
    assert store.handle_for_domain("https://acme-robotics.com/about") == "AcmeRobots"
    # The account lists no site; the classifier established the company_url.
    store.save_leads("run-1", [_lead("betafounder", website=None,
                                     company_url="https://beta.io/")])
    assert store.handle_for_domain("beta.io") == "betafounder"
    assert store.handle_for_domain("nobody.io") is None
    assert store.handle_for_domain("https://github.com/acme") is None
    assert store.handle_for_domain(None) is None


def test_handle_for_domain_prefers_a_real_handle_over_an_invented_slug(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="rss:acme-robotics", handle="acme-robotics",
                                 website=SITE, profile_url=SITE))
    assert store.handle_for_domain(SITE) == "acme-robotics"  # the only claim
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE))
    assert store.handle_for_domain(SITE) == "acmerobots"


# --- rename moves everything --------------------------------------------------


def test_rename_handle_moves_every_handle_keyed_row_and_rewrites_the_lead(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    store.actor = "partner:alan"
    slug = "acme-robotics"
    store.upsert_account(Account(id=f"rss:{slug}", handle=slug, website=SITE,
                                 profile_url=SITE, sources=["rss"]))
    store.save_leads("run-1", [_lead(slug, profile_url=SITE)])
    store.set_pipeline(slug, status="shortlisted", notes="met at demo day")
    store.set_vote(slug, "strong_yes", actor="partner:alan")
    store.add_comment(slug, "founders are ex-DeepMind", actor="partner:alan")
    store.record_bio(slug, "old bio")
    store.record_outcome(slug, company="Acme", round_stage="seed", evidence="press")
    store.record_query_hits("agentic robots", "intent", [slug])
    store.mark_refreshed(slug)

    moved = store.rename_handle(slug, "AcmeRobots")
    for table in ("leads", "accounts", "pipeline", "votes", "comments",
                  "bio_snapshots", "outcomes", "query_hits"):
        assert moved.get(table), f"{table} did not move: {moved}"

    assert store.latest_lead(slug) is None
    lead = store.latest_lead("acmerobots")
    assert lead is not None
    assert lead.account.handle == "AcmeRobots"  # the JSON inside agrees with the key
    assert lead.llm is not None and lead.llm.handle == "AcmeRobots"
    assert store.get_account(slug) is None
    assert store.get_account("acmerobots").handle == "AcmeRobots"
    row = store.get_pipeline("acmerobots")
    assert row["status"] == "shortlisted"
    assert row["notes"] == "met at demo day"
    assert row.get("researched_at")  # the refresh stamp followed
    assert not store.get_pipeline(slug)
    assert [v.actor for v in store.votes_for("acmerobots")] == ["partner:alan"]
    assert [c.body for c in store.comments_for("acmerobots")] == ["founders are ex-DeepMind"]
    assert store.record_bio("acmerobots", "new bio") == "old bio"  # snapshot history followed
    assert store.auto_outcomes()[0]["handle"] == "acmerobots"
    assert any(e.verb == "handle_merged" and e.handle == "acmerobots"
               for e in store.events(since=EPOCH, limit=50))


def test_rename_handle_keeps_the_targets_row_when_both_hold_one(tmp_path: Path) -> None:
    """Same company, seen twice: the target was tracked later with more
    evidence, so where both rows hold one unique thing the target's stands.
    Rows only the duplicate had still move."""
    store = Store(tmp_path / "t.db")
    store.actor = "partner:alan"
    store.set_pipeline("acme-robotics", status="longlisted")
    store.set_pipeline("acmerobots", status="shortlisted")
    store.set_vote("acme-robotics", "pass", actor="partner:alan")
    store.set_vote("acmerobots", "strong_yes", actor="partner:alan")
    store.set_vote("acme-robotics", "yes", actor="partner:bo")

    store.rename_handle("acme-robotics", "acmerobots")

    assert store.get_pipeline("acmerobots")["status"] == "shortlisted"
    assert not store.get_pipeline("acme-robotics")
    votes = {v.actor: v.stance for v in store.votes_for("acmerobots")}
    assert votes == {"partner:alan": "strong_yes", "partner:bo": "yes"}
    assert store.votes_for("acme-robotics") == []


def test_rename_handle_is_a_no_op_for_same_or_unknown_handles(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    assert store.rename_handle("acme", "@Acme") == {}
    assert store.rename_handle("ghost", "acme") == {}


# --- the pipeline reconciles before scoring -----------------------------------


def test_reconcile_lands_a_domain_keyed_newcomer_on_the_tracked_handle(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE,
                                 source="search", sources=["search"]))
    store.save_leads("run-1", [_lead("acmerobots")])
    newcomer = Account(id="rss:acme-robotics", handle="acme-robotics", website=SITE,
                       profile_url=SITE, source="rss", sources=["rss"])
    store.upsert_account(newcomer)  # sources persist their finds before the pipeline merges

    merged = _reconcile_identities([newcomer], store)

    assert [a.handle for a in merged] == ["acmerobots"]
    assert merged[0].profile_url is None  # the real handle's page is x.com/acmerobots
    assert merged[0].id == "x:acmerobots"
    # Both sightings on one account — the source_corroboration input.
    assert set(merged[0].sources) == {"rss", "search"}
    assert store.get_account("acme-robotics") is None  # the stray slug row was folded in
    assert "rss" in store.get_account("acmerobots").sources


def test_reconcile_renames_a_slug_row_when_the_real_handle_arrives(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="rss:acme-robotics", handle="acme-robotics", website=SITE,
                                 profile_url=SITE, source="rss", sources=["rss"]))
    store.save_leads("run-1", [_lead("acme-robotics", profile_url=SITE)])
    store.set_pipeline("acme-robotics", status="longlisted")
    newcomer = Account(id="x:acmerobots", handle="acmerobots", website=SITE,
                       source="search", sources=["search"])

    merged = _reconcile_identities([newcomer], store)

    assert [a.handle for a in merged] == ["acmerobots"]
    assert store.latest_lead("acme-robotics") is None
    assert store.latest_lead("acmerobots").account.handle == "acmerobots"
    assert store.get_pipeline("acmerobots")["status"] == "longlisted"
    assert set(merged[0].sources) == {"rss", "search"}


def test_reconcile_leaves_two_real_accounts_that_share_a_site_alone(tmp_path: Path) -> None:
    """A founder's personal account and the company account both list the
    company's domain. They are not the same row."""
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE))
    founder = Account(id="x:jane", handle="janebuilds", website=SITE, bio="CEO @acmerobots")
    merged = _reconcile_identities([founder], store)
    assert [a.handle for a in merged] == ["janebuilds"]
    assert store.get_account("acmerobots") is not None
    assert store.get_account("janebuilds") is None  # nothing was written for her either


def test_reconcile_merges_two_sightings_of_a_brand_new_company_in_one_run(
    tmp_path: Path,
) -> None:
    """Nothing in the store yet: RSS keyed the launch by its domain, X search
    found the handle the same morning. One lead, both sources on it, keyed
    by the handle — and the X account's own bio, not the feed headline."""
    store = Store(tmp_path / "t.db")
    feed = Account(id="rss:acme-robotics", handle="acme-robotics", website=SITE,
                   profile_url=SITE, bio="Acme launches — via YC", source="rss", sources=["rss"])
    x = Account(id="x:acmerobots", handle="acmerobots", website=SITE,
                bio="Agentic robots for warehouses", source="search", sources=["search"])
    merged = _reconcile_identities([feed, x], store)
    assert [a.handle for a in merged] == ["acmerobots"]
    assert merged[0].bio == "Agentic robots for warehouses"
    assert merged[0].profile_url is None
    assert set(merged[0].sources) == {"rss", "search"}
    # Same pair, other order: the slug arrives second and adopts the handle.
    merged = _reconcile_identities(
        [x.model_copy(deep=True), feed.model_copy(deep=True)], store)
    assert [a.handle for a in merged] == ["acmerobots"]
    assert set(merged[0].sources) == {"rss", "search"}


# --- the commands -------------------------------------------------------------


def test_add_by_domain_updates_the_tracked_handle_instead_of_duplicating(
    tmp_path: Path,
) -> None:
    db = tmp_path / "scout.db"
    store = Store(db)
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE,
                                 bio="agentic robots"))
    store.save_leads("run-1", [_lead("acmerobots")])

    result = runner.invoke(
        app, ["add", "acme-robotics.com", "--no-classify", "--no-research",
              "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(db)},
    )
    assert result.exit_code == 0, result.output
    assert "already tracked as @acmerobots" in result.output
    store = Store(db)
    assert store.get_account("acme-robotics") is None
    assert store.get_pipeline("acmerobots")["status"] == "longlisted"


def test_scout_merge_folds_the_duplicate_and_reports_the_tables(tmp_path: Path) -> None:
    db = tmp_path / "scout.db"
    store = Store(db)
    store.upsert_account(Account(id="rss:acme-robotics", handle="acme-robotics",
                                 website=SITE, profile_url=SITE))
    store.save_leads("run-1", [_lead("acme-robotics", profile_url=SITE)])
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", website=SITE))

    result = runner.invoke(app, ["merge", "acme-robotics", "@acmerobots"],
                           env={"DB_PATH": str(db)})
    assert result.exit_code == 0, result.output
    assert "leads" in result.output and "Merged" in result.output
    store = Store(db)
    assert store.latest_lead("acme-robotics") is None
    assert store.latest_lead("acmerobots").account.id == "x:acmerobots"

    result = runner.invoke(app, ["merge", "ghost", "acmerobots"], env={"DB_PATH": str(db)})
    assert result.exit_code == 1
    result = runner.invoke(app, ["merge", "acmerobots", "acmerobots"], env={"DB_PATH": str(db)})
    assert result.exit_code == 1
