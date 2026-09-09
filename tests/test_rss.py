"""RSS discovery source — the parsing rules, against real feed shapes.

What matters here is what an entry is allowed to BECOME. Bridging an entry
to an Account claims "this is a company"; getting that wrong writes a
publisher into the database as a startup. So the tests are mostly about
refusing to bridge: articles stay unlinked, publishers never become
companies, and off-thesis entries never reach the classifier at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import feedparser

from scout.config import Thesis
from scout.ingest.rss_src import company_domain, parse_entries

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
THESIS = Thesis(keywords=["agentic", "robotics"], sectors=["ai infra"])


def _rss(items: str, title: str = "Test Feed") -> list:
    return feedparser.parse(
        f"""<?xml version="1.0"?><rss version="2.0"><channel>
        <title>{title}</title><link>https://feed.example/</link>{items}
        </channel></rss>"""
    ).entries


def _item(title: str, link: str, desc: str = "", days_ago: int = 1) -> str:
    when = (NOW - timedelta(days=days_ago)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    return (f"<item><title>{title}</title><link>{link}</link>"
            f"<description>{desc}</description><pubDate>{when}</pubDate></item>")


def _parse(entries, feed_url="https://feed.example/rss", **kw):
    return parse_entries(entries, feed_url=feed_url, feed_title="Test Feed",
                         thesis=THESIS, now=NOW, **kw)


# --- company_domain: the bridging gate ----------------------------------------


def test_company_link_bridges_but_publisher_and_self_links_never_do() -> None:
    assert company_domain("https://acme-robotics.com/", "feed.example") == "acme-robotics.com"
    assert company_domain("https://www.acme.io/launch", "feed.example") == "acme.io"
    # An article ABOUT a company is not the company.
    assert company_domain("https://techcrunch.com/2026/09/acme-raises", "feed.example") is None
    assert company_domain("https://eu-startups.com/x", "feed.example") is None
    # …including on a publisher subdomain.
    assert company_domain("https://blog.medium.com/x", "feed.example") is None
    # A feed writing about itself is not a discovery.
    assert company_domain("https://feed.example/post/1", "feed.example") is None
    assert company_domain("not a url", "feed.example") is None
    assert company_domain("", "feed.example") is None


# --- what entries become ------------------------------------------------------


def test_company_feed_entry_becomes_a_domain_keyed_account() -> None:
    entries = _rss(_item("Acme Robotics launches agentic warehouse robots",
                         "https://acme-robotics.com/"))
    accounts, unlinked = _parse(entries)
    assert not unlinked
    assert len(accounts) == 1
    account = accounts[0]
    assert account.handle == "acme-robotics"
    assert account.source == "rss"
    # The identity rule from `scout add <domain>`: no invented x.com link.
    assert account.profile_url == "https://acme-robotics.com/"
    assert account.url == "https://acme-robotics.com/"
    assert "Acme Robotics launches" in account.bio
    assert "Test Feed" in account.bio  # provenance travels with the lead


def test_news_entry_stays_unlinked_rather_than_keying_the_publisher() -> None:
    entries = _rss(_item("Acme raises $14M for agentic robotics",
                         "https://techcrunch.com/2026/09/09/acme-raises"))
    accounts, unlinked = _parse(entries)
    assert not accounts
    assert len(unlinked) == 1
    assert unlinked[0].source == "rss"
    # ref is the article URL: the (source, ref) pk makes re-reads idempotent.
    assert unlinked[0].ref == "https://techcrunch.com/2026/09/09/acme-raises"


def test_an_x_handle_in_the_entry_bridges_even_from_a_news_feed() -> None:
    entries = _rss(_item("Acme raises $14M for agentic robotics",
                         "https://techcrunch.com/2026/09/09/acme",
                         desc="Follow them at https://x.com/acmerobots for updates."))
    accounts, unlinked = _parse(entries)
    assert not unlinked
    assert [a.handle for a in accounts] == ["acmerobots"]


def test_off_thesis_entries_are_dropped_before_they_cost_anything() -> None:
    """A funding feed carries every sector. Letting them all through would
    spend the classification budget on companies the thesis excludes."""
    entries = _rss(
        _item("Acme launches agentic robotics platform", "https://acme.com/")
        + _item("Dog grooming startup raises seed", "https://groomer.com/")
    )
    accounts, unlinked = _parse(entries)
    assert [a.handle for a in accounts] == ["acme"]
    assert not unlinked


def test_no_thesis_terms_configured_lets_everything_through() -> None:
    entries = _rss(_item("Anything at all", "https://whatever.com/"))
    accounts, _ = parse_entries(entries, feed_url="https://feed.example/rss",
                                feed_title="F", thesis=Thesis(), now=NOW)
    assert len(accounts) == 1


def test_stale_entries_fall_outside_the_freshness_window() -> None:
    entries = _rss(_item("Acme launches agentic robotics", "https://acme.com/",
                         days_ago=90))
    accounts, unlinked = _parse(entries)
    assert not accounts and not unlinked


def test_undated_entries_are_kept_not_silently_discarded() -> None:
    """Plenty of small company blogs emit no usable date; dropping them would
    make the source useless on exactly the feeds worth reading."""
    entries = _rss("<item><title>Acme ships agentic robotics</title>"
                   "<link>https://acme.com/</link></item>")
    accounts, _ = _parse(entries)
    assert len(accounts) == 1


# --- feed formats and malformed input -----------------------------------------


def test_atom_feeds_parse_the_same_way() -> None:
    atom = feedparser.parse("""<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><title>Atom Feed</title>
      <entry><title>Acme ships agentic robotics</title>
        <link href="https://acme-robotics.com/"/>
        <updated>2026-09-08T10:00:00Z</updated>
        <summary>Agentic robots for warehouses.</summary>
      </entry></feed>""").entries
    accounts, _ = _parse(atom)
    assert [a.handle for a in accounts] == ["acme-robotics"]


def test_malformed_feed_yields_nothing_instead_of_raising() -> None:
    """feedparser sets `bozo` and salvages what it can — an unattended daily
    reader must not crash the whole discovery leg on one broken feed."""
    broken = feedparser.parse("<rss><channel><item><title>unclosed")
    accounts, unlinked = _parse(list(broken.entries))
    assert isinstance(accounts, list) and isinstance(unlinked, list)


def test_entries_without_a_title_or_link_are_skipped() -> None:
    entries = _rss("<item><title>Agentic robotics thing</title></item>"
                   "<item><link>https://acme.com/</link></item>")
    accounts, unlinked = _parse(entries)
    assert not accounts and not unlinked
