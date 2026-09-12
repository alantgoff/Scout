"""RSS/Atom discovery source — free, keyless, no ToS risk, no rate limits.

The channel the query bank cannot reach. X search finds people TALKING about
building; feeds carry the structured announcements themselves: YC launch
posts, Product Hunt, portfolio "we invested in" notes, company engineering
blogs, funding coverage. A daily scan reads them for nothing.

Feeds come in two shapes and the difference decides what an entry can
honestly become:

- **Company feeds** (YC launches, Product Hunt, a company's own blog): the
  entry link IS the company or its product page. That bridges to a real
  Account keyed by the company's domain — the same identity `scout add
  <domain>` produces, with `profile_url` pointing at the site so nothing
  downstream renders a fabricated x.com/<slug>. The classifier then reads
  that site on the normal path and scores it like any other lead.
- **News feeds** (funding coverage): the entry link is an ARTICLE ABOUT a
  company, on the publisher's domain. Keying an entry to `techcrunch.com`
  would be a lie, so unless the entry text names an X handle these become
  UnlinkedLeads carrying the headline and the article URL.

Bridging is therefore never guessed: an Account appears only when the entry
yields an X handle or a company domain that isn't the feed's own publisher.

Pure parsing (`parse_entries`) is separated from I/O and unit-tested
against recorded feeds, the same split as the HN and arXiv sources. The
bridging gate itself (`web.company_domain`, `web.PUBLISHER_HOSTS`) lives in
scout.web because the HN and GitHub sources apply the identical rule.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from rich.console import Console

from scout.config import Seeds, Settings, Thesis
from scout.ingest.arxiv_src import extract_x_handle
from scout.ingest.base import DiscoverySource
from scout.models import Account, UnlinkedLead
from scout.signals.heuristics import matches_any
from scout.store import Store
from scout.web import company_domain, domain_slug, normalize_site_url

_console = Console()

_WINDOW_DAYS = 14  # feeds are a freshness instrument; older items are news
_MAX_ENTRIES_PER_FEED = 40
_FETCH_TIMEOUT_S = 10.0
_MAX_CONCURRENCY = 6
# A polite, honest UA: feed publishers block unidentified bulk readers, and
# an unattended daily reader should say what it is.
_UA = "Mozilla/5.0 (compatible; scout/0.1; +startup research; feed reader)"

def _entry_text(entry: Any) -> str:
    """Everything readable in one entry, for keyword matching and handles."""
    parts = [
        str(getattr(entry, "title", "") or ""),
        str(getattr(entry, "summary", "") or ""),
        str(getattr(entry, "author", "") or ""),
    ]
    for content in getattr(entry, "content", None) or []:
        parts.append(str((content or {}).get("value", "") or ""))
    return " ".join(parts)


def _entry_time(entry: Any, now: datetime) -> datetime:
    """The entry's publish time, or `now` when the feed omits/mangles it.

    Undated entries are kept rather than dropped: plenty of small company
    blogs emit no usable date, and silently discarding them would make the
    source quietly useless on exactly the feeds worth reading.
    """
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return now


def parse_entries(
    entries: list[Any],
    *,
    feed_url: str,
    feed_title: str,
    thesis: Thesis,
    now: datetime,
    window_days: int = _WINDOW_DAYS,
) -> tuple[list[Account], list[UnlinkedLead]]:
    """(pure, tested) Feed entries → bridged Accounts / UnlinkedLeads.

    Off-thesis entries are dropped here rather than downstream: a funding
    feed carries every sector, and letting them all through would spend the
    classification budget on companies the thesis already excludes.
    """
    feed_host = (urlparse(feed_url).hostname or "").lower().removeprefix("www.")
    cutoff = now - timedelta(days=window_days)
    terms = [*thesis.keywords, *thesis.sectors]
    accounts: list[Account] = []
    unlinked: list[UnlinkedLead] = []

    for entry in entries[:_MAX_ENTRIES_PER_FEED]:
        title = str(getattr(entry, "title", "") or "").strip()
        link = str(getattr(entry, "link", "") or "").strip()
        if not title or not link:
            continue
        if _entry_time(entry, now) < cutoff:
            continue
        text = _entry_text(entry)
        # No thesis terms configured → take everything and let scoring judge.
        if terms and not matches_any(text, terms):
            continue

        bio = f"{title} — via {feed_title}"[:280]
        handle = extract_x_handle(text)
        if handle:
            accounts.append(Account(
                id=f"rss-{handle.lower()}",
                handle=handle,
                name=title[:120],
                bio=bio,
                website=link,
                source="rss",
                fetched_at=now,
            ))
            continue

        domain = company_domain(link, feed_host)
        if domain:
            site = normalize_site_url(link)
            slug = domain_slug(domain)
            if slug:
                accounts.append(Account(
                    id=f"rss:{slug}",
                    handle=slug,
                    name=title[:120],
                    bio=bio,
                    website=site,
                    # No X presence known: the profile link must point at the
                    # real site, never an invented x.com/<slug>.
                    profile_url=site,
                    source="rss",
                    fetched_at=now,
                ))
                continue

        unlinked.append(UnlinkedLead(
            source="rss",
            ref=link,  # the (source, ref) pk — re-reading a feed is idempotent
            name=title[:120],
            bio=bio,
            url=link,
            found_at=now,
        ))
    return accounts, unlinked


class RSSSource(DiscoverySource):
    """Reads `seeds.rss_feeds`. Free, keyless, and cheap enough to run daily."""

    name = "rss"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(url)
            if response.status_code >= 400:
                _console.print(f"[yellow]feed {url} → HTTP {response.status_code}[/]")
                return None
            return response.text
        except Exception as exc:  # a dead feed must never sink discovery
            _console.print(f"[yellow]feed {url} failed:[/] {type(exc).__name__}")
            return None

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        feeds = [f.strip() for f in (seeds.rss_feeds or []) if f.strip()]
        if not feeds:
            return [], []
        now = datetime.now(timezone.utc)
        accounts: list[Account] = []
        unlinked: list[UnlinkedLead] = []
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def read(url: str) -> tuple[str, str | None]:
            async with semaphore:
                return url, await self._fetch(client, url)

        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S, follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            for url, body in await asyncio.gather(*(read(f) for f in feeds)):
                if not body:
                    continue
                # feedparser never raises on malformed input — it sets
                # `bozo` and returns whatever it could salvage, which is the
                # behaviour an unattended reader wants.
                parsed = feedparser.parse(body)
                entries = list(getattr(parsed, "entries", []) or [])
                if not entries:
                    _console.print(f"[dim]feed {url}: no entries[/dim]")
                    continue
                title = str(
                    (getattr(parsed, "feed", None) or {}).get("title", "") or url
                )
                got_accounts, got_unlinked = parse_entries(
                    entries, feed_url=url, feed_title=title, thesis=thesis, now=now,
                )
                accounts.extend(got_accounts)
                unlinked.extend(got_unlinked)

        _console.print(
            f"[dim]rss: {len(accounts)} bridged, {len(unlinked)} unlinked "
            f"from {len(feeds)} feed(s).[/dim]"
        )
        return accounts, unlinked
