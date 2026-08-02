"""Hacker News discovery source — free, keyless, zero ToS risk (Algolia API).

Two plays:
1. Show HN launches matching thesis terms in the last 30 days — the launch
   moment, often before any press.
2. The monthly "Who wants to be hired?" thread — individuals announcing
   availability, including just-left-a-lab engineers.

X/GitHub links found in posts bridge to Accounts; everything else becomes an
UnlinkedLead (HN username + story URL) for manual lookup.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Seeds, Settings, Thesis
from scout.ingest.base import DiscoverySource
from scout.models import Account, UnlinkedLead
from scout.store import Store

_console = Console()

_API = "https://hn.algolia.com/api/v1"
_WINDOW_DAYS = 30
_HITS_PER_QUERY = 50

_X_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/(@?[A-Za-z0-9_]{1,15})\b",
    re.IGNORECASE,
)
# Paths on x.com that are never user handles.
_X_RESERVED = {"i", "intent", "home", "search", "hashtag", "share", "settings"}

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
)


def extract_x_handle(text: str) -> str | None:
    """(pure, tested) First plausible X handle linked in a blob of text/HTML."""
    for match in _X_LINK_RE.finditer(text or ""):
        handle = match.group(1).lstrip("@")
        if handle.lower() not in _X_RESERVED:
            return handle
    return None


def parse_hn_hits(
    hits: list[dict[str, Any]], *, now: datetime
) -> tuple[list[Account], list[UnlinkedLead]]:
    """(pure, tested) Convert Algolia hits into bridged Accounts / UnlinkedLeads."""
    accounts: list[Account] = []
    unlinked: list[UnlinkedLead] = []
    for hit in hits:
        author = hit.get("author") or ""
        title = hit.get("title") or hit.get("story_title") or ""
        text = " ".join(
            str(hit.get(k) or "") for k in ("title", "story_text", "comment_text", "url")
        )
        story_url = hit.get("url") or (
            f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        )
        x_handle = extract_x_handle(text)
        if x_handle:
            accounts.append(
                Account(
                    id=f"hn-{author.lower()}",
                    handle=x_handle,
                    name=author,
                    bio=title[:280],
                    website=story_url,
                    source="hn",
                    fetched_at=now,
                )
            )
        elif author:
            unlinked.append(
                UnlinkedLead(
                    source="hn",
                    ref=author,
                    name=author,
                    bio=title[:280],
                    url=story_url,
                    found_at=now,
                )
            )
    return accounts, unlinked


class HackerNewsSource(DiscoverySource):
    name = "hn"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    @_retry
    async def _get(self, client: httpx.AsyncClient, path: str, **params: Any) -> Any:
        resp = await client.get(f"{_API}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        now = datetime.now(timezone.utc)
        cutoff = int((now - timedelta(days=_WINDOW_DAYS)).timestamp())
        terms = thesis.sectors or ["ai"]
        accounts: list[Account] = []
        unlinked: list[UnlinkedLead] = []

        async with httpx.AsyncClient() as client:
            # 1) Show HN launches matching thesis sectors
            for term in terms:
                try:
                    data = await self._get(
                        client,
                        "/search_by_date",
                        query=term,
                        tags="show_hn",
                        numericFilters=f"created_at_i>{cutoff}",
                        hitsPerPage=_HITS_PER_QUERY,
                    )
                except Exception as exc:
                    _console.print(f"[yellow]hn search {term!r} failed:[/] {exc}")
                    continue
                got_accounts, got_unlinked = parse_hn_hits(
                    data.get("hits", []), now=now
                )
                accounts.extend(got_accounts)
                unlinked.extend(got_unlinked)

            # 2) Latest "Who wants to be hired?" thread — comments matching thesis
            try:
                threads = await self._get(
                    client,
                    "/search_by_date",
                    query="Who wants to be hired",
                    tags="story,author_whoishiring",
                    hitsPerPage=5,
                )
                # Algolia matching is fuzzy and happily returns the employer
                # thread ("Who is hiring?") — filter on the exact title.
                thread_hits = [
                    h
                    for h in threads.get("hits", [])
                    if "who wants to be hired" in (h.get("title") or "").lower()
                ]
                if thread_hits:
                    thread_id = thread_hits[0]["objectID"]
                    for term in terms[:3]:
                        data = await self._get(
                            client,
                            "/search",
                            query=term,
                            tags=f"comment,story_{thread_id}",
                            hitsPerPage=_HITS_PER_QUERY,
                        )
                        got_accounts, got_unlinked = parse_hn_hits(
                            data.get("hits", []), now=now
                        )
                        accounts.extend(got_accounts)
                        unlinked.extend(got_unlinked)
            except Exception as exc:
                _console.print(f"[yellow]hn who-wants-to-be-hired failed:[/] {exc}")

        # Dedupe (an author can match several terms)
        accounts = list({a.handle.lower(): a for a in accounts}.values())
        unlinked = list({u.ref.lower(): u for u in unlinked}.values())
        self.store.upsert_accounts(accounts)
        self.store.upsert_unlinked_leads(unlinked)
        _console.print(
            f"[green]hn:[/] {len(accounts)} X-bridged accounts, "
            f"{len(unlinked)} unlinked leads"
        )
        return accounts, unlinked
