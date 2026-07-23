"""X API v2 fallback adapter — PAID (pay-per-use), guarded by a hard spend cap.

Every request is pre-checked against the persistent cross-run budget ledger
in scout.db using the WORST-CASE cost of the response; the ACTUAL objects
returned are recorded after. No pagination anywhere: one page per query,
`next_token` is never followed.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Seeds, Settings
from scout.ingest.base import SourceAdapter
from scout.models import Account, Tweet
from scout.store import Store

API_BASE = "https://api.x.com/2"

# Endpoint page-size bounds for /2/tweets/search/recent.
_SEARCH_PAGE_MIN = 10
_SEARCH_PAGE_MAX = 100
_TWEET_FIELDS = "public_metrics,created_at,text"
# "entities" gives us expanded_url for t.co links (profile url + bio links).
_USER_FIELDS = "description,public_metrics,url,pinned_tweet_id,entities"

# Engagement operators (twscrape-only search syntax) that X API v2
# /tweets/search/recent rejects with a 400.
_ENGAGEMENT_OPERATOR_RE = re.compile(
    r"\bmin_(?:faves|retweets|replies):\d+", re.IGNORECASE
)

# seeds.yaml is written in X advanced-search syntax so the free twscrape source
# keeps its full operator set (including the min_faves: quality filter stripped
# above, which has no v2 equivalent). v2 spells the same filters differently, so
# translate here rather than maintain two query banks that would drift apart.
_FILTER_EQUIVALENTS = {
    "links": "has:links",
    "media": "has:media",
    "images": "has:images",
    "replies": "is:reply",
    "retweets": "is:retweet",
    "quote": "is:quote",
    "quotes": "is:quote",
    "verified": "is:verified",
}
_FILTER_RE = re.compile(r"(-?)\bfilter:(\w+)", re.IGNORECASE)

# v2 rejects an over-long query with a 400. Error responses aren't billed, but
# the seed is still wasted, so check before sending. 512 is the Basic/Essential
# ceiling (Pro allows 1024) — the conservative choice, since sending under the
# limit is always safe and we cannot see which tier the token is on.
_MAX_QUERY_LEN = 512

console = Console()


def _clean_query(query: str) -> str:
    """Rewrite an advanced-search query into the operator set v2 accepts.

    Engagement operators have no v2 equivalent and are dropped; filter:X
    operators are translated to their has:/is: spelling. An unrecognised
    filter:X is passed through untouched so the resulting 400 surfaces loudly,
    rather than us silently searching for something other than what was asked.
    """

    def _translate(match: re.Match[str]) -> str:
        negation, name = match.group(1), match.group(2).lower()
        equivalent = _FILTER_EQUIVALENTS.get(name)
        return match.group(0) if equivalent is None else f"{negation}{equivalent}"

    cleaned = _ENGAGEMENT_OPERATOR_RE.sub("", query)
    cleaned = _FILTER_RE.sub(_translate, cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned == query:
        return query
    console.print(f"[dim]xapi: rewrote query for v2 — sending {cleaned!r}[/dim]")
    return cleaned


class BudgetExceededError(RuntimeError):
    """Raised BEFORE a request whose worst-case cost would break the cap."""

    def __init__(
        self, spent_usd: float, worst_case_usd: float, cap_usd: float
    ) -> None:
        super().__init__(
            f"X API spend guard: ${spent_usd:.2f} already spent across all runs, "
            f"and the next request could cost up to ${worst_case_usd:.2f}, which "
            f"would exceed the ${cap_usd:.2f} cap. If you really want to keep "
            f"spending, raise XAPI_SPEND_CAP_USD in .env — but remember the "
            f"token has a hard $25 total budget."
        )


class _RetryableAPIError(Exception):
    """429/5xx received before any body was counted — safe to retry."""


class XApiSource(SourceAdapter):
    """Official X API v2 — activated with `scout run --source xapi`.

    Only the spec-whitelisted endpoints: GET /2/users/by,
    GET /2/users/:id/tweets, GET /2/tweets/search/recent.
    """

    name = "xapi"

    def __init__(self, settings: Settings, store: Store) -> None:
        if not settings.x_bearer_token:
            raise RuntimeError(
                "X_BEARER_TOKEN is not set — add it to .env to use "
                "`--source xapi`, or use the free twscrape source instead."
            )
        self.settings = settings
        self.store = store
        self._timeline_fetches = 0
        self._timeline_cap_announced = False

    # ---------------------------------------------------------- budget guard

    def _guard(self, worst_case_usd: float) -> None:
        """Refuse to spend before the request — never request-then-check."""
        spent = self.store.xapi_spend_usd()
        if spent + worst_case_usd > self.settings.xapi_spend_cap_usd:
            raise BudgetExceededError(
                spent, worst_case_usd, self.settings.xapi_spend_cap_usd
            )

    def _record(self, endpoint: str, posts_read: int, users_read: int) -> None:
        self.store.record_xapi_usage(
            endpoint,
            posts_read,
            users_read,
            posts_read * self.settings.xapi_cost_per_post_read
            + users_read * self.settings.xapi_cost_per_user_read,
        )

    # ----------------------------------------------------------------- http

    @retry(
        # Double-billing constraint: retry ONLY failures that happen strictly
        # before X could have served (and billed) the request — connect-phase
        # errors, plus 429/5xx statuses (error responses aren't billed).
        # Read-side transport errors (ReadTimeout, ReadError, ...) can fire
        # AFTER X has already billed the response, so retrying them could
        # invisibly double-bill against the $25 budget — let them raise.
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ConnectTimeout, _RetryableAPIError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        # Fresh client per request: CLI commands invoke asyncio.run() more
        # than once, and a client bound to a closed loop raises "Event loop
        # is closed". Requests here are rare and paid, so losing connection
        # reuse costs nothing that matters.
        async with httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {self.settings.x_bearer_token}"},
            timeout=30.0,
        ) as client:
            resp = await client.get(path, params=params)
        if resp.status_code == 429:
            try:
                retry_after = int(resp.headers.get("retry-after", "5"))
            except ValueError:
                retry_after = 5
            await asyncio.sleep(min(retry_after, 30))  # within reason
            raise _RetryableAPIError(f"429 rate-limited on {path}")
        if resp.status_code >= 500:
            raise _RetryableAPIError(f"HTTP {resp.status_code} from {path}")
        if resp.status_code >= 400:
            raise RuntimeError(
                f"X API error {resp.status_code} on {path}: {resp.text[:200]}"
            )
        return resp.json()

    # -------------------------------------------------------------- parsing

    @staticmethod
    def _parse_user(u: dict[str, Any], source: str) -> Account:
        pm = u.get("public_metrics") or {}
        entities = u.get("entities") or {}
        # The bare `url` field is a t.co shortlink; prefer the expanded url
        # from entities so builder_evidence sees the real site.
        website = u.get("url") or None
        profile_urls = (entities.get("url") or {}).get("urls") or []
        if profile_urls and profile_urls[0].get("expanded_url"):
            website = profile_urls[0]["expanded_url"]
        # Same for t.co links inside the bio text.
        bio = u.get("description", "")
        for link in (entities.get("description") or {}).get("urls") or []:
            if link.get("url") and link.get("expanded_url"):
                bio = bio.replace(link["url"], link["expanded_url"])
        return Account(
            id=str(u["id"]),
            handle=u["username"],
            name=u.get("name", ""),
            bio=bio,
            website=website,
            followers=pm.get("followers_count", 0),
            following=pm.get("following_count", 0),
            pinned_tweet_id=u.get("pinned_tweet_id"),
            source=source,
            fetched_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _parse_tweet(t: dict[str, Any], account_id: str = "") -> Tweet:
        pm = t.get("public_metrics") or {}
        return Tweet(
            id=str(t["id"]),
            account_id=str(t.get("author_id") or account_id),
            text=t.get("text", ""),
            created_at=t.get("created_at"),
            likes=pm.get("like_count", 0),
            retweets=pm.get("retweet_count", 0),
            replies=pm.get("reply_count", 0),
        )

    # -------------------------------------------------------------- adapter

    async def fetch_accounts(
        self,
        seeds: Seeds,
        *,
        max_accounts: int,
        search_categories: set[str] | None = None,
    ) -> list[Account]:
        if seeds.lists or seeds.watchers:
            console.print(
                "[yellow]xapi mode: skipping seeds.lists and the watchlist — "
                "list members, bio search, and graph-hop require twscrape.[/yellow]"
            )
        accounts: dict[str, Account] = {}  # keyed by lowercased handle

        def merge(account: Account) -> None:
            existing = accounts.get(account.handle.lower())
            if existing is not None:
                account.followed_by = sorted(
                    set(existing.followed_by) | set(account.followed_by)
                )
                accounts[account.handle.lower()] = account
            elif len(accounts) < max_accounts:
                accounts[account.handle.lower()] = account

        for category, query in seeds.all_searches:
            if search_categories is not None and category not in search_categories:
                continue
            if len(accounts) >= max_accounts:
                break
            query = _clean_query(query)
            if len(query) > _MAX_QUERY_LEN:
                console.print(
                    f"[red]skipping query over the {_MAX_QUERY_LEN}-char v2 "
                    f"limit ({len(query)} chars): {query[:80]!r}…[/red]"
                )
                continue

            # TTL search cache: repeat runs of the same query are free.
            cached_handles = self.store.cached_search(query, self.settings.ttl_days)
            if cached_handles is not None:
                console.print(f"[dim]search {query!r} (cached)[/dim]")
                for handle in cached_handles:
                    account = self.store.get_account(handle)
                    if account is not None:
                        merge(account)
                continue

            # Only request as many results as we still need (endpoint min 10);
            # worst case: a full page of tweets, each from a distinct author
            # expanded as a user object. Capped by xapi_search_page_size —
            # breadth is cheaper bought with more narrow queries than with a
            # deeper page of one, since a query's 80th result is reliably worse
            # than another query's 5th and costs exactly the same.
            page_size = max(
                _SEARCH_PAGE_MIN,
                min(
                    _SEARCH_PAGE_MAX,
                    self.settings.xapi_search_page_size,
                    max_accounts - len(accounts),
                ),
            )
            try:
                self._guard(
                    page_size * self.settings.xapi_cost_per_post_read
                    + page_size * self.settings.xapi_cost_per_user_read
                )
                payload = await self._get(
                    "/tweets/search/recent",
                    {
                        "query": query,
                        "max_results": page_size,
                        "expansions": "author_id",
                        "tweet.fields": _TWEET_FIELDS,
                        "user.fields": _USER_FIELDS,
                    },
                )
            except BudgetExceededError:
                # Don't strand what earlier seeds already paid for — return
                # the partial account set so it still gets scored/exported.
                console.print(
                    "[yellow]X API budget reached mid-seeds — continuing with "
                    f"{len(accounts)} accounts fetched so far.[/yellow]"
                )
                break
            except (RuntimeError, httpx.HTTPError, _RetryableAPIError) as exc:
                console.print(
                    f"[red]search failed, skipping query {query!r}: {exc}[/red]"
                )
                continue
            data = payload.get("data") or []
            users = (payload.get("includes") or {}).get("users") or []
            self._record("/2/tweets/search/recent", len(data), len(users))
            # Search tweets double as each author's cached tweets, so
            # downstream fetch_tweets() reads the store instead of making
            # paid timeline calls.
            self.store.upsert_tweets([self._parse_tweet(t) for t in data])
            seen_handles: list[str] = []
            for u in users:
                account = self._parse_user(u, source="search")
                self.store.upsert_account(account)
                seen_handles.append(account.handle)
                merge(account)
            self.store.record_search(query, seen_handles)
        return list(accounts.values())

    async def fetch_tweets(self, account: Account, *, limit: int = 20) -> list[Tweet]:
        cached = self.store.get_tweets(account.id, limit=limit)
        if cached:
            return cached
        # Cache miss: the query bank never saw this account, so it came from
        # github/hn discovery. Checked before the id resolve below, because
        # that resolve is itself a paid user read.
        if self._timeline_fetches >= self.settings.xapi_max_timeline_fetches:
            if not self._timeline_cap_announced:
                self._timeline_cap_announced = True
                console.print(
                    "[dim]xapi: not paying for discovery-sourced timelines "
                    f"(xapi_max_timeline_fetches="
                    f"{self.settings.xapi_max_timeline_fetches}) — those "
                    "accounts score on bio signals[/dim]"
                )
            return []
        self._timeline_fetches += 1
        if not account.id.isdigit():
            # Discovery-sourced record (github/hn) with a synthetic id — resolve
            # the real X id first; without it the timeline endpoint 400s.
            resolved = await self.fetch_account(account.handle, fresh=True)
            if resolved is None:
                return []
            account = resolved
            cached = self.store.get_tweets(account.id, limit=limit)
            if cached:
                return cached
        max_results = max(5, min(limit, 100))  # endpoint floor is 5
        self._guard(max_results * self.settings.xapi_cost_per_post_read)
        payload = await self._get(
            f"/users/{account.id}/tweets",
            {"max_results": max_results, "tweet.fields": _TWEET_FIELDS},
        )
        data = payload.get("data") or []
        self._record("/2/users/:id/tweets", posts_read=len(data), users_read=0)
        tweets = [self._parse_tweet(t, account_id=account.id) for t in data]
        self.store.upsert_tweets(tweets)
        return tweets[:limit]

    async def fetch_account(self, handle: str, *, fresh: bool = False) -> Account | None:
        """Cache-first by default; `fresh=True` forces a paid lookup (used by
        `scout verify`, whose whole point is fresh data — and required when the
        cache holds a synthetic discovery id instead of a real X id)."""
        handle = handle.lstrip("@")
        if not fresh:
            cached = self.store.get_account(handle)
            if cached is not None and cached.id.isdigit():
                return cached
        self._guard(self.settings.xapi_cost_per_user_read)
        payload = await self._get(
            "/users/by", {"usernames": handle, "user.fields": _USER_FIELDS}
        )
        data = payload.get("data") or []
        self._record("/2/users/by", posts_read=0, users_read=len(data))
        if not data:
            return None
        account = self._parse_user(data[0], source="manual")
        self.store.upsert_account(account)
        return account
