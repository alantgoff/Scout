"""Primary ingestion adapter: twscrape (free, cookie-based).

Note on the `lists` seed strategy: the actual membership roll is fetched
via API.list_members (twscrape >= 0.19). If that call fails, we fall back
to deriving accounts from list *timeline* authors (API.list_timeline) —
a lossier proxy that misses members who haven't tweeted recently.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_random_exponential
from twscrape import API, gather
from twscrape.models import Tweet as TwTweet
from twscrape.models import User as TwUser

from scout.config import Seeds, Settings
from scout.ingest.base import SourceAdapter
from scout.models import Account, Tweet
from scout.store import Store

_console = Console()

# Tweets pulled per list/search seed; authors are deduped afterwards.
_TWEETS_PER_SEED = 100
# API.following is most-recent-first, so ~100 approximates "recent follows".
_FOLLOWS_PER_TASTEMAKER = 100
# Fixed twscrape pool entry; credentials are dummies, auth comes from cookies.
_POOL_USERNAME = "scout_cookies"

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=30),
)

_COOKIE_HELP = (
    "Set TW_COOKIES in .env to your X cookies file "
    "(see the README 'Cookie setup' section)."
)


def _parse_cookies(raw: str) -> str:
    """Normalize a cookies file to twscrape's "k1=v1; k2=v2" string.

    Accepts a JSON dict of name->value, a JSON list of {name, value} objects
    (the common browser-extension export), or an already-raw cookie string.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(data, dict):
        return "; ".join(f"{k}={v}" for k, v in data.items())
    if isinstance(data, list):
        pairs = [(c.get("name"), c.get("value")) for c in data if isinstance(c, dict)]
        return "; ".join(f"{k}={v}" for k, v in pairs if k)
    raise RuntimeError(f"Unrecognized cookies file format. {_COOKIE_HELP}")


def _to_account(user: TwUser, *, source: str) -> Account:
    # twscrape's descriptionLinks = bio links FIRST, then the profile's
    # dedicated website entity LAST — so [-1] is the actual website whenever
    # one is set; [0] would grab a random bio link (and misground the
    # classifier's website fetch).
    links = getattr(user, "descriptionLinks", None) or []
    pinned = getattr(user, "pinnedIds", None) or []
    return Account(
        id=str(user.id),
        handle=user.username,
        name=user.displayname,
        bio=user.rawDescription,
        website=links[-1].url if links else None,
        followers=user.followersCount,
        following=user.friendsCount,
        pinned_tweet_id=str(pinned[0]) if pinned else None,
        source=source,
        fetched_at=datetime.now(timezone.utc),
    )


def _to_tweet(t: TwTweet) -> Tweet:
    return Tweet(
        id=str(t.id),
        account_id=str(t.user.id),
        text=t.rawContent,
        created_at=t.date,
        likes=t.likeCount,
        retweets=t.retweetCount,
        replies=t.replyCount,
    )


class TwscrapeSource(SourceAdapter):
    name = "twscrape"
    # Free source — safe to fetch tweets concurrently (twscrape rate-limits
    # internally per account).
    parallel_safe = True

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.api = API()
        self._ready = False
        if settings.tw_cookies is None or not settings.tw_cookies.exists():
            raise RuntimeError(f"twscrape needs X session cookies. {_COOKIE_HELP}")
        self._cookies = _parse_cookies(
            settings.tw_cookies.read_text(encoding="utf-8")
        )
        if "ct0" not in self._cookies or "auth_token" not in self._cookies:
            _console.print(
                "[yellow]Warning:[/] cookies are missing 'ct0'/'auth_token' — "
                f"twscrape will likely fail. {_COOKIE_HELP}"
            )

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        # Delete-then-add so an updated cookies file takes effect; twscrape
        # marks the account active when a 'ct0' cookie is present (no login).
        await self.api.pool.delete_accounts(_POOL_USERNAME)
        await self.api.pool.add_account(
            username=_POOL_USERNAME,
            password="cookie-auth",
            email="cookie-auth@localhost",
            email_password="cookie-auth",
            cookies=self._cookies,
        )
        self._ready = True

    # -------------------------------------------------- retried network calls

    @_retry
    async def _list_members(self, list_id: int) -> list[TwUser]:
        return await gather(self.api.list_members(list_id))

    @_retry
    async def _list_timeline(self, list_id: int) -> list[TwTweet]:
        return await gather(self.api.list_timeline(list_id, limit=_TWEETS_PER_SEED))

    @_retry
    async def _search(self, query: str) -> list[TwTweet]:
        return await gather(self.api.search(query, limit=_TWEETS_PER_SEED))

    @_retry
    async def _search_user(self, query: str) -> list[TwUser]:
        # People search — hits BIOS, which tweet search cannot reach.
        return await gather(self.api.search_user(query, limit=_TWEETS_PER_SEED))

    @_retry
    async def _user_by_login(self, handle: str) -> TwUser | None:
        return await self.api.user_by_login(handle)

    @_retry
    async def _following(self, uid: int) -> list[TwUser]:
        return await gather(self.api.following(uid, limit=_FOLLOWS_PER_TASTEMAKER))

    @_retry
    async def _user_tweets(self, uid: int, limit: int) -> list[TwTweet]:
        return await gather(self.api.user_tweets(uid, limit=limit))

    # ------------------------------------------------------ SourceAdapter API

    async def fetch_accounts(
        self,
        seeds: Seeds,
        *,
        max_accounts: int,
        strategies: set[str] | None = None,
        search_categories: set[str] | None = None,
        recent_follow_days: int = 30,
        time_budget_s: float | None = None,
    ) -> list[Account]:
        """Run seed strategies; `strategies` filters to a subset of
        {"lists", "searches", "bio", "graph"} (None = all), and
        `search_categories` filters the query bank by category (None = all) —
        this is how thesis.target_stages steers what gets searched.

        `time_budget_s` is a hard wall-clock cap on sourcing: X rate limits
        (the follow-graph endpoint especially) make twscrape sleep through
        15-minute reset windows, so without a budget a run can hang for an
        hour. When the budget expires, whatever has been gathered is returned
        — every leg checks the deadline between items, and each network call
        is bounded to the remaining time so an in-call sleep can't overshoot.
        """
        await self._ensure_ready()
        found: dict[str, Account] = {}  # keyed by lowercased handle
        deadline = (
            time.monotonic() + time_budget_s if time_budget_s is not None else None
        )
        budget_announced = False

        def out_of_time() -> bool:
            nonlocal budget_announced
            if deadline is not None and time.monotonic() >= deadline:
                if not budget_announced:
                    budget_announced = True
                    _console.print(
                        f"[yellow]sourcing time budget ({time_budget_s:.0f}s) "
                        f"reached — continuing with {len(found)} accounts.[/]"
                    )
                return True
            return False

        async def bounded(coro):
            """Cap a network call at the remaining budget (twscrape sleeps
            through rate-limit windows INSIDE calls — this cuts that short)."""
            if deadline is None:
                return await coro
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                coro.close()
                raise TimeoutError("sourcing time budget reached")
            return await asyncio.wait_for(coro, timeout=remaining)

        def enabled(name: str) -> bool:
            return strategies is None or name in strategies

        def full() -> bool:
            return len(found) >= max_accounts

        def add(user: TwUser, source: str, watcher: str | None = None) -> None:
            key = user.username.lower()
            account = found.get(key)
            if account is None:
                if full():
                    return
                account = _to_account(user, source=source)
                found[key] = account
            # Track every strategy that independently surfaced this account —
            # feeds the source_corroboration signal. Persistence is one
            # batched upsert at the end of fetch_accounts: a search result
            # used to re-upsert the same author once per tweet.
            if source and source not in account.sources:
                account.sources.append(source)
            if watcher and watcher not in account.followed_by:
                account.followed_by.append(watcher)

        # 1) lists — membership roll first; timeline authors as the fallback
        #    (see module docstring)
        for list_id in seeds.lists if enabled("lists") else []:
            if full() or out_of_time():
                break
            try:
                members = await bounded(self._list_members(int(list_id)))
            except Exception as exc:
                _console.print(
                    f"[yellow]list {list_id} members failed ({exc}) — "
                    "falling back to timeline authors[/]"
                )
                members = None
            if members is not None:
                for user in members:
                    add(user, "list")
                continue
            try:
                tweets = await bounded(self._list_timeline(int(list_id)))
            except Exception as exc:
                _console.print(f"[yellow]list {list_id} failed:[/] {exc}")
                continue
            self.store.upsert_tweets([_to_tweet(t) for t in tweets])
            for t in tweets:
                add(t.user, "list")

        # 2) query bank — labeled searches; result tweets double as cached tweets
        for category, query in seeds.all_searches if enabled("searches") else []:
            if search_categories is not None and category not in search_categories:
                continue
            if full() or out_of_time():
                break
            try:
                tweets = await bounded(self._search(query))
            except Exception as exc:
                _console.print(f"[yellow]search {query!r} failed:[/] {exc}")
                continue
            self.store.upsert_tweets([_to_tweet(t) for t in tweets])
            for t in tweets:
                add(t.user, f"search:{category}")

        # 3) bio/people search — reaches BIOS ("ex-openai", "stealth"), which
        #    tweet search cannot; web-only capability, hence twscrape-only.
        for query in seeds.bio_searches if enabled("bio") else []:
            if full() or out_of_time():
                break
            try:
                users = await bounded(self._search_user(query))
            except Exception as exc:
                _console.print(f"[yellow]bio search {query!r} failed:[/] {exc}")
                continue
            for user in users:
                add(user, "bio_search")

        # 4) watchlist graph hop — snapshot each watcher's following list into
        #    the store (first_seen diffing happens there), then surface only
        #    accounts followed RECENTLY. Cold start: a watcher's first snapshot
        #    is baseline-only, so this strategy starts yielding from run 2.
        if enabled("graph"):
            cold_start = 0
            follow_lists: list[tuple[str, dict[str, TwUser]]] = []
            for watcher in seeds.watchers:
                if full() or out_of_time():
                    break
                handle = watcher.lstrip("@")
                try:
                    user = await bounded(self._user_by_login(handle))
                    if user is None:
                        _console.print(f"[yellow]watcher @{handle} not found[/]")
                        continue
                    follows = await bounded(self._following(user.id))
                except Exception as exc:
                    _console.print(f"[yellow]graph hop @{handle} failed:[/] {exc}")
                    continue
                is_first_snapshot = (
                    self.store._watcher_baseline(handle.lower()) is None
                )
                self.store.record_follow_snapshot(
                    handle, [u.username for u in follows]
                )
                if is_first_snapshot:
                    cold_start += 1
                    continue  # baseline only — new-follow diffing starts next run
                follow_lists.append(
                    (handle, {u.username.lower(): u for u in follows})
                )
            if follow_lists:
                # ONE batched recency query after all snapshots, replacing a
                # follow_edges scan per followee (that was O(watchers² ×
                # follows²) as the edge table grew run over run).
                recent_map = self.store.recent_watchers_map(recent_follow_days)
                for handle, by_handle in follow_lists:
                    for followee, user_obj in by_handle.items():
                        if recent_map.get(followee):
                            add(user_obj, "graph", watcher=handle)
            if cold_start:
                _console.print(
                    f"[dim]graph: baselined {cold_start} watcher(s) — new-follow "
                    "detection activates on the next run.[/dim]"
                )

        # Batched persistence for everything the legs above surfaced.
        self.store.upsert_accounts(list(found.values()))
        _console.print(
            f"[green]twscrape:[/] {len(found)} unique accounts from seeds"
        )
        return list(found.values())

    async def fetch_tweets(self, account: Account, *, limit: int = 20) -> list[Tweet]:
        # twscrape is free, so always fetch fresh; upserting keeps the cache warm.
        await self._ensure_ready()
        uid = account.id
        if not uid.isdigit():
            # Discovery-sourced account (github/hn) with a synthetic id —
            # resolve the real X user id by handle first.
            user = await self._user_by_login(account.handle)
            if user is None:
                return []
            uid = str(user.id)
        tweets = [_to_tweet(t) for t in await self._user_tweets(int(uid), limit)]
        self.store.upsert_tweets(tweets)
        return tweets

    async def fetch_account(self, handle: str) -> Account | None:
        await self._ensure_ready()
        user = await self._user_by_login(handle.lstrip("@"))
        if user is None:
            return None
        account = _to_account(user, source="manual")
        self.store.upsert_account(account)
        return account
