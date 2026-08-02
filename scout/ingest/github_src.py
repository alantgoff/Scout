"""GitHub discovery source — free build-evidence signal for technical founders.

Searches recent repos by thesis topics (created recently, minimum stars),
then bridges repo owners to X handles via the profile's twitter_username /
social accounts. This is the one free, ToS-clean identity bridge from code
to X. Owners without an X handle are still surfaced as UnlinkedLeads.

Rate limits: the Search API is the real throttle (~30 req/min authenticated,
~10/min anonymous) — we sleep between search calls and cap pages, so a run
stays well under it. GITHUB_TOKEN (free PAT) is optional but recommended.

Star-velocity diffing (re-poll stars and alert on spikes) is a future hook —
it needs scheduled runs to build a baseline.
"""

from __future__ import annotations

import asyncio
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

_API = "https://api.github.com"
_REPO_WINDOW_DAYS = 90
_MIN_STARS = 10
_REPOS_PER_TOPIC = 30
_SEARCH_PAUSE_S = 2.5  # stay far under the Search API per-minute limit
# Owner-profile fan-out. These are CORE API calls (5,000/hr with a token,
# 60/hr without) — not the throttled Search API — so a small concurrent
# burst is safe; serial fetches made this leg minutes of wall-clock.
_PROFILE_CONCURRENCY = 8

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
)


def parse_repo_owners(search_response: dict[str, Any]) -> list[dict[str, Any]]:
    """(pure, tested) Extract unique owner records from a repo-search response:
    [{login, repo_url, repo_name, stars, owner_type}]."""
    seen: dict[str, dict[str, Any]] = {}
    for repo in search_response.get("items", []):
        owner = repo.get("owner") or {}
        login = owner.get("login")
        if not login or login.lower() in seen:
            continue
        seen[login.lower()] = {
            "login": login,
            "repo_url": repo.get("html_url", ""),
            "repo_name": repo.get("full_name", ""),
            "stars": repo.get("stargazers_count", 0),
            "owner_type": owner.get("type", "User"),
        }
    return list(seen.values())


def profile_to_x_handle(
    profile: dict[str, Any], socials: list[dict[str, Any]]
) -> str | None:
    """(pure, tested) X handle from a GitHub profile: twitter_username field
    first, then any twitter/x social-account URL."""
    handle = profile.get("twitter_username")
    if handle:
        return str(handle).lstrip("@")
    for entry in socials:
        url = (entry.get("url") or "").lower()
        if "twitter.com/" in url or "x.com/" in url:
            tail = url.rstrip("/").rsplit("/", 1)[-1]
            if tail and tail not in ("twitter.com", "x.com"):
                return tail.lstrip("@")
    return None


class GitHubSource(DiscoverySource):
    name = "github"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    @_retry
    async def _get(self, client: httpx.AsyncClient, path: str, **params: Any) -> Any:
        resp = await client.get(f"{_API}{path}", params=params or None, timeout=20)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise RuntimeError(
                "GitHub rate limit hit — set GITHUB_TOKEN in .env (free PAT) "
                "or retry later."
            )
        resp.raise_for_status()
        return resp.json()

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        if not seeds.github_topics:
            return [], []
        created_after = (
            datetime.now(timezone.utc) - timedelta(days=_REPO_WINDOW_DAYS)
        ).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc)
        accounts: list[Account] = []
        unlinked: list[UnlinkedLead] = []

        async with httpx.AsyncClient(headers=self._headers()) as client:
            owners: dict[str, dict[str, Any]] = {}
            for topic in seeds.github_topics:
                query = f"topic:{topic} created:>{created_after} stars:>={_MIN_STARS}"
                try:
                    data = await self._get(
                        client,
                        "/search/repositories",
                        q=query,
                        sort="stars",
                        order="desc",
                        per_page=_REPOS_PER_TOPIC,
                    )
                except Exception as exc:
                    _console.print(f"[yellow]github topic {topic!r} failed:[/] {exc}")
                    continue
                for owner in parse_repo_owners(data):
                    owners.setdefault(owner["login"].lower(), owner)
                await asyncio.sleep(_SEARCH_PAUSE_S)

            semaphore = asyncio.Semaphore(_PROFILE_CONCURRENCY)

            async def fetch_owner(
                owner: dict[str, Any],
            ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
                login = owner["login"]
                async with semaphore:
                    try:
                        profile = await self._get(client, f"/users/{login}")
                        socials = (
                            await self._get(
                                client, f"/users/{login}/social_accounts"
                            )
                            if owner["owner_type"] == "User"
                            else []
                        )
                    except Exception as exc:
                        _console.print(
                            f"[yellow]github profile {login} failed:[/] {exc}"
                        )
                        return None
                return owner, profile, socials

            profiles = await asyncio.gather(
                *(fetch_owner(o) for o in owners.values())
            )
            for result in profiles:
                if result is None:
                    continue
                owner, profile, socials = result
                login = owner["login"]
                x_handle = profile_to_x_handle(profile, socials)
                bio = profile.get("bio") or profile.get("description") or ""
                if x_handle:
                    accounts.append(
                        Account(
                            id=f"gh-{login.lower()}",
                            handle=x_handle,
                            name=profile.get("name") or login,
                            bio=bio,
                            website=owner["repo_url"],
                            followers=profile.get("followers", 0),
                            source="github",
                            github_repo=owner["repo_url"],
                            fetched_at=now,
                        )
                    )
                else:
                    unlinked.append(
                        UnlinkedLead(
                            source="github",
                            ref=login,
                            name=profile.get("name") or login,
                            bio=bio,
                            url=owner["repo_url"],
                            found_at=now,
                        )
                    )

        self.store.upsert_accounts(accounts)
        self.store.upsert_unlinked_leads(unlinked)
        _console.print(
            f"[green]github:[/] {len(accounts)} X-bridged accounts, "
            f"{len(unlinked)} unlinked leads"
        )
        return accounts, unlinked
