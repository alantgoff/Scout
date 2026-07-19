"""Company-website evidence — fetch a candidate's site so the classifier
grounds "what the company does" in real product copy instead of bio vibes.

House split for testability: the pure functions (normalize_site_url,
extract_site_text) are unit-tested on fixtures; fetch_sites is a thin async
I/O wrapper mirroring cli._fetch_tweets (semaphore fan-out, per-call
timeout, cache-first via the store, negative caching for failures).
"""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from scout.config import Settings
from scout.models import SitePage
from scout.store import Store

# Hosts that never describe the product: X itself, link farms, socials,
# scheduling pages, app stores, code hosts (github_repo is separate
# evidence), newsletters. Checked against the host with any www. stripped.
_SKIP_HOSTS = {
    "x.com", "twitter.com", "t.co",
    "linktr.ee", "bio.link", "beacons.ai", "linkin.bio", "lnk.bio",
    "instagram.com", "youtube.com", "youtu.be", "facebook.com", "tiktok.com",
    "twitch.tv", "threads.net", "bsky.app", "linkedin.com",
    "discord.gg", "discord.com", "calendly.com", "cal.com",
    "github.com", "gitlab.com", "medium.com", "substack.com",
    "apps.apple.com", "play.google.com",
}

_THIN_TEXT_CHARS = 200  # below this the page is "thin" (JS-only SPA, mostly)
_MAX_HTML_BYTES = 500_000  # slice raw HTML before parsing
_MAX_CONTENT_BYTES = 2_000_000  # bail out on huge responses
_STORE_TEXT_CHARS = 20_000  # cached ceiling; prompt truncates further
_UA = "Mozilla/5.0 (compatible; scout/0.1; startup research bot)"


def normalize_site_url(url: str | None) -> str | None:
    """Root-page URL worth fetching, or None.

    Adds https:// when schemeless, lowercases the host, strips
    path/query/fragment (a /careers page describes hiring — the homepage
    describes the product), and rejects non-http(s) schemes, raw IPs,
    localhost, and _SKIP_HOSTS."""
    if not url or not url.strip():
        return None
    raw = url.strip()
    try:
        scheme = urlparse(raw).scheme
        # Reject non-web schemes BEFORE defaulting to https — otherwise
        # "mailto:a@b.co" re-parses with b.co as the host.
        if scheme and scheme not in ("http", "https"):
            return None
        if not scheme:
            raw = "https://" + raw
        parts = urlparse(raw)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower().strip(".")
    if not host or "." not in host:
        return None
    try:
        ipaddress.ip_address(host)
        return None  # a raw IP is never a startup's marketing site
    except ValueError:
        pass
    if host.removeprefix("www.") in _SKIP_HOSTS:
        return None
    return f"https://{host}/"


def extract_site_text(html: str, max_chars: int) -> str:
    """Visible page text, product-copy first: <title> and meta/og
    descriptions lead (JS-only SPAs almost always still have those), then
    body text with script/style/svg/etc. dropped and whitespace collapsed."""
    soup = BeautifulSoup(html[:_MAX_HTML_BYTES], "html.parser")
    parts: list[str] = []
    seen: set[str] = set()

    def add(text: str | None) -> None:
        cleaned = " ".join((text or "").split())
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            parts.append(cleaned)

    if soup.title is not None:
        add(soup.title.string)
    for attrs in ({"name": "description"}, {"property": "og:description"},
                  {"property": "og:title"}):
        tag = soup.find("meta", attrs=attrs)
        if tag is not None:
            add(tag.get("content"))

    for tag in soup(["script", "style", "noscript", "svg", "template",
                     "iframe", "head"]):
        tag.decompose()
    add(soup.get_text(" "))
    return "\n".join(parts)[:max_chars].strip()


async def _fetch_one(client: httpx.AsyncClient, url: str) -> SitePage:
    """One GET → SitePage (never raises; failures become status strings)."""
    page = SitePage(url=url, fetched_at=datetime.now(timezone.utc))
    try:
        resp = await client.get(url)
        page.final_url = str(resp.url)
        if resp.status_code >= 400:
            page.status = f"error:http:{resp.status_code}"
            return page
        ctype = (resp.headers.get("content-type") or "").lower()
        if ctype and "html" not in ctype and "text" not in ctype:
            page.status = "non-html"
            return page
        if len(resp.content) > _MAX_CONTENT_BYTES:
            page.status = "too-large"
            return page
        page.text = extract_site_text(resp.text, _STORE_TEXT_CHARS)
        page.status = "ok" if len(page.text) >= _THIN_TEXT_CHARS else "thin"
    except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
        page.status = "error:timeout"
    except httpx.HTTPError as exc:
        page.status = f"error:{type(exc).__name__.lower()}"
    except Exception as exc:  # never let one weird site kill the phase
        page.status = f"error:{type(exc).__name__.lower()}"
    return page


async def fetch_sites(
    urls: list[str],
    settings: Settings,
    store: Store | None = None,
    progress: Callable[[int, int], None] | None = None,
    fallbacks: dict[str, str] | None = None,
) -> dict[str, SitePage]:
    """Fetch many sites concurrently, cache-first. Keys are normalized URLs.

    Mirrors cli._fetch_tweets: Semaphore(web_fetch_concurrency), per-call
    wait_for(web_fetch_timeout_s), gather. Every outcome — including
    failures — is record_site()d so dead sites aren't re-tried every run
    (cached_site expires failures after 1 day). `fallbacks` maps a
    normalized URL to the original (path-bearing) URL: when the root fetch
    fails, the original is tried once before giving up."""
    result: dict[str, SitePage] = {}
    todo: list[str] = []
    for url in dict.fromkeys(urls):  # order-preserving dedupe
        cached = store.cached_site(url, settings.website_ttl_days) if store else None
        if cached is not None:
            result[url] = cached
        else:
            todo.append(url)
    if progress is not None:
        progress(0, len(todo))
    if not todo:
        return result

    semaphore = asyncio.Semaphore(max(1, settings.web_fetch_concurrency))
    timeout = settings.web_fetch_timeout_s
    done_n = 0
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
    ) as client:

        async def fetch(url: str) -> None:
            nonlocal done_n
            async with semaphore:
                page = await asyncio.wait_for(_fetch_one(client, url),
                                              timeout=timeout + 4)
                fallback = (fallbacks or {}).get(url)
                if page.status.startswith("error") and fallback and fallback != url:
                    retried = await asyncio.wait_for(_fetch_one(client, fallback),
                                                     timeout=timeout + 4)
                    if not retried.status.startswith("error"):
                        retried.url = url  # cache under the normalized key
                        page = retried
            page.fetched_at = datetime.now(timezone.utc)
            result[url] = page
            if store is not None:
                store.record_site(page)
            done_n += 1
            if progress is not None:
                progress(done_n, len(todo))

        await asyncio.gather(*(fetch(u) for u in todo))
    return result
