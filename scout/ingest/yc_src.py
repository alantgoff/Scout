"""Y Combinator directory discovery source — every new-batch company, with
its domain and one-liner, the day the batch starts.

The cleanest free seed-stage dataset there is: YC publishes its company
directory, and `yc-oss/api` mirrors it as static JSON rebuilt daily from
YC's own search index (no login, no scraping of ycombinator.com, one small
file per batch). A batch company is launched and seed-funded by
definition, and the record carries the two things that make a lead
scoreable at no cost — a website for the classifier to read and a
one-liner for the thesis filter — so nearly every entry becomes a
domain-keyed Account (the `scout add <domain>` identity) rather than an
unlinked lead.

"YC S25" in the bio is real, citable evidence: the classifier may ground a
`funding_stage` in it. Nothing else is inferred from membership.

Network shape: `meta.json` (batch names + counts) → the latest N batch files
(`batches/<slug>.json`), falling back to `companies/all.json` filtered by
batch if a batch file is missing. Pure parsing (`parse_batches`,
`parse_companies`) is unit-tested on recorded record shapes.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from rich.console import Console

from scout.config import Seeds, Settings, Thesis
from scout.ingest.base import DiscoverySource
from scout.models import Account, UnlinkedLead
from scout.signals.heuristics import matches_any
from scout.store import Store
from scout.web import company_domain, domain_slug, normalize_site_url

_console = Console()

_BASE = "https://yc-oss.github.io/api"
_FETCH_TIMEOUT_S = 20.0
_UA = "Mozilla/5.0 (compatible; scout/0.1; +startup research; yc directory reader)"
_SEASONS = {"winter": 1, "spring": 2, "summer": 3, "fall": 4}
_BATCH_NAME = re.compile(r"^(winter|spring|summer|fall)\s+(\d{4})$", re.I)
# Companies that are no longer a seed-stage investment. "Active" is the
# directory's word for everything else, including not-yet-launched.
_DONE_STATUSES = {"acquired", "inactive", "public"}


def batch_slug(name: str) -> str:
    """"Summer 2025" → "summer-2025", the per-batch file name."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _batch_key(name: str) -> tuple[int, int] | None:
    match = _BATCH_NAME.match(name.strip())
    if not match:
        return None
    return int(match.group(2)), _SEASONS[match.group(1).lower()]


def parse_batches(meta: Any) -> list[str]:
    """(pure, tested) Batch names from meta.json, newest first. The file's
    exact shape is not load-bearing: names are taken from dict keys, from
    `name`/`batch` fields of listed objects, or from a list of strings —
    anything that reads as "<Season> <Year>"."""
    names: list[str] = []

    def consider(value: Any) -> None:
        if isinstance(value, str) and _batch_key(value):
            names.append(value.strip())

    if isinstance(meta, dict):
        if isinstance(meta.get("batches"), (dict, list)):
            return parse_batches(meta["batches"])
        for key, value in meta.items():
            consider(key)
            if isinstance(value, dict):
                consider(value.get("name"))
                consider(value.get("batch"))
    elif isinstance(meta, list):
        for item in meta:
            if isinstance(item, dict):
                consider(item.get("name"))
                consider(item.get("batch"))
            else:
                consider(item)
    unique = list(dict.fromkeys(names))
    unique.sort(key=lambda n: _batch_key(n) or (0, 0), reverse=True)
    return unique


def _text_of(record: dict[str, Any]) -> str:
    bits = [str(record.get(k) or "") for k in ("name", "one_liner", "long_description")]
    for key in ("tags", "industries"):
        value = record.get(key)
        if isinstance(value, list):
            bits.extend(str(v) for v in value)
        elif isinstance(value, str):
            bits.append(value)
    return " ".join(bits)


def parse_companies(
    records: list[dict[str, Any]],
    *,
    thesis: Thesis,
    now: datetime,
    batches: set[str] | None = None,
) -> tuple[list[Account], list[UnlinkedLead]]:
    """(pure, tested) Directory records → domain-keyed Accounts / UnlinkedLeads.

    Off-thesis companies are dropped here (heuristics.matches_any over the
    one-liner, description, tags and industries) so a 250-company batch
    does not spend classification budget on the sectors the thesis
    excludes; no thesis terms configured means everything passes and the
    classifier judges. Acquired / public / inactive companies are not
    seed-stage investments and are skipped outright.
    """
    terms = [*thesis.keywords, *thesis.sectors]
    accounts: dict[str, Account] = {}
    unlinked: list[UnlinkedLead] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = " ".join(str(record.get("name") or "").split())
        if not name:
            continue
        batch = str(record.get("batch") or "").strip()
        if batches and batch not in batches:
            continue
        if str(record.get("status") or "").strip().lower() in _DONE_STATUSES:
            continue
        if terms and not matches_any(_text_of(record), terms):
            continue
        one_liner = " ".join(str(record.get("one_liner") or "").split())
        bio = (f"{one_liner} — YC {batch}" if one_liner else f"YC {batch}").strip(" —")[:280]
        yc_url = str(record.get("url") or "").strip()
        website = str(record.get("website") or "").strip()
        domain = company_domain(website, "ycombinator.com") if website else None
        slug = domain_slug(domain) if domain else ""
        if slug:
            site = normalize_site_url(website)
            if slug not in accounts:
                accounts[slug] = Account(
                    id=f"yc:{record.get('slug') or slug}",
                    handle=slug,
                    name=name[:120],
                    bio=bio,
                    website=site,
                    # No X presence known: the profile link must point at the
                    # real site, never an invented x.com/<slug>.
                    profile_url=site,
                    source="yc",
                    fetched_at=now,
                )
            continue
        unlinked.append(UnlinkedLead(
            source="yc", ref=yc_url or f"yc:{record.get('slug') or name}",
            name=name[:120], bio=bio, url=yc_url or website, found_at=now,
        ))
    return list(accounts.values()), unlinked


class YCSource(DiscoverySource):
    """Reads the latest `settings.yc_batches` batches once a run. Free."""

    name = "yc"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> Any | None:
        """None on 404 (a batch file not yet published); anything else that
        is not a 200 is printed loudly — a blocked host must not read as
        an empty batch."""
        try:
            resp = await client.get(url)
        except Exception as exc:  # a dead mirror must never sink discovery
            _console.print(f"[yellow]yc: {url} failed: {type(exc).__name__}[/]")
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            _console.print(f"[yellow]yc: {url} → HTTP {resp.status_code}[/]")
            return None
        try:
            return resp.json()
        except ValueError:
            _console.print(f"[yellow]yc: {url} is not JSON[/]")
            return None

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        now = datetime.now(timezone.utc)
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S, follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            meta = await self._get_json(client, f"{_BASE}/meta.json")
            names = parse_batches(meta)[:max(self.settings.yc_batches, 0)]
            if not names:
                _console.print("[yellow]yc: no batches found in meta.json — "
                               "nothing read.[/]")
                return [], []
            records: list[dict[str, Any]] = []
            missing: list[str] = []
            for name in names:
                data = await self._get_json(client, f"{_BASE}/batches/{batch_slug(name)}.json")
                if isinstance(data, list):
                    records.extend(r for r in data if isinstance(r, dict))
                else:
                    missing.append(name)
            if missing:
                # A batch file the mirror has not split out yet: the full
                # directory, filtered by batch, says the same thing.
                everything = await self._get_json(client, f"{_BASE}/companies/all.json")
                if isinstance(everything, list):
                    wanted = set(missing)
                    records.extend(r for r in everything
                                   if isinstance(r, dict) and str(r.get("batch") or "") in wanted)
        await asyncio.sleep(0)
        accounts, unlinked = parse_companies(records, thesis=thesis, now=now, batches=set(names))
        self.store.upsert_accounts(accounts)
        _console.print(
            f"[dim]yc: {len(accounts)} companies bridged, {len(unlinked)} unlinked "
            f"from {', '.join(names)} ({len(records)} records).[/dim]"
        )
        return accounts, unlinked
