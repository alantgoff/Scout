"""Typer CLI entrypoint — orchestration only; all logic lives in the modules."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.progress_bar import ProgressBar
from rich.table import Table

from scout.config import (
    Seeds,
    Settings,
    Thesis,
    ensure_thesis_id,
    load_seeds,
    load_thesis,
    save_thesis,
    slugify,
    switch_thesis,
    thesis_version,
)
# `thesis_path` is a local variable name throughout this module (the --thesis
# option); alias the library-path helper so the two can never shadow.
from scout.config import thesis_path as thesis_library_path
from scout import web
from scout.export import print_top_table, write_csv, write_markdown
from scout.ingest.base import DiscoverySource, SourceAdapter
from scout.ingest.xapi_src import BudgetExceededError
from scout.models import Account, Lead, Signal, SitePage, Tweet, UnlinkedLead
from scout.score import score_breakdown, score_leads
from scout.signals.heuristics import (
    intent_appeared,
    run_heuristics,
    verdict_disqualified,
)
from scout.signals.llm import classify, verify_leads
from scout.store import Store

app = typer.Typer(
    help="scout — lightweight Twitter/X founder-sourcing engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


class Source(str, Enum):
    twscrape = "twscrape"
    xapi = "xapi"


class ExportFormat(str, Enum):
    md = "md"
    csv = "csv"
    both = "both"


# --------------------------------------------------------------------- helpers


def _build_adapter(source: Source, settings: Settings, store: Store) -> SourceAdapter:
    # Imported lazily so one adapter's missing extras never break the other path.
    if source is Source.xapi:
        from scout.ingest.xapi_src import XApiSource

        return XApiSource(settings, store)
    from scout.ingest.twscrape_src import TwscrapeSource

    return TwscrapeSource(settings, store)


def _load_thesis_or_exit(path: Path) -> Thesis:
    try:
        return load_thesis(path)
    except FileNotFoundError:
        console.print(f"[red]Thesis file not found:[/red] {path}")
        raise typer.Exit(1) from None
    except (yaml.YAMLError, ValidationError) as exc:
        console.print(f"[red]Could not parse {path}:[/red] {exc}")
        raise typer.Exit(1) from None


def _load_seeds_or_exit(path: Path) -> Seeds:
    try:
        return load_seeds(path)
    except FileNotFoundError:
        console.print(f"[red]Seeds file not found:[/red] {path}")
        raise typer.Exit(1) from None
    except (yaml.YAMLError, ValidationError) as exc:
        console.print(f"[red]Could not parse {path}:[/red] {exc}")
        raise typer.Exit(1) from None


def _load_seeds_or_default(path: Path = Path("seeds.yaml")) -> Seeds:
    """Best-effort seeds load for run provenance (verify/demo don't need seeds
    to work, but the strategy fingerprint wants the full config)."""
    try:
        return load_seeds(path)
    except Exception:
        return Seeds()


def _record_run(
    store: Store, run_id: str, source_name: str, thesis: Thesis, seeds: Seeds
) -> None:
    """Stamp run provenance: which thesis, at which version, produced this run.

    Registering the thesis here (rather than in a separate bookkeeping step)
    keeps the registry correct for free — every run declares the identity and
    version it ran under, which is what later makes staleness computable.
    """
    thesis_id = ensure_thesis_id(thesis)
    version = thesis_version(thesis, seeds)
    store.adopt_history(thesis_id, thesis.thesis)
    store.upsert_thesis(
        thesis_id,
        name=thesis.name or thesis_id,
        statement=thesis.thesis,
        version=version,
        make_active=True,
    )
    store.record_run(
        run_id,
        source=source_name,
        strategy_hash=version,
        thesis_statement=thesis.thesis,
        thesis_id=thesis_id,
        thesis_version=version,
        config_json=json.dumps(
            {
                "thesis": thesis.model_dump(mode="json"),
                "seeds": seeds.model_dump(mode="json"),
            },
            sort_keys=True,
        ),
    )


def _thesis_banner(thesis: Thesis, seeds: Seeds, store: Store) -> str:
    """`Novel Architectures · v3` — the header line for run/reclassify."""
    thesis_id = ensure_thesis_id(thesis)
    version = thesis_version(thesis, seeds)
    history = store.thesis_version_history(thesis_id)
    n = next((h["n"] for h in history if h["version"] == version), len(history) + 1)
    return f"{thesis.name or thesis_id} · v{n}"


async def _fetch_tweets(
    adapter: SourceAdapter,
    accounts: list[Account],
    limit: int,
    concurrency: int = 1,
    time_budget_s: float | None = None,
    store: Store | None = None,
) -> dict[str, list[Tweet]]:
    """Fetch recent tweets per account.

    Free adapters (parallel_safe) fan out under a semaphore — the single
    biggest wall-clock win on large runs — under an optional wall-clock
    budget: X rate-limits user timelines to ~50 reads per 15-minute window
    per account, and twscrape sleeps through the window. When the budget
    expires (or a call is cut short), remaining accounts fall back to the
    store's cached tweets from earlier runs instead of waiting. Callers
    should pass accounts strongest-first so the budget goes to the leads
    that matter. Paid adapters stay strictly sequential (and unbudgeted —
    the $ guard governs them) so the spend pre-check cannot be raced.
    """
    tweets_by_handle: dict[str, list[Tweet]] = {}
    fetched_n = 0
    last_scan_tick = 0.0

    def _scan_tick() -> None:
        # Store heartbeat for the UI progress bar — throttled to ~1 write/s.
        nonlocal fetched_n, last_scan_tick
        fetched_n += 1
        if store is None:
            return
        now = time.monotonic()
        if now - last_scan_tick >= 1.0 or fetched_n == len(accounts):
            last_scan_tick = now
            store.scan_update("fetching timelines",
                              f"{fetched_n}/{len(accounts)} timelines fetched",
                              done=fetched_n, total=len(accounts), unit="items")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Fetching tweets...", total=len(accounts))

        if adapter.parallel_safe and concurrency > 1 and len(accounts) > 1:
            semaphore = asyncio.Semaphore(concurrency)
            deadline = (
                time.monotonic() + time_budget_s if time_budget_s is not None else None
            )
            budget_announced = False

            def cached(account: Account) -> list[Tweet]:
                return store.get_tweets(account.id, limit) if store is not None else []

            def announce_budget() -> None:
                nonlocal budget_announced
                if not budget_announced:
                    budget_announced = True
                    console.print(
                        "[yellow]tweet-fetch time budget reached — using cached "
                        "tweets for the remaining accounts.[/yellow]"
                    )

            async def fetch_one(account: Account) -> None:
                async with semaphore:
                    remaining = (
                        deadline - time.monotonic() if deadline is not None else None
                    )
                    if remaining is not None and remaining <= 0:
                        announce_budget()
                        tweets_by_handle[account.handle] = cached(account)
                        progress.advance(task)
                        _scan_tick()
                        return
                    try:
                        coro = adapter.fetch_tweets(account, limit=limit)
                        tweets_by_handle[account.handle] = (
                            await asyncio.wait_for(coro, timeout=remaining)
                            if remaining is not None
                            else await coro
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        announce_budget()
                        tweets_by_handle[account.handle] = cached(account)
                    except Exception as exc:
                        console.print(
                            f"[yellow]tweet fetch failed for @{account.handle} "
                            f"({exc}) — using cached/bio signals only.[/yellow]"
                        )
                        tweets_by_handle[account.handle] = cached(account)
                    progress.advance(task)
                    _scan_tick()

            await asyncio.gather(*(fetch_one(a) for a in accounts))
            return tweets_by_handle

        for account in accounts:
            try:
                tweets_by_handle[account.handle] = await adapter.fetch_tweets(
                    account, limit=limit
                )
            except BudgetExceededError:
                console.print(
                    "[yellow]X API budget reached — scoring what we have "
                    "(skipping remaining tweet fetches).[/yellow]"
                )
                break
            except Exception as exc:
                console.print(
                    f"[yellow]tweet fetch failed for @{account.handle} "
                    f"({exc}) — scoring on bio signals only.[/yellow]"
                )
                tweets_by_handle[account.handle] = []
            progress.advance(task)
            _scan_tick()
    return tweets_by_handle


def _print_budget_line(store: Store, settings: Settings) -> None:
    console.print(
        f"X API spend: [bold]${store.xapi_spend_usd():.2f}[/bold] of "
        f"${settings.xapi_spend_cap_usd:.2f} cap"
    )


def _enrich_accounts(
    accounts: list[Account], store: Store, thesis: Thesis, settings: Settings
) -> None:
    """Fill enrichment fields from store history (in place):
    bio_changed (bio snapshot diff) and recent_followed_by (follow-edge diff)."""
    # One batched recency query for every account, instead of a follow-edge
    # scan per account (recent_watchers_map keys are lowercased followees).
    recent_map = store.recent_watchers_map(settings.recent_follow_days)
    for account in accounts:
        previous_bio = store.record_bio(account.handle, account.bio)
        account.bio_changed = intent_appeared(previous_bio, account.bio, thesis)
        account.recent_followed_by = recent_map.get(
            account.handle.lstrip("@").lower(), []
        )


def _discovery_sources(
    names: set[str], settings: Settings, store: Store
) -> list[DiscoverySource]:
    sources: list[DiscoverySource] = []
    if "github" in names:
        from scout.ingest.github_src import GitHubSource

        sources.append(GitHubSource(settings, store))
    if "hn" in names:
        from scout.ingest.hn_src import HackerNewsSource

        sources.append(HackerNewsSource(settings, store))
    return sources


def _run_discovery(
    names: set[str],
    seeds: Seeds,
    thesis: Thesis,
    settings: Settings,
    store: Store,
) -> tuple[list[Account], list[UnlinkedLead]]:
    """Run free supplementary discovery sources; failures never abort a run."""
    accounts: list[Account] = []
    unlinked: list[UnlinkedLead] = []
    for src in _discovery_sources(names, settings, store):
        try:
            with console.status(f"Discovering via {src.name}..."):
                found, missing = asyncio.run(src.discover(seeds, thesis))
            accounts.extend(found)
            unlinked.extend(missing)
        except Exception as exc:
            console.print(f"[yellow]{src.name} discovery failed ({exc}) — skipped.[/]")
    return accounts, unlinked


def _drop_product_disqualified(leads: list[Lead], thesis: Thesis) -> list[Lead]:
    """Drop leads the classified product disqualifies (thesis.
    product_disqualifiers), reporting what went and why.

    Belongs everywhere a verdict is produced — `run` and `reclassify` both —
    or reclassifying would quietly resurrect the crypto companies the run had
    already dropped. Call it BEFORE the audit so a dropped lead never costs an
    audit call.
    """
    hits = [(x, term) for x in leads if (term := verdict_disqualified(x.llm, thesis))]
    if not hits:
        return leads
    shown = ", ".join(f"@{x.account.handle} ({term})" for x, term in hits[:4])
    console.print(
        f"Dropped [bold]{len(hits)}[/bold] disqualified on the classified "
        f"product: {shown}{'…' if len(hits) > 4 else ''}."
    )
    dropped = {x.account.handle for x, _ in hits}
    return [x for x in leads if x.account.handle not in dropped]


def _rank_candidates(
    leads: list[Lead], thesis: Thesis, cap: int
) -> tuple[list[Lead], int]:
    """Leads worth Claude tokens, ranked by deterministic pre-score and
    capped. Returns (candidates, skipped_count). Pure — unit-tested.

    Signal-bearing leads rank first. Search-sourced leads with NO signal
    follow, because the deterministic signals describe a FOUNDER account —
    bio_intent reads a personal bio, departure_signal wants an ex-employer,
    smart_money_follow wants investors following a person — and the query
    bank now hunts COMPANY accounts, which fire none of them. A company
    posting "we're hiring ML engineers" from a bio-less account scores zero
    signals and is not thereby uninteresting: @BiggerMax19 (no bio, 0
    followers) classified at thesis fit 0.70 once it finally reached Claude.

    They are also already paid for. A search-sourced account cost a post
    read plus a user read to discover, and the query that surfaced it
    already encoded the thesis; dropping it before classification spends
    that money for nothing. Discovery-sourced (github/hn) leads without a
    signal stay excluded — those arrive in bulk and cost nothing to skip.
    """
    def pre_score(lead: Lead) -> float:
        return -score_breakdown(lead, thesis)[0][1]

    with_signals = sorted((x for x in leads if x.signals_hit), key=pre_score)
    paid_signalless = sorted(
        (x for x in leads if not x.signals_hit and x.account.source == "search"),
        key=pre_score,
    )
    ordered = with_signals + paid_signalless
    return ordered[:cap], max(len(ordered) - cap, 0)


def _fetch_candidate_sites(
    candidates: list[tuple[Account, list[Tweet]]],
    settings: Settings,
    store: Store,
) -> dict[str, SitePage]:
    """Lowercased handle → the candidate's company website, fetched — so the
    classifier grounds "what do they build" in real product copy instead of
    the founder's bio. Cache-first (websites table, TTL); failures are
    negative-cached AND returned so the prompt can say 'unreachable'."""
    targets: dict[str, str] = {}  # normalized root URL -> original URL
    by_handle: dict[str, str] = {}
    for account, _tweets in candidates:
        normalized = web.normalize_site_url(account.website)
        if normalized is None:
            continue
        targets.setdefault(normalized, account.website or "")
        by_handle[account.handle.lower()] = normalized
    if not targets:
        return {}
    urls = list(targets)
    console.print(f"Reading [bold]{len(urls)}[/bold] company websites...")
    store.scan_update("reading websites", f"0/{len(urls)} sites",
                      done=0, total=len(urls), unit="items")
    last_tick = 0.0

    def progress(done: int, total: int) -> None:
        # Throttled store heartbeat for the UI progress bar (~1 write/s).
        nonlocal last_tick
        now = time.monotonic()
        if done == total or now - last_tick >= 1.0:
            last_tick = now
            store.scan_update("reading websites", f"{done}/{total} sites read",
                              done=done, total=total, unit="items")

    pages = asyncio.run(web.fetch_sites(
        urls, settings, store=store, progress=progress,
        fallbacks={norm: orig for norm, orig in targets.items() if orig},
    ))
    return {handle: pages[url] for handle, url in by_handle.items() if url in pages}


def _verification_pass(
    leads: list[Lead],
    tweets_by_handle: dict[str, list[Tweet]],
    sites: dict[str, SitePage],
    thesis: Thesis,
    settings: Settings,
    store: Store,
) -> None:
    """Adversarial audit of the TOP leads' verdicts before final scoring —
    a second Claude pass that catches product claims sourced from founder
    pedigree instead of evidence. Mutates lead.llm in place; corrected
    verdicts are re-cached under the same fingerprint (llm.verify_leads)."""
    if settings.verify_top_n <= 0:
        return
    ranked = [x for x in score_leads(list(leads), thesis) if x.llm is not None]
    top = ranked[: settings.verify_top_n]
    if not top:
        return
    tweets_lower = {h.lower(): t for h, t in tweets_by_handle.items()}
    # The classifier may have surfaced a company_url the account row lacked —
    # fetch those sites (cache-first) so the audit sees the best evidence.
    extra: dict[str, str] = {}
    for lead in top:
        key = lead.account.handle.lower()
        if key in sites or lead.llm is None:
            continue
        url = web.normalize_site_url(lead.llm.company_url or lead.account.website)
        if url is not None:
            extra[key] = url
    if extra:
        pages = asyncio.run(web.fetch_sites(
            list(dict.fromkeys(extra.values())), settings, store=store))
        for key, url in extra.items():
            page = pages.get(url)
            if page is not None and page.usable:
                sites[key] = page
    console.print(f"Auditing [bold]{len(top)}[/bold] top verdicts against their evidence...")
    store.scan_update("verifying", f"0/{len(top)} verdicts audited",
                      done=0, total=len(top), unit="items")
    verify_leads(
        top, tweets_lower, sites, thesis, settings, store=store,
        progress=lambda done, total: store.scan_update(
            "verifying", f"{done}/{total} verdicts audited",
            done=done, total=total, unit="items"),
    )


def _merge_accounts(*groups: list[Account]) -> list[Account]:
    """Dedupe by handle; earlier groups win, followed_by / sources merge.

    `sources` accumulates every strategy that independently surfaced the
    account — the input to the source_corroboration signal.
    """
    merged: dict[str, Account] = {}
    for group in groups:
        for account in group:
            key = account.handle.lower()
            existing = merged.get(key)
            if existing is None:
                merged[key] = account
                if account.source and account.source not in account.sources:
                    account.sources.append(account.source)
            else:
                for src in [account.source, *account.sources]:
                    if src and src not in existing.sources:
                        existing.sources.append(src)
                for watcher in account.followed_by:
                    if watcher not in existing.followed_by:
                        existing.followed_by.append(watcher)
                existing.github_repo = existing.github_repo or account.github_repo
    return list(merged.values())


# ------------------------------------------------------------------------- run


def _run_pipeline(
    adapter: SourceAdapter,
    source: Source,
    settings: Settings,
    thesis: Thesis,
    seeds: Seeds,
    store: Store,
    max_accounts: int,
    min_score: float,
    ttl_days: int,
) -> None:
    console.print(f"Thesis: [bold]{_thesis_banner(thesis, seeds, store)}[/bold]")
    console.print(
        f"Targeting stages: [bold]{', '.join(thesis.target_stages)}[/bold] → "
        f"queries: {', '.join(sorted(thesis.active_search_categories))}"
        + (
            f" · discovery: {', '.join(sorted(thesis.active_discovery_sources))}"
            if thesis.active_discovery_sources
            else ""
        )
    )
    # One wall-clock budget spans the whole network phase (discovery legs
    # first, then tweet fetching gets whatever remains).
    network_deadline = time.monotonic() + settings.sourcing_time_budget_s
    store.scan_update("discovering", f"X {source.value} + github/hn legs",
                      total=settings.sourcing_time_budget_s, unit="s")
    with console.status(f"Fetching accounts via {source.value}..."):
        if source is Source.twscrape:
            strategies = {"lists", "searches"} | (
                {"bio", "graph"} if thesis.bio_graph_active else set()
            )
            accounts = asyncio.run(
                adapter.fetch_accounts(
                    seeds,
                    max_accounts=max_accounts,
                    strategies=strategies,
                    search_categories=thesis.active_search_categories,
                    recent_follow_days=settings.recent_follow_days,
                    time_budget_s=settings.sourcing_time_budget_s,
                )
            )
        else:
            accounts = asyncio.run(
                adapter.fetch_accounts(
                    seeds,
                    max_accounts=max_accounts,
                    search_categories=thesis.active_search_categories,
                )
            )

    # Free supplementary discovery, filtered by target stages — it costs
    # nothing; tweet fetching stays subject to the xapi bio gate below.
    discovered, unlinked = _run_discovery(
        thesis.active_discovery_sources, seeds, thesis, settings, store
    )
    accounts = _merge_accounts(accounts, discovered)
    console.print(f"Fetched [bold]{len(accounts)}[/bold] candidate accounts.")
    if unlinked:
        console.print(
            f"[dim]{len(unlinked)} unlinked leads (no X handle) recorded — "
            "see `scout source` output or the report appendix.[/dim]"
        )

    _enrich_accounts(accounts, store, thesis, settings)

    # One grouped query for the TTL skip, not one leads-table query per account.
    scored_at = store.last_scored_map()
    ttl_cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    fresh = [
        a for a in accounts
        if (last := scored_at.get(a.handle.lower())) is None or last <= ttl_cutoff
    ]
    skipped = len(accounts) - len(fresh)
    if skipped:
        console.print(
            f"Skipped [bold]{skipped}[/bold] accounts already scored within "
            f"{ttl_days} days (incremental run)."
        )

    # Cheap bio-only pass: disqualify before spending anything on tweets.
    bio_signals: dict[str, list[Signal]] = {}
    kept: list[Account] = []
    for account in fresh:
        signals, disqualified = run_heuristics(account, [], thesis)
        if disqualified:
            continue
        bio_signals[account.handle] = signals
        kept.append(account)
    dropped = len(fresh) - len(kept)
    if dropped:
        console.print(f"Dropped [bold]{dropped}[/bold] disqualified accounts (bio pass).")

    if source is Source.xapi:
        # Frugality tradeoff: every tweet read bills against the $25 grant, so
        # only accounts that already show >=1 bio-level signal earn a timeline
        # fetch; the rest are scored on bio signals alone (launch_traction=0).
        # The adapter is cache-first, so repeat runs cost nothing.
        to_fetch = [a for a in kept if any(s.value > 0 for s in bio_signals[a.handle])]
    else:
        to_fetch = kept

    tweets_by_handle: dict[str, list[Tweet]] = {}
    if to_fetch:
        # Strongest bio signals first — if the time budget cuts the phase
        # short, the timelines that matter most are already in.
        def _bio_pre_score(account: Account) -> float:
            probe = Lead(account=account, signals=bio_signals[account.handle])
            return score_breakdown(probe, thesis)[0][1]

        to_fetch.sort(key=_bio_pre_score, reverse=True)
        remaining_budget = max(network_deadline - time.monotonic(), 0.0)
        store.scan_update("fetching timelines",
                          f"{len(to_fetch)} accounts · {remaining_budget:.0f}s budget left",
                          done=0, total=len(to_fetch), unit="items")
        if source is Source.twscrape:
            # Belt AND suspenders: the per-call budget inside _fetch_tweets is
            # the mechanism; this outer cap is the guarantee. If anything in
            # the adapter stack ever swallows a cancellation again, the whole
            # phase is still severed here and the rest comes from cache.
            async def _capped_fetch() -> dict[str, list[Tweet]]:
                try:
                    return await asyncio.wait_for(
                        _fetch_tweets(
                            adapter, to_fetch, settings.tweets_per_account,
                            concurrency=settings.tweet_fetch_concurrency,
                            time_budget_s=remaining_budget, store=store,
                        ),
                        timeout=remaining_budget + 60,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    console.print(
                        "[yellow]tweet phase hit the hard cap — continuing "
                        "with cached tweets.[/yellow]"
                    )
                    return {}

            tweets_by_handle = asyncio.run(_capped_fetch())
            for account in to_fetch:  # cache fallback for anything unfetched
                if account.handle not in tweets_by_handle:
                    tweets_by_handle[account.handle] = store.get_tweets(
                        account.id, settings.tweets_per_account
                    )
        else:
            tweets_by_handle = asyncio.run(
                _fetch_tweets(
                    adapter, to_fetch, settings.tweets_per_account,
                    concurrency=settings.tweet_fetch_concurrency,
                    store=store,
                )
            )

    leads: list[Lead] = []
    for account in kept:
        tweets = tweets_by_handle.get(account.handle, [])
        signals, disqualified = run_heuristics(account, tweets, thesis)
        if disqualified:
            continue
        leads.append(Lead(account=account, signals=signals))

    # Spend Claude tokens on the most promising accounts first: rank by the
    # deterministic pre-score and cap (cache hits inside classify are free).
    ranked, skipped_llm = _rank_candidates(leads, thesis, settings.llm_max_candidates)
    if skipped_llm:
        console.print(
            f"[dim]Capping Claude classification to the top "
            f"{settings.llm_max_candidates} candidates by heuristic score "
            f"({skipped_llm} skipped — raise LLM_MAX_CANDIDATES to widen).[/dim]"
        )
    candidates = [
        (lead.account, tweets_by_handle.get(lead.account.handle, []))
        for lead in ranked
    ]
    sites: dict[str, SitePage] = {}
    if candidates:
        sites = _fetch_candidate_sites(candidates, settings, store)
        console.print(f"Classifying [bold]{len(candidates)}[/bold] candidates with Claude...")
        store.scan_update("classifying", f"{len(candidates)} candidates",
                          done=0, total=len(candidates), unit="items")
        verdicts = classify(
            candidates, thesis, settings, store=store,
            progress=lambda done, total: store.scan_update(
                "classifying", f"{done}/{total} classified by Claude",
                done=done, total=total, unit="items"),
            sites=sites,
        )
        for lead in leads:
            lead.llm = verdicts.get(lead.account.handle.lower())
        leads = _drop_product_disqualified(leads, thesis)

    _verification_pass(leads, tweets_by_handle, sites, thesis, settings, store)

    store.scan_update("scoring & saving")
    leads = score_leads(leads, thesis)
    if min_score > 0:
        before = len(leads)
        leads = [lead for lead in leads if lead.score >= min_score]
        if len(leads) < before:
            console.print(
                f"Filtered [bold]{before - len(leads)}[/bold] leads below "
                f"--min-score {min_score:g}."
            )
        for i, lead in enumerate(leads, start=1):  # re-rank after the cut
            lead.rank = i
    if not leads:
        console.print("[yellow]No leads survived scoring — nothing to export.[/yellow]")
        return

    # Microseconds keep back-to-back runs (cheap with a warm cache) from
    # sharing a run_id and upsert-merging into each other's lead set.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    store.save_leads(run_id, leads)
    _record_run(store, run_id, source.value, thesis, seeds)
    csv_path = write_csv(leads, settings.out_dir, thesis)
    md_path = write_markdown(leads, thesis, settings.out_dir)

    print_top_table(leads)
    console.print(f"Saved [bold]{len(leads)}[/bold] leads (run {run_id}).")
    console.print(f"CSV:    [bold]{csv_path}[/bold]")
    console.print(f"Report: [bold]{md_path}[/bold]")
    if source is Source.xapi:
        _print_budget_line(store, settings)


@app.command()
def run(
    source: Annotated[
        Source, typer.Option(help="Data source: twscrape (free) or xapi (paid).")
    ] = Source.twscrape,
    max_accounts: Annotated[
        int | None, typer.Option(help="Cap accounts ingested per run (default: settings).")
    ] = None,
    min_score: Annotated[
        float, typer.Option(help="Drop leads scoring below this (0-100).")
    ] = 0.0,
    ttl_days: Annotated[
        int | None,
        typer.Option(help="Skip accounts scored within N days (default: settings)."),
    ] = None,
    thesis_path: Annotated[
        Path, typer.Option("--thesis", help="Path to thesis.yaml.")
    ] = Path("thesis.yaml"),
    seeds_path: Annotated[
        Path, typer.Option("--seeds", help="Path to seeds.yaml.")
    ] = Path("seeds.yaml"),
    watch: Annotated[
        bool, typer.Option("--watch", help="Scheduled runs + Slack alerts (future hook).")
    ] = False,
) -> None:
    """Run the full sourcing pipeline: ingest -> signals -> LLM -> score -> export."""
    if watch:
        console.print("[yellow]--watch: not implemented — future hook.[/yellow]")
        raise typer.Exit()

    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    seeds = _load_seeds_or_exit(seeds_path)
    store = Store(settings.db_path)
    effective_max = max_accounts if max_accounts is not None else settings.max_accounts
    effective_ttl = ttl_days if ttl_days is not None else settings.ttl_days

    run_phases = ["discovering", "fetching timelines", "reading websites",
                  "classifying"]
    if settings.verify_top_n > 0:
        run_phases.append("verifying")
    run_phases.append("scoring & saving")
    store.scan_start("run", os.getpid(), phases=run_phases)
    ok = False
    try:
        adapter = _build_adapter(source, settings, store)
        _run_pipeline(
            adapter,
            source,
            settings,
            thesis,
            seeds,
            store,
            effective_max,
            min_score,
            effective_ttl,
        )
        ok = True
    except typer.Exit:
        raise
    except BudgetExceededError as exc:
        console.print(f"[red]X API spend cap reached:[/red] {exc}")
        _print_budget_line(store, settings)
        console.print("Raise XAPI_SPEND_CAP_USD in .env or switch to --source twscrape.")
        raise typer.Exit(1) from None
    except RuntimeError as exc:
        console.print(f"[red]Cannot run:[/red] {exc}")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None
    finally:
        store.scan_finish("done" if ok else "failed")


# ------------------------------------------------------------------ reclassify


@app.command()
def reclassify(
    top: Annotated[
        int | None,
        typer.Option(help="Only reclassify the top N leads (of the latest run, "
                          "or of the whole ledger with --all)."),
    ] = None,
    all_runs: Annotated[
        bool,
        typer.Option("--all", help="Reclassify EVERY startup across ALL runs (the "
                     "whole ledger), not just the latest run. Slower and costs more "
                     "(one Claude call per distinct startup), but corrects the entire "
                     "database in a single pass — use after a thesis or prompt change."),
    ] = False,
    skip_verify: Annotated[
        bool, typer.Option("--skip-verify", help="Skip the adversarial audit pass.")
    ] = False,
    stale_only: Annotated[
        bool,
        typer.Option("--stale-only", help="Only startups scored under an older "
                     "version of this thesis — the cheap way to bring the "
                     "database current after a tuning change."),
    ] = False,
    thesis_path: Annotated[
        Path, typer.Option("--thesis", help="Path to thesis.yaml.")
    ] = Path("thesis.yaml"),
) -> None:
    """Re-run classification + audit + scoring — no discovery, no timeline
    fetching. Cache-first everywhere (tweets, websites, unchanged verdicts),
    so this is the fast loop for iterating on the thesis or prompt: minutes
    and cents, not an hour. Defaults to the LATEST run's leads; --all covers
    every distinct startup across the whole ledger; --stale-only covers just
    what the current thesis has outgrown."""
    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    store = Store(settings.db_path)
    seeds = _load_seeds_or_default()
    console.print(f"Thesis: [bold]{_thesis_banner(thesis, seeds, store)}[/bold]")
    scope_label = "the whole ledger" if all_runs else "the latest run"
    if stale_only:
        all_runs, scope_label = True, "the stale set"
    prior = (
        [e.lead for e in store.load_lead_ledger(include_demo=False)]
        if all_runs else store.load_latest_leads()
    )
    if stale_only:
        stale = set(store.stale_handles(ensure_thesis_id(thesis)))
        if not stale:
            console.print("[green]Nothing stale[/green] — every startup is scored "
                          "under the current version of this thesis.")
            return
        prior = [x for x in prior if x.account.handle.lower() in stale]
        console.print(f"[bold]{len(prior)}[/bold] startup(s) scored under an older version.")
    if not prior:
        console.print("[red]No leads yet[/red] — run `scout run` first.")
        raise typer.Exit(1)
    if all(x.account.source == "demo" for x in prior):
        console.print(
            "[yellow]Only offline-demo leads found — reclassify works on real runs.[/yellow]"
        )
        raise typer.Exit(1)

    phases = ["preparing", "reading websites", "classifying"]
    if settings.verify_top_n > 0 and not skip_verify:
        phases.append("verifying")
    phases.append("scoring & saving")
    store.scan_start("reclassify", os.getpid(), phases=phases)
    ok = False
    try:
        store.scan_update("preparing", f"{len(prior)} leads from {scope_label}")
        console.print(
            f"Reclassifying [bold]{len(prior)}[/bold] leads from {scope_label} "
            "(no discovery — cached tweets and sites)."
        )
        # The store row may be fresher than the lead snapshot (a later run
        # can refresh bio/website) — prefer it.
        accounts = [store.get_account(x.account.handle) or x.account for x in prior]
        _enrich_accounts(accounts, store, thesis, settings)
        tweets_by_handle = {
            account.handle: store.get_tweets(account.id, settings.tweets_per_account)
            for account in accounts
        }
        leads: list[Lead] = []
        for account in accounts:
            signals, disqualified = run_heuristics(
                account, tweets_by_handle.get(account.handle, []), thesis
            )
            if not disqualified:
                leads.append(Lead(account=account, signals=signals))

        # --all means "correct everything": don't throttle to
        # llm_max_candidates the way a fresh run does. An explicit --top still
        # caps either scope.
        if top is not None:
            cap = top
        elif all_runs:
            cap = len(leads)
        else:
            cap = settings.llm_max_candidates
        ranked, _skipped = _rank_candidates(leads, thesis, cap)
        candidates = [
            (lead.account, tweets_by_handle.get(lead.account.handle, []))
            for lead in ranked
        ]
        sites: dict[str, SitePage] = {}
        if candidates:
            sites = _fetch_candidate_sites(candidates, settings, store)
            console.print(
                f"Classifying [bold]{len(candidates)}[/bold] candidates with Claude..."
            )
            store.scan_update("classifying", f"{len(candidates)} candidates",
                              done=0, total=len(candidates), unit="items")
            verdicts = classify(
                candidates, thesis, settings, store=store,
                progress=lambda done, total: store.scan_update(
                    "classifying", f"{done}/{total} classified by Claude",
                    done=done, total=total, unit="items"),
                sites=sites,
            )
            for lead in leads:
                lead.llm = verdicts.get(lead.account.handle.lower())
            leads = _drop_product_disqualified(leads, thesis)

        if not skip_verify:
            _verification_pass(leads, tweets_by_handle, sites, thesis, settings, store)

        store.scan_update("scoring & saving")
        leads = score_leads(leads, thesis)
        if not leads:
            console.print("[yellow]No leads survived — check disqualifiers.[/yellow]")
            ok = True
            return
        run_id = datetime.now(timezone.utc).strftime("reclass-%Y%m%d-%H%M%S-%f")
        store.save_leads(run_id, leads)
        _record_run(store, run_id, "reclassify", thesis, _load_seeds_or_default())
        csv_path = write_csv(leads, settings.out_dir, thesis)
        md_path = write_markdown(leads, thesis, settings.out_dir)
        print_top_table(leads)
        console.print(f"Saved [bold]{len(leads)}[/bold] reclassified leads (run {run_id}).")
        console.print(f"CSV:    [bold]{csv_path}[/bold]")
        console.print(f"Report: [bold]{md_path}[/bold]")
        ok = True
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None
    finally:
        store.scan_finish("done" if ok else "failed")


# --------------------------------------------------------------------- inspect


@app.command()
def inspect(
    handle: Annotated[str, typer.Argument(help="Account handle, with or without @.")],
    source: Annotated[
        Source, typer.Option(help="Data source: twscrape (free) or xapi (paid).")
    ] = Source.twscrape,
) -> None:
    """Score a single account and explain every signal (cache-first)."""
    handle = handle.lstrip("@").lower()
    settings = Settings()
    thesis = _load_thesis_or_exit(Path("thesis.yaml"))
    store = Store(settings.db_path)

    try:
        account = store.get_account(handle)
        adapter: SourceAdapter | None = None
        if account is None:
            adapter = _build_adapter(source, settings, store)
            with console.status(f"Looking up @{handle} via {source.value}..."):
                account = asyncio.run(adapter.fetch_account(handle))
        if account is None:
            console.print(f"[red]@{handle} not found.[/red]")
            raise typer.Exit(1)

        tweets = store.get_tweets(account.id, settings.tweets_per_account)
        if not tweets:
            if adapter is None:
                adapter = _build_adapter(source, settings, store)
            with console.status(f"Fetching tweets for @{handle}..."):
                tweets = asyncio.run(
                    adapter.fetch_tweets(account, limit=settings.tweets_per_account)
                )

        signals, disqualified = run_heuristics(account, tweets, thesis)
        lead = Lead(account=account, signals=signals, disqualified=disqualified)
        if not disqualified:
            sites = _fetch_candidate_sites([(account, tweets)], settings, store)
            if sites:
                site = sites[account.handle.lower()]
                console.print(f"[dim]website: {site.url} → {site.status} "
                              f"({len(site.text)} chars extracted)[/dim]")
            verdicts = classify([(account, tweets)], thesis, settings,
                                store=store, sites=sites)
            lead.llm = verdicts.get(account.handle.lower())
        lead = score_leads([lead], thesis)[0]

        console.print(
            f"\n[bold]@{account.handle}[/bold] — {account.name} "
            f"({account.followers:,} followers, {len(tweets)} cached tweets)"
        )
        if account.bio:
            console.print(f"[dim]{account.bio}[/dim]")
        console.print(account.url)

        table = Table(title="Signal breakdown")
        table.add_column("signal")
        table.add_column("value", justify="right")
        table.add_column("weight", justify="right")
        table.add_column("contribution", justify="right")
        table.add_column("detail", overflow="fold")
        for s in lead.signals:
            table.add_row(
                s.name, f"{s.value:.2f}", f"{s.weight:g}", f"{s.contribution:.1f}", s.detail
            )
        console.print(table)

        if disqualified:
            console.print(
                "[red bold]DISQUALIFIED[/red bold] — would be dropped in a full run."
            )
        if lead.llm is not None:
            v = lead.llm
            verdict_table = Table(title="LLM verdict")
            verdict_table.add_column("field")
            verdict_table.add_column("value", overflow="fold")
            verdict_table.add_row("is_founder", str(v.is_founder))
            verdict_table.add_row("stage", v.stage or "-")
            verdict_table.add_row("sector", v.sector or "-")
            verdict_table.add_row("subsector", v.subsector or "-")
            verdict_table.add_row("business_model", v.business_model or "-")
            verdict_table.add_row("product_summary", v.product_summary or "-")
            verdict_table.add_row("grounding", v.grounding or "-")
            verdict_table.add_row("thesis_fit",
                                  f"{v.thesis_fit:.2f}" if v.thesis_fit is not None else "-")
            verdict_table.add_row("fit_reason", v.fit_reason or "-")
            verdict_table.add_row("one_line_summary", v.one_line_summary)
            verdict_table.add_row("why_interesting", v.why_interesting)
            verdict_table.add_row("confidence", f"{v.confidence:.2f}")
            console.print(verdict_table)
        elif not disqualified:
            console.print("[dim]No LLM verdict (missing API key).[/dim]")

        console.print(f"Final score: [bold]{lead.score:.1f}[/bold] / 100")
    except typer.Exit:
        raise
    except BudgetExceededError as exc:
        console.print(f"[red]X API spend cap reached:[/red] {exc}")
        _print_budget_line(store, settings)
        raise typer.Exit(1) from None
    except RuntimeError as exc:
        console.print(f"[red]Cannot inspect:[/red] {exc}")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


# ---------------------------------------------------------------------- export


@app.command()
def export(
    fmt: Annotated[
        ExportFormat,
        typer.Option("--format", help="Export format: md, csv, or both."),
    ] = ExportFormat.md,
    pipeline: Annotated[
        bool,
        typer.Option(
            "--pipeline",
            help="Export the deal-flow pipeline (status, notes, outreach, briefs) "
            "instead of the last run.",
        ),
    ] = False,
) -> None:
    """Re-export the most recent run (or, with --pipeline, the deal flow)."""
    settings = Settings()
    store = Store(settings.db_path)

    if pipeline:
        from scout.export import pipeline_rows, write_pipeline_csv

        try:
            thesis = load_thesis(Path("thesis.yaml"))
        except OSError:
            thesis = None
        rows = pipeline_rows(store, thesis)
        if not rows:
            console.print("[yellow]Pipeline is empty — shortlist leads first.[/yellow]")
            raise typer.Exit()
        path = write_pipeline_csv(rows, settings.out_dir)
        console.print(f"Exported [bold]{len(rows)}[/bold] pipeline rows: [bold]{path}[/bold]")
        return

    leads = store.load_latest_leads()
    if not leads:
        console.print("[yellow]No previous runs found — try `scout run` first.[/yellow]")
        raise typer.Exit()

    paths: list[Path] = []
    thesis = _load_thesis_or_exit(Path("thesis.yaml"))
    if fmt in (ExportFormat.csv, ExportFormat.both):
        paths.append(write_csv(leads, settings.out_dir, thesis))
    if fmt in (ExportFormat.md, ExportFormat.both):
        paths.append(write_markdown(leads, thesis, settings.out_dir))
    rendered = ", ".join(f"[bold]{p}[/bold]" for p in paths)
    console.print(f"Re-exported [bold]{len(leads)}[/bold] leads: {rendered}")


# ---------------------------------------------------------------------- budget


@app.command()
def budget() -> None:
    """Show cumulative X API spend against the hard cap."""
    settings = Settings()
    store = Store(settings.db_path)
    spent = store.xapi_spend_usd()
    cap = settings.xapi_spend_cap_usd
    remaining = max(cap - spent, 0.0)
    reads_left = int(remaining / settings.xapi_cost_per_post_read)

    console.print(f"X API budget: [bold]${spent:.2f}[/bold] spent of ${cap:.2f} cap")
    console.print(ProgressBar(total=cap, completed=min(spent, cap), width=40))
    console.print(
        f"\n~[bold]{reads_left:,}[/bold] tweet reads remaining "
        f"@ ${settings.xapi_cost_per_post_read}/post"
    )
    console.print(f"Ledger: [dim]{store.db_path.resolve()}[/dim]")


# ---------------------------------------------------------------------- source


_X_STRATEGIES = {"lists", "searches", "bio", "graph"}
_DISCOVERY_STRATEGIES = {"github", "hn"}


@app.command("source")
def source_preview(
    strategy: Annotated[
        str,
        typer.Option(
            help="Comma-separated subset of: lists,searches,bio,graph,github,hn "
            "(default: all)."
        ),
    ] = "all",
    max_accounts: Annotated[
        int | None, typer.Option(help="Cap X-sourced accounts (default: settings).")
    ] = None,
    thesis_path: Annotated[Path, typer.Option("--thesis")] = Path("thesis.yaml"),
    seeds_path: Annotated[Path, typer.Option("--seeds")] = Path("seeds.yaml"),
) -> None:
    """Preview DISCOVERY only: which accounts each strategy finds, raw.

    No scoring, no LLM, no exports — the recall/precision eyeball tool.
    Free: twscrape + GitHub + HN (no paid X API calls).
    """
    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    seeds = _load_seeds_or_exit(seeds_path)
    store = Store(settings.db_path)
    effective_max = max_accounts if max_accounts is not None else settings.max_accounts

    if strategy == "all":
        # Stage-aware default: only the strategies that fit thesis.target_stages.
        chosen = {"lists", "searches"} | (
            {"bio", "graph"} if thesis.bio_graph_active else set()
        )
        chosen |= thesis.active_discovery_sources
        console.print(
            f"Targeting stages: [bold]{', '.join(thesis.target_stages)}[/bold] → "
            f"strategies: {', '.join(sorted(chosen))}"
        )
    else:
        chosen = {s.strip() for s in strategy.split(",") if s.strip()}
    unknown = chosen - _X_STRATEGIES - _DISCOVERY_STRATEGIES
    if unknown:
        console.print(f"[red]Unknown strategies:[/red] {', '.join(sorted(unknown))}")
        raise typer.Exit(1)

    store.scan_start("source preview", os.getpid(), phases=["discovering"])
    store.scan_update("discovering", ", ".join(sorted(chosen)))
    x_accounts: list[Account] = []
    x_chosen = chosen & _X_STRATEGIES
    if x_chosen:
        try:
            from scout.ingest.twscrape_src import TwscrapeSource

            adapter = TwscrapeSource(settings, store)
            with console.status(f"X discovery ({', '.join(sorted(x_chosen))})..."):
                x_accounts = asyncio.run(
                    adapter.fetch_accounts(
                        seeds,
                        max_accounts=effective_max,
                        strategies=x_chosen,
                        search_categories=thesis.active_search_categories,
                        recent_follow_days=settings.recent_follow_days,
                        time_budget_s=settings.sourcing_time_budget_s,
                    )
                )
        except RuntimeError as exc:
            console.print(f"[yellow]X strategies skipped:[/yellow] {exc}")
        except Exception as exc:
            console.print(f"[yellow]X discovery failed ({exc}) — continuing.[/yellow]")

    discovered, unlinked = _run_discovery(
        chosen & _DISCOVERY_STRATEGIES, seeds, thesis, settings, store
    )
    accounts = _merge_accounts(x_accounts, discovered)
    _enrich_accounts(accounts, store, thesis, settings)

    if not accounts and not unlinked:
        console.print(
            "[yellow]Nothing discovered — check credentials/seeds, or loosen queries.[/yellow]"
        )
        store.scan_finish("done", "nothing discovered")
        raise typer.Exit()

    by_strategy: dict[str, int] = {}
    for account in accounts:
        by_strategy[account.source] = by_strategy.get(account.source, 0) + 1
    console.print(
        "Found via: "
        + "  ·  ".join(f"[bold]{k}[/bold] {v}" for k, v in sorted(by_strategy.items()))
    )

    table = Table(title=f"Discovered accounts ({len(accounts)})")
    table.add_column("via")
    table.add_column("handle")
    table.add_column("name", overflow="fold")
    table.add_column("bio", overflow="fold", max_width=60)
    table.add_column("new follows", overflow="fold")
    for account in sorted(accounts, key=lambda a: (a.source, a.handle.lower())):
        table.add_row(
            account.source,
            f"@{account.handle}",
            account.name,
            " ".join(account.bio.split())[:120],
            ", ".join(account.recent_followed_by),
        )
    console.print(table)

    if unlinked:
        missing = Table(title=f"Unlinked leads — no X handle ({len(unlinked)})")
        missing.add_column("via")
        missing.add_column("ref")
        missing.add_column("what", overflow="fold", max_width=60)
        missing.add_column("url", overflow="fold")
        for lead in unlinked:
            missing.add_row(lead.source, lead.ref, lead.bio[:100], lead.url)
        console.print(missing)

    console.print(
        "[dim]Preview only — nothing scored or exported. "
        "Run `scout run` for the full pipeline.[/dim]"
    )
    store.scan_finish("done")


# ---------------------------------------------------------------------- verify


@app.command()
def verify(
    max_leads: Annotated[
        int, typer.Option("--max", help="How many top leads to hydrate.")
    ] = 50,
    tweets_each: Annotated[
        int, typer.Option("--tweets", help="Tweets to pull per account (paid).")
    ] = 10,
    min_score: Annotated[
        float, typer.Option(help="Only hydrate leads at/above this score.")
    ] = 0.0,
    discovered: Annotated[
        bool,
        typer.Option(
            "--discovered",
            help="Hydrate recently-DISCOVERED accounts (github/hn/twscrape finds) "
            "instead of the latest scored run.",
        ),
    ] = False,
) -> None:
    """Hydrate the current shortlist with FRESH paid X API data and re-score.

    The precision instrument at the end of the funnel: discovery stays free,
    only the shortlist costs money (~$0.01/profile + $0.005/tweet, hard-capped
    by XAPI_SPEND_CAP_USD).
    """
    settings = Settings()
    thesis = _load_thesis_or_exit(Path("thesis.yaml"))
    store = Store(settings.db_path)

    if discovered:
        shortlist = store.recent_discovered_accounts(days=7)[:max_leads]
        leads = [Lead(account=account) for account in shortlist]
        if not leads:
            console.print(
                "[yellow]No recently discovered accounts — run `scout source` first.[/yellow]"
            )
            raise typer.Exit(1)
    else:
        leads = [
            lead
            for lead in store.load_latest_leads()
            if lead.score >= min_score and not lead.disqualified
        ][:max_leads]
        if leads and all(lead.account.source == "demo" for lead in leads):
            console.print(
                "[yellow]Latest run is the demo (fake handles) — refusing to spend "
                "money on it. Use `scout verify --discovered` or run a real "
                "`scout run` first.[/yellow]"
            )
            raise typer.Exit(1)
        if not leads:
            console.print(
                "[yellow]No shortlist to verify — run `scout run` (or `scout source` "
                "+ `scout run`) first.[/yellow]"
            )
            raise typer.Exit(1)

    est = len(leads) * (
        settings.xapi_cost_per_user_read
        + tweets_each * settings.xapi_cost_per_post_read
    )
    console.print(
        f"Hydrating [bold]{len(leads)}[/bold] leads via X API "
        f"(worst case ≈ [bold]${est:.2f}[/bold], cap ${settings.xapi_spend_cap_usd:.2f})."
    )

    verify_phases = ["hydrating", "reading websites", "classifying"]
    if settings.verify_top_n > 0:
        verify_phases.append("verifying")
    verify_phases.append("scoring & saving")
    store.scan_start("verify", os.getpid(), phases=verify_phases)
    ok = False
    try:
        from scout.ingest.xapi_src import XApiSource

        adapter = XApiSource(settings, store)
        store.scan_update("hydrating", f"{len(leads)} leads via paid X API",
                          done=0, total=len(leads), unit="items")

        async def hydrate() -> list[tuple[Account, list[Tweet]]]:
            out: list[tuple[Account, list[Tweet]]] = []
            for i, lead in enumerate(leads, start=1):
                stale = lead.account
                store.scan_update("hydrating", f"{i}/{len(leads)} leads hydrated",
                                  done=i, total=len(leads), unit="items")
                try:
                    account = await adapter.fetch_account(stale.handle, fresh=True)
                    if account is None:
                        console.print(
                            f"[yellow]@{stale.handle} not found — skipped.[/yellow]"
                        )
                        continue
                    # Preserve discovery provenance on the fresh record.
                    account.source = stale.source or account.source
                    account.github_repo = stale.github_repo
                    for watcher in stale.followed_by:
                        if watcher not in account.followed_by:
                            account.followed_by.append(watcher)
                    tweets = await adapter.fetch_tweets(account, limit=tweets_each)
                    out.append((account, tweets))
                except BudgetExceededError:
                    console.print(
                        "[yellow]Budget cap reached — verifying what we have.[/yellow]"
                    )
                    break
                except Exception as exc:
                    console.print(
                        f"[yellow]@{stale.handle} hydration failed ({exc}) — skipped.[/yellow]"
                    )
            return out

        with console.status("Hydrating shortlist..."):
            hydrated = asyncio.run(hydrate())
        if not hydrated:
            console.print("[red]Nothing hydrated — check token/budget.[/red]")
            raise typer.Exit(1)

        accounts = [account for account, _ in hydrated]
        _enrich_accounts(accounts, store, thesis, settings)
        tweets_by_handle = {
            account.handle.lower(): tweets for account, tweets in hydrated
        }
        fresh_leads: list[Lead] = []
        for account, tweets in hydrated:
            signals, disqualified = run_heuristics(account, tweets, thesis)
            if not disqualified:
                fresh_leads.append(Lead(account=account, signals=signals))

        candidates = [
            (lead.account, tweets_by_handle.get(lead.account.handle.lower(), []))
            for lead in fresh_leads
            if lead.signals_hit
        ]
        v_sites: dict[str, SitePage] = {}
        if candidates:
            v_sites = _fetch_candidate_sites(candidates, settings, store)
            console.print(
                f"Classifying [bold]{len(candidates)}[/bold] candidates with Claude..."
            )
            store.scan_update("classifying", f"{len(candidates)} candidates",
                              done=0, total=len(candidates), unit="items")
            verdicts = classify(
                candidates, thesis, settings, store=store,
                progress=lambda done, total: store.scan_update(
                    "classifying", f"{done}/{total} classified by Claude",
                    done=done, total=total, unit="items"),
                sites=v_sites,
            )
            for lead in fresh_leads:
                lead.llm = verdicts.get(lead.account.handle.lower())

        _verification_pass(fresh_leads, tweets_by_handle, v_sites,
                           thesis, settings, store)

        store.scan_update("scoring & saving")
        fresh_leads = score_leads(fresh_leads, thesis)
        run_id = datetime.now(timezone.utc).strftime("verify-%Y%m%d-%H%M%S-%f")
        store.save_leads(run_id, fresh_leads)
        _record_run(store, run_id, "xapi", thesis, _load_seeds_or_default())
        csv_path = write_csv(fresh_leads, settings.out_dir, thesis)
        md_path = write_markdown(fresh_leads, thesis, settings.out_dir)
        print_top_table(fresh_leads)
        console.print(f"Verified [bold]{len(fresh_leads)}[/bold] leads (run {run_id}).")
        console.print(f"CSV: [bold]{csv_path}[/bold]  Report: [bold]{md_path}[/bold]")
        _print_budget_line(store, settings)
        ok = True
    except typer.Exit:
        raise
    except BudgetExceededError as exc:
        console.print(f"[red]X API spend cap reached:[/red] {exc}")
        _print_budget_line(store, settings)
        raise typer.Exit(1) from None
    except RuntimeError as exc:
        console.print(f"[red]Cannot verify:[/red] {exc}")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None
    finally:
        store.scan_finish("done" if ok else "failed")


# -------------------------------------------------------------------- strategy


@app.command()
def strategy(
    description: Annotated[
        str, typer.Argument(help="Your thesis in plain language, quoted.")
    ],
    apply: Annotated[
        bool, typer.Option("--apply", help="Write the proposal to thesis.yaml + seeds.yaml.")
    ] = False,
    thesis_path: Annotated[Path, typer.Option("--thesis")] = Path("thesis.yaml"),
    seeds_path: Annotated[Path, typer.Option("--seeds")] = Path("seeds.yaml"),
) -> None:
    """AI strategy agent: thesis in plain language -> full sourcing config.

    Prints the proposed targeting, query bank, and watchlist with the agent's
    rationale. Add --apply to write it (weights/params/prompt are untouched).
    """
    from scout.agents import apply_strategy, generate_strategy, validate_watchlist

    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    seeds = _load_seeds_or_exit(seeds_path)
    store = Store(settings.db_path)

    try:
        with console.status("Designing sourcing strategy with Claude..."):
            proposal = generate_strategy(description, thesis, seeds, settings)
    except RuntimeError as exc:
        console.print(f"[red]Cannot generate strategy:[/red] {exc}")
        raise typer.Exit(1) from None

    with console.status("Validating watchlist handles (free, via twscrape)..."):
        _, invalid_handles, validated = validate_watchlist(
            proposal.watchlist, settings, store
        )
    invalid_set = set(invalid_handles)

    console.print(f"\n[bold]Thesis:[/bold] {proposal.thesis}")
    console.print(f"[dim]{proposal.rationale}[/dim]\n")
    table = Table(title="Proposed sourcing strategy")
    table.add_column("field")
    table.add_column("proposal", overflow="fold")
    rows = [
        ("target_stages", ", ".join(proposal.target_stages)),
        ("keywords", ", ".join(proposal.keywords)),
        ("target_bios", ", ".join(proposal.target_bios)),
        ("sectors", ", ".join(proposal.sectors)),
        ("disqualifiers", ", ".join(proposal.disqualifiers)),
        ("searches_departure", "\n".join(proposal.searches_departure)),
        ("searches_stealth_intent", "\n".join(proposal.searches_stealth_intent)),
        ("searches_hiring", "\n".join(proposal.searches_hiring)),
        ("searches_launch", "\n".join(proposal.searches_launch)),
        ("bio_searches", ", ".join(proposal.bio_searches)),
        ("github_topics", ", ".join(proposal.github_topics)),
        ("watchlist", ", ".join(
            f"[red strike]{h}[/red strike]" if h in invalid_set else h
            for h in proposal.watchlist
        )),
    ]
    for field, value in rows:
        table.add_row(field, value or "[dim](keep current)[/dim]")
    console.print(table)
    if invalid_set:
        console.print(
            f"[yellow]{len(invalid_set)} watchlist handle(s) not found on X "
            f"(struck through) — dropped on --apply.[/yellow]"
        )
    elif not validated:
        console.print(
            "[dim]Watchlist handles not validated — configure twscrape cookies "
            "to check they exist.[/dim]"
        )

    if apply:
        from scout.config import save_seeds, save_thesis

        if invalid_set:
            proposal = proposal.model_copy(update={
                "watchlist": [h for h in proposal.watchlist if h not in invalid_set]
            })
            console.print(
                f"Dropped [bold]{len(invalid_set)}[/bold] unresolvable handles: "
                + ", ".join(sorted(invalid_set))
            )
        new_thesis, new_seeds = apply_strategy(proposal, thesis, seeds)
        save_thesis(new_thesis, thesis_path)
        save_seeds(new_seeds, seeds_path)
        console.print(
            f"Applied to [bold]{thesis_path}[/bold] + [bold]{seeds_path}[/bold]. "
            "Run `scout source` to preview discovery."
        )
    else:
        console.print("[dim]Preview only — re-run with --apply to write the config.[/dim]")


# --------------------------------------------------------------------- publish


@app.command()
def publish(
    push: Annotated[
        bool,
        typer.Option("--push", help="Commit docs/ and push — updates GitHub Pages."),
    ] = False,
    thesis_path: Annotated[Path, typer.Option("--thesis")] = Path("thesis.yaml"),
) -> None:
    """Render the phone digest (docs/index.html) for GitHub Pages.

    A read-only, mobile-first snapshot of the deal flow: launched startups
    grouped by company, the pre-launch watchlist, briefs. Lead data only —
    never secrets or config.
    """
    import subprocess as sp

    from scout.publish import build_digest

    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    store = Store(settings.db_path)
    docs = Path("docs")
    path = build_digest(store, thesis, docs)
    console.print(f"Digest written: [bold]{path}[/bold]")
    if not push:
        console.print("[dim]Re-run with --push to publish it to GitHub Pages.[/dim]")
        return
    if not settings.digest_repo:
        console.print(
            "[red]DIGEST_REPO is not set in .env[/red] — point it at a PUBLIC "
            "repo (e.g. https://github.com/you/scout-digest.git). The digest "
            "repo is separate from the code repo on purpose: only the rendered "
            "page goes public."
        )
        raise typer.Exit(1)

    def git(*args: str) -> sp.CompletedProcess:
        return sp.run(["git", "-C", str(docs), *args], capture_output=True, text=True)

    # docs/ is its own tiny checkout of the digest repo (ignored by the code repo).
    if not (docs / ".git").exists():
        git("init", "-b", "main")
        git("remote", "add", "origin", settings.digest_repo)
    git("add", "-A")
    commit = git("commit", "-m", "Update digest")
    if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr):
        console.print("Digest unchanged — nothing to push.")
        return
    pushed = git("push", "-u", "origin", "main")
    if pushed.returncode != 0:
        console.print(f"[red]git push failed:[/red] {pushed.stderr.strip()}")
        raise typer.Exit(1)
    console.print("Pushed — GitHub Pages refreshes in about a minute.")


# ----------------------------------------------------------------------- probe


@app.command()
def probe(
    list_id: Annotated[
        str | None,
        typer.Option(help="Optionally probe list-member billing with this List ID."),
    ] = None,
) -> None:
    """Empirically settle the unverified X API facts (~$0.20–0.50, budget-guarded).

    Checks: auth works; /2/users/search availability on pay-per-use; recent
    search accepts our departure queries; engagement operators are rejected.
    Every call is recorded in the spend ledger.
    """
    import httpx as _httpx

    settings = Settings()
    store = Store(settings.db_path)
    if not settings.x_bearer_token:
        console.print("[red]X_BEARER_TOKEN not set — cannot probe.[/red]")
        raise typer.Exit(1)

    headers = {"Authorization": f"Bearer {settings.x_bearer_token}"}
    results: list[tuple[str, str]] = []

    def guard(worst_case: float) -> None:
        spent = store.xapi_spend_usd()
        if spent + worst_case > settings.xapi_spend_cap_usd:
            raise BudgetExceededError(
                f"probe would exceed cap (${spent:.2f} + ${worst_case:.2f} "
                f"> ${settings.xapi_spend_cap_usd:.2f})"
            )

    def record(endpoint: str, posts: int, users: int) -> None:
        cost = (
            posts * settings.xapi_cost_per_post_read
            + users * settings.xapi_cost_per_user_read
        )
        store.record_xapi_usage(endpoint, posts, users, cost)

    try:
        with _httpx.Client(
            base_url="https://api.x.com/2", headers=headers, timeout=20
        ) as client:
            # 1. auth + user read
            guard(settings.xapi_cost_per_user_read)
            r = client.get("/users/by", params={"usernames": "jack"})
            record("probe:users/by", 0, len(r.json().get("data", [])) if r.is_success else 0)
            results.append(
                ("auth + GET /2/users/by", "OK" if r.is_success else f"HTTP {r.status_code}")
            )
            if not r.is_success:
                raise RuntimeError(f"auth failed: HTTP {r.status_code} — {r.text[:200]}")

            # 2. users/search availability (the big unknown)
            guard(10 * settings.xapi_cost_per_user_read)
            r = client.get("/users/search", params={"query": "stealth", "max_results": 10})
            record("probe:users/search", 0, len(r.json().get("data", [])) if r.is_success else 0)
            results.append(
                (
                    "GET /2/users/search (people/bio search)",
                    "AVAILABLE on this tier ✓"
                    if r.is_success
                    else f"NOT available (HTTP {r.status_code})",
                )
            )

            # 3. departure query on recent search
            guard(10 * settings.xapi_cost_per_post_read)
            r = client.get(
                "/tweets/search/recent",
                params={"query": '"some personal news" openai', "max_results": 10},
            )
            record(
                "probe:search/recent", len(r.json().get("data", [])) if r.is_success else 0, 0
            )
            results.append(
                (
                    "recent search: departure query",
                    f"OK ({len(r.json().get('data', []))} hits)"
                    if r.is_success
                    else f"HTTP {r.status_code}",
                )
            )

            # 4. engagement operator should be REJECTED (free to confirm)
            r = client.get(
                "/tweets/search/recent",
                params={"query": "stealth min_faves:20", "max_results": 10},
            )
            if r.is_success:
                record(
                    "probe:min_faves", len(r.json().get("data", [])), 0
                )
            results.append(
                (
                    "min_faves: operator",
                    "REJECTED as expected (web-only) ✓"
                    if r.status_code == 400
                    else f"unexpectedly HTTP {r.status_code}",
                )
            )

            # 5. optional list-member billing probe
            if list_id:
                guard(5 * settings.xapi_cost_per_user_read)
                r = client.get(f"/lists/{list_id}/members", params={"max_results": 5})
                record(
                    "probe:list-members",
                    0,
                    len(r.json().get("data", [])) if r.is_success else 0,
                )
                results.append(
                    (
                        f"GET /2/lists/{list_id}/members",
                        "OK" if r.is_success else f"HTTP {r.status_code}",
                    )
                )

        table = Table(title="X API probe results")
        table.add_column("check")
        table.add_column("result", overflow="fold")
        for name, verdict in results:
            table.add_row(name, verdict)
        console.print(table)
        console.print(
            "[dim]Costs are worst-case estimates recorded in the ledger; "
            "compare against the X dev console for exact billing.[/dim]"
        )
        _print_budget_line(store, settings)
    except typer.Exit:
        raise
    except BudgetExceededError as exc:
        console.print(f"[red]X API spend cap reached:[/red] {exc}")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from None


@app.command()
def demo(
    thesis_path: Annotated[Path, typer.Option("--thesis")] = Path("thesis.yaml"),
) -> None:
    """Score a set of built-in SAMPLE founders — a $0, offline end-to-end test.

    Seeds synthetic accounts into the cache and runs the full heuristics ->
    Claude -> score -> export pipeline with no live X calls. Uses your Anthropic
    key if set (a few cents); otherwise runs heuristics-only.
    """
    from scout.demo_data import sample_data

    settings = Settings()
    thesis = _load_thesis_or_exit(thesis_path)
    store = Store(settings.db_path)

    accounts, tweets = sample_data(datetime.now(timezone.utc))
    for account in accounts:
        store.upsert_account(account)
    store.upsert_tweets(tweets)
    tweets_by_handle: dict[str, list[Tweet]] = {}
    for tweet in tweets:
        acct = next(a for a in accounts if a.id == tweet.account_id)
        tweets_by_handle.setdefault(acct.handle, []).append(tweet)

    console.print(
        f"[dim]Seeded {len(accounts)} sample accounts (source=demo) — "
        f"no X API calls, $0 spent.[/dim]"
    )

    leads: list[Lead] = []
    for account in accounts:
        acct_tweets = tweets_by_handle.get(account.handle, [])
        signals, disqualified = run_heuristics(account, acct_tweets, thesis)
        if disqualified:
            console.print(f"[dim]Dropped @{account.handle} (disqualified).[/dim]")
            continue
        leads.append(Lead(account=account, signals=signals))

    candidates = [
        (lead.account, tweets_by_handle.get(lead.account.handle, []))
        for lead in leads
        if lead.signals_hit
    ]
    if candidates:
        console.print(f"Classifying [bold]{len(candidates)}[/bold] candidates with Claude...")
        verdicts = classify(candidates, thesis, settings, store=store)
        for lead in leads:
            lead.llm = verdicts.get(lead.account.handle.lower())

    leads = score_leads(leads, thesis)
    run_id = datetime.now(timezone.utc).strftime("demo-%Y%m%d-%H%M%S-%f")
    store.save_leads(run_id, leads)
    _record_run(store, run_id, "demo", thesis, _load_seeds_or_default())
    csv_path = write_csv(leads, settings.out_dir, thesis)
    md_path = write_markdown(leads, thesis, settings.out_dir)

    print_top_table(leads)
    console.print(f"\nSaved [bold]{len(leads)}[/bold] demo leads — view them on the UI's Startups page.")
    console.print(f"CSV:    [bold]{csv_path}[/bold]")
    console.print(f"Report: [bold]{md_path}[/bold]")


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Port to serve the panel on.")] = 8501,
) -> None:
    """Launch the browser control panel (edit thesis, scoring, sources; run and inspect)."""
    import subprocess
    import sys

    ui_path = Path(__file__).resolve().parent / "ui.py"
    console.print(f"Opening scout control panel at [bold]http://localhost:{port}[/bold] …")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", str(ui_path),
                "--server.port", str(port),
                "--server.address", "localhost",
            ],
            check=False,
        )
    except FileNotFoundError:
        console.print(
            "[red]streamlit is not installed.[/red] Run [bold]uv sync[/bold] "
            "(or [bold]pip install streamlit[/bold]) first."
        )
        raise typer.Exit(1)


# --------------------------------------------------------------------- thesis

thesis_app = typer.Typer(
    help="Manage the thesis library: what you source and score against.",
    no_args_is_help=True,
)
app.add_typer(thesis_app, name="thesis")


def _thesis_store() -> Store:
    return Store(Settings().db_path)


@thesis_app.command("list")
def thesis_list(
    archived: Annotated[
        bool, typer.Option("--archived", help="Include archived theses.")
    ] = False,
) -> None:
    """Every thesis, with the runs and startups each produced."""
    store = _thesis_store()
    store.backfill_thesis_ids()
    path = Path("thesis.yaml")
    if path.exists():
        active = load_thesis(path)
        active_id = ensure_thesis_id(active)
        store.adopt_history(active_id, active.thesis)
        store.upsert_thesis(
            active_id, name=active.name or active_id,
            statement=active.thesis,
            version=thesis_version(active, _load_seeds_or_default()),
            make_active=True,
        )
    rows = store.list_theses(include_archived=archived)
    if not rows:
        console.print("[yellow]No theses recorded yet[/yellow] — run `scout run`.")
        return
    table = Table(title="Theses", box=box.SIMPLE)
    for col in ("", "Id", "Name", "Runs", "Startups", "Version", "Last run"):
        table.add_column(col)
    for row in rows:
        versions = store.thesis_version_history(row["id"])
        n = next(
            (v["n"] for v in versions if v["version"] == row["current_version"]),
            len(versions),
        )
        table.add_row(
            "●" if row["is_active"] else "",
            row["id"],
            (row["name"] or "")[:34],
            str(row["run_count"]),
            str(row["lead_count"]),
            f"v{n}",
            (row["last_run_at"] or "—")[:10],
        )
    console.print(table)


@thesis_app.command("show")
def thesis_show(
    thesis_id: Annotated[str, typer.Argument(help="Thesis id (see `scout thesis list`).")],
) -> None:
    """One thesis: its statement, version history, and staleness."""
    store = _thesis_store()
    row = store.get_thesis(thesis_id)
    if row is None:
        console.print(f"[red]No thesis {thesis_id!r}[/red] — see `scout thesis list`.")
        raise typer.Exit(1)
    console.print(f"[bold]{row['name']}[/bold] ({row['id']})")
    console.print(f"  {row['statement']}\n")
    for version in store.thesis_version_history(thesis_id):
        marker = " ← current" if version["version"] == row["current_version"] else ""
        console.print(
            f"  v{version['n']}  {version['version'][:12]}  "
            f"{version['run_count']} run(s)  {version['created_at'][:10]}{marker}"
        )
    stale = store.stale_handles(thesis_id)
    if stale:
        console.print(
            f"\n[yellow]{len(stale)} startup(s) scored under an older version[/yellow]"
            " — `scout reclassify --stale-only` to bring them current."
        )


@thesis_app.command("new")
def thesis_new(
    name: Annotated[str, typer.Argument(help="Display name, e.g. 'Novel Architectures'.")],
    statement: Annotated[
        str, typer.Option("--statement", help="The thesis in plain language.")
    ] = "",
) -> None:
    """Start a new thesis from the current one, then edit it in the UI."""
    path = Path("thesis.yaml")
    base = load_thesis(path) if path.exists() else Thesis()
    base.id = slugify(name)
    base.name = name
    if statement:
        base.thesis = statement
    save_thesis(base, path)
    _thesis_store().upsert_thesis(
        base.id, name=name, statement=base.thesis, version="", make_active=True
    )
    console.print(
        f"Created [bold]{name}[/bold] ({base.id}) and made it active — "
        f"library copy at [bold]{thesis_library_path(base.id, path)}[/bold]."
    )


@thesis_app.command("use")
def thesis_use(
    thesis_id: Annotated[str, typer.Argument(help="Thesis id to activate.")],
) -> None:
    """Switch the active thesis. The one you leave is saved, not lost."""
    try:
        activated = switch_thesis(thesis_id, Path("thesis.yaml"))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    store = _thesis_store()
    store.set_active_thesis(ensure_thesis_id(activated))
    console.print(f"Now sourcing and scoring against [bold]{activated.name or thesis_id}[/bold].")
    stale = store.stale_handles(ensure_thesis_id(activated))
    if stale:
        console.print(
            f"[yellow]{len(stale)} startup(s) were scored under an older version "
            "of this thesis[/yellow] — `scout reclassify --stale-only` to update."
        )


@thesis_app.command("clone")
def thesis_clone(
    thesis_id: Annotated[str, typer.Argument(help="Thesis id to copy.")],
    name: Annotated[str, typer.Argument(help="Name for the copy.")],
) -> None:
    """Fork a thesis so you can diverge without losing the original."""
    source = thesis_library_path(thesis_id, Path("thesis.yaml"))
    if not source.exists():
        console.print(f"[red]No thesis {thesis_id!r}[/red] — see `scout thesis list`.")
        raise typer.Exit(1)
    clone = load_thesis(source)
    clone.id = slugify(name)
    clone.name = name
    save_thesis(clone, Path("thesis.yaml"))
    _thesis_store().upsert_thesis(
        clone.id, name=name, statement=clone.thesis, version="", make_active=True
    )
    console.print(f"Cloned [bold]{thesis_id}[/bold] → [bold]{name}[/bold] ({clone.id}), now active.")


@thesis_app.command("archive")
def thesis_archive(
    thesis_id: Annotated[str, typer.Argument(help="Thesis id to archive.")],
    undo: Annotated[bool, typer.Option("--undo", help="Un-archive instead.")] = False,
) -> None:
    """Hide a thesis from the pickers. Its runs and startups are kept."""
    _thesis_store().archive_thesis(thesis_id, archived=not undo)
    console.print(
        f"{'Un-archived' if undo else 'Archived'} [bold]{thesis_id}[/bold]"
        f"{'' if undo else ' — its history is kept.'}"
    )


if __name__ == "__main__":
    app()
