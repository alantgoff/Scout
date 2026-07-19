"""run_analysis — the deep-analysis orchestrator.

  Evidence pack ($0, from store) -> Recon agent (web)
    -> 8 dimension agents (parallel, web, budget-guarded)
    -> Cross-examination (synth model, no web; confidence downgrades only)
    -> Composite (pure math, composite.py)
    -> Memo synthesis (synth model, no web; restates findings only)
    -> Persist Memo (fingerprint cache) -> KG ingest ($0)

Budget: a shared tracker checks spend-so-far before LAUNCHING each call; at
DILIGENCE_COST_CAP_USD it stops launching, keeps completed findings, and
flags the gap ("budget_capped"). A single failed dimension becomes a gap
finding (score=None) — it never aborts the run.

Every stage advances the scan banner via store.scan_update.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from scout.companies import company_key as lead_company_key
from scout.companies import group_by_company, normalize_company, startup_key
from scout.config import Settings, Thesis
from scout.diligence import graph, rubrics
from scout.diligence.composite import diligence_breakdown, hard_flags
from scout.diligence.research import ResearchError, research_call
from scout.diligence.schema import (
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    DimensionFinding,
    Memo,
    ReconReport,
    apply_cross_exam,
    gap_finding,
    parse_cross_exam,
    parse_finding,
    parse_recon,
)
from scout.models import Lead
from scout.store import Store

TWEETS_IN_PACK = 10
TWEET_MAX_CHARS = 280
# recon + the 8 dimensions + cross-exam + memo synthesis — the count quoted
# in every user-facing cost line, derived so it tracks the engine.
AGENT_COUNT = len(DIMENSION_KEYS) + 3


class EvidencePack(BaseModel):
    """The $0 dossier assembled from the store before any API call."""

    company_key: str
    company_name: str = ""
    primary_handle: str = ""
    website: str = ""
    urls: list[str] = Field(default_factory=list)  # verbatim, for web_fetch
    bios: list[str] = Field(default_factory=list)  # fingerprint input
    sectors: list[str] = Field(default_factory=list)
    kg_competitors: list[str] = Field(default_factory=list)
    digest: str = ""
    is_demo: bool = False  # synthetic sample lead — analysis refuses to spend


class _Budget:
    """Thread-safe per-memo spend tracker. `allows_next()` gates LAUNCHING a
    call — an in-flight call may overshoot slightly, which is the documented
    tradeoff (the cap stops new spend, it doesn't kill running calls)."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost

    def allows_next(self) -> bool:
        with self._lock:
            return self.spent < self.cap_usd


def resolve_company(store: Store, query: str) -> tuple[str, Lead, list[Lead]] | None:
    """Resolve a handle-or-company query to (company_key, primary lead,
    all member leads) via the ledger + company grouping. None when nothing
    tracked matches."""
    entries = store.load_lead_ledger(include_demo=True)
    if not entries:
        return None
    grouped = group_by_company([(e.lead, e) for e in entries])
    normalized = normalize_company(query)
    handle_query = query.lstrip("@").lower()
    for primary, _entry, secondaries in grouped:
        members = [primary, *secondaries]
        key = lead_company_key(primary)
        if key is not None and normalized is not None and key == normalized:
            return key, primary, members
        if any(m.account.handle.lower() == handle_query for m in members):
            return startup_key(primary), primary, members
    return None


def _member_urls(members: list[Lead]) -> list[str]:
    """Candidate URLs across a company's accounts, verbatim and deduped —
    web_fetch only fetches URLs already present in the conversation, so
    these must reach the prompts unmodified."""
    urls: list[str] = []
    for lead in members:
        if lead.account.website:
            urls.append(lead.account.website)
        if lead.llm and lead.llm.company_url:
            urls.append(lead.llm.company_url)
    return list(dict.fromkeys(u for u in urls if u))


def build_evidence_pack(store: Store, query: str) -> EvidencePack | None:
    """Assemble everything scout already knows — grouped accounts, bios,
    cached tweets, signals, verdict, ledger history, pipeline notes, and KG
    neighbors. Free; no network."""
    resolved = resolve_company(store, query)
    if resolved is None:
        return None
    key, primary, members = resolved
    verdict = primary.llm
    company_name = (verdict.company_name if verdict else "") or primary.account.name \
        or primary.account.handle

    urls = _member_urls(members)
    website = urls[0] if urls else ""

    sectors = []
    if verdict:
        sectors = [s for s in (verdict.sector, verdict.subsector) if s]
    sector_keys = [k for k in (graph.normalize_name(s) for s in sectors) if k]
    kg_competitors = store.kg_competitors_for_sectors(sector_keys, exclude_company_key=key)
    neighbor_competitors = [
        n["name"] for n in store.kg_neighbors(key) if n["rel"] == "competes_with"
    ]
    kg_competitors = sorted(set(kg_competitors) | set(neighbor_competitors))

    lines = [f"STARTUP UNDER ANALYSIS: {company_name}  (key: {key})"]
    for i, lead in enumerate(members):
        account = lead.account
        role = "Primary account" if i == 0 else "Also tracked"
        lines.append(
            f"{role}: @{account.handle} ({account.name or 'name unknown'}) — "
            f"{account.followers:,} followers"
        )
        if account.bio:
            lines.append(f"  bio: {account.bio}")
        if account.followed_by:
            lines.append(f"  followed by watchlist investors: {', '.join(account.followed_by)}")
    if urls:
        lines.append("Known URLs (verbatim, fetchable): " + ", ".join(urls))
    if verdict:
        lines.append(
            f"Screening classifier: stage {verdict.stage or '?'} · "
            f"{verdict.sector or '?'} / {verdict.subsector or '?'} · "
            f"{verdict.business_model or '?'}"
        )
        if verdict.one_line_summary:
            lines.append(f"  summary: {verdict.one_line_summary}")
        if verdict.why_interesting:
            lines.append(f"  why interesting: {verdict.why_interesting}")
        if verdict.thesis_fit is not None:
            lines.append(f"  thesis fit: {verdict.thesis_fit:.2f} — {verdict.fit_reason}")
    hits = [s for s in primary.signals if s.value > 0]
    if hits:
        lines.append("Screening signals hit: " + "; ".join(
            f"{s.name} ({s.detail})" if s.detail else s.name for s in hits
        ))
    lines.append(f"Screening score: {primary.score:.0f}/100 (ranking instrument only)")
    row = store.get_pipeline(primary.account.handle)
    if row.get("status"):
        lines.append(f"Pipeline status: {row['status']}")
    if row.get("notes"):
        lines.append(f"Analyst notes: {row['notes']}")
    tweets: list[str] = []
    for lead in members:
        for tweet in store.get_tweets(lead.account.id, limit=TWEETS_IN_PACK):
            tweets.append(f"@{lead.account.handle}: {tweet.text[:TWEET_MAX_CHARS]}")
    if tweets:
        lines.append("Recent tweets (x-data):")
        lines += [f"  - {t}" for t in tweets[:TWEETS_IN_PACK]]
    if kg_competitors:
        lines.append(
            "Knowledge graph — previously mapped in this space: "
            + ", ".join(kg_competitors)
        )

    return EvidencePack(
        company_key=key,
        company_name=company_name,
        primary_handle=primary.account.handle,
        website=website,
        urls=urls,
        bios=[m.account.bio for m in members],
        sectors=sectors,
        kg_competitors=kg_competitors,
        digest="\n".join(lines),
        is_demo=any(m.account.source == "demo" for m in members),
    )


def _fingerprint(
    bios: list[str], website: str, thesis: Thesis, settings: Settings
) -> str:
    payload = "\x1f".join(
        [
            "\x1e".join(bios),
            website,
            thesis.thesis,
            settings.claude_model,
            settings.diligence_synth_model,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def memo_fingerprint(pack: EvidencePack, thesis: Thesis, settings: Settings) -> str:
    """Cache key: any change to the company's bios, website, the thesis
    statement, or either model invalidates the stored memo."""
    return _fingerprint(pack.bios, pack.website, thesis, settings)


def members_fingerprint(
    members: list[Lead], thesis: Thesis, settings: Settings
) -> str:
    """Same fingerprint computed straight from in-memory leads — lets the UI
    check memo staleness without rebuilding an evidence pack (no store I/O)."""
    urls = _member_urls(members)
    return _fingerprint(
        [m.account.bio for m in members], urls[0] if urls else "", thesis, settings
    )


# ------------------------------------------------------------- user prompts


def _recon_user(pack: EvidencePack) -> str:
    return (
        pack.digest
        + "\n\nEstablish ground truth for this startup. Start from the known "
        "URLs above (fetch the website)."
    )


def _dimension_user(pack: EvidencePack, recon: ReconReport, key: str) -> str:
    lines = [pack.digest]
    if recon.key_sources:
        lines.append("\nKey sources from recon (fetchable URLs):")
        lines += [
            f"- {s.url} — {s.title or s.what}" for s in recon.key_sources if s.url
        ]
    if key == "competition" and pack.kg_competitors:
        lines.append(
            "\nPreviously mapped in this space (verify still relevant, drop "
            "dead/pivoted ones): " + ", ".join(pack.kg_competitors)
        )
    lines.append(
        f"\nAnalyze the '{DIMENSION_LABELS[key]}' dimension for "
        f"{recon.company_name or pack.company_name}."
    )
    return "\n".join(lines)


def _findings_json(findings: list[DimensionFinding]) -> str:
    return json.dumps([f.model_dump() for f in findings], ensure_ascii=False)


def _cross_exam_user(pack: EvidencePack, findings: list[DimensionFinding]) -> str:
    return (
        f"Startup: {pack.company_name}\n\nDimension findings:\n"
        + _findings_json(findings)
    )


def _memo_user(
    pack: EvidencePack,
    recon: ReconReport,
    findings: list[DimensionFinding],
    composite: float | None,
    steps: list[tuple[str, float]],
    contradictions: list[str],
    flags: list[str],
) -> str:
    return "\n".join(
        [
            f"Startup: {pack.company_name} (@{pack.primary_handle})",
            "Composite Diligence Score: "
            + (f"{composite:.1f}/100" if composite is not None else "not computable"),
            "Composite steps: " + "; ".join(f"{d} -> {v:.1f}" for d, v in steps),
            "Flags: " + (", ".join(flags) or "none"),
            "Cross-examination contradictions: "
            + ("; ".join(contradictions) or "none"),
            "",
            "Recon report:",
            json.dumps(recon.model_dump(), ensure_ascii=False),
            "",
            "Dimension findings:",
            _findings_json(findings),
        ]
    )


def _parse_memo_md(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or "## " not in cleaned:
        raise ValueError("memo must be markdown with the required sections")
    return cleaned


def _fallback_memo_md(
    pack: EvidencePack, findings: list[DimensionFinding], composite: float | None
) -> str:
    """Data-only memo when synthesis was skipped (budget) or failed — the
    scorecard still renders and nothing pretends to be prose that isn't."""
    lines = [
        "## Recommendation",
        "_Memo synthesis unavailable — dimension findings below are the raw output._",
        "",
        "| Dimension | Score | Confidence | Summary |",
        "|---|---|---|---|",
    ]
    for f in findings:
        score = f"{f.score:.1f}" if f.score is not None else "—"
        lines.append(
            f"| {DIMENSION_LABELS.get(f.key, f.key)} | {score} | "
            f"{f.confidence:.0%} | {f.summary or '—'} |"
        )
    if composite is not None:
        lines.insert(1, f"Composite Diligence Score: **{composite:.1f}/100**")
    unknowns = [u for f in findings for u in f.unknowns]
    if unknowns:
        lines += ["", "## Risks & open questions"] + [f"- {u}" for u in unknowns]
    return "\n".join(lines)


# ---------------------------------------------------------------- orchestrator


def _run_dimensions(
    pack: EvidencePack,
    recon: ReconReport,
    thesis: Thesis,
    settings: Settings,
    db_path,
    budget: _Budget,
) -> list[DimensionFinding]:
    """The 8 dimension agents, budget-gated, in a thread pool. Each worker
    opens its OWN Store (sqlite connections don't cross threads); order of
    the result matches DIMENSION_KEYS."""

    def one(key: str) -> DimensionFinding:
        if not budget.allows_next():
            return gap_finding(
                key,
                f"skipped — cost cap ${settings.diligence_cost_cap_usd:.2f} reached",
                flag="skipped_budget_cap",
            )
        thread_store = Store(db_path)
        system = rubrics.render(rubrics.RUBRICS[key], thesis, recon)
        try:
            finding, cost = research_call(
                system,
                _dimension_user(pack, recon, key),
                lambda text: parse_finding(text, key),
                settings,
                thread_store,
                company_key=pack.company_key,
                stage=key,
            )
            budget.add(cost)
            return finding
        except ResearchError as exc:
            budget.add(exc.cost_usd)
            return gap_finding(key, str(exc))
        except Exception as exc:  # a single bad dimension never sinks the run
            return gap_finding(key, f"{type(exc).__name__}: {exc}")

    workers = max(1, min(settings.diligence_concurrency, len(DIMENSION_KEYS)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, DIMENSION_KEYS))


def run_analysis(
    company_key: str,
    store: Store,
    thesis: Thesis,
    settings: Settings,
    refresh: bool = False,
    pack: EvidencePack | None = None,
) -> Memo:
    """Full deep analysis of one startup. Cached: an unchanged fingerprint is
    served from the memos table unless refresh=True. Raises RuntimeError when
    the company isn't tracked, is a demo lead, or the key is missing. Callers
    that already built the evidence pack (the CLI's confirm gate) pass it in
    to skip a second assembly."""
    pack = pack or build_evidence_pack(store, company_key)
    if pack is None:
        raise RuntimeError(
            f"no tracked startup matches '{company_key}' — check the name/handle "
            "or run discovery first."
        )
    if pack.is_demo:
        # Mirrors `scout verify`'s guard: never spend real money on synthetic
        # sample founders (it would also pollute the knowledge graph).
        raise RuntimeError(
            f"'{pack.company_name}' is a demo lead (fake handle) — refusing to "
            "spend money analyzing it."
        )
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — deep analysis needs Claude.")

    fingerprint = memo_fingerprint(pack, thesis, settings)
    if not refresh:
        cached = store.load_memo(pack.company_key)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached

    label = f"analyzing {pack.company_name}"
    budget = _Budget(settings.diligence_cost_cap_usd)

    # 1. Recon — shared ground truth. A recon failure degrades to the pack's
    # own facts rather than aborting (dimensions can still research).
    store.scan_update(label, "recon")
    recon = ReconReport(company_name=pack.company_name, website=pack.website)
    try:
        recon, cost = research_call(
            rubrics.render(rubrics.RECON_PROMPT, thesis),
            _recon_user(pack),
            parse_recon,
            settings,
            store,
            company_key=pack.company_key,
            stage="recon",
        )
        budget.add(cost)
        if not recon.company_name:
            recon.company_name = pack.company_name
        if not recon.website:
            recon.website = pack.website
    except ResearchError as exc:
        budget.add(exc.cost_usd)

    # 2. Eight dimensions in parallel, budget-gated.
    store.scan_update(label, f"researching {len(DIMENSION_KEYS)} dimensions")
    findings = _run_dimensions(pack, recon, thesis, settings, store.db_path, budget)

    # 3. Cross-examination (synth model, no web): downgrades only.
    contradictions: list[str] = []
    skipped_synth = False
    if budget.allows_next():
        store.scan_update(label, "cross-examination")
        try:
            exam, cost = research_call(
                rubrics.render(rubrics.CROSS_EXAM_PROMPT, thesis),
                _cross_exam_user(pack, findings),
                parse_cross_exam,
                settings,
                store,
                model=settings.diligence_synth_model,
                company_key=pack.company_key,
                stage="cross_exam",
                web=False,
            )
            budget.add(cost)
            apply_cross_exam(findings, exam)
            contradictions = exam.contradictions
        except ResearchError as exc:
            budget.add(exc.cost_usd)
    else:
        skipped_synth = True

    # 4. Composite — pure math.
    steps = diligence_breakdown(findings, thesis.diligence_weights)
    composite = round(steps[-1][1], 1) if steps else None
    flags = hard_flags(findings)

    # 5. Memo synthesis (synth model, no web) — may only restate findings.
    memo_md = ""
    if budget.allows_next():
        store.scan_update(label, "writing memo")
        try:
            memo_md, cost = research_call(
                rubrics.render(rubrics.MEMO_PROMPT, thesis),
                _memo_user(pack, recon, findings, composite, steps, contradictions, flags),
                _parse_memo_md,
                settings,
                store,
                model=settings.diligence_synth_model,
                company_key=pack.company_key,
                stage="memo",
                web=False,
                max_tokens=8192,
            )
            budget.add(cost)
        except ResearchError as exc:
            budget.add(exc.cost_usd)
    else:
        skipped_synth = True

    if any("skipped_budget_cap" in f.flags for f in findings) or skipped_synth:
        flags.append("budget_capped")
    if not memo_md:
        memo_md = _fallback_memo_md(pack, findings, composite)
        flags.append("memo_synthesis_unavailable")

    memo = Memo(
        company_key=pack.company_key,
        company_name=pack.company_name,
        fingerprint=fingerprint,
        composite=composite,
        flags=flags,
        findings=findings,
        recon=recon,
        memo_md=memo_md,
        cost_usd=round(budget.spent, 4),
        model=settings.claude_model,
        synth_model=settings.diligence_synth_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # 6. Persist + knowledge-graph ingest ($0).
    store.scan_update(label, "saving memo + knowledge graph")
    store.save_memo(memo)
    nodes, edges = graph.ingest(memo, sectors=pack.sectors)
    for node in nodes:
        store.kg_upsert_node(**node)
    for edge in edges:
        store.kg_upsert_edge(**edge)
    return memo


def resynthesize_memo(
    memo: Memo, store: Store, thesis: Thesis, settings: Settings
) -> Memo:
    """Re-run ONLY memo synthesis from stored findings (scout memo --refresh).
    Cheap by construction — one synth-model call, no web tools — which is why
    it runs without a _Budget tracker; usage still lands in the ledger via
    research_call."""
    steps = diligence_breakdown(memo.findings, thesis.diligence_weights)
    pack = EvidencePack(
        company_key=memo.company_key,
        company_name=memo.company_name,
        primary_handle="",
        website=memo.recon.website,
    )
    memo_md, cost = research_call(
        rubrics.render(rubrics.MEMO_PROMPT, thesis),
        _memo_user(
            pack, memo.recon, memo.findings, memo.composite, steps, [], memo.flags
        ),
        _parse_memo_md,
        settings,
        store,
        model=settings.diligence_synth_model,
        company_key=memo.company_key,
        stage="memo",
        web=False,
        max_tokens=8192,
    )
    memo.memo_md = memo_md
    memo.flags = [f for f in memo.flags if f != "memo_synthesis_unavailable"]
    memo.cost_usd = round(memo.cost_usd + cost, 4)
    memo.created_at = datetime.now(timezone.utc).isoformat()
    store.save_memo(memo)
    return memo
