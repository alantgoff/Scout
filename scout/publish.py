"""Static mobile digest — renders the deal flow to docs/ for GitHub Pages.

`scout publish` turns the database into a small read-only APP, not just a
page: four hash-routed views behind a bottom tab bar —

  #/startups   the triage list, with search, sort (score / fit / newest /
               round) and filters (status, stage, round) running client-side
               off data-* attributes stamped at render time
  #/funnel     the pipeline by status, in FUNNEL_STAGES order
  #/graph      the knowledge-graph canvas (scout.graph_view, embedded as-is)
  #/alerts     what changed: acquired/merged/shut-down companies, new
               arrivals, score movers

plus a web-app manifest and a service worker, so "Add to Home Screen" is a
real installable app that still opens on a plane. Everything is one
self-contained index.html — server-rendered cards (testable as strings),
no build step, no JS dependencies.

The output directory is deployable as-is to GitHub Pages OR Vercel: the
Vercel files (vercel.json, middleware.js, robots.txt) are inert on Pages
and, on Vercel, add the one thing Pages cannot — a real password on a deal
flow digest (Edge Middleware, Basic auth from a DIGEST_PASSWORD env var;
open until it is set). `scout publish --vercel` deploys it.

The privacy boundary is the module's contract: lead data, verdicts,
pipeline STATUS and briefs go public; notes, votes, comments, spend and
config never do. The page stays <meta name="robots" content="noindex">.
"""

from __future__ import annotations

import html
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

from scout import graph_view, rubric
from scout.companies import group_by_company, startup_identity
from scout.config import Thesis
from scout.models import (
    COMPANY_STATUS_LABELS,
    FUNDING_STAGE_LABELS,
    FUNDING_STAGE_ORDER,
    Lead,
    LedgerEntry,
)
from scout.score import scorecard_score
from scout.status import FUNNEL_STAGES, STATUS_LABELS
from scout.store import Store

LAUNCHED = {"launched", "scaling"}
PRELAUNCH = {"idea", "stealth"}
WATCH_SIGNALS = {"departure_signal", "bio_change", "bio_intent"}

# Companies in these states get an Alerts entry and a leading warning chip —
# their sites and feeds keep looking healthy, which is exactly the problem.
GONE_STATUSES = ("acquired", "merged", "shut_down")

MOVER_DELTA = 10.0  # score jump that makes the Alerts view


def _e(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def _is_startup(lead: Lead) -> bool:
    verdict = lead.llm
    stage = verdict.stage if verdict else None
    return stage in LAUNCHED and bool(verdict) and (verdict.account_type or "other") != "other"


def _is_prelaunch(lead: Lead) -> bool:
    verdict = lead.llm
    stage = verdict.stage if verdict else None
    if stage in PRELAUNCH:
        return True
    return stage is None and any(
        s.name in WATCH_SIGNALS and s.value > 0 for s in lead.signals
    )


def _chips(lead: Lead, entry: LedgerEntry | None, status: str | None,
           firm: str = "", thesis: Thesis | None = None) -> str:
    verdict = lead.llm
    chips: list[tuple[str, str]] = []
    # A company that is no longer its own company leads — every other chip
    # can look healthy while the deal does not exist (same rule as the app).
    if verdict and verdict.company_status in GONE_STATUSES:
        label = COMPANY_STATUS_LABELS[verdict.company_status]
        note = (verdict.company_status_note or "").strip()
        chips.append((f"⚠ {label}" + (f" — {note}" if note else ""), "warn"))
    if verdict and verdict.thesis_fit is not None:
        chips.append((f"Fit {verdict.thesis_fit:.0%}", "accent"))
    if thesis is not None:
        scorecard = scorecard_score(verdict, thesis)
        if scorecard is not None:
            result = scorecard[0]
            chips.append((f"{rubric.BAND_LABELS[result.band]} {result.total:.0f}",
                          "accent" if result.band == "strong" else ""))
    if verdict and verdict.funding_stage not in (None, "unknown"):
        round_label = FUNDING_STAGE_LABELS.get(verdict.funding_stage,
                                               verdict.funding_stage)
        if verdict.funding_amount:
            round_label += f" · {verdict.funding_amount}"
        chips.append((round_label, "accent"))
    if verdict and verdict.value_add_fit is not None:
        chips.append((f"{firm or 'Firm'} lift {verdict.value_add_fit:.0%}", "accent"))
    if status:
        chips.append((STATUS_LABELS.get(status, status), "status"))
    if entry and entry.is_new:
        chips.append(("New", "accent"))
    if verdict and verdict.stage:
        chips.append((verdict.stage.capitalize(), ""))
    if verdict and verdict.customer_type:
        chips.append((verdict.customer_type.upper() if len(verdict.customer_type) <= 5
                      else verdict.customer_type, ""))
    sector = " · ".join(x for x in [verdict.sector if verdict else "",
                                    verdict.subsector if verdict else ""] if x)
    if sector:
        chips.append((sector, ""))
    if verdict and verdict.business_model:
        chips.append((verdict.business_model, ""))
    spans = "".join(f'<span class="chip {cls}">{_e(text)}</span>'
                    for text, cls in chips[:7] if text)
    return f'<div class="chips">{spans}</div>' if spans else ""


def _card(primary: Lead, entry: LedgerEntry | None, secondaries: list[Lead],
          pipeline_row: dict, startup: bool, firm: str = "",
          thesis: Thesis | None = None,
          connections: list[str] | None = None) -> str:
    account, verdict = primary.account, primary.llm
    # Startup-first titling, same system as the app: real company name, or the
    # synthesized stealth identity tied to the founder.
    identity = startup_identity(primary)
    if identity:
        title, synthesized = identity
        # Don't restate the title: "Acme Robotics — Acme Robotics · @acmebot"
        # when the account is the company account itself.
        subtitle = (f"@{account.handle}"
                    if synthesized or (account.name or "").strip() == title
                    else f"{account.name or account.handle} · @{account.handle}")
    else:
        title = account.name or f"@{account.handle}"
        subtitle = f"@{account.handle}"
    company_url = (verdict.company_url if verdict else None) or account.website
    title_html = (f'<a href="{_e(company_url)}">{_e(title)}</a>'
                  if company_url else _e(title))
    summary = ((verdict.product_summary if verdict else "")
               or (verdict.one_line_summary if verdict else "")
               or account.bio or "")
    why = (verdict.why_interesting if verdict else "") or ""

    # The researched facts, one quiet line: HQ · founded · founders.
    facts_bits = []
    if verdict and verdict.hq:
        facts_bits.append(verdict.hq)
    if verdict and verdict.founded_year:
        facts_bits.append(f"founded {verdict.founded_year}")
    if verdict and verdict.founders:
        facts_bits.append("; ".join(verdict.founders[:3]))
    facts = (f'<div class="why">{_e(" · ".join(facts_bits))}</div>'
             if facts_bits else "")
    investors = ""
    if verdict and verdict.funding_investors:
        investors = (f'<div class="why">Backed by '
                     f'{_e(", ".join(verdict.funding_investors[:5]))}</div>')
    conn_html = ""
    if connections:
        items = "".join(f"<div>· {line}</div>" for line in connections[:4])
        conn_html = f'<div class="why conn"><b>Connections</b>{items}</div>'
    also = ""
    if secondaries:
        others = ", ".join(f"@{x.account.handle}" for x in secondaries[:3])
        also = f'<div class="why">Also tracking: {_e(others)}</div>'
    brief = pipeline_row.get("brief") or ""
    brief_html = (
        f'<details><summary>Research brief</summary>'
        f'<div class="brief">{_e(brief)}</div></details>' if brief else ""
    )
    status = pipeline_row.get("status")
    handle = account.handle.lower()
    fit = verdict.thesis_fit if verdict and verdict.thesis_fit is not None else -1
    round_rank = FUNDING_STAGE_ORDER.get(
        (verdict.funding_stage if verdict else None) or "unknown", 6)
    first_seen = (entry.first_seen_at.isoformat()
                  if entry and entry.first_seen_at else "")
    search_blob = " ".join([
        title, account.handle, account.name, summary,
        (verdict.sector or "") if verdict else "",
        " ".join(verdict.tags) if verdict else "",
        " ".join(verdict.founders) if verdict else "",
        (verdict.hq or "") if verdict else "",
    ]).lower()
    return f"""<article class="card" id="card-{_e(handle)}" data-search="{_e(search_blob)}"
  data-score="{primary.score:.1f}" data-fit="{fit:.2f}"
  data-status="{_e(status or 'new')}" data-stage="{_e((verdict.stage if verdict else '') or '')}"
  data-roundrank="{round_rank}" data-new="{1 if entry and entry.is_new else 0}"
  data-first="{_e(first_seen)}">
  <div class="row">
    <div class="grow">
      <div class="name">{title_html} <span class="sub">{_e(subtitle)}</span></div>
      <div class="summary">{_e(summary)}</div>
      {_chips(primary, entry, status if status and status != "new" else None, firm, thesis)}
      {facts}
      {investors}
      {f'<div class="why">{_e(why)}</div>' if why else ''}
      {conn_html}
      {also}
      {brief_html}
    </div>
    <div class="scorecol"><div class="score">{primary.score:.0f}</div>
      <a class="xlink" href="{_e(account.url)}">Profile →</a></div>
  </div>
</article>"""


# ------------------------------------------------------------------ context


def _connection_lines(edges: list[dict]) -> dict[str, list[str]]:
    """company key → cross-link lines ("Bpifrance also backs Acme, Beta").

    Only what links SIDEWAYS earns a line — a card's own investors already
    render on the card; the graph's value is the other end.
    """
    by_src: dict[str, list[dict]] = {}
    by_dst: dict[str, list[dict]] = {}
    for e in edges:
        by_src.setdefault(e["src_key"], []).append(e)
        by_dst.setdefault(e["dst_key"], []).append(e)
    out: dict[str, list[str]] = {}
    for ckey, inbound in by_dst.items():
        lines: list[str] = []
        for e in inbound:
            if e["rel"] != "invested_in":
                continue
            siblings = sorted({
                x["dst_label"] for x in by_src.get(e["src_key"], [])
                if x["rel"] == "invested_in" and x["dst_key"] != ckey
            })
            if siblings:
                lines.append(f"{_e(e['src_label'])} also backs "
                             + ", ".join(_e(s) for s in siblings[:3]))
        for e in by_src.get(ckey, []):
            if e["rel"] != "acquired_by":
                continue
            others = sorted({
                x["src_label"] for x in by_dst.get(e["dst_key"], [])
                if x["rel"] == "acquired_by" and x["src_key"] != ckey
            })
            line = f"Acquired by {_e(e['dst_label'])}"
            if others:
                line += " — which also bought " + ", ".join(
                    _e(o) for o in others[:3])
            lines.append(line)
        if lines:
            out[ckey] = lines
    return out


def digest_context(store: Store, thesis: Thesis) -> dict:
    """Every store read the page needs, in one dict — so rendering is pure
    and each view is testable against the same context."""
    from scout.graph import company_node

    ledger = store.load_lead_ledger()
    pipeline = store.all_pipeline()
    pairs = [(e.lead, e) for e in ledger]
    edges = store.all_graph_edges()
    if not edges and ledger:
        store.rebuild_graph(ledger)
        edges = store.all_graph_edges()

    startups = group_by_company([p for p in pairs if _is_startup(p[0])])
    watch = [(x, e) for x, e in pairs if _is_prelaunch(x)][:30]

    # Funnel: EVERY triaged lead, grouped by status in funnel order.
    by_handle = {e.lead.account.handle.lower(): e for e in ledger}
    funnel: list[tuple[str, list[LedgerEntry]]] = []
    for status in [*FUNNEL_STAGES, "passed"]:
        members = sorted(
            (by_handle[h] for h, row in pipeline.items()
             if (row.get("status") or "new") == status and h in by_handle),
            key=lambda e: -e.lead.score,
        )
        funnel.append((status, members))

    # Alerts: gone companies first, then arrivals, then movers.
    gone = [e for e in ledger if e.lead.llm
            and e.lead.llm.company_status in GONE_STATUSES]
    arrivals = sorted((e for e in ledger if e.is_new),
                      key=lambda e: -e.lead.score)[:8]
    movers = sorted((e for e in ledger
                     if e.score_delta is not None and e.score_delta >= MOVER_DELTA),
                    key=lambda e: -(e.score_delta or 0))[:8]

    return {
        "ledger": ledger,
        "pipeline": pipeline,
        "startups": startups,
        "watch": watch,
        "funnel": funnel,
        "gone": gone,
        "arrivals": arrivals,
        "movers": movers,
        "edges": edges,
        "connections": _connection_lines(edges),
        "company_key": lambda lead: company_node(lead)[0],
        "n_new": sum(1 for _, e in pairs if e and e.is_new),
        "strong": sum(1 for x, _ in pairs
                      if x.llm and x.llm.thesis_fit is not None
                      and x.llm.thesis_fit >= 0.7),
        "updated": datetime.now(timezone.utc).strftime("%b %d, %H:%M UTC"),
    }


# ------------------------------------------------------------------- views


def _display_title(lead: Lead) -> str:
    identity = startup_identity(lead)
    return identity[0] if identity else (lead.account.name
                                         or f"@{lead.account.handle}")


def _funnel_html(context: dict) -> str:
    sections = []
    for status, members in context["funnel"]:
        if not members:
            continue
        rows = "".join(
            f'<div class="frow" data-target="card-{_e(e.lead.account.handle.lower())}">'
            f'<span class="fname">{_e(_display_title(e.lead))}</span>'
            f'<span class="fmeta">{FUNDING_STAGE_LABELS.get((e.lead.llm.funding_stage if e.lead.llm else None) or "unknown", "")}'
            f'</span><span class="fscore">{e.lead.score:.0f}</span></div>'
            for e in members
        )
        label = STATUS_LABELS.get(status, status)
        body = f'<div class="fsec"><h2>{_e(label)} <span class="count">{len(members)}</span></h2>{rows}</div>'
        if status == "passed":
            body = (f'<details class="fsec"><summary><h2 style="display:inline">'
                    f'{_e(label)} <span class="count">{len(members)}</span></h2>'
                    f'</summary>{rows}</details>')
        sections.append(body)
    return "".join(sections) or '<div class="why">Nothing triaged yet.</div>'


def _alerts_html(context: dict) -> str:
    parts = []
    if context["gone"]:
        rows = ""
        for e in context["gone"]:
            v = e.lead.llm
            label = COMPANY_STATUS_LABELS.get(v.company_status, v.company_status)
            note = (v.company_status_note or "").strip()
            evidence = (v.company_status_evidence or "").strip()
            rows += (f'<div class="alert"><b>⚠ {_e(_display_title(e.lead))} — '
                     f'{_e(label)}</b>'
                     + (f'<div class="why">{_e(note)}</div>' if note else "")
                     + (f'<div class="why">source: {_e(evidence)}</div>'
                        if evidence else "")
                     + "</div>")
        parts.append(f"<h2>No longer independent</h2>{rows}")
    if context["arrivals"]:
        rows = "".join(
            f'<div class="frow" data-target="card-{_e(e.lead.account.handle.lower())}">'
            f'<span class="fname">{_e(_display_title(e.lead))}</span>'
            f'<span class="fscore">{e.lead.score:.0f}</span></div>'
            for e in context["arrivals"]
        )
        parts.append(f"<h2>New this run</h2>{rows}")
    if context["movers"]:
        rows = "".join(
            f'<div class="frow" data-target="card-{_e(e.lead.account.handle.lower())}">'
            f'<span class="fname">{_e(_display_title(e.lead))}</span>'
            f'<span class="fmeta">▲ {e.score_delta:.0f}</span>'
            f'<span class="fscore">{e.lead.score:.0f}</span></div>'
            for e in context["movers"]
        )
        parts.append(f"<h2>Moving up</h2>{rows}")
    return "".join(parts) or '<div class="why">Nothing new — quiet is fine.</div>'


def _graph_html(context: dict) -> str:
    nodes, links = graph_view.graph_data(context["edges"], cross_links_only=True)
    if not nodes:
        return ('<div class="why">No cross-links yet — the graph grows as '
                'scans and refreshes add investors, founders and acquirers.</div>')
    return graph_view.graph_page_html(nodes, links, height=470)


def render_page(context: dict, thesis: Thesis) -> str:
    """The whole app as one HTML document. Pure given a context."""
    pipeline = context["pipeline"]
    connections = context["connections"]
    ckey_of = context["company_key"]

    def row_for(lead: Lead) -> dict:
        return pipeline.get(lead.account.handle.lower(), {})

    startup_cards = "\n".join(
        _card(p, e, secs, row_for(p), startup=True, firm=thesis.firm_name,
              thesis=thesis, connections=connections.get(ckey_of(p)))
        for p, e, secs in context["startups"]
    )
    watch_cards = "\n".join(
        _card(x, e, [], row_for(x), startup=False, firm=thesis.firm_name,
              thesis=thesis, connections=connections.get(ckey_of(x)))
        for x, e in context["watch"]
    )
    statuses_present = sorted({
        (row.get("status") or "new") for row in pipeline.values()
    } - {"new"})
    status_options = "".join(
        f'<option value="{_e(s)}">{_e(STATUS_LABELS.get(s, s))}</option>'
        for s in statuses_present
    )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Scout">
<meta name="theme-color" content="#f4ebe0">
<link rel="apple-touch-icon" href="icon.png">
<link rel="manifest" href="manifest.webmanifest">
<title>Scout — deal flow</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..700&family=Figtree:wght@300..700&display=swap');
:root {{ --bg:#f4ebe0; --surface:#fbf5ec; --ink:#20180f; --ink2:#4a4034; --muted:#6b6052;
  --hair:rgba(32,24,15,0.14); --accent:#20180f; --butter:#f2dc6c;
  --soft:rgba(226,196,90,0.28); --warn:#7c2d20; --warnsoft:rgba(178,58,44,0.12);
  --serif:"Fraunces","Iowan Old Style",Georgia,serif;
  --sans:"Figtree",-apple-system,system-ui,sans-serif; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:var(--sans);
  background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased;
  padding:max(env(safe-area-inset-top),12px) 14px
          calc(64px + env(safe-area-inset-bottom));
  max-width:640px; margin:0 auto; }}
h1 {{ font-family:var(--serif); font-size:2rem; font-weight:600;
  letter-spacing:-0.015em; margin:10px 0 2px; text-align:center; }}
.thesis {{ font-family:var(--serif); font-style:italic; color:var(--ink2);
  font-size:0.92rem; line-height:1.45; text-align:center; }}
.meta {{ color:var(--muted); font-size:0.75rem; margin:6px 0 14px; text-align:center; }}
.stats {{ display:flex; gap:8px; margin-bottom:14px; }}
.stat {{ flex:1; background:var(--surface); border:1px solid var(--hair); border-radius:12px;
  padding:8px 10px; }}
.stat b {{ font-family:var(--serif); font-size:1.3rem; font-weight:600; display:block; }}
.stat span {{ font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.09em; }}
.toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
input[type=search] {{ flex:1 1 100%; padding:10px 14px; border-radius:999px;
  border:1px solid var(--hair); font-size:1rem; font-family:var(--sans);
  color:var(--ink); background:var(--surface); -webkit-appearance:none; }}
select {{ flex:1; min-width:0; padding:7px 8px; border-radius:10px;
  border:1px solid var(--hair); background:var(--surface); color:var(--ink);
  font-family:var(--sans); font-size:0.8rem; }}
h2 {{ font-family:var(--serif); font-size:1.2rem; font-weight:600;
  margin:18px 0 8px; letter-spacing:-0.005em; }}
.count {{ color:var(--muted); font-size:0.8rem; font-weight:400; }}
.card {{ background:var(--surface); border:1px solid var(--hair); border-radius:14px;
  padding:12px 14px; margin-bottom:10px; }}
.card.flash {{ outline:2px solid var(--butter); }}
.row {{ display:flex; gap:10px; }}
.grow {{ flex:1; min-width:0; }}
.name {{ font-family:var(--serif); font-weight:600; font-size:1.05rem; }}
.name a {{ color:var(--ink); text-decoration:none; border-bottom:1px solid var(--hair); }}
.sub {{ font-family:var(--sans); color:var(--muted); font-weight:400; font-size:0.78rem; }}
.summary {{ color:var(--ink2); font-size:0.86rem; line-height:1.4; margin-top:3px; }}
.why {{ color:var(--muted); font-size:0.78rem; line-height:1.4; margin-top:6px; }}
.conn b {{ color:var(--ink2); }}
.chips {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }}
.chip {{ padding:2.5px 9px; border-radius:999px; font-size:0.6rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.07em;
  background:rgba(32,24,15,0.07); color:var(--ink2); }}
.chip.accent {{ background:var(--butter); color:var(--ink); font-weight:600; }}
.chip.status {{ background:var(--ink); color:var(--bg); }}
.chip.warn {{ background:var(--warnsoft); color:var(--warn); }}
.scorecol {{ text-align:right; flex:0 0 56px; }}
.score {{ font-family:var(--serif); font-size:1.35rem; font-weight:600; }}
.xlink {{ font-size:0.72rem; color:var(--ink); text-decoration:underline;
  text-underline-offset:2px; }}
details {{ margin-top:8px; }} summary {{ font-size:0.78rem; color:var(--muted); cursor:pointer; }}
.brief {{ font-size:0.8rem; color:var(--ink2); white-space:pre-wrap; margin-top:6px; }}
.frow {{ display:flex; gap:8px; align-items:baseline; background:var(--surface);
  border:1px solid var(--hair); border-radius:10px; padding:8px 12px;
  margin-bottom:6px; cursor:pointer; }}
.fname {{ flex:1; font-weight:600; font-size:0.9rem; min-width:0;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.fmeta {{ color:var(--muted); font-size:0.72rem; }}
.fscore {{ font-family:var(--serif); font-weight:600; }}
.alert {{ background:var(--surface); border:1px solid var(--warnsoft);
  border-left:3px solid var(--warn); border-radius:10px; padding:10px 12px;
  margin-bottom:8px; font-size:0.88rem; }}
.view {{ display:none; }} .view.on {{ display:block; }}
nav {{ position:fixed; left:0; right:0; bottom:0; display:flex; z-index:5;
  background:var(--surface); border-top:1px solid var(--hair);
  padding-bottom:env(safe-area-inset-bottom); max-width:640px; margin:0 auto; }}
nav button {{ flex:1; padding:11px 0 9px; background:none; border:none;
  font-family:var(--sans); font-size:0.72rem; color:var(--muted);
  letter-spacing:0.04em; cursor:pointer; }}
nav button.on {{ color:var(--ink); font-weight:700;
  box-shadow:inset 0 3px 0 var(--butter); }}
footer {{ color:var(--muted); font-size:0.72rem; margin-top:24px; text-align:center; }}
</style></head><body>
<h1>Scout</h1>
<div class="thesis">{_e(thesis.thesis)}</div>
<div class="meta">Updated {context['updated']} · read-only digest — triage in the desktop app</div>

<section id="view-startups" class="view on">
<div class="stats">
  <div class="stat"><b>{len(context['startups'])}</b><span>Startups</span></div>
  <div class="stat"><b>{context['strong']}</b><span>Strong fit</span></div>
  <div class="stat"><b>{context['n_new']}</b><span>New this run</span></div>
  <div class="stat"><b>{len(context['watch'])}</b><span>Watchlist</span></div>
</div>
<div class="toolbar">
  <input id="q" type="search" placeholder="Search startups, sectors, founders…">
  <select id="sort">
    <option value="score">Sort: score</option>
    <option value="fit">Sort: fit</option>
    <option value="new">Sort: newest</option>
    <option value="round">Sort: round</option>
  </select>
  <select id="fstatus"><option value="">Status: all</option>{status_options}</select>
  <select id="fstage"><option value="">Stage: all</option>
    <option value="stealth">Stealth</option><option value="launched">Launched</option>
    <option value="scaling">Scaling</option><option value="idea">Idea</option></select>
  <select id="fround"><option value="">Round: all</option>
    <option value="0">Bootstrapped</option><option value="1">Pre-seed</option>
    <option value="2">Seed</option><option value="3">Series A</option>
    <option value="4">Series B</option><option value="5">Series C+</option>
    <option value="6">Unknown</option></select>
</div>
<h2>Launched startups</h2>
<div id="list-startups">
{startup_cards or '<div class="why">Nothing yet — run a scan.</div>'}
</div>
<h2>Pre-launch watch</h2>
<div id="list-watch">
{watch_cards or '<div class="why">Nothing on watch.</div>'}
</div>
</section>

<section id="view-funnel" class="view">
{_funnel_html(context)}
</section>

<section id="view-graph" class="view">
<h2>Knowledge graph</h2>
<div class="why" style="margin-bottom:8px">Who connects to whom — every edge
derived from cited evidence. Drag, zoom, click to trace.</div>
{_graph_html(context)}
</section>

<section id="view-alerts" class="view">
{_alerts_html(context)}
</section>

<nav>
  <button data-view="startups" class="on">Startups</button>
  <button data-view="funnel">Funnel</button>
  <button data-view="graph">Graph</button>
  <button data-view="alerts">Alerts</button>
</nav>
<footer>Generated by scout · not indexed</footer>
<script>
(function () {{
  // ---- tabs + hash routing -------------------------------------------------
  var views = ["startups", "funnel", "graph", "alerts"];
  function show(view) {{
    if (views.indexOf(view) === -1) view = "startups";
    views.forEach(function (v) {{
      document.getElementById("view-" + v).classList.toggle("on", v === view);
    }});
    document.querySelectorAll("nav button").forEach(function (b) {{
      b.classList.toggle("on", b.dataset.view === view);
    }});
    if (location.hash !== "#/" + view) history.replaceState(null, "", "#/" + view);
    // The graph canvas measures 0×0 while hidden — re-measure on reveal.
    if (view === "graph") window.dispatchEvent(new Event("resize"));
    window.scrollTo(0, 0);
  }}
  document.querySelectorAll("nav button").forEach(function (b) {{
    b.addEventListener("click", function () {{ show(b.dataset.view); }});
  }});
  window.addEventListener("hashchange", function () {{
    show(location.hash.replace("#/", ""));
  }});
  show(location.hash.replace("#/", "") || "startups");

  // Funnel/alert rows jump to the startup's card.
  document.querySelectorAll(".frow[data-target]").forEach(function (r) {{
    r.addEventListener("click", function () {{
      show("startups");
      ["q"].forEach(function (id) {{ document.getElementById(id).value = ""; }});
      ["fstatus", "fstage", "fround"].forEach(function (id) {{
        document.getElementById(id).value = "";
      }});
      apply();
      var card = document.getElementById(r.dataset.target);
      if (card) {{
        card.scrollIntoView({{ behavior: "smooth", block: "center" }});
        card.classList.add("flash");
        setTimeout(function () {{ card.classList.remove("flash"); }}, 1600);
      }}
    }});
  }});

  // ---- search + filters + sort --------------------------------------------
  var q = document.getElementById("q"), sort = document.getElementById("sort");
  var fstatus = document.getElementById("fstatus"),
      fstage = document.getElementById("fstage"),
      fround = document.getElementById("fround");
  function apply() {{
    var needle = q.value.toLowerCase();
    document.querySelectorAll(".card").forEach(function (c) {{
      var ok = (!needle || c.dataset.search.indexOf(needle) !== -1)
        && (!fstatus.value || c.dataset.status === fstatus.value)
        && (!fstage.value || c.dataset.stage === fstage.value)
        && (!fround.value || c.dataset.roundrank === fround.value);
      c.style.display = ok ? "" : "none";
    }});
    ["list-startups", "list-watch"].forEach(function (id) {{
      var list = document.getElementById(id);
      var cards = Array.prototype.slice.call(list.querySelectorAll(".card"));
      cards.sort(function (a, b) {{
        switch (sort.value) {{
          case "fit": return (+b.dataset.fit) - (+a.dataset.fit);
          case "new": return (b.dataset.first || "").localeCompare(a.dataset.first || "");
          case "round": return (+a.dataset.roundrank) - (+b.dataset.roundrank)
                            || (+b.dataset.score) - (+a.dataset.score);
          default: return (+b.dataset.score) - (+a.dataset.score);
        }}
      }});
      cards.forEach(function (c) {{ list.appendChild(c); }});
    }});
  }}
  [q, sort, fstatus, fstage, fround].forEach(function (el) {{
    el.addEventListener("input", apply);
    el.addEventListener("change", apply);
  }});

  // ---- offline -------------------------------------------------------------
  if ("serviceWorker" in navigator) {{
    navigator.serviceWorker.register("sw.js").catch(function () {{}});
  }}
}})();
</script>
</body></html>
"""


# ------------------------------------------------------------------- output


_MANIFEST = """{
  "name": "Scout",
  "short_name": "Scout",
  "start_url": ".",
  "display": "standalone",
  "background_color": "#f4ebe0",
  "theme_color": "#f4ebe0",
  "icons": [{ "src": "icon.png", "sizes": "180x180", "type": "image/png" }]
}
"""

# Network-first with cache fallback: a fresh publish lands on the next online
# open, and a plane/tunnel serves the last one instead of a dinosaur. The
# cache name carries the publish stamp so a new deploy's activate step
# evicts every older cache, and only OK responses are cached — behind an
# auth gate, caching a 401 would make "offline" mean "locked out".
_SW_JS = """const CACHE = "scout-digest-__STAMP__";
const SHELL = ["./", "index.html", "icon.png", "manifest.webmanifest"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    fetch(e.request)
      .then((resp) => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return resp;
      })
      .catch(() => caches.match(e.request, { ignoreSearch: true }))
  );
});
"""

# Vercel project config for the output directory. Static site, no build:
# never indexed, HTML and the service worker always revalidated (a publish
# must land on the next open), the icon cached for a week.
_VERCEL_JSON = """{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "cleanUrls": true,
  "trailingSlash": false,
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Robots-Tag", "value": "noindex, nofollow, noarchive" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "no-referrer" }
      ]
    },
    {
      "source": "/(index.html|sw.js|)",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }]
    },
    {
      "source": "/icon.png",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=604800" }]
    }
  ]
}
"""

# Vercel Edge Middleware: HTTP Basic auth on everything except the manifest
# and icon (the browser fetches those without credentials, and "Add to Home
# Screen" needs them). The password is the DIGEST_PASSWORD env var set in
# the Vercel project — nothing here holds a secret, so the file is safe to
# publish. Unset = open, the bootstrap state; set it before sharing a URL.
_MIDDLEWARE_JS = """export const config = {
  matcher: ["/((?!manifest\\.webmanifest|icon\\.png).*)"],
};

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export default function middleware(request) {
  const expected = process.env.DIGEST_PASSWORD || "";
  if (!expected) return;  // no password configured yet: open
  const header = request.headers.get("authorization") || "";
  const [scheme, encoded] = header.split(" ");
  if (scheme === "Basic" && encoded) {
    let decoded = "";
    try { decoded = atob(encoded); } catch (_) { decoded = ""; }
    const password = decoded.slice(decoded.indexOf(":") + 1);
    if (timingSafeEqual(password, expected)) return;
  }
  return new Response(
    "<!doctype html><meta name=viewport content=width=device-width>"
    + "<title>Scout</title><p style=font-family:system-ui;padding:2rem>"
    + "Scout deal flow — sign in with the shared password.</p>",
    {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="Scout digest", charset="UTF-8"',
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
      },
    }
  );
}
"""

_ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

# The output directory is its own git checkout (the digest repo); the Vercel
# CLI's link file must not ride along in it.
_OUT_GITIGNORE = ".vercel/\n"


def build_digest(store: Store, thesis: Thesis, out_dir: Path,
                 *, stamp: str | None = None) -> Path:
    """Render the app into out_dir: index.html + manifest + service worker
    + icon, plus the Vercel files (inert on GitHub Pages). `stamp` names
    the service-worker cache for this publish; a new stamp evicts the old
    cache on the next open. Returns the index path."""
    context = digest_context(store, thesis)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    path = out_dir / "index.html"
    path.write_text(render_page(context, thesis), encoding="utf-8")
    (out_dir / "manifest.webmanifest").write_text(_MANIFEST, encoding="utf-8")
    (out_dir / "sw.js").write_text(_SW_JS.replace("__STAMP__", stamp), encoding="utf-8")
    (out_dir / "icon.png").write_bytes(_icon_png())
    (out_dir / "vercel.json").write_text(_VERCEL_JSON, encoding="utf-8")
    (out_dir / "middleware.js").write_text(_MIDDLEWARE_JS, encoding="utf-8")
    (out_dir / "robots.txt").write_text(_ROBOTS_TXT, encoding="utf-8")
    (out_dir / ".gitignore").write_text(_OUT_GITIGNORE, encoding="utf-8")
    return path


def _icon_png(size: int = 180) -> bytes:
    """Solid butter-yellow apple-touch-icon (Headline's brand accent),
    generated without image deps (iOS rounds the corners itself)."""
    r, g, b = 0xF2, 0xDC, 0x6C
    row = b"\x00" + bytes((r, g, b)) * size  # filter byte + RGB pixels
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
