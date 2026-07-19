"""Static mobile digest — renders the lead ledger to docs/ for GitHub Pages.

`scout publish` turns the current deal flow into one self-contained,
phone-first HTML page (Apple design language, same track semantics as the
UI: launched startups grouped by company first, pre-launch watch second).
Push docs/ and GitHub Pages serves it; on a phone, "Add to Home Screen"
makes it a Scout app that refreshes with every published scan.

REMOTE CONTROL: the page is static, so its "backend" is the GitHub API.
After a one-time setup on the phone (private code-repo slug + a
fine-grained PAT, both stored ONLY in that browser's localStorage), the
page can:
  - triage and tag startups — each tap PUTs one JSON decision file into
    the private repo's remote/inbox/ (unique filenames: no write races),
    which `scout inbox --apply` folds into the store on the next run;
  - start runs — workflow_dispatch on the scout-run / scout-analyze
    GitHub Actions workflows;
  - show run status by polling the Actions API.
Without a token the page stays exactly what it was: a read-only digest.

Contains ONLY lead data — no secrets, no keys, no watchlist. The private
repo's SLUG is embedded only when CONTROL_REPO is explicitly set (it can
always be typed into the setup panel instead). The page carries
<meta name="robots" content="noindex"> since Pages URLs are public.
"""

from __future__ import annotations

import html
import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

from scout.companies import group_by_company
from scout.config import Thesis
from scout.models import Lead, LedgerEntry
from scout.store import Store, pipeline_tags

# Workflow files the page dispatches (must match .github/workflows/).
RUN_WORKFLOW = "scout-run.yml"
ANALYZE_WORKFLOW = "scout-analyze.yml"
DISPATCH_BRANCH = "main"

LAUNCHED = {"launched", "scaling"}
PRELAUNCH = {"idea", "stealth"}
WATCH_SIGNALS = {"departure_signal", "bio_change", "bio_intent"}

STATUS_LABELS = {
    "shortlisted": "To reach out", "contacted": "Contacted", "meeting": "Meeting",
    "diligence": "Diligence", "won": "Allocated", "passed": "Passed",
}


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


def _chips(lead: Lead, entry: LedgerEntry | None, status: str | None) -> str:
    verdict = lead.llm
    chips: list[tuple[str, str]] = []
    if verdict and verdict.thesis_fit is not None:
        chips.append((f"Fit {verdict.thesis_fit:.0%}", "accent"))
    if status:
        chips.append((STATUS_LABELS.get(status, status), "status"))
    if entry and entry.is_new:
        chips.append(("New", "accent"))
    if verdict and verdict.stage:
        chips.append((verdict.stage.capitalize(), ""))
    sector = " · ".join(x for x in [verdict.sector if verdict else "",
                                    verdict.subsector if verdict else ""] if x)
    if sector:
        chips.append((sector, ""))
    if verdict and verdict.business_model:
        chips.append((verdict.business_model, ""))
    spans = "".join(f'<span class="chip {cls}">{_e(text)}</span>'
                    for text, cls in chips[:6] if text)
    return f'<div class="chips">{spans}</div>' if spans else ""


def _card(primary: Lead, entry: LedgerEntry | None, secondaries: list[Lead],
          pipeline_row: dict, startup: bool) -> str:
    account, verdict = primary.account, primary.llm
    if startup and verdict and verdict.company_name:
        title = verdict.company_name
        subtitle = f"{account.name or account.handle} · @{account.handle}"
    else:
        title = account.name or f"@{account.handle}"
        subtitle = f"@{account.handle}"
    company_url = (verdict.company_url if verdict else None) or account.website
    title_html = (f'<a href="{_e(company_url)}">{_e(title)}</a>'
                  if company_url else _e(title))
    summary = (verdict.one_line_summary if verdict else "") or account.bio or ""
    why = (verdict.why_interesting if verdict else "") or ""
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
    # Analyst tags (pipeline table) — tappable on the phone to remove.
    tags = pipeline_tags(pipeline_row)
    tags_html = ""
    if tags:
        tags_html = '<div class="chips">' + "".join(
            f'<span class="chip tag" data-tag="{_e(t)}">#{_e(t)}</span>' for t in tags
        ) + "</div>"
    # Remote triage actions — hidden until the setup panel connects a token.
    actions = (
        '<div class="actions" data-remote hidden>'
        '<button data-act="shortlist">Shortlist</button>'
        '<button data-act="pass">Pass</button>'
        '<button data-act="tag">+ Tag</button>'
        '<button data-act="note">+ Note</button>'
        + ('<button data-act="analyze" class="spend">Analyze</button>' if startup else "")
        + "</div>"
    )
    search_blob = " ".join(
        [title, account.handle, account.name, summary,
         (verdict.sector or "") if verdict else "",
         " ".join(verdict.tags) if verdict else "", " ".join(tags)]
    ).lower()
    return f"""<article class="card" data-handle="{_e(account.handle)}" data-search="{_e(search_blob)}">
  <div class="row">
    <div class="grow">
      <div class="name">{title_html} <span class="sub">{_e(subtitle)}</span></div>
      <div class="summary">{_e(summary)}</div>
      {_chips(primary, entry, status if status and status != "new" else None)}
      {tags_html}
      {f'<div class="why">{_e(why)}</div>' if why else ''}
      {also}
      {brief_html}
      {actions}
    </div>
    <div class="scorecol"><div class="score">{primary.score:.0f}</div>
      <a class="xlink" href="{_e(account.url)}">Profile →</a></div>
  </div>
</article>"""


def build_digest(
    store: Store, thesis: Thesis, out_dir: Path, control_repo: str = ""
) -> Path:
    """Render the ledger into out_dir/index.html (+ apple-touch icon).

    `control_repo` ("owner/repo" of the private code repo) pre-fills the
    remote setup panel; empty keeps the private repo's name off the page.
    """
    ledger = store.load_lead_ledger()
    pipeline = store.all_pipeline()
    pairs = [(e.lead, e) for e in ledger]

    startups = group_by_company([p for p in pairs if _is_startup(p[0])])
    watch = [(x, e) for x, e in pairs if _is_prelaunch(x)][:30]
    n_new = sum(1 for _, e in pairs if e and e.is_new)
    strong = sum(1 for x, _ in pairs
                 if x.llm and x.llm.thesis_fit is not None and x.llm.thesis_fit >= 0.7)

    def row_for(lead: Lead) -> dict:
        return pipeline.get(lead.account.handle.lower(), {})

    startup_cards = "\n".join(
        _card(p, e, secs, row_for(p), startup=True) for p, e, secs in startups
    )
    watch_cards = "\n".join(
        _card(x, e, [], row_for(x), startup=False) for x, e in watch
    )
    updated = datetime.now(timezone.utc).strftime("%b %d, %H:%M UTC")

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Scout">
<meta name="theme-color" content="#fbfbfd">
<link rel="apple-touch-icon" href="icon.png">
<title>Scout — deal flow</title>
<style>
:root {{ --bg:#fbfbfd; --surface:#fff; --ink:#1d1d1f; --ink2:#494949; --muted:#6e6e73;
  --hair:rgba(0,0,0,0.08); --accent:#0071e3; --soft:rgba(0,113,227,0.10); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
  background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased;
  padding:max(env(safe-area-inset-top),12px) 14px 40px; max-width:640px; margin:0 auto; }}
h1 {{ font-size:1.9rem; letter-spacing:-0.03em; margin:10px 0 2px; }}
.thesis {{ color:var(--muted); font-size:0.85rem; line-height:1.4; }}
.meta {{ color:var(--muted); font-size:0.75rem; margin:6px 0 14px; }}
.stats {{ display:flex; gap:8px; margin-bottom:14px; }}
.stat {{ flex:1; background:var(--surface); border:1px solid var(--hair); border-radius:12px;
  padding:8px 10px; }}
.stat b {{ font-size:1.25rem; display:block; }}
.stat span {{ font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; }}
input {{ width:100%; padding:10px 14px; border-radius:12px; border:1px solid var(--hair);
  font-size:1rem; background:var(--surface); margin-bottom:14px; -webkit-appearance:none; }}
h2 {{ font-size:1.05rem; margin:18px 0 8px; letter-spacing:-0.01em; }}
.card {{ background:var(--surface); border:1px solid var(--hair); border-radius:14px;
  padding:12px 14px; margin-bottom:10px; }}
.row {{ display:flex; gap:10px; }}
.grow {{ flex:1; min-width:0; }}
.name {{ font-weight:650; font-size:0.98rem; }}
.name a {{ color:var(--ink); text-decoration:none; border-bottom:1px solid var(--hair); }}
.sub {{ color:var(--muted); font-weight:400; font-size:0.78rem; }}
.summary {{ color:var(--ink2); font-size:0.86rem; line-height:1.4; margin-top:3px; }}
.why {{ color:var(--muted); font-size:0.78rem; line-height:1.4; margin-top:6px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }}
.chip {{ padding:2px 9px; border-radius:999px; font-size:0.68rem; font-weight:500;
  background:rgba(0,0,0,0.05); color:var(--ink2); }}
.chip.accent {{ background:var(--soft); color:var(--accent); font-weight:600; }}
.chip.status {{ background:var(--ink); color:var(--bg); }}
.scorecol {{ text-align:right; flex:0 0 56px; }}
.score {{ font-size:1.3rem; font-weight:650; }}
.xlink {{ font-size:0.72rem; color:var(--accent); text-decoration:none; }}
details {{ margin-top:8px; }} summary {{ font-size:0.78rem; color:var(--accent); }}
.brief {{ font-size:0.8rem; color:var(--ink2); white-space:pre-wrap; margin-top:6px; }}
footer {{ color:var(--muted); font-size:0.72rem; margin-top:24px; text-align:center; }}
.chip.tag {{ background:var(--soft); color:var(--accent); font-weight:600; }}
.chip.tag.gone {{ opacity:0.35; text-decoration:line-through; }}
.chip.queued {{ background:var(--accent); color:#fff; }}
.actions {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }}
.actions button, .runrow button, .setup button {{ -webkit-appearance:none; appearance:none;
  border:1px solid var(--hair); background:var(--surface); color:var(--ink);
  border-radius:999px; padding:6px 13px; font-size:0.78rem; font-weight:550;
  font-family:inherit; }}
.actions button:active, .runrow button:active {{ background:var(--soft); }}
.actions .spend {{ color:var(--accent); border-color:var(--accent); }}
.runrow {{ margin:0 0 12px; display:flex; gap:8px; align-items:center; }}
.runrow button {{ background:var(--accent); border-color:var(--accent); color:#fff;
  font-size:0.85rem; padding:8px 16px; }}
#remotebar {{ margin:0 0 12px; padding:8px 12px; background:var(--soft);
  border-radius:10px; }}
#remotebar.bad {{ background:rgba(196,66,66,0.08); color:#a33; }}
.setup input {{ margin-bottom:8px; }}
.setup .why {{ margin:6px 0 10px; }}
</style></head><body>
<h1>Scout</h1>
<div class="thesis">{_e(thesis.thesis)}</div>
<div class="meta">Updated {updated} · <span id="modehint">read-only — connect Remote
(bottom of page) to triage and start runs from here</span></div>
<div id="remotebar" class="why" hidden></div>
<div class="runrow" data-remote hidden>
  <button id="runbtn">Run scan now</button>
  <span class="why">free sources · runs in GitHub Actions</span>
</div>
<div class="stats">
  <div class="stat"><b>{len(startups)}</b><span>Startups</span></div>
  <div class="stat"><b>{strong}</b><span>Strong fit</span></div>
  <div class="stat"><b>{n_new}</b><span>New this run</span></div>
  <div class="stat"><b>{len(watch)}</b><span>Watchlist</span></div>
</div>
<input id="q" type="search" placeholder="Search startups, sectors, tags…">
<h2>Launched startups</h2>
{startup_cards or '<div class="why">Nothing yet — run a scan.</div>'}
<h2>Pre-launch watch</h2>
{watch_cards or '<div class="why">Nothing on watch.</div>'}
<details class="setup" id="setup"><summary>⚙︎ Remote — connect this page to scout</summary>
  <div class="why">Paste the private code repo and a fine-grained GitHub token
  (Contents + Actions read/write on that repo only). Both stay in THIS
  browser's localStorage — they are never published or sent anywhere except
  api.github.com. Triage taps queue as files in remote/inbox/; the next scan
  folds them in and republishes this page.</div>
  <input id="r-repo" autocapitalize="none" autocomplete="off" placeholder="owner/private-code-repo">
  <input id="r-token" type="password" autocomplete="off" placeholder="Fine-grained personal access token">
  <button id="r-save">Connect on this device</button>
  <button id="r-clear">Disconnect &amp; forget token</button>
</details>
<footer>Generated by scout · not indexed</footer>
"""
    remote_defaults = json.dumps(
        {
            "controlRepo": control_repo or "",
            "branch": DISPATCH_BRANCH,
            "runWorkflow": RUN_WORKFLOW,
            "analyzeWorkflow": ANALYZE_WORKFLOW,
        }
    )
    page += (
        "<script>window.SCOUT_REMOTE_DEFAULTS = " + remote_defaults + ";</script>\n"
        "<script>\n" + _REMOTE_JS + "\n</script>\n</body></html>\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    (out_dir / "icon.png").write_bytes(_icon_png())
    return path


# Plain (non-f) string so the JS braces stay readable. The page's dynamic
# config arrives via window.SCOUT_REMOTE_DEFAULTS above.
_REMOTE_JS = r"""
(function () {
  'use strict';
  var D = window.SCOUT_REMOTE_DEFAULTS || {};
  var LS = window.localStorage;
  function cfg() {
    return { repo: LS.getItem('scout.repo') || D.controlRepo || '',
             token: LS.getItem('scout.token') || '' };
  }
  function ready() { var c = cfg(); return !!(c.repo && c.token); }

  function api(path, method, body) {
    var c = cfg();
    return fetch('https://api.github.com' + path.replace('{repo}', c.repo), {
      method: method || 'GET',
      headers: { 'Authorization': 'Bearer ' + c.token,
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28' },
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      if (!r.ok) return r.text().then(function (t) {
        throw new Error('HTTP ' + r.status + ' ' + t.slice(0, 140));
      });
      return r.status === 204 ? null : r.json();
    });
  }

  function b64(s) { return btoa(unescape(encodeURIComponent(s))); }

  // One decision = one new file: unique names mean no SHA dance, no races.
  function decide(handle, action, extra) {
    var payload = { handle: handle, action: action,
                    tag: (extra && extra.tag) || '', note: (extra && extra.note) || '',
                    at: new Date().toISOString() };
    var name = Date.now() + '-' + action + '-' + handle.replace(/[^A-Za-z0-9_-]/g, '')
               + '-' + Math.random().toString(36).slice(2, 6) + '.json';
    return api('/repos/{repo}/contents/remote/inbox/' + name, 'PUT', {
      message: 'phone: ' + action + ' @' + handle,
      content: b64(JSON.stringify(payload))
    });
  }

  function dispatch(workflow, inputs) {
    return api('/repos/{repo}/actions/workflows/' + workflow + '/dispatches', 'POST',
               { ref: D.branch || 'main', inputs: inputs || {} });
  }

  function toast(msg, bad) {
    var el = document.getElementById('remotebar');
    if (!el) return;
    el.hidden = false; el.textContent = msg;
    el.className = 'why' + (bad ? ' bad' : '');
  }

  function queued(card, label) {
    if (!card) return;
    var host = card.querySelector('.chips') || card.querySelector('.grow');
    var span = document.createElement('span');
    span.className = 'chip queued'; span.textContent = label + ' ✓';
    host.appendChild(span);
  }

  var pollTimer = null;
  function pollStatus() {
    if (!ready()) return;
    api('/repos/{repo}/actions/runs?per_page=1').then(function (data) {
      var run = data && data.workflow_runs && data.workflow_runs[0];
      if (!run) return;
      var mins = Math.max(0, Math.round((Date.now() - new Date(run.created_at)) / 60000));
      var age = mins < 60 ? mins + 'm ago' : Math.round(mins / 60) + 'h ago';
      var state = run.status === 'completed' ? (run.conclusion || 'done') : run.status;
      toast(run.name + ': ' + state + ' · ' + age +
            (run.status !== 'completed' ? ' — refresh when done' : ''));
      clearTimeout(pollTimer);
      if (run.status !== 'completed') pollTimer = setTimeout(pollStatus, 20000);
    }).catch(function () { /* status is best-effort */ });
  }

  function refreshMode() {
    var on = ready();
    document.querySelectorAll('[data-remote]').forEach(function (el) { el.hidden = !on; });
    var hint = document.getElementById('modehint');
    if (hint) hint.textContent = on
      ? 'remote connected — taps queue to remote/inbox and fold in on the next scan'
      : 'read-only — connect Remote (bottom of page) to triage and start runs from here';
    if (on) pollStatus();
  }

  document.addEventListener('click', function (ev) {
    var btn = ev.target.closest('[data-act]');
    if (btn) {
      var card = btn.closest('.card');
      var handle = card ? card.dataset.handle : '';
      var act = btn.dataset.act;
      if (!handle) return;
      if (act === 'analyze') {
        if (!confirm('Deep analysis of @' + handle +
                     ' costs ~$4–8 and takes a few minutes. Dispatch it?')) return;
        dispatch(D.analyzeWorkflow || 'scout-analyze.yml', { query: handle })
          .then(function () { queued(card, 'analysis queued'); toast('Analysis dispatched for @' + handle); })
          .catch(function (e) { toast('Analyze failed: ' + e.message, true); });
        return;
      }
      var extra = {};
      if (act === 'tag') {
        var t = prompt('Tag for @' + handle + ':'); if (!t || !t.trim()) return;
        extra.tag = t.trim();
      }
      if (act === 'note') {
        var n = prompt('Note for @' + handle + ':'); if (!n || !n.trim()) return;
        extra.note = n.trim();
      }
      decide(handle, act, extra)
        .then(function () { queued(card, act === 'tag' ? '#' + extra.tag : act); })
        .catch(function (e) { toast('Sync failed: ' + e.message, true); });
      return;
    }
    var chip = ev.target.closest('.chip.tag');
    if (chip && ready() && !chip.classList.contains('gone')) {
      var card2 = chip.closest('.card');
      var h2 = card2 ? card2.dataset.handle : '';
      var tg = chip.dataset.tag;
      if (h2 && tg && confirm('Remove tag "' + tg + '" from @' + h2 + '?'))
        decide(h2, 'untag', { tag: tg })
          .then(function () { chip.classList.add('gone'); })
          .catch(function (e) { toast('Sync failed: ' + e.message, true); });
    }
  });

  var runbtn = document.getElementById('runbtn');
  if (runbtn) runbtn.addEventListener('click', function () {
    if (!confirm('Start a discovery scan now? Free sources; queued phone triage is folded in first.')) return;
    dispatch(D.runWorkflow || 'scout-run.yml', {})
      .then(function () {
        toast('Scan dispatched — this page republishes when it finishes.');
        clearTimeout(pollTimer); pollTimer = setTimeout(pollStatus, 15000);
      })
      .catch(function (e) { toast('Dispatch failed: ' + e.message, true); });
  });

  var repoIn = document.getElementById('r-repo');
  var tokIn = document.getElementById('r-token');
  if (repoIn) repoIn.value = cfg().repo;
  var save = document.getElementById('r-save');
  if (save) save.addEventListener('click', function () {
    if (repoIn.value.trim()) LS.setItem('scout.repo', repoIn.value.trim());
    if (tokIn.value.trim()) LS.setItem('scout.token', tokIn.value.trim());
    tokIn.value = '';
    refreshMode();
    toast(ready() ? 'Remote connected.' : 'Enter both the repo and a token.', !ready());
  });
  var clear = document.getElementById('r-clear');
  if (clear) clear.addEventListener('click', function () {
    LS.removeItem('scout.repo'); LS.removeItem('scout.token');
    refreshMode(); toast('Disconnected — token removed from this device.');
  });

  var q = document.getElementById('q');
  if (q) q.addEventListener('input', function () {
    var v = this.value.toLowerCase();
    document.querySelectorAll('.card').forEach(function (c) {
      c.style.display = !v || c.dataset.search.indexOf(v) !== -1 ? '' : 'none';
    });
  });

  refreshMode();
})();
"""


def _icon_png(size: int = 180) -> bytes:
    """Solid Apple-blue apple-touch-icon, generated without image deps
    (iOS rounds the corners itself)."""
    r, g, b = 0x00, 0x71, 0xE3
    row = b"\x00" + bytes((r, g, b)) * size  # filter byte + RGB pixels
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
