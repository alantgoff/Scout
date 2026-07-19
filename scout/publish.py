"""Static mobile digest — renders the lead ledger to docs/ for GitHub Pages.

`scout publish` turns the current deal flow into one self-contained,
phone-first HTML page (Apple design language, same track semantics as the
UI: launched startups grouped by company first, pre-launch watch second).
Push docs/ and GitHub Pages serves it; on a phone, "Add to Home Screen"
makes it a read-only Scout app that refreshes with every published scan.

Contains ONLY lead data — no secrets, no config. The page carries
<meta name="robots" content="noindex"> since Pages URLs are public.
"""

from __future__ import annotations

import html
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

from scout.companies import group_by_company, startup_identity
from scout.config import Thesis
from scout.models import Lead, LedgerEntry
from scout.store import Store

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


def _chips(lead: Lead, entry: LedgerEntry | None, status: str | None,
           firm: str = "") -> str:
    verdict = lead.llm
    chips: list[tuple[str, str]] = []
    if verdict and verdict.thesis_fit is not None:
        chips.append((f"Fit {verdict.thesis_fit:.0%}", "accent"))
    if verdict and verdict.value_add_fit is not None:
        chips.append((f"{firm or 'Firm'} lift {verdict.value_add_fit:.0%}", "accent"))
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
          pipeline_row: dict, startup: bool, firm: str = "") -> str:
    account, verdict = primary.account, primary.llm
    # Startup-first titling, same system as the app: real company name, or the
    # synthesized stealth identity tied to the founder.
    identity = startup_identity(primary)
    if identity:
        title, synthesized = identity
        subtitle = (f"@{account.handle}" if synthesized
                    else f"{account.name or account.handle} · @{account.handle}")
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
    return f"""<article class="card" data-search="{_e(' '.join([title, account.handle, account.name, summary, (verdict.sector or '') if verdict else '', ' '.join(verdict.tags) if verdict else '']).lower())}">
  <div class="row">
    <div class="grow">
      <div class="name">{title_html} <span class="sub">{_e(subtitle)}</span></div>
      <div class="summary">{_e(summary)}</div>
      {_chips(primary, entry, status if status and status != "new" else None, firm)}
      {f'<div class="why">{_e(why)}</div>' if why else ''}
      {also}
      {brief_html}
    </div>
    <div class="scorecol"><div class="score">{primary.score:.0f}</div>
      <a class="xlink" href="{_e(account.url)}">Profile →</a></div>
  </div>
</article>"""


def build_digest(store: Store, thesis: Thesis, out_dir: Path) -> Path:
    """Render the ledger into out_dir/index.html (+ apple-touch icon)."""
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
        _card(p, e, secs, row_for(p), startup=True, firm=thesis.firm_name)
        for p, e, secs in startups
    )
    watch_cards = "\n".join(
        _card(x, e, [], row_for(x), startup=False, firm=thesis.firm_name)
        for x, e in watch
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
</style></head><body>
<h1>Scout</h1>
<div class="thesis">{_e(thesis.thesis)}</div>
<div class="meta">Updated {updated} · read-only digest — triage in the desktop app</div>
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
<footer>Generated by scout · not indexed</footer>
<script>
document.getElementById('q').addEventListener('input', function () {{
  var q = this.value.toLowerCase();
  document.querySelectorAll('.card').forEach(function (c) {{
    c.style.display = !q || c.dataset.search.indexOf(q) !== -1 ? '' : 'none';
  }});
}});
</script>
</body></html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    (out_dir / "icon.png").write_bytes(_icon_png())
    return path


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
