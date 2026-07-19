"""scout — a deal-flow workspace over the CLI, in Apple design language.

Five surfaces, content first:

  LEADS     — the ranked lead feed: information-rich cards, triage inline
  PIPELINE  — work the shortlist to allocation: status, notes, outreach, briefs
  SOURCING  — the AI strategy agent, run controls, and (behind disclosure)
              every manual knob: query bank, watchlist, weights, prompt
  DATABASE  — the raw store, browsable: any table, auto-generated filters,
              full-text search, CSV export, read-only SQL
  SETTINGS  — keys, budget, defaults

Edits write back to thesis.yaml / seeds.yaml / .env; deal-flow state lives in
scout.db. The UI never stores secrets. Launch with `./scout-cli ui`.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from scout.agents import (
    apply_strategy,
    generate_strategy,
    research_brief,
    suggest_weights,
    validate_watchlist,
)
from scout.config import (
    STAGES,
    Seeds,
    Settings,
    SignalParams,
    Thesis,
    load_seeds,
    load_thesis,
    save_seeds,
    save_thesis,
)
from scout.companies import group_by_company
from scout.export import pipeline_rows, write_pipeline_csv
from scout.insights import stats_prompt, triage_stats
from scout.models import Lead, LedgerEntry
from scout.outreach import CHANNELS, draft_outreach
from scout.score import score_breakdown
from scout.signals.llm import DEFAULT_PROMPT_TEMPLATE
from scout.store import Store

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THESIS_PATH = PROJECT_ROOT / "thesis.yaml"
SEEDS_PATH = PROJECT_ROOT / "seeds.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

SIGNAL_HELP = {
    "bio_intent": "Bio matches a thesis keyword ('stealth', 'building something new', …)",
    "smart_money_convergence": "Multiple watchlist investors NEWLY followed this account — the strongest X signal",
    "departure_signal": "Bio shows a departure marker from target_bios ('ex-OpenAI', …)",
    "bio_change": "Stealth/intent language newly APPEARED in the bio since last seen",
    "smart_money_follow": "Followed by watchlist members at all, any age (saturates at 3)",
    "launch_traction": "Recent tweet with launch language and high engagement/followers",
    "builder_evidence": "GitHub or personal-site link in the bio",
    "github_evidence": "Discovered via a recent, starred GitHub repo in your thesis topics",
    "source_corroboration": "Independently surfaced by 2+ discovery strategies (search + GitHub + graph…)",
}

STATUS_LABELS = {
    "new": "New",
    "shortlisted": "To reach out",
    "contacted": "Contacted",
    "meeting": "Meeting",
    "diligence": "Diligence",
    "won": "Allocated",
    "passed": "Passed",
}
LABEL_TO_STATUS = {v: k for k, v in STATUS_LABELS.items()}
WIN_STAGES = ["shortlisted", "contacted", "meeting", "diligence", "won"]

STAGE_LABEL = {"idea": "Idea", "stealth": "Stealth", "launched": "Launched", "scaling": "Scaling"}
TYPE_LABEL = {"founder": "Founder", "startup": "Startup", "other": "Other"}


# --------------------------------------------------------------------- styling


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Committed light appearance — matches .streamlit/config.toml [theme],
           which pins Streamlit's native widgets to the same look. */
        :root {
          --bg:#fbfbfd; --surface:#ffffff; --ink:#1d1d1f; --ink-2:#494949;
          /* #6e6e73 = Apple's secondary label; ~4.9:1 on --bg, passes AA for text */
          --muted:#6e6e73; --hair:rgba(0,0,0,0.08); --hair-strong:rgba(0,0,0,0.14);
          --accent:#0071e3; --accent-soft:rgba(0,113,227,0.10);
          --good:#1d7a3a; --track:rgba(0,0,0,0.06);
          --shadow:0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.05);
        }
        html, body, [class*="css"], .stMarkdown, button, input, textarea {
          font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
            "Helvetica Neue",system-ui,sans-serif !important;
          -webkit-font-smoothing:antialiased;
        }
        .stApp { background:var(--bg); }
        .block-container { padding-top:2.2rem; padding-bottom:4rem; max-width:1080px; }
        #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"] { visibility:hidden; height:0; }

        /* Typography */
        h1,h2,h3,h4 { letter-spacing:-0.022em; color:var(--ink); }
        .hero-title { font-size:2.35rem; font-weight:700; letter-spacing:-0.03em;
          color:var(--ink); line-height:1.05; margin:0; }
        .hero-sub { color:var(--muted); font-size:1.02rem; margin-top:6px;
          font-weight:400; max-width:44rem; }
        .section-title { font-size:1.35rem; font-weight:650; letter-spacing:-0.02em;
          color:var(--ink); margin:0 0 2px; }
        .section-sub { color:var(--muted); font-size:0.92rem; margin:0 0 14px; }
        .subtle { color:var(--muted); font-size:0.88rem; }

        /* Tabs → centered pill control (Streamlit ≥1.59 react-aria markup) */
        .stTabs [role="tablist"] { gap:2px; background:var(--track); padding:3px;
          border-radius:12px; width:fit-content; margin:1.4rem auto 1.6rem;
          border-bottom:none !important; }
        .stTabs [data-testid="stTab"] { height:34px; border-radius:9px; padding:0 22px;
          background:transparent; border:none; display:flex; align-items:center; }
        .stTabs [data-testid="stTab"] p { font-size:0.9rem !important; font-weight:500;
          color:var(--ink-2); }
        .stTabs [data-testid="stTab"][aria-selected="true"] { background:var(--surface);
          box-shadow:0 1px 4px rgba(0,0,0,0.14); }
        .stTabs [data-testid="stTab"][aria-selected="true"] p { color:var(--ink);
          font-weight:600; }
        .stTabs .react-aria-SelectionIndicator { display:none !important; }

        /* Buttons */
        .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
          border-radius:980px; font-weight:500; font-size:0.88rem;
          border:1px solid var(--hair-strong); background:var(--surface); color:var(--ink);
          padding:0.32rem 1rem; transition:all .12s ease; box-shadow:none; }
        .stButton>button:hover, .stDownloadButton>button:hover { border-color:var(--accent);
          color:var(--accent); background:var(--surface); }
        .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
          background:var(--accent); border-color:var(--accent); color:#fff; }
        .stButton>button[kind="primary"]:hover { background:#0077ed; color:#fff; }
        .stButton>button:active { transform:scale(0.97); }
        .stButton>button:disabled { opacity:0.4; cursor:not-allowed; }
        .stButton>button:disabled:hover { border-color:var(--hair-strong); color:var(--ink); }
        .stButton>button[kind="primary"]:disabled:hover { background:var(--accent);
          border-color:var(--accent); color:#fff; }

        /* Segmented controls → the same pill group as the tabs. The track is
           the inner radiogroup (the outer testid node also wraps the label). */
        [data-testid="stButtonGroup"] [role="radiogroup"] { gap:2px;
          background:var(--track); padding:3px; border-radius:12px; width:fit-content; }
        [data-testid="stButtonGroup"] button { border:none !important;
          border-radius:9px !important; min-height:30px;
          padding:0.18rem 0.9rem !important; background:transparent !important;
          box-shadow:none !important; }
        [data-testid="stButtonGroup"] button p { font-size:0.86rem !important;
          font-weight:500; color:var(--ink-2); }
        [data-testid="stButtonGroup"] button:hover p { color:var(--ink); }
        [data-testid="stButtonGroup"] button[aria-checked="true"] {
          background:var(--surface) !important;
          box-shadow:0 1px 4px rgba(0,0,0,0.14) !important; }
        [data-testid="stButtonGroup"] button[aria-checked="true"] p {
          color:var(--ink); font-weight:600; }

        /* Cards & tiles */
        [data-testid="stVerticalBlockBorderWrapper"] {
          background:var(--surface); border:1px solid var(--hair); border-radius:16px;
          box-shadow:var(--shadow);
          transition:box-shadow .18s ease, border-color .18s ease; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
          border-color:var(--hair-strong);
          box-shadow:0 2px 4px rgba(0,0,0,0.05), 0 14px 36px rgba(0,0,0,0.08); }
        [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
          padding:0.65rem 0.75rem; }
        .tile { background:var(--surface); border:1px solid var(--hair); border-radius:16px;
          padding:16px 18px; box-shadow:var(--shadow); }
        .tile .label { color:var(--muted); font-size:0.72rem; text-transform:uppercase;
          letter-spacing:0.06em; font-weight:600; }
        .tile .value { color:var(--ink); font-size:1.75rem; font-weight:650;
          letter-spacing:-0.02em; line-height:1.15; margin-top:4px; }
        .tile .sub { color:var(--muted); font-size:0.8rem; margin-top:2px; }

        /* Lead card internals */
        .lead-name { font-size:1.04rem; font-weight:600; color:var(--ink);
          letter-spacing:-0.01em; }
        .lead-name a { color:var(--ink); text-decoration:none; }
        .lead-name a:hover { color:var(--accent); }
        .lead-handle { color:var(--muted); font-weight:400; font-size:0.9rem; }
        .lead-summary { color:var(--ink-2); font-size:0.92rem; line-height:1.45;
          margin-top:3px; }
        .avatar { width:40px; height:40px; border-radius:50%; background:var(--accent-soft);
          color:var(--accent); display:flex; align-items:center; justify-content:center;
          font-weight:600; font-size:0.95rem; letter-spacing:0; flex:0 0 40px; }
        .lead-row { display:flex; gap:13px; align-items:flex-start; }

        .chiprow { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
        .chip { display:inline-block; padding:3px 10px; border-radius:980px;
          font-size:0.74rem; font-weight:500; background:var(--track); color:var(--ink-2);
          white-space:nowrap; }
        .chip.accent { background:var(--accent-soft); color:var(--accent); font-weight:600; }
        .chip.status { background:var(--ink); color:var(--bg); }
        .chip.invalid { text-decoration:line-through; opacity:0.55; }

        .scoreblock { text-align:right; }
        .scorenum { font-size:1.6rem; font-weight:650; letter-spacing:-0.02em;
          color:var(--ink); line-height:1; }
        .scorecap { color:var(--muted); font-size:0.7rem; text-transform:uppercase;
          letter-spacing:0.06em; font-weight:600; margin-top:2px; }
        .scoretrack { width:92px; height:4px; border-radius:2px; background:var(--track);
          margin-top:8px; margin-left:auto; }
        .scorefill { height:4px; border-radius:2px; background:var(--accent); }

        /* Signal bars (single hue — magnitude) */
        .sigrow { display:flex; align-items:center; gap:10px; margin:5px 0; }
        .signame { flex:0 0 190px; font-size:0.82rem; color:var(--ink-2); }
        .sigtrack { flex:1; height:5px; border-radius:2.5px; background:var(--track); }
        .sigfill { height:5px; border-radius:2.5px; background:var(--accent); }
        .sigpts { flex:0 0 46px; text-align:right; font-size:0.82rem; color:var(--ink);
          font-variant-numeric:tabular-nums; font-weight:550; }
        .sigdetail { flex:0 0 34%; font-size:0.76rem; color:var(--muted);
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

        .math-step { font-size:0.84rem; color:var(--muted); padding:2px 0;
          font-variant-numeric:tabular-nums; }
        .math-step b { color:var(--ink); font-weight:600; }

        /* Expanders — quiet */
        [data-testid="stExpander"] { border:none !important; box-shadow:none !important;
          background:transparent !important; }
        [data-testid="stExpander"] details { border:none !important; background:transparent; }
        [data-testid="stExpander"] summary { font-size:0.84rem; color:var(--muted);
          font-weight:500; }
        [data-testid="stExpander"] summary:hover { color:var(--accent); }

        .nudge { background:var(--accent-soft); border-radius:12px; padding:10px 16px;
          font-size:0.88rem; color:var(--ink-2); margin:0 0 16px; }
        .nudge b { color:var(--ink); }

        /* Live scan banner (auto-refreshing fragment) */
        .scanbar { background:var(--accent-soft); border-radius:12px; padding:10px 16px;
          font-size:0.88rem; color:var(--ink-2); margin:14px 0 0;
          display:flex; align-items:center; gap:9px; }
        .scanbar b { color:var(--ink); }
        .scanbar.failed { background:rgba(196,66,66,0.08); }
        .scandot { width:8px; height:8px; border-radius:50%; background:var(--accent);
          flex:0 0 8px; animation:scanpulse 1.6s ease-in-out infinite; }
        @keyframes scanpulse { 0%,100% { opacity:1; transform:scale(1); }
          50% { opacity:0.35; transform:scale(0.8); } }

        hr { border-color:var(--hair) !important; margin:1.9rem 0 1.5rem !important; }
        [data-testid="stWidgetLabel"] p { font-size:0.83rem; color:var(--ink-2);
          font-weight:500; }

        /* Inputs — hairline borders, soft focus ring, one radius everywhere */
        .stTextInput [data-baseweb="input"], .stNumberInput [data-baseweb="input"],
        .stTextArea [data-baseweb="textarea"] {
          border-radius:10px !important; border-color:var(--hair-strong) !important;
          background:var(--surface) !important; transition:border-color .12s ease,
          box-shadow .12s ease; }
        .stTextInput [data-baseweb="input"]:focus-within,
        .stNumberInput [data-baseweb="input"]:focus-within,
        .stTextArea [data-baseweb="textarea"]:focus-within {
          border-color:var(--accent) !important;
          box-shadow:0 0 0 3px var(--accent-soft); }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
          border-radius:10px !important; border-color:var(--hair-strong) !important;
          background:var(--surface) !important; }
        .stSelectbox [data-baseweb="select"]:focus-within > div,
        .stMultiSelect [data-baseweb="select"]:focus-within > div {
          border-color:var(--accent) !important;
          box-shadow:0 0 0 3px var(--accent-soft); }

        /* Data tables — framed like cards */
        [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
          border-radius:12px; overflow:hidden; }
        [data-testid="stDataFrame"] > div { border-radius:12px; }

        /* Alerts & toasts — same rounding as everything else */
        [data-testid="stAlert"] { border-radius:12px; }
        [data-testid="stPopoverBody"] { border-radius:14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _e(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'


def _chips(items: list[tuple[str, str]]) -> str:
    """items: (text, css_class) — css_class in {"", "accent", "status"}.
    Deduped case-insensitively (a tag often repeats the sector or model)."""
    seen: set[str] = set()
    spans = []
    for text, cls in items:
        key = (text or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        spans.append(f'<span class="chip {cls}">{_e(text)}</span>')
    return f'<div class="chiprow">{"".join(spans)}</div>' if spans else ""


def _initials(name: str, handle: str) -> str:
    src = (name or handle).strip()
    parts = [p for p in src.split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return src[:2].upper() if src else "?"


# --------------------------------------------------------------------- helpers


def _ago(ts: str | None) -> str:
    """Compact relative time for ISO timestamps ('just now', '2h ago')."""
    if not ts:
        return ""
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - then).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400 * 2:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _to_lines(items: list[str]) -> str:
    return "\n".join(items)


def _from_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _set_env_var(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    for i, line in enumerate(lines):
        if line.split("#", 1)[0].strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stream_command(args: list[str]) -> None:
    # Click-time guard: the page may have rendered before another scan
    # started (buttons enabled), so re-check right before launching.
    if (store.current_scan() or {}).get("status") == "running":
        st.warning("A scan is already running — wait for the banner to clear.")
        return
    box = st.empty()
    lines: list[str] = []
    env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "COLUMNS": "120"}
    proc = subprocess.Popen([sys.executable, "-m", "scout.cli", *args], cwd=PROJECT_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]) or "…", language=None)
    proc.wait()
    (st.success if proc.returncode == 0 else st.error)(
        "Done." if proc.returncode == 0 else f"Exited {proc.returncode} — see output.")


# ----------------------------------------------------------------------- page


st.set_page_config(page_title="Scout", page_icon="🔭", layout="wide")
_inject_css()

# Toasts queued before an st.rerun() would be lost with a direct call —
# actions stash the message here and it fires on the following run.
if _pending_toast := st.session_state.pop("toast", None):
    st.toast(_pending_toast)

settings = Settings(_env_file=ENV_PATH if ENV_PATH.exists() else None)
thesis = load_thesis(THESIS_PATH)
seeds = load_seeds(SEEDS_PATH)
store = Store(Path(settings.db_path), cross_thread=True)

leads = store.load_latest_leads()
pipeline = store.all_pipeline()
counts = store.pipeline_counts()
# The ledger is the person-centric view: best-known state per handle across
# ALL runs. It backs the "All leads" scope and — always — the Pipeline tab,
# so a shortlisted lead never degrades just because it missed the latest run.
latest_is_demo = bool(leads) and all(x.account.source == "demo" for x in leads)
ledger = store.load_lead_ledger(include_demo=latest_is_demo)
entry_by_handle = {e.lead.account.handle.lower(): e for e in ledger}
lead_by_handle = {h: e.lead for h, e in entry_by_handle.items()}
latest_handles = {x.account.handle.lower() for x in leads}


def _status_of(lead: Lead) -> str:
    return pipeline.get(lead.account.handle.lower(), {}).get("status") or "new"


# Header — large title, thesis as the subtitle
st.markdown(
    f'<div class="hero-title">Scout</div>'
    f'<div class="hero-sub">{_e(thesis.thesis) or "No thesis yet — open Sourcing and describe one."}</div>',
    unsafe_allow_html=True,
)

st.session_state.setdefault(
    "page_loaded_at", datetime.now(timezone.utc).isoformat()
)


@st.fragment(run_every="4s")
def _scan_indicator() -> None:
    """Live scan status, visible on every tab. Auto-polls the store every few
    seconds (fragment rerun — the rest of the page is untouched). Shows a
    pulsing banner while a scan runs, and a refresh prompt when one finished
    after this page was loaded."""
    scan = store.current_scan()
    if not scan:
        return
    kind = scan.get("kind") or "scan"
    detail = f' · {scan["detail"]}' if scan.get("detail") else ""
    if scan.get("status") == "running":
        st.markdown(
            f'<div class="scanbar"><span class="scandot"></span>'
            f'<b>{_e(kind.capitalize())} running</b> — {_e(scan.get("phase") or "…")}'
            f'{_e(detail)} · started {_ago(scan.get("started_at"))}</div>',
            unsafe_allow_html=True,
        )
    elif (scan.get("finished_at") or "") > st.session_state["page_loaded_at"]:
        outcome = "finished" if scan.get("status") == "done" else "failed"
        st.markdown(
            f'<div class="scanbar {"" if outcome == "finished" else "failed"}">'
            f'<b>{_e(kind.capitalize())} {outcome}</b> {_ago(scan.get("finished_at"))}'
            f'{_e(detail)}</div>',
            unsafe_allow_html=True,
        )
        if scan.get("status") == "done":
            if st.button("Refresh results", key="scan_refresh"):
                st.session_state["page_loaded_at"] = datetime.now(timezone.utc).isoformat()
                st.rerun(scope="app")


_scan_indicator()

tab_leads, tab_pipeline, tab_sourcing, tab_data, tab_settings = st.tabs(
    ["Leads", "Pipeline", "Sourcing", "Database", "Settings"]
)


# ============================================================ LEADS


def _lead_card(
    lead: Lead,
    entry: LedgerEntry | None = None,
    fresh: bool = True,
    view_max: float = 100.0,
    pct_label: str = "",
    secondary: list[Lead] | None = None,
) -> None:
    """One lead card. `entry` carries cross-run movement (delta / new); `fresh`
    means the lead is from the latest run — run-scoped signals like new
    watchlist follows are suppressed on stale entries. `view_max` normalizes
    the score bar to the strongest lead in view; `pct_label` is an optional
    percentile caption ("top 12%"). `secondary` holds other X accounts folded
    into this startup (Startups track) — the card is titled by the company."""
    account, verdict = lead.account, lead.llm
    handle_key = account.handle.lower()
    status = _status_of(lead)
    secondary = secondary or []

    chips: list[tuple[str, str]] = []
    if verdict and verdict.thesis_fit is not None:
        chips.append((f"Fit {verdict.thesis_fit:.0%}", "accent"))
    if verdict and verdict.value_add_fit is not None:
        chips.append((f"{thesis.firm_name or 'Firm'} lift {verdict.value_add_fit:.0%}", "accent"))
    if status != "new":
        chips.append((STATUS_LABELS.get(status, status), "status"))
    if entry and entry.is_new:
        chips.append(("New", "accent"))
    if entry and entry.score_delta is not None and abs(entry.score_delta) >= 1:
        arrow = "▲" if entry.score_delta > 0 else "▼"
        chips.append(
            (f"{arrow} {abs(entry.score_delta):.0f}",
             "accent" if entry.score_delta > 0 else "")
        )
    if verdict and verdict.account_type:
        chips.append((TYPE_LABEL.get(verdict.account_type, ""), ""))
    if verdict and verdict.stage:
        chips.append((STAGE_LABEL.get(verdict.stage, verdict.stage), ""))
    sector = " · ".join(x for x in [verdict.sector if verdict else "", verdict.subsector if verdict else ""] if x)
    if sector:
        chips.append((sector, ""))
    if verdict and verdict.business_model:
        chips.append((verdict.business_model, ""))
    if fresh and account.recent_followed_by:
        chips.append((f"New follows: {', '.join(account.recent_followed_by[:3])}", "accent"))
    # Keep the card face calm; tags, provenance, and any overflow live in Details.
    face_chips = chips[:7]
    detail_chips: list[tuple[str, str]] = [(t, "") for t in (verdict.tags if verdict else [])]
    if len({s for s in account.sources if s}) > 1:
        detail_chips.append((f"sources: {', '.join(sorted({s for s in account.sources if s}))}", ""))
    detail_chips += chips[7:]

    summary = (verdict.one_line_summary if verdict else "") or account.bio or "—"

    # Startup-first identity: when the classifier named the company, the card
    # is titled by the STARTUP; the account becomes the byline.
    company = (verdict.company_name or "").strip() if verdict else ""
    company_url = (verdict.company_url or "").strip() if verdict else ""
    if company:
        title_href = _e(company_url or account.url)
        title_html = (
            f'<a href="{title_href}" target="_blank">{_e(company)}</a> '
            f'<span class="lead-handle">{_e(account.name or account.handle)} · '
            f'@{_e(account.handle)} · {account.followers:,} followers</span>'
        )
        avatar_text = _initials(company, account.handle)
    else:
        title_html = (
            f'<a href="{account.url}" target="_blank">{_e(account.name or account.handle)}</a> '
            f'<span class="lead-handle">@{_e(account.handle)} · {account.followers:,} followers</span>'
        )
        avatar_text = _initials(account.name, account.handle)

    secondary_html = ""
    if secondary:
        others = ", ".join(f"@{_e(x.account.handle)}" for x in secondary)
        secondary_html = (
            f'<div class="chiprow"><span class="chip">Also tracking: {others}</span></div>'
        )

    with st.container(border=True):
        col_main, col_score = st.columns([5.2, 1.1])
        with col_main:
            st.markdown(
                f"""
                <div class="lead-row">
                  <div class="avatar">{_e(avatar_text)}</div>
                  <div style="min-width:0">
                    <div class="lead-name">{title_html}</div>
                    <div class="lead-summary">{_e(summary)}</div>
                    {_chips(face_chips)}
                    {secondary_html}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_score:
            bar_pct = min(max(lead.score / (view_max or 1.0) * 100, 0), 100)
            pct_html = (f'<div class="scorecap" style="margin-top:5px">{_e(pct_label)}</div>'
                        if pct_label else "")
            st.markdown(
                f"""
                <div class="scoreblock">
                  <div class="scorenum">{lead.score:.0f}</div>
                  <div class="scorecap">score</div>
                  <div class="scoretrack"><div class="scorefill" style="width:{bar_pct:.0f}%"></div></div>
                  {pct_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Triage lives on the card face — no expanding needed to act.
            if status in WIN_STAGES:
                if st.button("Remove", key=f"rm_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="new")
                    st.session_state["toast"] = f"Removed @{account.handle} from the pipeline"
                    st.rerun()
            elif status == "passed":
                if st.button("Restore", key=f"restore_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="new")
                    st.session_state["toast"] = f"Restored @{account.handle}"
                    st.rerun()
            else:
                if st.button("Shortlist", key=f"short_{handle_key}", type="primary",
                             use_container_width=True):
                    store.set_pipeline(account.handle, status="shortlisted")
                    st.session_state["toast"] = f"Shortlisted @{account.handle}"
                    st.rerun()
                if st.button("Pass", key=f"pass_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="passed")
                    st.session_state["toast"] = f"Passed on @{account.handle}"
                    st.rerun()

        # Stateless expander: the `expanded` prop is re-applied only when its
        # value CHANGES between reruns, so user toggles persist. Actions that
        # must keep a card open (brief generation) set open_card before rerun.
        with st.expander("Details", expanded=(handle_key == st.session_state.get("open_card"))):
            if verdict and verdict.why_interesting:
                st.markdown(f'<div class="lead-summary">{_e(verdict.why_interesting)}</div>',
                            unsafe_allow_html=True)
            if verdict and verdict.fit_reason:
                st.markdown(f'<div class="subtle" style="margin-top:6px">Fit — {_e(verdict.fit_reason)}</div>',
                            unsafe_allow_html=True)
            # The value-add dimension: which of the firm's specific levers
            # would accelerate this startup, per the classifier.
            if verdict and verdict.value_add_fit is not None:
                firm = thesis.firm_name or "Firm"
                reason = f" — {verdict.value_add_reason}" if verdict.value_add_reason else ""
                st.markdown(
                    f'<div class="subtle" style="margin-top:6px">{_e(firm)} lift '
                    f'{verdict.value_add_fit:.0%}{_e(reason)}</div>',
                    unsafe_allow_html=True,
                )
                lever_labels = {x.key: x.label for x in thesis.firm_value_add}
                lever_help = {x.key: x.description for x in thesis.firm_value_add}
                levers = [(k, min(max(v, 0.0), 1.0))
                          for k, v in verdict.value_add_levers.items() if v > 0]
                if levers:
                    rows = "".join(
                        f'<div class="sigrow"><div class="signame" title="{_e(lever_help.get(k, ""))}">{_e(lever_labels.get(k, k))}</div>'
                        f'<div class="sigtrack"><div class="sigfill" style="width:{100 * v:.0f}%"></div></div>'
                        f'<div class="sigpts">{v:.0%}</div>'
                        f'<div class="sigdetail"></div></div>'
                        for k, v in sorted(levers, key=lambda kv: -kv[1])
                    )
                    st.markdown(f'<div style="margin-top:4px">{rows}</div>', unsafe_allow_html=True)
            if entry and entry.times_seen > 1 and entry.first_seen_at:
                st.markdown(
                    f'<div class="subtle" style="margin-top:6px">Seen {entry.times_seen}× '
                    f'since {entry.first_seen_at:%b %d}'
                    + (f" · previous score {entry.prev_score:.0f}" if entry.prev_score is not None else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            st.write("")

            hits = [s for s in lead.signals if s.value > 0]
            if hits:
                max_pts = max(s.contribution for s in hits) or 1.0
                rows = "".join(
                    f'<div class="sigrow"><div class="signame" title="{_e(SIGNAL_HELP.get(s.name, ""))}">{_e(s.name)}</div>'
                    f'<div class="sigtrack"><div class="sigfill" style="width:{100 * s.contribution / max_pts:.0f}%"></div></div>'
                    f'<div class="sigpts">{s.contribution:.1f}</div>'
                    f'<div class="sigdetail" title="{_e(s.detail)}">{_e(s.detail)}</div></div>'
                    for s in sorted(hits, key=lambda s: -s.contribution)
                )
                st.markdown(rows, unsafe_allow_html=True)

            steps = "".join(
                f'<div class="math-step">{_e(desc)} → <b>{running:.1f}</b></div>'
                for desc, running in score_breakdown(lead, thesis)
            )
            st.markdown(f'<div style="margin-top:10px">{steps}</div>', unsafe_allow_html=True)

            link_urls = list(lead.evidence_links)
            if company_url and company_url not in link_urls:
                link_urls.insert(1, company_url)
            links = " · ".join(
                f'<a href="{_e(u)}" target="_blank">{_e(u.removeprefix("https://").removeprefix("http://"))}</a>'
                for u in link_urls
            )
            st.markdown(f'<div class="subtle" style="margin-top:8px">{links}</div>',
                        unsafe_allow_html=True)
            if detail_chips:
                st.markdown(_chips(detail_chips), unsafe_allow_html=True)
            st.write("")

            row = pipeline.get(handle_key, {})
            b1, _sp = st.columns([1.4, 4.6])
            if b1.button("Research brief", key=f"brief_{handle_key}"):
                with st.spinner("Compiling brief…"):
                    brief, is_ai = research_brief(lead, thesis, settings)
                store.set_pipeline(account.handle, brief=brief)
                st.session_state["open_card"] = handle_key  # keep this card open
                st.session_state["toast"] = (
                    f"Brief ready for @{account.handle}" if is_ai
                    else "No Anthropic key — brief is data-only."
                )
                st.rerun()

            if row.get("brief"):
                st.markdown("---")
                if row.get("brief_at"):
                    st.markdown(f'<div class="subtle">Generated {_ago(row["brief_at"])}</div>',
                                unsafe_allow_html=True)
                st.markdown(row["brief"])


with tab_leads:
    if not leads and not ledger:
        st.markdown(
            '<div class="section-title">No leads yet</div>'
            '<div class="section-sub">Open <b>Sourcing</b>, describe your thesis, and run discovery. '
            'Or run <code>./scout-cli demo</code> for an offline sample.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Cadence nudge: follow-graph diffing and bio-change tracking only work
        # with regular runs — surface staleness instead of scheduling anything.
        last_run = store.last_real_run_at()
        if last_run is None:
            st.markdown(
                '<div class="nudge">No real discovery run yet — these are sample leads. '
                'When your X cookies are set, open <b>Sourcing → Run discovery</b>.</div>',
                unsafe_allow_html=True,
            )
        elif (datetime.now(timezone.utc) - last_run).total_seconds() > 24 * 3600:
            st.markdown(
                f'<div class="nudge">Last discovery run {_ago(last_run.isoformat())} — '
                'daily runs keep the follow-graph and bio-change signals meaningful. '
                'Open <b>Sourcing → Run discovery</b>.</div>',
                unsafe_allow_html=True,
            )

        strategies = store.list_strategies()
        sc1, sc2, sc3 = st.columns([2.3, 1.4, 2.3])
        with sc1:
            # THE product split: real launched startups first; people the
            # system expects to launch soon are the completeness track.
            track = st.segmented_control(
                "Track", ["Startups", "Pre-launch watch", "Everything"],
                default="Startups", key="leads_track", label_visibility="collapsed",
            ) or "Startups"
        with sc2:
            scope = st.segmented_control(
                "Scope", ["All runs", "Latest run"], default="All runs",
                key="leads_time_scope", label_visibility="collapsed",
            ) or "All runs"
        strategy_hash = None
        with sc3:
            if scope == "All runs" and len(strategies) >= 2:
                def _strategy_label(h: str | None) -> str:
                    if h is None:
                        return "All strategies"
                    s = next(x for x in strategies if x["strategy_hash"] == h)
                    statement = (s["thesis_statement"] or "(no thesis statement)")[:60]
                    plural = "s" if s["run_count"] != 1 else ""
                    return f"{statement} · {s['run_count']} run{plural}"
                strategy_hash = st.selectbox(
                    "Strategy", [None] + [s["strategy_hash"] for s in strategies],
                    format_func=_strategy_label, label_visibility="collapsed",
                    key="leads_strategy",
                    help="Runs made with identical thesis + seeds group into one strategy.",
                )

        # (lead, ledger entry) pairs for the chosen time scope
        if scope == "Latest run":
            pairs = [(x, entry_by_handle.get(x.account.handle.lower())) for x in leads]
        elif strategy_hash:
            filtered = store.load_lead_ledger(
                include_demo=latest_is_demo, strategy_hash=strategy_hash
            )
            pairs = [(e.lead, e) for e in filtered]
        else:
            pairs = [(e.lead, e) for e in ledger]

        LAUNCHED = {"launched", "scaling"}
        PRELAUNCH = {"idea", "stealth"}
        WATCH_SIGNALS = {"departure_signal", "bio_change", "bio_intent"}

        def _in_track(lead: Lead) -> bool:
            verdict = lead.llm
            stage = verdict.stage if verdict else None
            if track == "Startups":
                # A real, launched company — reached via its founder or its own account.
                return (stage in LAUNCHED
                        and (verdict.account_type or "other") != "other")
            if track == "Pre-launch watch":
                if stage in PRELAUNCH:
                    return True
                # Unclassified but showing pre-launch tells (departure, fresh
                # stealth language) — that's exactly what this track watches.
                return stage is None and any(
                    s.name in WATCH_SIGNALS and s.value > 0 for s in lead.signals
                )
            return True  # Everything

        pairs = [(x, e) for x, e in pairs if _in_track(x)]
        base_leads = [x for x, _ in pairs]

        n_new = sum(1 for _, e in pairs if e and e.is_new)
        n_convergence = sum(1 for x in leads if x.account.recent_followed_by)
        fits = [x.llm.thesis_fit for x in base_leads if x.llm and x.llm.thesis_fit is not None]
        strong_fit = sum(1 for f in fits if f >= 0.7)
        in_pipeline = sum(counts.get(s, 0) for s in WIN_STAGES)
        sub = f"{n_new} new this run" if scope == "All runs" else "latest run"
        t1, t2, t3, t4 = st.columns(4)
        if track == "Startups":
            n_companies = len(group_by_company(pairs))
            t1.markdown(_tile("Launched startups", str(n_companies), sub), unsafe_allow_html=True)
        elif track == "Pre-launch watch":
            t1.markdown(_tile("Pre-launch people", str(len(base_leads)), sub), unsafe_allow_html=True)
        else:
            t1.markdown(_tile("Tracked leads", str(len(base_leads)), sub), unsafe_allow_html=True)
        t2.markdown(_tile("Strong fit", str(strong_fit), "thesis fit ≥ 70%"), unsafe_allow_html=True)
        t3.markdown(_tile("Smart-money events", str(n_convergence), "new follows · latest run"),
                    unsafe_allow_html=True)
        t4.markdown(_tile("In pipeline", str(in_pipeline), f"{counts.get('won', 0)} allocated"), unsafe_allow_html=True)
        st.write("")

        # Reset must land BEFORE the filter widgets instantiate.
        FILTER_DEFAULTS: dict = {
            "f_type": ["founder", "startup"], "f_stage": list(STAGES),
            "f_minscore": 0, "f_minfit": 0, "f_hidepassed": True,
        }
        if st.session_state.pop("filters_reset", False):
            for k, v in FILTER_DEFAULTS.items():
                st.session_state[k] = v
        n_active = sum([
            set(st.session_state.get("f_type", FILTER_DEFAULTS["f_type"])) != {"founder", "startup"},
            set(st.session_state.get("f_stage", FILTER_DEFAULTS["f_stage"])) != set(STAGES),
            st.session_state.get("f_minscore", 0) > 0,
            st.session_state.get("f_minfit", 0) > 0,
            not st.session_state.get("f_hidepassed", True),
        ])

        f1, f2, f3, f4 = st.columns([2.6, 1.2, 1.2, 2])
        with f1:
            query = st.text_input("Search", placeholder="Search name, bio, sector, tags…",
                                  label_visibility="collapsed")
        with f2:
            lift_sort = f"{thesis.firm_name or 'Value-add'} lift"
            sort_by = st.selectbox("Sort", ["Score", "Score change", "Thesis fit", lift_sort,
                                            "Followers"],
                                   label_visibility="collapsed")
        with f3:
            with st.popover(f"Filters · {n_active}" if n_active else "Filters"):
                type_filter = st.multiselect("Type", ["founder", "startup", "other"],
                                             default=FILTER_DEFAULTS["f_type"], key="f_type",
                                             format_func=lambda t: TYPE_LABEL[t])
                stage_filter = st.multiselect("Stage", list(STAGES),
                                              default=FILTER_DEFAULTS["f_stage"], key="f_stage",
                                              format_func=lambda s: STAGE_LABEL[s])
                min_score = st.slider("Minimum score", 0, 100, 0, key="f_minscore")
                min_fit = st.slider("Minimum thesis fit", 0, 100, 0, format="%d%%", key="f_minfit")
                hide_passed = st.toggle("Hide passed", value=True, key="f_hidepassed")
                if n_active and st.button("Reset filters"):
                    st.session_state["filters_reset"] = True
                    st.rerun()
        with f4:
            pass

        HIDE_LABELS = {"type": "type filter", "stage": "stage filter",
                       "score": "min score", "fit": "min fit",
                       "passed": "passed", "search": "search"}

        def _hide_reason(lead: Lead) -> str | None:
            """None = visible; otherwise which filter hides this lead."""
            verdict = lead.llm
            stage = verdict.stage if verdict else None
            # Type filter applies only when the classifier explicitly typed the
            # account — unclassified/legacy leads are unknown, not "other"
            # (matters for the pre-launch watch, which is full of them).
            if verdict is not None and verdict.account_type is not None:
                if verdict.account_type not in type_filter:
                    return "type"
            if stage_filter and stage and stage not in stage_filter:
                return "stage"
            if lead.score < min_score:
                return "score"
            if min_fit and (not verdict or verdict.thesis_fit is None
                            or verdict.thesis_fit * 100 < min_fit):
                return "fit"
            if hide_passed and _status_of(lead) == "passed":
                return "passed"
            if query:
                haystack = " ".join(
                    [lead.account.name, lead.account.handle, lead.account.bio]
                    + ([verdict.sector or "", verdict.subsector or "",
                        verdict.one_line_summary, " ".join(verdict.tags)] if verdict else [])
                ).lower()
                if query.lower() not in haystack:
                    return "search"
            return None

        shown = []
        hidden_counts: dict[str, int] = {}
        for x, e in pairs:
            reason = _hide_reason(x)
            if reason is None:
                shown.append((x, e))
            else:
                hidden_counts[reason] = hidden_counts.get(reason, 0) + 1
        if sort_by == "Thesis fit":
            shown.sort(key=lambda p: -(p[0].llm.thesis_fit
                                       if p[0].llm and p[0].llm.thesis_fit is not None else -1))
        elif sort_by == lift_sort:
            shown.sort(key=lambda p: -(p[0].llm.value_add_fit
                                       if p[0].llm and p[0].llm.value_add_fit is not None else -1))
        elif sort_by == "Followers":
            shown.sort(key=lambda p: -p[0].account.followers)
        elif sort_by == "Score change":
            shown.sort(key=lambda p: (p[1].score_delta
                                      if p[1] and p[1].score_delta is not None
                                      else float("-inf")), reverse=True)

        # In the Startups track the unit is the company: fold founder +
        # company accounts attributed to the same startup into one card.
        if track == "Startups":
            display = group_by_company(shown)
        else:
            display = [(lead, entry, []) for lead, entry in shown]

        hidden_note = ""
        if hidden_counts:
            parts = [f"{count} by {HIDE_LABELS[reason]}"
                     for reason, count in sorted(hidden_counts.items(), key=lambda kv: -kv[1])]
            hidden_note = " · hidden: " + ", ".join(parts)
        if track == "Startups" and len(display) != len(shown):
            count_text = f"{len(display)} startups · {len(shown)} accounts of {len(pairs)}"
        elif track == "Startups":
            count_text = f"{len(display)} startups of {len(pairs)} accounts"
        else:
            count_text = f"{len(shown)} of {len(pairs)} leads"
        st.markdown(
            f'<div class="subtle" style="margin:4px 0 10px">{count_text}{hidden_note}</div>',
            unsafe_allow_html=True,
        )

        # Score bar + percentile are relative to what's actually in view.
        scores_desc = sorted((x.score for x, _ in shown), reverse=True)
        view_max = scores_desc[0] if scores_desc else 100.0

        def _pct_label(score: float) -> str:
            if len(scores_desc) < 10:
                return ""
            pct = -(-100 * (scores_desc.index(score) + 1) // len(scores_desc))  # ceil
            return f"top {pct}%" if pct <= 50 else ""

        # Reset pagination whenever the view changes (track, scope, filters…)
        page_size = 25
        view_key = repr((track, scope, strategy_hash, tuple(type_filter), tuple(stage_filter),
                         min_score, min_fit, hide_passed, query, sort_by))
        if st.session_state.get("leads_view_key") != view_key:
            st.session_state["leads_view_key"] = view_key
            st.session_state["leads_limit"] = page_size
        limit = st.session_state.get("leads_limit", page_size)

        for lead, entry, secondary in display[:limit]:
            _lead_card(lead, entry,
                       fresh=lead.account.handle.lower() in latest_handles,
                       view_max=view_max, pct_label=_pct_label(lead.score),
                       secondary=secondary)
        if len(display) > limit:
            if st.button(f"Show more ({len(display) - limit} remaining)"):
                st.session_state["leads_limit"] = limit + page_size
                st.rerun()
        if not display:
            if track == "Startups":
                st.markdown('<div class="subtle">No launched startups in view yet — try the '
                            '<b>Pre-launch watch</b> track, widen the filters, or run discovery.</div>',
                            unsafe_allow_html=True)
            elif scope == "Latest run" and ledger:
                st.markdown('<div class="subtle">No leads in the latest run — switch the scope '
                            'to <b>All runs</b> to see everything scout is tracking.</div>',
                            unsafe_allow_html=True)

        out_dir = PROJECT_ROOT / settings.out_dir
        artifacts = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        if artifacts:
            st.write("")
            st.markdown('<div class="subtle">Export</div>', unsafe_allow_html=True)
            for col, path in zip(st.columns(max(len(artifacts), 3)), artifacts):
                col.download_button(path.name, path.read_bytes(), file_name=path.name,
                                    use_container_width=True)


# ============================================================ PIPELINE


with tab_pipeline:
    shortlist = [h for h, p in pipeline.items() if (p.get("status") or "") in WIN_STAGES]
    if not shortlist:
        st.markdown(
            '<div class="section-title">Nothing in the pipeline</div>'
            '<div class="section-sub">Shortlist promising leads from the Leads tab — '
            'they land here as “To reach out”.</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(len(WIN_STAGES))
        for col, stage_key in zip(cols, WIN_STAGES):
            col.markdown(_tile(STATUS_LABELS[stage_key], str(counts.get(stage_key, 0))),
                         unsafe_allow_html=True)
        st.write("")

        editor_rows = []
        for handle_key in sorted(shortlist,
                                 key=lambda h: -(lead_by_handle[h].score if h in lead_by_handle else 0)):
            row = pipeline[handle_key]
            lead = lead_by_handle.get(handle_key)
            verdict = lead.llm if lead else None
            editor_rows.append({
                "handle": handle_key,
                "Lead": lead.account.url if lead else f"https://x.com/{handle_key}",
                "Fit": (f"{verdict.thesis_fit:.0%}" if verdict and verdict.thesis_fit is not None else "—"),
                "Score": lead.score if lead else 0,
                "Status": STATUS_LABELS.get(row.get("status", "shortlisted"), "To reach out"),
                "Notes": row.get("notes") or "",
            })
        edited = st.data_editor(
            editor_rows, hide_index=True, use_container_width=True, key="pipeline_editor",
            column_order=["Lead", "Fit", "Score", "Status", "Notes"],
            column_config={
                "handle": None,
                "Lead": st.column_config.LinkColumn("Lead", display_text=r"x\.com/(.+)",
                                                    disabled=True, width="small"),
                "Fit": st.column_config.TextColumn("Fit", disabled=True, width="small"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                         format="%d", width="medium"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=[STATUS_LABELS[s] for s in WIN_STAGES] + [STATUS_LABELS["passed"]],
                    required=True),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
        )
        for orig, new in zip(editor_rows, edited):
            if new["Status"] != orig["Status"] or new["Notes"] != orig["Notes"]:
                store.set_pipeline(orig["handle"],
                                   status=LABEL_TO_STATUS.get(new["Status"], "shortlisted"),
                                   notes=new["Notes"])
                st.rerun()

        st.write("")
        st.markdown('<div class="section-title">Work a lead</div>'
                    '<div class="section-sub">AI-drafted outreach and the research brief, side by side.</div>',
                    unsafe_allow_html=True)
        pick = st.selectbox("Lead", sorted(shortlist), format_func=lambda h: f"@{h}",
                            label_visibility="collapsed")
        picked_lead = lead_by_handle.get(pick)
        picked_row = pipeline.get(pick, {})

        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            st.markdown("**Outreach**")
            channel = st.selectbox("Channel", list(CHANNELS))
            if st.button("Draft with AI", type="primary", disabled=picked_lead is None):
                with st.spinner("Drafting…"):
                    message, is_ai = draft_outreach(picked_lead, thesis, settings, channel)
                store.set_pipeline(pick, outreach=message, channel=channel)
                if not is_ai:
                    st.info("No Anthropic key — used a template.")
                st.rerun()
            if picked_row.get("outreach_at"):
                st.markdown(f'<div class="subtle">Drafted {_ago(picked_row["outreach_at"])}</div>',
                            unsafe_allow_html=True)
            draft = st.text_area("Message", value=picked_row.get("outreach") or "",
                                 height=170, label_visibility="collapsed",
                                 placeholder="Draft with AI, or write your own…")
            if st.button("Save message"):
                store.set_pipeline(pick, outreach=draft, channel=channel)
                st.success("Saved.")
        with col_right:
            st.markdown("**Research brief**")
            if st.button("Generate brief", disabled=picked_lead is None):
                with st.spinner("Compiling brief…"):
                    brief, is_ai = research_brief(picked_lead, thesis, settings)
                store.set_pipeline(pick, brief=brief)
                if not is_ai:
                    st.info("No Anthropic key — brief is data-only.")
                st.rerun()
            existing_brief = picked_row.get("brief")
            if existing_brief:
                if picked_row.get("brief_at"):
                    st.markdown(f'<div class="subtle">Generated {_ago(picked_row["brief_at"])}</div>',
                                unsafe_allow_html=True)
                st.markdown(existing_brief)
            else:
                st.markdown('<div class="subtle">No brief yet — generate one for a pre-call memo: '
                            'evidence, thesis fit, risks, and questions to ask.</div>',
                            unsafe_allow_html=True)

        st.write("")
        export_rows = pipeline_rows(store)
        if export_rows:
            export_path = write_pipeline_csv(export_rows, PROJECT_ROOT / settings.out_dir)
            st.download_button(
                f"Export pipeline CSV ({len(export_rows)} leads)",
                export_path.read_bytes(), file_name=export_path.name,
            )


# ============================================================ SOURCING


with tab_sourcing:
    # --- AI strategy designer -------------------------------------------------
    st.markdown('<div class="section-title">Design the sourcing strategy</div>'
                '<div class="section-sub">Describe your thesis in plain language. The agent writes '
                'the targeting, the X query bank, bio searches, GitHub topics, and a watchlist — '
                'you review before anything is saved.</div>',
                unsafe_allow_html=True)
    description = st.text_area("Thesis", value=thesis.thesis, height=90,
                               label_visibility="collapsed",
                               placeholder="e.g. Technical founders leaving top AI labs to build vertical agents on proprietary data…")
    generate_col, _ = st.columns([1.6, 4])
    if generate_col.button("Design strategy with AI", type="primary",
                           disabled=not description.strip()):
        try:
            with st.spinner("Designing strategy — Claude is writing your query bank…"):
                proposal_new = generate_strategy(description, thesis, seeds, settings)
                _, wl_invalid, wl_validated = validate_watchlist(
                    proposal_new.watchlist, settings, store
                )
            st.session_state["proposal"] = proposal_new
            st.session_state["watchlist_invalid"] = wl_invalid
            st.session_state["watchlist_validated"] = wl_validated
        except RuntimeError as exc:
            st.error(str(exc))

    proposal = st.session_state.get("proposal")
    if proposal is not None:
        with st.container(border=True):
            st.markdown(f'**{_e(proposal.thesis)}**')
            st.markdown(f'<div class="subtle">{_e(proposal.rationale)}</div>', unsafe_allow_html=True)
            st.write("")
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("**Targeting**")
                st.markdown(_chips([(STAGE_LABEL.get(s, s), "accent") for s in proposal.target_stages]
                                   + [(k, "") for k in proposal.keywords]),
                            unsafe_allow_html=True)
                st.markdown("**Departure markers**")
                st.markdown(_chips([(b, "") for b in proposal.target_bios]), unsafe_allow_html=True)
                st.markdown("**Watchlist**")
                wl_invalid = set(st.session_state.get("watchlist_invalid", []))
                st.markdown(
                    _chips([(f"@{w}", "invalid" if w in wl_invalid else "")
                            for w in proposal.watchlist]),
                    unsafe_allow_html=True,
                )
                if wl_invalid:
                    st.markdown(
                        f'<div class="subtle">{len(wl_invalid)} handle'
                        f'{"s" if len(wl_invalid) != 1 else ""} not found on X — '
                        'dropped automatically on apply.</div>',
                        unsafe_allow_html=True,
                    )
                elif not st.session_state.get("watchlist_validated", True):
                    st.markdown(
                        '<div class="subtle">Handles not validated — add twscrape '
                        'cookies to check they exist before applying.</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("**GitHub topics**")
                st.markdown(_chips([(t, "") for t in proposal.github_topics]), unsafe_allow_html=True)
            with p2:
                st.markdown("**X query bank**")
                for label, queries in [("Departures", proposal.searches_departure),
                                       ("Stealth intent", proposal.searches_stealth_intent),
                                       ("Hiring", proposal.searches_hiring),
                                       ("Launches", proposal.searches_launch)]:
                    if queries:
                        st.markdown(f'<div class="subtle" style="margin-top:6px">{label}</div>',
                                    unsafe_allow_html=True)
                        st.code("\n".join(queries), language=None)
            a1, a2, _sp = st.columns([1.2, 1, 4])
            if a1.button("Apply strategy", type="primary"):
                dropped = set(st.session_state.get("watchlist_invalid", []))
                to_apply = proposal
                if dropped:
                    to_apply = proposal.model_copy(update={
                        "watchlist": [w for w in proposal.watchlist if w not in dropped]
                    })
                new_thesis, new_seeds = apply_strategy(to_apply, thesis, seeds)
                save_thesis(new_thesis, THESIS_PATH)
                save_seeds(new_seeds, SEEDS_PATH)
                for k in ("proposal", "watchlist_invalid", "watchlist_validated"):
                    st.session_state.pop(k, None)
                msg = "Strategy applied to thesis.yaml + seeds.yaml."
                if dropped:
                    msg += f" Dropped {len(dropped)} unresolvable handles: " + ", ".join(sorted(dropped))
                st.session_state["toast"] = msg
                st.rerun()
            if a2.button("Discard"):
                for k in ("proposal", "watchlist_invalid", "watchlist_validated"):
                    st.session_state.pop(k, None)
                st.rerun()

    st.write("")
    st.markdown("---")

    # --- Run --------------------------------------------------------------------
    st.markdown('<div class="section-title">Run discovery</div>'
                '<div class="section-sub">Free sources (X scraping, GitHub, Hacker News) unless you '
                'pick the paid X API. Runs are incremental — recently scored accounts are skipped.</div>',
                unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns([1.4, 1, 1, 1])
    with r1:
        source = st.segmented_control("Source", ["twscrape (free)", "xapi (paid)"],
                                      default="twscrape (free)")
    with r2:
        max_accounts = st.number_input("Max accounts", 10, 2000, settings.max_accounts, step=10)
    with r3:
        min_score_run = st.number_input("Min score", 0, 100, 0, step=5)
    with r4:
        ttl = st.number_input("Skip if scored < N days", 0, 90, settings.ttl_days)
    paid_run = "xapi" in (source or "")
    scan_active = ((store.current_scan() or {}).get("status") == "running")
    if scan_active:
        st.markdown('<div class="subtle">A scan is already running (see the banner above) — '
                    'the buttons unlock when it finishes.</div>', unsafe_allow_html=True)
    run_ready = not scan_active
    if paid_run:
        spent_now = store.xapi_spend_usd()
        remaining = max(settings.xapi_spend_cap_usd - spent_now, 0.0)
        est_profiles = int(max_accounts) * settings.xapi_cost_per_user_read
        est_tweets = (int(max_accounts) * settings.tweets_per_account
                      * settings.xapi_cost_per_post_read)
        est_total = est_profiles + est_tweets
        st.markdown(
            f'<div class="subtle">Worst case ≈ <b>${est_total:.2f}</b> '
            f'(${est_profiles:.2f} profiles + up to ${est_tweets:.2f} tweets; the '
            f'bio-signal gate usually cuts the tweet part sharply) · '
            f'<b>${remaining:.2f}</b> left of the ${settings.xapi_spend_cap_usd:.0f} cap.</div>',
            unsafe_allow_html=True,
        )
        run_ready = run_ready and st.checkbox(
            f"Spend up to ${est_total:.2f} of the X API budget",
            key="confirm_run_spend",
        )
    run_col, preview_col, _sp = st.columns([1, 1.6, 3])
    if run_col.button("Run scout", type="primary", disabled=not run_ready):
        _stream_command(["run", "--source", "xapi" if paid_run else "twscrape",
                         "--max-accounts", str(int(max_accounts)),
                         "--min-score", str(int(min_score_run)), "--ttl-days", str(int(ttl))])
    if preview_col.button("Preview discovery (free, no scoring)", disabled=scan_active):
        _stream_command(["source", "--max-accounts", str(int(max_accounts))])

    st.write("")
    st.markdown("---")

    # --- Precision pass (paid verify) ------------------------------------------
    st.markdown('<div class="section-title">Precision pass</div>'
                '<div class="section-sub">Before outreach, re-score the top leads of the latest '
                'run with fresh official X API data. Discovery stays free — only this step '
                'spends, and only after you confirm the cost.</div>',
                unsafe_allow_html=True)
    v1, v2, _v3 = st.columns([1, 1, 2])
    with v1:
        n_verify = st.number_input("Top leads to hydrate", 1, 200, 20)
    with v2:
        v_tweets = st.number_input("Tweets per lead", 0, 50, 10)
    v_est = int(n_verify) * (settings.xapi_cost_per_user_read
                             + int(v_tweets) * settings.xapi_cost_per_post_read)
    v_remaining = max(settings.xapi_spend_cap_usd - store.xapi_spend_usd(), 0.0)
    st.markdown(
        f'<div class="subtle">Worst case ≈ <b>${v_est:.2f}</b> · '
        f'<b>${v_remaining:.2f}</b> left of the cap. Results land as a verify run '
        f'in the ledger.</div>',
        unsafe_allow_html=True,
    )
    v_ready = st.checkbox(f"Spend up to ${v_est:.2f} of the X API budget",
                          key="confirm_verify_spend")
    if st.button("Run precision pass", disabled=not v_ready or scan_active):
        _stream_command(["verify", "--max", str(int(n_verify)), "--tweets", str(int(v_tweets))])

    st.write("")
    st.markdown("---")
    st.markdown('<div class="section-title">Fine-tune</div>'
                '<div class="section-sub">Everything the agent wrote is editable by hand.</div>',
                unsafe_allow_html=True)

    with st.expander("Targeting — stages, keywords, markers"):
        st.markdown('<div class="subtle">Stages steer the whole engine: which searches run, which '
                    'discovery sources fire, and how leads are scored for fit.</div>',
                    unsafe_allow_html=True)
        stage_cols = st.columns(len(STAGES))
        chosen_stages = []
        for col, stage_key in zip(stage_cols, STAGES):
            with col:
                if st.toggle(STAGE_LABEL[stage_key], value=stage_key in thesis.target_stages,
                             key=f"stage_{stage_key}"):
                    chosen_stages.append(stage_key)
        with st.form("targeting_form"):
            c1, c2 = st.columns(2)
            with c1:
                keywords = st.text_area("Intent keywords → bio_intent", _to_lines(thesis.keywords), height=120)
                target_bios = st.text_area("Departure markers → departure_signal", _to_lines(thesis.target_bios), height=120)
                launch_phrases = st.text_area("Launch phrases → launch_traction", _to_lines(thesis.launch_phrases), height=110)
            with c2:
                sectors = st.text_area("Sectors (classifier context)", _to_lines(thesis.sectors), height=120)
                disqualifiers = st.text_area("Disqualifiers (drop account)", _to_lines(thesis.disqualifiers), height=120)
            if st.form_submit_button("Save targeting", type="primary"):
                save_thesis(thesis.model_copy(update={
                    "target_stages": chosen_stages or list(STAGES),
                    "keywords": _from_lines(keywords), "target_bios": _from_lines(target_bios),
                    "launch_phrases": _from_lines(launch_phrases), "sectors": _from_lines(sectors),
                    "disqualifiers": _from_lines(disqualifiers)}), THESIS_PATH)
                st.success("Saved."); st.rerun()

    with st.expander("Query bank — X searches and bio search"):
        with st.form("seeds_form"):
            c1, c2 = st.columns(2)
            with c1:
                s_launch = st.text_area("Just-launched", _to_lines(seeds.searches_launch), height=110)
                s_departure = st.text_area("Departures", _to_lines(seeds.searches_departure), height=110)
                s_stealth = st.text_area("Stealth / intent", _to_lines(seeds.searches_stealth_intent), height=100)
            with c2:
                s_hiring = st.text_area("Founding-team hiring", _to_lines(seeds.searches_hiring), height=110)
                bio_searches = st.text_area("Bio / people search", _to_lines(seeds.bio_searches), height=100)
                s_legacy = st.text_area("Other searches", _to_lines(seeds.searches), height=70)
            if st.form_submit_button("Save queries", type="primary"):
                save_seeds(seeds.model_copy(update={
                    "searches": _from_lines(s_legacy),
                    "searches_departure": _from_lines(s_departure),
                    "searches_stealth_intent": _from_lines(s_stealth),
                    "searches_hiring": _from_lines(s_hiring),
                    "searches_launch": _from_lines(s_launch),
                    "bio_searches": _from_lines(bio_searches)}), SEEDS_PATH)
                st.success("Saved."); st.rerun()

    with st.expander("Watchlist & discovery — follow-graph, GitHub, lists"):
        with st.form("watchlist_form"):
            c3, c4 = st.columns(2)
            with c3:
                watchlist = st.text_area("Watchlist — investors to follow-diff",
                                         _to_lines(seeds.watchers), height=150,
                                         help="Their NEW follows become candidates; 2+ following the same account fires smart_money_convergence.")
            with c4:
                github_topics = st.text_area("GitHub topics", _to_lines(seeds.github_topics), height=70)
                lists = st.text_area("Public X List IDs", _to_lines(seeds.lists), height=60)
            if st.form_submit_button("Save watchlist", type="primary"):
                save_seeds(seeds.model_copy(update={
                    "watchlist": _from_lines(watchlist), "tastemakers": [],
                    "github_topics": _from_lines(github_topics),
                    "lists": _from_lines(lists)}), SEEDS_PATH)
                st.success("Saved."); st.rerun()

    with st.expander("Signals & scoring — weights, parameters, classifier prompt"):
        stats = triage_stats(ledger, pipeline)
        if stats is None:
            st.markdown('<div class="subtle">Triage at least 5 leads (shortlist or pass) and '
                        'insights appear here: how your decisions cluster by sector and signal, '
                        'plus AI-suggested weight adjustments.</div>', unsafe_allow_html=True)
        else:
            st.markdown("**Triage insights** — "
                        f"{stats.shortlisted} shortlisted · {stats.passed} passed")
            for line in stats.findings or ["No strong contrasts yet — keep triaging."]:
                st.markdown(f'<div class="subtle">• {_e(line)}</div>', unsafe_allow_html=True)
            st.write("")
            if st.button("Suggest weight adjustments with AI"):
                try:
                    with st.spinner("Analyzing your triage decisions…"):
                        st.session_state["weight_proposal"] = suggest_weights(
                            stats_prompt(stats), thesis, settings
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
            weight_proposal = st.session_state.get("weight_proposal")
            if weight_proposal is not None:
                st.markdown(f'<div class="subtle">{_e(weight_proposal.rationale)}</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    [{"signal": name,
                      "current": thesis.weights.get(name, 0.0),
                      "proposed": value,
                      "change": round(value - thesis.weights.get(name, 0.0), 1)}
                     for name, value in sorted(weight_proposal.weights.items())],
                    hide_index=True, use_container_width=True,
                )
                w1, w2, _w3 = st.columns([1.1, 1, 3])
                if w1.button("Apply weights", type="primary"):
                    save_thesis(thesis.model_copy(update={"weights": weight_proposal.weights}),
                                THESIS_PATH)
                    st.session_state.pop("weight_proposal", None)
                    st.session_state["toast"] = "Weights updated — re-rank with a run or demo."
                    st.rerun()
                if w2.button("Discard", key="weights_discard"):
                    st.session_state.pop("weight_proposal", None)
                    st.rerun()
        st.markdown("---")
        st.markdown('<div class="subtle">Score = 100 × Σ(value × weight) / Σ(weights), then × Claude '
                    'confidence, × 0.2 if not-a-founder, × stage-fit multiplier, × thesis-fit '
                    'multiplier, × value-add multiplier (off by default). All editable.</div>',
                    unsafe_allow_html=True)
        with st.form("signals_form"):
            names = list(SIGNAL_HELP) + [n for n in thesis.weights if n not in SIGNAL_HELP]
            new_weights: dict[str, float] = {}
            for i in range(0, len(names), 2):
                row = st.columns(2)
                for col, name in zip(row, names[i:i + 2]):
                    with col:
                        new_weights[name] = float(st.slider(name, 0, 50, int(thesis.weights.get(name, 0)),
                                                             help=SIGNAL_HELP.get(name, "")))
            st.divider()
            params = thesis.signal_params
            q1, q2, q3 = st.columns(3)
            with q1:
                tf = st.number_input("Traction floor (eng/followers)", 0.0, 1.0, float(params.traction_floor), 0.01)
                ts = st.number_input("Traction saturation", 0.01, 2.0, float(params.traction_saturation), 0.01)
            with q2:
                tw = st.number_input("Traction window (days)", 1, 180, int(params.traction_window_days))
                cf = st.number_input("Convergence full-credit follows", 1, 10, int(params.convergence_full_credit))
            with q3:
                sm = st.number_input("Off-target stage multiplier", 0.0, 1.0, float(params.stage_mismatch_multiplier), 0.05)
                fw = st.number_input("Thesis-fit weight", 0.0, 1.0, float(params.thesis_fit_weight), 0.05,
                                     help="0 ignores Claude's thesis_fit; 1 lets it scale the score fully.")
                vw = st.number_input(f"{thesis.firm_name or 'Firm'} value-add weight", 0.0, 1.0,
                                     float(params.value_add_weight), 0.05,
                                     help="How much value_add_fit (would the firm's value-add "
                                          "accelerate this startup?) sways the score. 0 = "
                                          "informational only — chips, sort, and exports still "
                                          "show it.")
            st.divider()
            st.markdown("**Classifier prompt** — placeholders `{thesis}` `{sectors}` `{stages}` "
                        "`{firm}` `{value_add}`")
            llm_prompt = st.text_area("Prompt", thesis.llm_prompt or DEFAULT_PROMPT_TEMPLATE,
                                      height=260, label_visibility="collapsed")
            if st.form_submit_button("Save signals", type="primary"):
                new_prompt = "" if llm_prompt.strip() == DEFAULT_PROMPT_TEMPLATE.strip() else llm_prompt
                save_thesis(thesis.model_copy(update={
                    "weights": {k: float(v) for k, v in new_weights.items()},
                    "signal_params": SignalParams(
                        traction_floor=tf, traction_saturation=ts,
                        traction_window_days=int(tw), convergence_full_credit=int(cf),
                        stage_mismatch_multiplier=sm, thesis_fit_weight=fw,
                        value_add_weight=vw),
                    "llm_prompt": new_prompt}), THESIS_PATH)
                st.success("Saved."); st.rerun()

        with st.expander(f"{thesis.firm_name or 'Firm'} value-add levers"):
            st.markdown(
                '<div class="subtle">The classifier scores every lead against these levers '
                '(the per-lever bars in each card\'s Details). Edit them under '
                '<code>firm_value_add</code> in <code>thesis.yaml</code>.</div>',
                unsafe_allow_html=True,
            )
            for lever in thesis.firm_value_add:
                st.markdown(f"**{lever.label}** (`{lever.key}`) — {lever.description}")


# ============================================================ DATABASE


# Friendly presentation order + one-liners; unknown tables still appear after.
DB_TABLE_ORDER = [
    "leads", "accounts", "tweets", "llm_verdicts", "pipeline", "runs",
    "unlinked_leads", "follow_edges", "follow_meta", "bio_snapshots",
    "searches", "xapi_usage", "scan",
]
DB_TABLE_HELP = {
    "leads": "Every scored lead — one row per handle per run.",
    "accounts": "The fetch cache: every X account scout has seen.",
    "tweets": "Cached tweets per account.",
    "llm_verdicts": "Claude classification cache, keyed by input fingerprint.",
    "pipeline": "Deal-flow state: status, notes, outreach, briefs.",
    "runs": "Run provenance — source, strategy hash, config snapshot.",
    "unlinked_leads": "GitHub/HN founders with no X handle (manual lookup).",
    "follow_edges": "Investor follow-graph snapshots (the smart-money signals).",
    "follow_meta": "Per-watcher snapshot baselines.",
    "bio_snapshots": "Bio history behind the bio_change signal.",
    "searches": "Per-query result cache with TTL.",
    "xapi_usage": "The paid X API budget ledger — every billed call.",
    "scan": "Live scan status (drives the banner).",
}


def _qi(name: str) -> str:
    """Quote an SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


def _db_filter_meta(db, table: str, text_cols: list[str],
                    num_cols: list[str]) -> tuple[list[tuple[str, list]], list[tuple[str, float, float]]]:
    """Auto-derive filters: low-cardinality text columns → value pickers,
    ranged numeric columns → min/max sliders."""
    value_filters: list[tuple[str, list]] = []
    for col in text_cols:
        if col.endswith("_json") or len(value_filters) >= 6:
            continue
        n = db.execute(f"SELECT COUNT(DISTINCT {_qi(col)}) FROM {_qi(table)}").fetchone()[0]
        if 2 <= n <= 30:
            values = [r[0] for r in db.execute(
                f"SELECT DISTINCT {_qi(col)} FROM {_qi(table)} "
                f"WHERE {_qi(col)} IS NOT NULL ORDER BY 1"
            ).fetchall()]
            value_filters.append((col, values))
    range_filters: list[tuple[str, float, float]] = []
    for col in num_cols:
        if len(range_filters) >= 3:
            break
        lo, hi = db.execute(
            f"SELECT MIN({_qi(col)}), MAX({_qi(col)}) FROM {_qi(table)}"
        ).fetchone()
        if lo is not None and hi is not None and float(lo) < float(hi):
            range_filters.append((col, float(lo), float(hi)))
    return value_filters, range_filters


with tab_data:
    db = store.db
    known_tables = [t for t in DB_TABLE_ORDER if db[t].exists()]
    extra_tables = sorted(
        t for t in db.table_names()
        if t not in known_tables and not t.startswith("sqlite_")
    )
    db_tables = known_tables + extra_tables
    db_file = Path(store.db_path)

    st.markdown(
        '<div class="section-title">Database</div>'
        f'<div class="section-sub">Everything scout knows, raw — <code>{_e(str(db_file))}</code>. '
        'Browse any table, filter, search, export. Read-only.</div>',
        unsafe_allow_html=True,
    )

    if not db_tables:
        st.markdown(
            '<div class="subtle">The database is empty — run <code>./scout-cli demo</code> '
            'or a discovery run first.</div>',
            unsafe_allow_html=True,
        )
    else:
        row_counts = {t: db[t].count for t in db_tables}
        size_mb = db_file.stat().st_size / 1e6 if db_file.exists() else 0.0
        d1, d2, d3 = st.columns(3)
        d1.markdown(_tile("Tables", str(len(db_tables))), unsafe_allow_html=True)
        d2.markdown(_tile("Rows", f"{sum(row_counts.values()):,}", "across all tables"),
                    unsafe_allow_html=True)
        d3.markdown(_tile("On disk", f"{size_mb:.1f} MB", db_file.name), unsafe_allow_html=True)
        st.write("")

        table = st.selectbox(
            "Table", db_tables,
            format_func=lambda t: f"{t}  ·  {row_counts[t]:,} rows",
            key="db_table",
        )
        if DB_TABLE_HELP.get(table):
            st.markdown(f'<div class="subtle" style="margin:-4px 0 10px">{_e(DB_TABLE_HELP[table])}</div>',
                        unsafe_allow_html=True)

        cols_meta = db[table].columns
        col_names = [c.name for c in cols_meta]
        text_cols = [c.name for c in cols_meta
                     if "INT" not in (c.type or "").upper()
                     and "REAL" not in (c.type or "").upper()
                     and "FLOA" not in (c.type or "").upper()]
        num_cols = [c.name for c in cols_meta if c.name not in text_cols]
        value_filters, range_filters = _db_filter_meta(db, table, text_cols, num_cols)

        f1, f2, f3, f4, f5 = st.columns([2.5, 1.35, 0.8, 1.0, 0.95])
        with f1:
            db_query = st.text_input("Search", key=f"db_q_{table}",
                                     placeholder=f"Search {table} — any text column…",
                                     label_visibility="collapsed")
        with f2:
            default_sort = col_names.index("created_at") if "created_at" in col_names else 0
            sort_col = st.selectbox("Sort by", col_names, index=default_sort,
                                    key=f"db_sort_{table}", label_visibility="collapsed")
        with f3:
            sort_desc = (st.segmented_control("Order", ["↓", "↑"], default="↓",
                                              key=f"db_dir_{table}",
                                              label_visibility="collapsed") or "↓") == "↓"
        with f4:
            selected_values: dict[str, list] = {}
            selected_ranges: dict[str, tuple[float, float]] = {}
            n_active_db = 0
            with st.popover("Filters"):
                visible_cols = st.multiselect(
                    "Columns", col_names,
                    default=[c for c in col_names if not c.endswith("_json")],
                    key=f"db_cols_{table}",
                )
                for col, values in value_filters:
                    picked = st.multiselect(col, values, key=f"db_f_{table}_{col}")
                    if picked:
                        selected_values[col] = picked
                        n_active_db += 1
                for col, lo, hi in range_filters:
                    picked_lo, picked_hi = st.slider(col, lo, hi, (lo, hi),
                                                     key=f"db_r_{table}_{col}")
                    if (picked_lo, picked_hi) != (lo, hi):
                        selected_ranges[col] = (picked_lo, picked_hi)
                        n_active_db += 1
        with f5:
            db_limit = st.selectbox("Rows", [100, 500, 1000, 5000], index=1,
                                    key=f"db_limit_{table}", label_visibility="collapsed",
                                    format_func=lambda n: f"{n:,} rows")

        where: list[str] = []
        params: list = []
        if db_query.strip():
            like_cols = text_cols or col_names
            where.append("(" + " OR ".join(f"{_qi(c)} LIKE ?" for c in like_cols) + ")")
            params += [f"%{db_query.strip()}%"] * len(like_cols)
        for col, picked in selected_values.items():
            where.append(f"{_qi(col)} IN ({','.join('?' * len(picked))})")
            params += picked
        for col, (lo, hi) in selected_ranges.items():
            where.append(f"{_qi(col)} BETWEEN ? AND ?")
            params += [lo, hi]
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total = db.execute(f"SELECT COUNT(*) FROM {_qi(table)}{where_sql}", params).fetchone()[0]
        select_cols = ", ".join(_qi(c) for c in (visible_cols or col_names))
        order_sql = f" ORDER BY {_qi(sort_col)} {'DESC' if sort_desc else 'ASC'}"
        df = pd.read_sql_query(
            f"SELECT {select_cols} FROM {_qi(table)}{where_sql}{order_sql} LIMIT ?",
            db.conn, params=params + [int(db_limit)],
        )

        shown_note = f"{len(df):,} of {total:,} matching rows"
        if n_active_db or db_query.strip():
            shown_note += f" · {row_counts[table]:,} in table"
        st.markdown(f'<div class="subtle" style="margin:2px 0 8px">{shown_note}</div>',
                    unsafe_allow_html=True)
        # Height hugs the rows (35px each + header) instead of padding the
        # grid with empty rows; caps at ~12 rows then scrolls.
        st.dataframe(df, use_container_width=True, hide_index=True,
                     height=min(460, 37 * (len(df) + 1) + 5))

        st.download_button(
            f"Download view as CSV ({len(df):,} row{'s' if len(df) != 1 else ''})",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{table}_view.csv",
        )

        with st.expander("SQL — read-only queries against the store"):
            sql_text = st.text_area(
                "Query", value=f"SELECT * FROM {table} LIMIT 50",
                height=90, label_visibility="collapsed", key="db_sql",
            )
            if st.button("Run query"):
                statement = sql_text.strip().rstrip(";")
                if not statement.lower().startswith(("select", "with")):
                    st.error("Read-only: only SELECT / WITH queries run here.")
                else:
                    try:
                        # sqlite3 executes a single statement — a piggybacked
                        # second statement raises rather than running.
                        sql_df = pd.read_sql_query(statement, db.conn)
                        st.dataframe(sql_df, use_container_width=True, hide_index=True)
                        st.markdown(f'<div class="subtle">{len(sql_df):,} rows</div>',
                                    unsafe_allow_html=True)
                    except Exception as exc:  # surface SQL errors inline
                        st.error(f"{type(exc).__name__}: {exc}")


# ============================================================ SETTINGS


with tab_settings:
    spent = store.xapi_spend_usd()
    cap = settings.xapi_spend_cap_usd
    s1, s2, s3 = st.columns(3)
    s1.markdown(_tile("X API spend", f"${spent:.2f}", f"of ${cap:.0f} cap"), unsafe_allow_html=True)
    s2.markdown(_tile("Latest leads", str(len(leads)), "most recent run"), unsafe_allow_html=True)
    s3.markdown(_tile("Verdict cache TTL", f"{settings.verdict_ttl_days}d",
                      "re-runs reuse Claude verdicts"), unsafe_allow_html=True)
    st.write("")

    k1, k2, k3 = st.columns(3)
    cookie_ok = bool(settings.tw_cookies and Path(settings.tw_cookies).exists())
    k1.markdown(("✓ " if cookie_ok else "○ ") + "twscrape cookies")
    k2.markdown(("✓ " if settings.x_bearer_token else "○ ") + "X API token")
    k3.markdown(("✓ " if settings.anthropic_api_key else "○ ") + "Anthropic key")
    st.markdown('<div class="subtle">Secrets live in <code>.env</code> — edit them there. '
                'Non-secret defaults below.</div>', unsafe_allow_html=True)
    st.write("")

    with st.form("settings_form"):
        c1, c2 = st.columns(2)
        with c1:
            cap_in = st.number_input("X API spend cap (USD)", 0.0, 25.0, float(cap), 1.0)
            model_in = st.text_input("Claude model", settings.claude_model)
            llm_cap_in = st.number_input("Max accounts sent to Claude per run", 10, 2000,
                                         settings.llm_max_candidates)
        with c2:
            max_in = st.number_input("Default max accounts / run", 10, 5000, settings.max_accounts)
            ttl_in = st.number_input("Default TTL days", 0, 90, settings.ttl_days)
            verdict_ttl_in = st.number_input("Verdict cache TTL (days)", 0, 90,
                                             settings.verdict_ttl_days)
        if st.form_submit_button("Save settings", type="primary"):
            _set_env_var("XAPI_SPEND_CAP_USD", f"{cap_in:.2f}")
            _set_env_var("CLAUDE_MODEL", model_in.strip())
            _set_env_var("MAX_ACCOUNTS", str(int(max_in)))
            _set_env_var("TTL_DAYS", str(int(ttl_in)))
            _set_env_var("LLM_MAX_CANDIDATES", str(int(llm_cap_in)))
            _set_env_var("VERDICT_TTL_DAYS", str(int(verdict_ttl_in)))
            st.success(".env updated."); st.rerun()
