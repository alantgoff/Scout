"""scout — a deal-flow workspace over the CLI, in Headline's editorial design
language (headline.com): warm cream paper, ink serif display, butter-yellow
accents, uppercase tracked labels.

Six surfaces, funnel-ordered (session-state nav, so any action can route to
any page — a card's Memo button lands on the Memos page):

  THESIS    — define how the scrape runs: the AI strategy agent, run controls,
              and (behind disclosure) every manual knob: query bank, watchlist,
              weights, prompt
  STARTUPS  — what sourcing found: the lead feed (latest run or the all-runs
              ledger) plus a Database sub-page — every tracked startup as a
              dossier table (select a row for the full record), with the raw
              SQLite browser behind an expander
  LONGLIST  — the first cut: candidates worth a closer look, Claude's
              per-dimension scoring open on every card
  SHORTLIST — the working set: stage/notes tracking to allocation,
              per-dimension scoring, CRM-ready CSV export
  MEMOS     — the full investment memo per startup (overview, product,
              tech, competition, market, acquisition dynamics,
              recommendation): generate, edit in place, export .md/.pdf,
              outreach draft alongside
  SETTINGS  — keys, budget, defaults

Scoring is transparent and overridable: every card breaks the score into
quality / fit / signals with per-dimension evidence, and an Adjust-scoring
panel persists the investor's own numbers (store.score_overrides) on top.

Edits write back to thesis.yaml / seeds.yaml / .env; deal-flow state lives in
scout.db. The UI never stores secrets. Launch with `./scout-cli ui`.
"""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from scout.agents import (
    apply_strategy,
    generate_strategy,
    investment_memo,
    suggest_weights,
    validate_watchlist,
)
from scout.config import (
    CUSTOMER_TYPES,
    QUALITY_DIMENSIONS,
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
from scout.companies import display_name, group_by_company, startup_identity
from scout.export import memo_pdf_bytes, pipeline_rows, write_pipeline_csv
from scout.insights import stats_prompt, triage_stats
from scout.models import Lead, LedgerEntry, LLMVerdict
from scout.outreach import CHANNELS, draft_outreach
from scout.score import (
    apply_override,
    quality_score,
    score_breakdown,
    score_components,
)
from scout.signals.llm import DEFAULT_PROMPT_TEMPLATE
from scout.store import Store
from scout.web import fetch_sites, normalize_site_url

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
    "longlisted": "Longlisted",
    "shortlisted": "Shortlisted",
    "contacted": "Contacted",
    "meeting": "Meeting",
    "diligence": "Diligence",
    "won": "Allocated",
    "passed": "Passed",
}
LABEL_TO_STATUS = {v: k for k, v in STATUS_LABELS.items()}
WIN_STAGES = ["shortlisted", "contacted", "meeting", "diligence", "won"]
# The triage funnel: feed → longlist → shortlist (→ stages to allocation).
FUNNEL_STAGES = ["longlisted", *WIN_STAGES]

STAGE_LABEL = {"idea": "Idea", "stealth": "Stealth", "launched": "Launched", "scaling": "Scaling"}
TYPE_LABEL = {"founder": "Founder", "startup": "Startup", "other": "Other"}
CUSTOMER_TYPE_LABEL = {"b2b": "B2B", "b2c": "B2C", "b2b2c": "B2B2C", "mixed": "B2B+B2C"}
QUALITY_DIM_LABEL = {
    "team": "Team", "tech_product": "Tech & product", "market": "Market",
    "defensibility": "Defensibility", "traction": "Traction", "investors": "Investors",
}

GROUNDED_SOURCES = {"website", "pinned_tweet", "tweets", "github"}


def _grounding_chip(verdict: LLMVerdict) -> tuple[str, str] | None:
    """The evidence-trust chip: audit outcome first, else the classifier's
    own grounding claim. None for legacy verdicts (no claim either way)."""
    domain = urlparse(verdict.company_url or "").netloc.removeprefix("www.")
    suffix = f" · {domain}" if domain else ""
    if verdict.verification == "confirmed":
        return (f"✓ verified{suffix}", "accent")
    if verdict.verification == "corrected":
        return (f"✓ corrected{suffix}", "accent")
    if verdict.verification == "unverifiable":
        return ("⚠ unverifiable", "")
    if verdict.grounding in GROUNDED_SOURCES:
        return (f"grounded: {verdict.grounding.replace('_', ' ')}", "")
    if verdict.grounding is not None:
        return ("⚠ ungrounded", "")
    return None


# --------------------------------------------------------------------- styling


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Committed light appearance — matches .streamlit/config.toml [theme],
           which pins Streamlit's native widgets to the same look.
           Headline (headline.com) design language: warm cream paper, warm-black
           ink, butter-yellow brand accent, high-contrast serif display type
           (Fraunces ≈ Headline's serif), geometric sans for UI text (Figtree),
           uppercase letter-spaced micro-labels, hairline rules, pill buttons. */
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..700&family=Figtree:ital,wght@0,300..700;1,300..700&display=swap');
        :root {
          --bg:#f4ebe0; --surface:#fbf5ec; --ink:#20180f; --ink-2:#4a4034;
          /* warm taupe secondary label; ≥4.5:1 on --bg, passes AA for text */
          --muted:#6b6052; --hair:rgba(32,24,15,0.14); --hair-strong:rgba(32,24,15,0.26);
          --accent:#20180f; --butter:#f2dc6c; --accent-soft:rgba(226,196,90,0.28);
          --good:#2e6b34; --track:rgba(32,24,15,0.08);
          --shadow:0 1px 2px rgba(32,24,15,0.04), 0 8px 24px rgba(32,24,15,0.05);
          --serif:"Fraunces","Iowan Old Style",Georgia,"Times New Roman",serif;
          --sans:"Figtree",-apple-system,BlinkMacSystemFont,"Helvetica Neue",
            system-ui,sans-serif;
        }
        html, body, [class*="css"], .stMarkdown, button, input, textarea {
          font-family:var(--sans) !important;
          -webkit-font-smoothing:antialiased;
        }
        .stApp { background:var(--bg); }
        .block-container { padding-top:2.2rem; padding-bottom:4rem; max-width:1080px; }
        #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"] { visibility:hidden; height:0; }

        /* Typography — serif display over sans body, like the Headline site */
        h1,h2,h3,h4 { font-family:var(--serif) !important; font-weight:600;
          letter-spacing:-0.01em; color:var(--ink); }
        /* Centered serif masthead — echoes the headline.com wordmark */
        .hero-title { font-family:var(--serif); font-size:2.7rem; font-weight:600;
          letter-spacing:-0.015em; color:var(--ink); line-height:1.05; margin:0;
          text-align:center; }
        .hero-sub { font-family:var(--serif); font-style:italic; color:var(--ink-2);
          font-size:1.08rem; margin:8px auto 0; font-weight:400; max-width:44rem;
          text-align:center; line-height:1.45; }
        .section-title { font-family:var(--serif); font-size:1.45rem; font-weight:600;
          letter-spacing:-0.01em; color:var(--ink); margin:0 0 2px; }
        .section-sub { color:var(--muted); font-size:0.92rem; margin:0 0 14px; }
        .subtle { color:var(--muted); font-size:0.88rem; }

        /* Tabs → centered pill control (Streamlit ≥1.59 react-aria markup);
           uppercase tracked labels, selected pill inverts to ink */
        .stTabs [role="tablist"] { gap:2px; background:var(--track); padding:3px;
          border-radius:980px; width:fit-content; margin:1.4rem auto 1.6rem;
          border-bottom:none !important; }
        .stTabs [data-testid="stTab"] { height:34px; border-radius:980px; padding:0 20px;
          background:transparent; border:none; display:flex; align-items:center; }
        .stTabs [data-testid="stTab"] p { font-size:0.74rem !important; font-weight:600;
          text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-2); }
        .stTabs [data-testid="stTab"][aria-selected="true"] { background:var(--ink);
          box-shadow:none; }
        .stTabs [data-testid="stTab"][aria-selected="true"] p { color:var(--bg);
          font-weight:600; }
        .stTabs .react-aria-SelectionIndicator { display:none !important; }

        /* Buttons — thin-outline pills with uppercase tracked labels ("MORE") */
        .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button,
        [data-testid="stPopoverButton"] {
          border-radius:980px; font-weight:600; font-size:0.74rem;
          text-transform:uppercase; letter-spacing:0.08em;
          border:1px solid var(--hair-strong); background:transparent; color:var(--ink);
          padding:0.38rem 1.05rem; transition:all .12s ease; box-shadow:none; }
        .stButton>button:hover, .stDownloadButton>button:hover,
        [data-testid="stPopoverButton"]:hover { border-color:var(--ink);
          color:var(--bg); background:var(--ink); }
        [data-testid="stPopoverButton"] p { font-size:0.74rem !important; }
        .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {
          background:var(--ink); border-color:var(--ink); color:var(--bg); }
        .stButton>button[kind="primary"]:hover { background:#3a2e1f;
          border-color:#3a2e1f; color:var(--bg); }
        .stButton>button:active { transform:scale(0.97); }
        .stButton>button:disabled { opacity:0.4; cursor:not-allowed; }
        .stButton>button:disabled:hover { border-color:var(--hair-strong);
          color:var(--ink); background:transparent; }
        .stButton>button[kind="primary"]:disabled:hover { background:var(--ink);
          border-color:var(--ink); color:var(--bg); }

        /* Top-level nav — a segmented control styled exactly like the old
           tab pills, centered. Session-state-driven so buttons anywhere in
           the app can route to a page (card → Memos). */
        .st-key-topnav [role="radiogroup"] { margin:1.4rem auto 1.6rem; }
        .st-key-topnav [data-testid="stButtonGroup"] { display:flex;
          justify-content:center; }

        /* Segmented controls → the same pill group as the tabs. The track is
           the inner radiogroup (the outer testid node also wraps the label). */
        [data-testid="stButtonGroup"] [role="radiogroup"] { gap:2px;
          background:var(--track); padding:3px; border-radius:980px; width:fit-content; }
        [data-testid="stButtonGroup"] button { border:none !important;
          border-radius:980px !important; min-height:30px;
          padding:0.18rem 0.9rem !important; background:transparent !important;
          box-shadow:none !important; }
        [data-testid="stButtonGroup"] button p { font-size:0.72rem !important;
          font-weight:600; text-transform:uppercase; letter-spacing:0.07em;
          color:var(--ink-2); }
        [data-testid="stButtonGroup"] button:hover p { color:var(--ink); }
        [data-testid="stButtonGroup"] button[aria-checked="true"] {
          background:var(--ink) !important; box-shadow:none !important; }
        [data-testid="stButtonGroup"] button[aria-checked="true"] p {
          color:var(--bg); font-weight:600; }

        /* Cards & tiles — cream paper panels with hairline rules */
        [data-testid="stVerticalBlockBorderWrapper"] {
          background:var(--surface); border:1px solid var(--hair); border-radius:16px;
          box-shadow:var(--shadow);
          transition:box-shadow .18s ease, border-color .18s ease; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
          border-color:var(--hair-strong);
          box-shadow:0 2px 4px rgba(32,24,15,0.05), 0 14px 36px rgba(32,24,15,0.08); }
        [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
          padding:0.65rem 0.75rem; }
        .tile { background:var(--surface); border:1px solid var(--hair); border-radius:16px;
          padding:16px 18px; box-shadow:var(--shadow); }
        .tile .label { color:var(--muted); font-size:0.68rem; text-transform:uppercase;
          letter-spacing:0.1em; font-weight:600; }
        .tile .value { font-family:var(--serif); color:var(--ink); font-size:1.8rem;
          font-weight:600; letter-spacing:-0.01em; line-height:1.15; margin-top:4px; }
        .tile .sub { color:var(--muted); font-size:0.8rem; margin-top:2px; }

        /* Lead card internals — startup names in serif, like the portfolio page */
        .lead-name { font-family:var(--serif); font-size:1.16rem; font-weight:600;
          color:var(--ink); letter-spacing:-0.005em; }
        .lead-name a { color:var(--ink); text-decoration:none; }
        .lead-name a:hover { text-decoration:underline;
          text-underline-offset:3px; text-decoration-thickness:1px; }
        .lead-handle { font-family:var(--sans); color:var(--muted); font-weight:400;
          font-size:0.85rem; letter-spacing:0; }
        .lead-summary { color:var(--ink-2); font-size:0.92rem; line-height:1.45;
          margin-top:3px; }
        .avatar { width:40px; height:40px; border-radius:50%; background:var(--butter);
          color:var(--ink); display:flex; align-items:center; justify-content:center;
          font-family:var(--serif); font-weight:600; font-size:0.95rem;
          letter-spacing:0; flex:0 0 40px; }
        .lead-row { display:flex; gap:13px; align-items:flex-start; }

        /* Chips — uppercase tracked micro-labels in tonal pills
           ("INFRASTRUCTURE" / "FINTECH" on the Headline portfolio page) */
        .chiprow { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
        .chip { display:inline-block; padding:3.5px 11px; border-radius:980px;
          font-size:0.66rem; font-weight:600; text-transform:uppercase;
          letter-spacing:0.07em; background:var(--track); color:var(--ink-2);
          white-space:nowrap; }
        .chip.accent { background:var(--butter); color:var(--ink); font-weight:600; }
        .chip.status { background:var(--ink); color:var(--bg); }
        .chip.invalid { text-decoration:line-through; opacity:0.55; }

        .scoreblock { text-align:right; }
        .scorenum { font-family:var(--serif); font-size:1.7rem; font-weight:600;
          letter-spacing:-0.01em; color:var(--ink); line-height:1; }
        .scorecap { color:var(--muted); font-size:0.66rem; text-transform:uppercase;
          letter-spacing:0.1em; font-weight:600; margin-top:2px; }
        .scoretrack { width:92px; height:4px; border-radius:2px; background:var(--track);
          margin:8px 0 10px auto; }
        /* deep mustard — the butter accent, dark enough to read on the track */
        .scorefill { height:4px; border-radius:2px; background:#d9b83f; }

        /* Signal bars (single hue — magnitude) */
        .sigrow { display:flex; align-items:center; gap:10px; margin:5px 0; }
        .signame { flex:0 0 190px; font-size:0.82rem; color:var(--ink-2); }
        .sigtrack { flex:1; height:5px; border-radius:2.5px; background:var(--track); }
        .sigfill { height:5px; border-radius:2.5px; background:var(--ink); }
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
        [data-testid="stExpander"] summary:hover { color:var(--ink);
          text-decoration:underline; text-underline-offset:3px; }

        .nudge { background:var(--accent-soft); border-radius:12px; padding:10px 16px;
          font-size:0.88rem; color:var(--ink-2); margin:0 0 16px; }
        .nudge b { color:var(--ink); }

        /* Live scan banner (auto-refreshing fragment) — butter-yellow wash */
        .scanbar { background:var(--accent-soft); border-radius:12px; padding:10px 16px;
          font-size:0.88rem; color:var(--ink-2); margin:14px 0 0;
          display:flex; align-items:center; gap:9px; }
        .scanbar b { color:var(--ink); }
        .scanbar.failed { background:rgba(178,58,44,0.10); }
        .scandot { width:8px; height:8px; border-radius:50%; background:var(--ink);
          flex:0 0 8px; animation:scanpulse 1.6s ease-in-out infinite; }
        @keyframes scanpulse { 0%,100% { opacity:1; transform:scale(1); }
          50% { opacity:0.35; transform:scale(0.8); } }

        /* Run progress panel — the pipeline stepper (Thesis page) */
        .runpanel { background:var(--surface); border:1px solid var(--hair);
          border-radius:16px; padding:16px 20px 14px; box-shadow:var(--shadow);
          margin:14px 0 4px; }
        .rp-head { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
        .rp-title { font-family:var(--serif); font-size:1.12rem; font-weight:600;
          color:var(--ink); }
        .rp-title .scandot { display:inline-block; margin-right:7px;
          vertical-align:1px; }
        .rp-meta { margin-left:auto; color:var(--muted); font-size:0.82rem;
          font-variant-numeric:tabular-nums; }
        .rp-meta b { color:var(--ink); font-weight:600; }
        .rp-steps { margin-top:10px; }
        .rp-step { display:flex; align-items:center; gap:12px; padding:8px 0;
          border-top:1px solid var(--hair); }
        .rp-dot { width:10px; height:10px; border-radius:50%; flex:0 0 10px; }
        .rp-dot.done { background:var(--good); }
        .rp-dot.current { background:var(--ink);
          animation:scanpulse 1.6s ease-in-out infinite; }
        .rp-dot.pending { background:transparent; border:1.5px solid var(--hair-strong); }
        .rp-name { flex:0 0 190px; font-size:0.88rem; color:var(--ink);
          font-weight:550; }
        .rp-name.pending { color:var(--muted); font-weight:400; }
        .rp-track { flex:1; height:5px; border-radius:2.5px; background:var(--track);
          overflow:hidden; position:relative; }
        .rp-fill { height:5px; border-radius:2.5px; background:#d9b83f; }
        .rp-fill.done { background:var(--good); opacity:0.45; }
        .rp-fill.indet { width:32%; position:absolute; left:0; top:0;
          animation:rp-slide 1.9s ease-in-out infinite; }
        @keyframes rp-slide { 0% { left:-32%; } 100% { left:100%; } }
        .rp-right { flex:0 0 175px; text-align:right; font-size:0.79rem;
          color:var(--muted); font-variant-numeric:tabular-nums;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .rp-detail { margin-top:9px; color:var(--ink-2); font-size:0.85rem; }
        .rp-est { color:var(--muted); font-size:0.84rem; margin:6px 0 10px; }
        .rp-est b { color:var(--ink); }

        hr { border-color:var(--hair) !important; margin:1.9rem 0 1.5rem !important; }
        [data-testid="stWidgetLabel"] p { font-size:0.83rem; color:var(--ink-2);
          font-weight:500; }

        /* Inputs — hairline borders, butter focus ring, one radius everywhere */
        .stTextInput [data-baseweb="input"], .stNumberInput [data-baseweb="input"],
        .stTextArea [data-baseweb="textarea"] {
          border-radius:10px !important; border-color:var(--hair-strong) !important;
          background:var(--surface) !important; transition:border-color .12s ease,
          box-shadow .12s ease; }
        .stTextInput [data-baseweb="input"]:focus-within,
        .stNumberInput [data-baseweb="input"]:focus-within,
        .stTextArea [data-baseweb="textarea"]:focus-within {
          border-color:var(--ink) !important;
          box-shadow:0 0 0 3px var(--accent-soft); }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
          border-radius:10px !important; border-color:var(--hair-strong) !important;
          background:var(--surface) !important; }
        .stSelectbox [data-baseweb="select"]:focus-within > div,
        .stMultiSelect [data-baseweb="select"]:focus-within > div {
          border-color:var(--ink) !important;
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


# --------------------------------------------------- run launching & progress

PHASE_LABELS = {
    "starting": "Starting",
    "discovering": "Discover accounts",
    "fetching timelines": "Fetch timelines",
    "reading websites": "Read company websites",
    "classifying": "Claude classification",
    "verifying": "Adversarial verification",
    "scoring & saving": "Score & save",
    "hydrating": "Hydrate via X API",
    "preparing": "Load cached leads",
}
PHASE_BLURB = {
    "discovering": "X searches, watchlist follow-diff, GitHub & Hacker News legs",
    "fetching timelines": "recent tweets per candidate (wall-clock budgeted)",
    "reading websites": "fetch each candidate's site so Claude classifies from "
                        "real product copy, not bio pedigree",
    "classifying": "Claude reads each dossier: product, stage, sector, thesis fit",
    "verifying": "Claude re-reads each top lead's evidence and corrects "
                 "speculative verdicts",
    "scoring & saving": "weighted signals × fit multipliers → ranked leads",
    "hydrating": "fresh official X API reads for the shortlist",
    "preparing": "reload the latest run's leads and cached evidence",
}


def _fmt_dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s" if seconds % 60 else f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _iso_ts(ts: str | None) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp() if ts else 0.0
    except ValueError:
        return 0.0


def _as_int(value) -> int | None:
    """Robust int coercion — SQLite may hand counters back as TEXT when the
    column was created from a NULL first write."""
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _tail_log(path: str, max_lines: int = 30) -> str:
    """Last lines of a scan log without reading the whole file."""
    p = Path(path)
    if not path or not p.exists():
        return ""
    try:
        with open(p, "rb") as fh:
            fh.seek(max(p.stat().st_size - 16_000, 0))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:])


def _launch_scan(args: list[str], kind_hint: str) -> None:
    """Start a CLI command as a DETACHED background process, logging to
    ~/.scout/logs/. The UI never blocks on it — the progress panel and the
    scan banner poll the store, and the log file is tailed on demand.
    (The child records the log path in the scan row via SCOUT_SCAN_LOG.)"""
    # Click-time guard: the page may have rendered before another scan
    # started (buttons enabled), so re-check right before launching.
    if (store.current_scan() or {}).get("status") == "running":
        st.warning("A scan is already running — wait for it to finish.")
        return
    log_dir = Path(settings.db_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{kind_hint}-{datetime.now():%Y%m%d-%H%M%S}.log"
    env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "COLUMNS": "120",
           "PYTHONUNBUFFERED": "1", "SCOUT_SCAN_LOG": str(log_path)}
    with open(log_path, "wb") as fh:
        fh.write(f"$ scout {' '.join(args)}\n".encode())
        subprocess.Popen([sys.executable, "-u", "-m", "scout.cli", *args],
                         cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    st.session_state["launch_pending"] = time.time()
    st.session_state["last_log_path"] = str(log_path)
    st.session_state["toast"] = "Pipeline started — live progress below."
    st.rerun()


def _last_scan_durations(kind: str) -> dict[str, float] | None:
    """Per-phase seconds from the most recent successful scan of this kind."""
    if not store.db["scan_history"].exists():
        return None
    rows = list(store.db["scan_history"].rows_where(
        "kind = ? AND status = 'done'", [kind],
        order_by="finished_at DESC", limit=1))
    if not rows:
        return None
    log = json.loads(rows[0].get("phase_log_json") or "[]")
    if not log:
        return None
    out: dict[str, float] = {}
    for entry, nxt in zip(log, log[1:] + [{"at": rows[0].get("finished_at")}]):
        if entry["phase"] == "starting":
            continue
        dur = _iso_ts(nxt.get("at")) - _iso_ts(entry.get("at"))
        out[entry["phase"]] = out.get(entry["phase"], 0.0) + max(dur, 0.0)
    return out or None


def _estimate_scan(kind: str, max_accounts: int | None = None,
                   n_verify: int | None = None) -> tuple[float, float, dict[str, tuple[float, float]], str]:
    """(lo_s, hi_s, per-phase (lo, hi), caption). Empirical when a completed
    scan of this kind exists; otherwise a settings-based first-run estimate."""
    hist = _last_scan_durations(kind)
    if hist:
        per = {ph: (d * 0.6, d * 1.5) for ph, d in hist.items()}
        label = f"based on your last completed {kind}"
    elif kind == "run":
        budget = float(settings.sourcing_time_budget_s)
        n = float(max_accounts or settings.max_accounts)
        n_class = min(max(n * 0.5, 10.0), float(settings.llm_max_candidates))
        batch_waves = (math.ceil(n_class / max(settings.classify_batch_size, 1))
                       / max(settings.llm_concurrency, 1))
        per = {
            # discovery + timelines share one wall-clock budget — the pair
            # can't exceed it, which caps the whole network phase.
            "discovering": (min(60.0, budget * 0.15), budget * 0.45),
            "fetching timelines": (budget * 0.10, budget * 0.55),
            "reading websites": (10.0, min(100.0, n_class * 0.7)),
            "classifying": (batch_waves * 10.0, batch_waves * 25.0),
            "verifying": (settings.verify_top_n * 1.5, settings.verify_top_n * 4.0)
            if settings.verify_top_n else (0.0, 0.0),
            "scoring & saving": (5.0, 20.0),
        }
        label = ("first-run estimate — the network phase is hard-capped at "
                 f"{_fmt_dur(budget)}; real timings are learned after one run")
    elif kind == "verify":
        n = float(n_verify or 20)
        waves = (math.ceil(min(n, settings.llm_max_candidates)
                           / max(settings.classify_batch_size, 1))
                 / max(settings.llm_concurrency, 1))
        per = {"hydrating": (n * 0.6, n * 1.5),
               "reading websites": (5.0, min(60.0, n * 0.7)),
               "classifying": (waves * 10.0, waves * 25.0),
               "scoring & saving": (5.0, 15.0)}
        label = "first-run estimate"
    elif kind == "reclassify":
        n_class = float(settings.llm_max_candidates)
        waves = (math.ceil(n_class / max(settings.classify_batch_size, 1))
                 / max(settings.llm_concurrency, 1))
        per = {"preparing": (2.0, 10.0),
               "reading websites": (5.0, 60.0),
               "classifying": (waves * 10.0, waves * 25.0),
               "verifying": (settings.verify_top_n * 1.5, settings.verify_top_n * 4.0)
               if settings.verify_top_n else (0.0, 0.0),
               "scoring & saving": (5.0, 15.0)}
        label = "cache-first — unchanged verdicts and cached websites are free"
    else:  # source preview
        budget = float(settings.sourcing_time_budget_s)
        per = {"discovering": (min(60.0, budget * 0.2), budget)}
        label = "first-run estimate"
    lo = sum(v[0] for v in per.values())
    hi = sum(v[1] for v in per.values())
    return lo, hi, per, label


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
# ALL runs. It backs the "All runs" scope and — always — the Longlist,
# Shortlist, and Memos pages, so a triaged lead never degrades just because
# it missed the latest run.
latest_is_demo = bool(leads) and all(x.account.source == "demo" for x in leads)
ledger = store.load_lead_ledger(include_demo=latest_is_demo)

# The investor's manual scoring, layered on top of every loaded lead BEFORE
# anything renders or ranks — adjusted dims/fit re-enter the score math, a
# pinned score wins outright (score.apply_override).
overrides = store.all_overrides()
if overrides:
    for _lead in leads:
        apply_override(_lead, overrides.get(_lead.account.handle.lower()), thesis)
    leads.sort(key=lambda x: -x.score)
    for _entry in ledger:
        apply_override(
            _entry.lead, overrides.get(_entry.lead.account.handle.lower()), thesis
        )
    ledger.sort(key=lambda e: -e.lead.score)

entry_by_handle = {e.lead.account.handle.lower(): e for e in ledger}
lead_by_handle = {h: e.lead for h, e in entry_by_handle.items()}
latest_handles = {x.account.handle.lower() for x in leads}


def _status_of(lead: Lead) -> str:
    return pipeline.get(lead.account.handle.lower(), {}).get("status") or "new"


# Header — large title, thesis as the subtitle
st.markdown(
    f'<div class="hero-title">Scout</div>'
    f'<div class="hero-sub">{_e(thesis.thesis) or "No thesis yet — open Thesis and describe one."}</div>',
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
            f'{_e(detail)} · started {_ago(scan.get("started_at"))} · '
            f'live progress on the <b>Thesis</b> page</div>',
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


def _phase_durations(plog: list[dict], finished_at: str | None) -> dict[str, float]:
    """Seconds spent per phase, from the scan's phase log. The open phase
    (no successor) runs until finished_at, or until now while live."""
    end_default = _iso_ts(finished_at) if finished_at else time.time()
    durs: dict[str, float] = {}
    for entry, nxt in zip(plog, plog[1:] + [{"at": None}]):
        end = _iso_ts(nxt["at"]) if nxt.get("at") else end_default
        durs[entry["phase"]] = durs.get(entry["phase"], 0.0) + max(end - _iso_ts(entry.get("at")), 0.0)
    return durs


@st.fragment(run_every="2s")
def _run_panel() -> None:
    """The run cockpit (Thesis page): a phase stepper with live progress bars,
    per-phase timings, an ETA, the current detail line, a Stop button, and a
    tail of the console log. Polls the store every 2s."""
    scan = store.current_scan()
    running = bool(scan) and scan.get("status") == "running"

    # Smooth the launch gap: the child takes a few seconds to import and
    # write its first scan row — show a starting state instead of nothing.
    pending = st.session_state.get("launch_pending")
    if pending:
        if running and _iso_ts(scan.get("started_at")) >= pending - 5:
            st.session_state.pop("launch_pending", None)
        elif time.time() - pending < 45:
            st.markdown(
                '<div class="runpanel"><div class="rp-head">'
                '<span class="rp-title"><span class="scandot"></span>Starting the pipeline…</span>'
                '</div><div class="rp-detail">Booting the scout process — the first phase '
                'reports in a few seconds.</div></div>',
                unsafe_allow_html=True,
            )
            return
        else:
            st.session_state.pop("launch_pending", None)
            st.error("The run never reported back — the log below has the reason.")
            tail = _tail_log(st.session_state.get("last_log_path", ""))
            if tail:
                st.code(tail, language=None)
            return

    just_finished = (
        bool(scan) and not running
        and (scan.get("finished_at") or "") > st.session_state["page_loaded_at"]
    )
    if not scan or (not running and not just_finished):
        return

    kind = scan.get("kind") or "scan"
    phases: list[str] = json.loads(scan.get("phases_json") or "[]")
    current = scan.get("phase") or ""
    if not phases:  # legacy row (run started under old code)
        phases = [current] if current else []
    plog = json.loads(scan.get("phase_log_json") or "[]")
    durs = _phase_durations(plog, scan.get("finished_at"))
    _, _, per_est, _ = _estimate_scan(kind)
    cur_idx = phases.index(current) if current in phases else -1
    elapsed = max(time.time() - _iso_ts(scan.get("started_at")), 0.0)
    done_n, total_n = _as_int(scan.get("done")), _as_int(scan.get("total"))
    unit = scan.get("unit") or ""

    # ETA: finish the current phase (items → observed rate; seconds → budget
    # remainder; else estimate midpoint), then estimate midpoints for the rest.
    remaining = 0.0
    if running:
        phase_elapsed = durs.get(current, 0.0)
        if unit == "items" and done_n and total_n:
            rate = done_n / max(phase_elapsed, 1e-6)
            remaining += (total_n - done_n) / max(rate, 1e-6)
        elif unit == "s" and total_n:
            remaining += max(float(total_n) - phase_elapsed, 0.0)
        else:
            lo, hi = per_est.get(current, (30.0, 120.0))
            remaining += max((lo + hi) / 2 - phase_elapsed, (lo + hi) * 0.08)
        for ph in phases[cur_idx + 1:]:
            lo, hi = per_est.get(ph, (10.0, 30.0))
            remaining += (lo + hi) / 2

    rows_html = []
    for i, ph in enumerate(phases):
        if not running and scan.get("status") == "done":
            state = "done"
        elif i < cur_idx or (i == cur_idx and not running):
            state = "done" if i < cur_idx or scan.get("status") == "done" else "failed"
        elif i == cur_idx:
            state = "current"
        else:
            state = "pending"
        label = PHASE_LABELS.get(ph, ph.capitalize())
        blurb = PHASE_BLURB.get(ph, "")

        if state == "done":
            bar = '<div class="rp-fill done" style="width:100%"></div>'
            right = _fmt_dur(durs[ph]) if ph in durs else "—"
        elif state == "current":
            if unit == "items" and done_n is not None and total_n:
                frac = min(done_n / total_n, 1.0)
                bar = f'<div class="rp-fill" style="width:{frac * 100:.0f}%"></div>'
                right = f"{done_n}/{total_n} · {_fmt_dur(durs.get(ph, 0))}"
            elif unit == "s" and total_n:
                frac = min(durs.get(ph, 0.0) / float(total_n), 1.0)
                bar = f'<div class="rp-fill" style="width:{frac * 100:.0f}%"></div>'
                right = f"{_fmt_dur(durs.get(ph, 0))} of ≤{_fmt_dur(float(total_n))}"
            else:
                bar = '<div class="rp-fill indet"></div>'
                right = _fmt_dur(durs.get(ph, 0))
        elif state == "failed":
            bar = ""
            right = "failed here"
        else:
            lo, hi = per_est.get(ph, (0.0, 0.0))
            bar = ""
            right = f"~{_fmt_dur((lo + hi) / 2)}" if hi else ""
        dot_cls = {"done": "done", "current": "current",
                   "failed": "pending", "pending": "pending"}[state]
        name_cls = " pending" if state == "pending" else ""
        rows_html.append(
            f'<div class="rp-step"><div class="rp-dot {dot_cls}"></div>'
            f'<div class="rp-name{name_cls}" title="{_e(blurb)}">{_e(label)}</div>'
            f'<div class="rp-track">{bar}</div>'
            f'<div class="rp-right">{_e(right)}</div></div>'
        )

    if running:
        title = f'<span class="scandot"></span>{_e(kind.capitalize())} in progress'
        meta = (f'elapsed <b>{_fmt_dur(elapsed)}</b> · '
                f'≈ <b>{_fmt_dur(remaining)}</b> left')
    elif scan.get("status") == "done":
        title = f'{_e(kind.capitalize())} finished'
        meta = f'took <b>{_fmt_dur(_iso_ts(scan.get("finished_at")) - _iso_ts(scan.get("started_at")))}</b>'
    else:
        title = f'{_e(kind.capitalize())} failed'
        meta = f'after {_fmt_dur(_iso_ts(scan.get("finished_at")) - _iso_ts(scan.get("started_at")))}'
    detail = scan.get("detail") or ""
    st.markdown(
        f'<div class="runpanel"><div class="rp-head">'
        f'<span class="rp-title">{title}</span>'
        f'<span class="rp-meta">{meta}</span></div>'
        f'<div class="rp-steps">{"".join(rows_html)}</div>'
        + (f'<div class="rp-detail">{_e(detail)}</div>' if detail else "")
        + '</div>',
        unsafe_allow_html=True,
    )

    b1, b2, _sp = st.columns([1, 1.3, 3.7])
    if running:
        if b1.button("Stop run", key="rp_stop"):
            try:
                os.kill(int(scan.get("pid") or 0), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
            store.scan_finish("failed", "stopped from the UI")
            st.session_state["toast"] = "Run stopped."
            st.rerun(scope="app")
    elif scan.get("status") == "done":
        if b1.button("View results", key="rp_refresh", type="primary"):
            st.session_state["page_loaded_at"] = datetime.now(timezone.utc).isoformat()
            st.rerun(scope="app")
    log_path = scan.get("log_path") or st.session_state.get("last_log_path", "")
    tail = _tail_log(log_path)
    if tail:
        with st.expander("Console log"):
            st.code(tail, language=None)

# Top-level navigation — session-state-driven (unlike st.tabs) so any button
# can route to a page: set st.session_state["nav_target"] and rerun.
PAGES = ["Thesis", "Startups", "Longlist", "Shortlist", "Memos", "Settings"]
if (_nav_target := st.session_state.pop("nav_target", None)) in PAGES:
    st.session_state["nav"] = _nav_target
st.session_state.setdefault("nav", "Thesis")
with st.container(key="topnav"):
    nav = st.segmented_control("Page", PAGES, key="nav",
                               label_visibility="collapsed")
if nav is None:  # clicking the active pill deselects — snap back
    st.session_state["nav_target"] = st.session_state.get("nav_last", "Thesis")
    st.rerun()
st.session_state["nav_last"] = nav


# ============================================================ STARTUPS · feed


def _override_editor(lead: Lead, ov: dict, key_ns: str) -> None:
    """The manual-scoring form (inside a card's Adjust-scoring popover).

    Sliders default to the CURRENT effective value; a dimension is persisted
    only when it was already overridden or the slider moved — untouched
    model scores are never frozen, so reclassification keeps updating them."""
    handle_key = lead.account.handle.lower()
    verdict = lead.llm
    st.markdown(
        '<div class="subtle">Your judgment on top of Claude\'s: adjusted dimensions '
        're-enter the same score math (bars, chips, and steps update everywhere); '
        'pinning the final score wins outright. Saved per startup, kept across '
        'runs. Use <b>Clear</b> to return to Claude\'s scoring.</div>',
        unsafe_allow_html=True,
    )
    existing_q = {k: float(v) for k, v in (ov.get("quality") or {}).items()}
    new_quality: dict[str, float] = {}
    fit_out: float | None = None
    if verdict is not None:
        st.markdown("**Quality dimensions**")
        for dim, weight in thesis.quality_weights.items():
            if weight <= 0:
                continue
            effective = verdict.quality.get(dim)
            default = int(round((effective if effective is not None else 0.0) * 100))
            suffix = "" if effective is not None else " (no evidence — was excluded)"
            val = st.slider(
                f"{QUALITY_DIM_LABEL.get(dim, dim)}{suffix}", 0, 100, default,
                key=f"{key_ns}_ovq_{dim}_{handle_key}", format="%d%%",
                help=QUALITY_DIMENSIONS.get(dim, ""),
            )
            if dim in existing_q or val != default:
                new_quality[dim] = val / 100.0
        st.markdown("**Thesis fit**")
        fit_default = int(round((verdict.thesis_fit
                                 if verdict.thesis_fit is not None else 0.0) * 100))
        fit_val = st.slider("Thesis fit", 0, 100, fit_default,
                            key=f"{key_ns}_ovf_{handle_key}", format="%d%%",
                            label_visibility="collapsed",
                            help="How squarely the product matches the thesis.")
        if ov.get("fit") is not None or fit_val != fit_default:
            fit_out = fit_val / 100.0
    else:
        st.markdown(
            '<div class="subtle">No Claude verdict on this lead yet — only a '
            'pinned final score applies here.</div>',
            unsafe_allow_html=True,
        )
    pin = st.checkbox(
        "Pin the final score", value=ov.get("score") is not None,
        key=f"{key_ns}_ovp_{handle_key}",
        help="Bypass the math entirely and set the 0–100 score yourself.",
    )
    pin_val = None
    if pin:
        pin_val = float(st.number_input(
            "Pinned score", 0.0, 100.0,
            float(ov["score"]) if ov.get("score") is not None else float(lead.score),
            1.0, key=f"{key_ns}_ovs_{handle_key}", label_visibility="collapsed",
        ))
    note = st.text_input(
        "Why (kept with the adjustment)", value=ov.get("note") or "",
        key=f"{key_ns}_ovn_{handle_key}",
        placeholder="e.g. spoke to a customer — traction is real",
    )
    c1, c2 = st.columns(2)
    if c1.button("Save adjustments", type="primary",
                 key=f"{key_ns}_ovsave_{handle_key}", use_container_width=True):
        store.set_override(handle_key, quality=new_quality or None,
                           fit=fit_out, score=pin_val, note=note)
        st.session_state["toast"] = f"Scoring adjusted — {display_name(lead)}"
        st.rerun()
    if ov and c2.button("Clear", key=f"{key_ns}_ovclear_{handle_key}",
                        use_container_width=True):
        store.clear_override(handle_key)
        st.session_state["toast"] = f"Back to Claude's scoring — {display_name(lead)}"
        st.rerun()


def _lead_card(
    lead: Lead,
    entry: LedgerEntry | None = None,
    fresh: bool = True,
    view_max: float = 100.0,
    pct_label: str = "",
    secondary: list[Lead] | None = None,
    details_open: bool = False,
    key_ns: str = "feed",
) -> None:
    """One lead card. `entry` carries cross-run movement (delta / new); `fresh`
    means the lead is from the latest run — run-scoped signals like new
    watchlist follows are suppressed on stale entries. `view_max` normalizes
    the score bar to the strongest lead in view; `pct_label` is an optional
    percentile caption ("top 12%"). `secondary` holds other X accounts folded
    into this startup (Startups track) — the card is titled by the company.
    `details_open` renders the per-dimension scoring expanded (Longlist /
    Shortlist pages); `key_ns` namespaces widget keys so the same lead can
    render on more than one page."""
    account, verdict = lead.account, lead.llm
    handle_key = account.handle.lower()
    status = _status_of(lead)
    secondary = secondary or []

    chips: list[tuple[str, str]] = []
    if verdict and verdict.thesis_fit is not None:
        chips.append((f"Fit {verdict.thesis_fit:.0%}", "accent"))
    if verdict is not None and (g_chip := _grounding_chip(verdict)) is not None:
        chips.append(g_chip)
    if verdict and verdict.customer_type:
        chips.append((CUSTOMER_TYPE_LABEL.get(verdict.customer_type,
                                              verdict.customer_type), ""))
    if verdict and verdict.value_add_fit is not None:
        chips.append((f"{thesis.firm_name or 'Firm'} lift {verdict.value_add_fit:.0%}", "accent"))
    if status != "new":
        chips.append((STATUS_LABELS.get(status, status), "status"))
    if overrides.get(handle_key):
        chips.append(("Adjusted", "accent"))
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

    # Product truth first: the grounded product_summary beats the one-liner,
    # which beats the bio.
    summary = ((verdict.product_summary if verdict else "")
               or (verdict.one_line_summary if verdict else "")
               or account.bio or "—")

    # STARTUP-FIRST identity: every founder-like lead is titled by its startup
    # — the classifier's company when named, otherwise a synthesized stealth
    # identity tied to the founder ("Ada Lin's stealth startup"). Only
    # corporate/commentator accounts keep a person/account title.
    company_url = (verdict.company_url or "").strip() if verdict else ""
    identity = startup_identity(lead)
    if identity:
        startup_title, synthesized = identity
        # A synthesized title already carries the founder's name — the byline
        # drops it rather than repeat it.
        byline = (
            f'@{_e(account.handle)} · {account.followers:,} followers'
            if synthesized else
            f'{_e(account.name or account.handle)} · @{_e(account.handle)} · '
            f'{account.followers:,} followers'
        )
        title_href = _e(company_url or account.url)
        title_html = (
            f'<a href="{title_href}" target="_blank">{_e(startup_title)}</a> '
            f'<span class="lead-handle">{byline}</span>'
        )
        avatar_text = _initials(account.name if synthesized else startup_title,
                                account.handle)
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
            # Component caption: which of quality / fit / signals this score
            # is made of (present components only).
            comp_html = ""
            if verdict is not None:
                comps = score_components(lead, thesis)
                bits = [f"{tag} {value:.0f}"
                        for tag, value in (("Q", comps["quality"]),
                                           ("F", comps["fit"]),
                                           ("S", comps["signals"]))
                        if value is not None]
                if bits:
                    comp_html = (f'<div class="scorecap" style="margin-top:5px">'
                                 f'{_e(" · ".join(bits))}</div>')
            st.markdown(
                f"""
                <div class="scoreblock">
                  <div class="scorenum">{lead.score:.0f}</div>
                  <div class="scorecap">score</div>
                  <div class="scoretrack"><div class="scorefill" style="width:{bar_pct:.0f}%"></div></div>
                  {comp_html}
                  {pct_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Triage lives on the card face — no expanding needed to act.
            # The funnel: new → Longlist → Shortlist (→ stages to allocation).
            if status == "longlisted":
                if st.button("Shortlist", key=f"{key_ns}_short_{handle_key}", type="primary",
                             use_container_width=True):
                    store.set_pipeline(account.handle, status="shortlisted")
                    st.session_state["toast"] = f"Shortlisted @{account.handle}"
                    st.rerun()
                if st.button("Remove", key=f"{key_ns}_rm_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="new")
                    st.session_state["toast"] = f"Removed @{account.handle} from the longlist"
                    st.rerun()
            elif status == "shortlisted":
                if st.button("To longlist", key=f"{key_ns}_demote_{handle_key}",
                             use_container_width=True):
                    store.set_pipeline(account.handle, status="longlisted")
                    st.session_state["toast"] = f"Moved @{account.handle} back to the longlist"
                    st.rerun()
            elif status in WIN_STAGES:  # contacted and beyond — manage on Shortlist
                if st.button("Remove", key=f"{key_ns}_rm_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="new")
                    st.session_state["toast"] = f"Removed @{account.handle} from the shortlist"
                    st.rerun()
            elif status == "passed":
                if st.button("Restore", key=f"{key_ns}_restore_{handle_key}",
                             use_container_width=True):
                    store.set_pipeline(account.handle, status="new")
                    st.session_state["toast"] = f"Restored @{account.handle}"
                    st.rerun()
            else:
                if st.button("Longlist", key=f"{key_ns}_long_{handle_key}", type="primary",
                             use_container_width=True):
                    store.set_pipeline(account.handle, status="longlisted")
                    st.session_state["toast"] = f"Longlisted @{account.handle}"
                    st.rerun()
                if st.button("Pass", key=f"{key_ns}_pass_{handle_key}", use_container_width=True):
                    store.set_pipeline(account.handle, status="passed")
                    st.session_state["toast"] = f"Passed on @{account.handle}"
                    st.rerun()

        # Stateless expander: the `expanded` prop is re-applied only when its
        # value CHANGES between reruns, so user toggles persist.
        with st.expander("Details", expanded=details_open):
            ov = overrides.get(handle_key) or {}
            params = thesis.signal_params
            comps = score_components(lead, thesis)
            if verdict and verdict.why_interesting:
                st.markdown(f'<div class="lead-summary">{_e(verdict.why_interesting)}</div>',
                            unsafe_allow_html=True)
            if verdict and verdict.verification_note:
                st.markdown(f'<div class="subtle" style="margin-top:6px">Audit — {_e(verdict.verification_note)}</div>',
                            unsafe_allow_html=True)
            if ov:
                ov_note = f" — “{ov['note']}”" if ov.get("note") else ""
                st.markdown(
                    f'<div class="subtle" style="margin-top:6px">✎ Manually adjusted'
                    f'{_e(ov_note)} · your numbers are folded into everything below; '
                    'revisit them in <b>Adjust scoring</b>.</div>',
                    unsafe_allow_html=True,
                )
            # ---- Company quality (Q): the full rubric, dimension by
            # dimension — hover a name for what it measures and its weight;
            # the note is the evidence each score rests on. Dims Claude
            # could NOT evidence show as excluded, never guessed.
            if verdict is not None:
                weighted_dims = [(k, w) for k, w in thesis.quality_weights.items() if w > 0]
                wsum_q = sum(w for _, w in weighted_dims) or 1.0
                q = quality_score(verdict, thesis)
                if q is not None:
                    lens = CUSTOMER_TYPE_LABEL.get(verdict.customer_type or "", "")
                    known_n = sum(1 for k, _w in weighted_dims if k in verdict.quality)
                    st.markdown(
                        f'<div class="subtle" style="margin-top:10px">Company quality '
                        f'<b>Q {q:.0f}</b> · {params.score_weight_quality:.0%} of the blend — '
                        f'{known_n} of {len(weighted_dims)} dimensions evidence-backed'
                        + (f", scored through the {_e(lens)} lens" if lens else "")
                        + '. Hover a dimension for what it measures.</div>',
                        unsafe_allow_html=True,
                    )
                    q_rows: list[str] = []
                    for k, w in sorted(weighted_dims,
                                       key=lambda kw: -(verdict.quality.get(kw[0], -1.0))):
                        label = QUALITY_DIM_LABEL.get(k, k)
                        tip = (f"{QUALITY_DIMENSIONS.get(k, '')} — "
                               f"{100 * w / wsum_q:.0f}% of the quality score")
                        value = verdict.quality.get(k)
                        if value is None:
                            q_rows.append(
                                f'<div class="sigrow" style="opacity:0.55">'
                                f'<div class="signame" title="{_e(tip)}">{_e(label)}</div>'
                                f'<div class="sigtrack"></div>'
                                f'<div class="sigpts">—</div>'
                                f'<div class="sigdetail">no evidence — excluded, weight renormalizes</div></div>'
                            )
                        else:
                            value = min(max(value, 0.0), 1.0)
                            reason = verdict.quality_reasons.get(k, "")
                            q_rows.append(
                                f'<div class="sigrow"><div class="signame" title="{_e(tip)}">{_e(label)}</div>'
                                f'<div class="sigtrack"><div class="sigfill" style="width:{100 * value:.0f}%"></div></div>'
                                f'<div class="sigpts">{value:.0%}</div>'
                                f'<div class="sigdetail" title="{_e(reason)}">{_e(reason)}</div></div>'
                            )
                    st.markdown(f'<div style="margin-top:4px">{"".join(q_rows)}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="subtle" style="margin-top:10px">No quality rubric on '
                        'this verdict (older cache) — <b>Reclassify latest run</b> on the '
                        'Thesis page scores it, or set the dimensions yourself in '
                        '<b>Adjust scoring</b>.</div>',
                        unsafe_allow_html=True,
                    )
                # ---- Thesis fit (F): one number, with Claude's reasoning.
                if verdict.thesis_fit is not None:
                    st.markdown(
                        f'<div class="subtle" style="margin-top:10px">Thesis fit '
                        f'<b>F {100 * min(max(verdict.thesis_fit, 0.0), 1.0):.0f}</b> · '
                        f'{params.score_weight_fit:.0%} of the blend — '
                        f'{_e(verdict.fit_reason or "no reasoning recorded")}</div>',
                        unsafe_allow_html=True,
                    )
            # ---- X signals (S): the deterministic momentum score — every
            # signal that fired, its weighted points, and the evidence.
            hits = [s for s in lead.signals if s.value > 0]
            if comps["signals"] is not None:
                s_note = (f'{len(hits)} signal{"s" if len(hits) != 1 else ""} fired '
                          f'of {len(thesis.weights)} tracked'
                          if hits else "no signals fired")
                st.markdown(
                    f'<div class="subtle" style="margin-top:10px">X signals '
                    f'<b>S {comps["signals"]:.0f}</b> · {params.score_weight_signals:.0%} '
                    f'of the blend — {s_note}. Hover a signal for what it means; '
                    'points = value × weight.</div>',
                    unsafe_allow_html=True,
                )
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

            # The value-add dimension: which of the firm's specific levers
            # would accelerate this startup, per the classifier.
            if verdict and verdict.value_add_fit is not None:
                firm = thesis.firm_name or "Firm"
                reason = f" — {verdict.value_add_reason}" if verdict.value_add_reason else ""
                st.markdown(
                    f'<div class="subtle" style="margin-top:10px">{_e(firm)} lift '
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

            # ---- The score math, step by step: how Q, F, and S blend and
            # which multipliers moved the number. The last line IS the score.
            st.markdown(
                '<div class="subtle" style="margin-top:10px">Score math — the blend '
                'renormalizes over the components this lead actually evidences, '
                'then trust multipliers apply:</div>',
                unsafe_allow_html=True,
            )
            steps = "".join(
                f'<div class="math-step">{_e(desc)} → <b>{running:.1f}</b></div>'
                for desc, running in score_breakdown(
                    lead, thesis, manual_score=ov.get("score"))
            )
            st.markdown(f'<div style="margin-top:2px">{steps}</div>', unsafe_allow_html=True)

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
            b1, b2, _sp = st.columns([1.4, 1.9, 2.7])
            memo_label = "Open memo" if row.get("brief") else "Memo"
            if b1.button(memo_label, key=f"{key_ns}_brief_{handle_key}",
                         help="The full investment memo lives on the Memos page — "
                              "this writes one (or opens the existing one) there."):
                st.session_state["nav_target"] = "Memos"
                st.session_state["memo_target"] = handle_key
                st.rerun()
            with b2.popover("Adjust scoring"):
                _override_editor(lead, ov, key_ns)


def _render_startup_feed() -> None:
    """The sourcing feed: startup-first lead cards over the latest run (or the
    all-runs ledger), with track / scope / filters and inline triage."""
    if not leads and not ledger:
        st.markdown(
            '<div class="section-title">No startups yet</div>'
            '<div class="section-sub">Open <b>Thesis</b>, describe your thesis, and run discovery. '
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
                'When your X cookies are set, open <b>Thesis → Run discovery</b>.</div>',
                unsafe_allow_html=True,
            )
        elif (datetime.now(timezone.utc) - last_run).total_seconds() > 24 * 3600:
            st.markdown(
                f'<div class="nudge">Last discovery run {_ago(last_run.isoformat())} — '
                'daily runs keep the follow-graph and bio-change signals meaningful. '
                'Open <b>Thesis → Run discovery</b>.</div>',
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
                "Scope", ["Latest run", "All runs"], default="Latest run",
                key="leads_time_scope", label_visibility="collapsed",
            ) or "Latest run"
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
        n_long = counts.get("longlisted", 0)
        n_short = sum(counts.get(s, 0) for s in WIN_STAGES)
        sub = f"{n_new} new this run" if scope == "All runs" else "latest run"
        t1, t2, t3, t4 = st.columns(4)
        if track == "Startups":
            n_companies = len(group_by_company(pairs))
            t1.markdown(_tile("Launched startups", str(n_companies), sub), unsafe_allow_html=True)
        elif track == "Pre-launch watch":
            t1.markdown(_tile("Pre-launch startups", str(len(base_leads)), sub), unsafe_allow_html=True)
        else:
            t1.markdown(_tile("Tracked leads", str(len(base_leads)), sub), unsafe_allow_html=True)
        t2.markdown(_tile("Strong fit", str(strong_fit), "thesis fit ≥ 70%"), unsafe_allow_html=True)
        t3.markdown(_tile("Smart-money events", str(n_convergence), "new follows · latest run"),
                    unsafe_allow_html=True)
        t4.markdown(_tile("In funnel", str(n_long + n_short),
                          f"{n_long} longlisted · {n_short} shortlisted"), unsafe_allow_html=True)
        st.write("")

        # Reset must land BEFORE the filter widgets instantiate.
        FILTER_DEFAULTS: dict = {
            "f_type": ["founder", "startup"], "f_stage": list(STAGES),
            "f_ctype": list(CUSTOMER_TYPES),
            "f_minscore": 0, "f_minfit": 0, "f_hidepassed": True,
        }
        if st.session_state.pop("filters_reset", False):
            for k, v in FILTER_DEFAULTS.items():
                st.session_state[k] = v
        n_active = sum([
            set(st.session_state.get("f_type", FILTER_DEFAULTS["f_type"])) != {"founder", "startup"},
            set(st.session_state.get("f_stage", FILTER_DEFAULTS["f_stage"])) != set(STAGES),
            set(st.session_state.get("f_ctype", FILTER_DEFAULTS["f_ctype"])) != set(CUSTOMER_TYPES),
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
            sort_by = st.selectbox("Sort", ["Score", "Quality", "Score change", "Thesis fit",
                                            lift_sort, "Followers"],
                                   label_visibility="collapsed")
        with f3:
            with st.popover(f"Filters · {n_active}" if n_active else "Filters"):
                type_filter = st.multiselect("Type", ["founder", "startup", "other"],
                                             default=FILTER_DEFAULTS["f_type"], key="f_type",
                                             format_func=lambda t: TYPE_LABEL[t])
                stage_filter = st.multiselect("Stage", list(STAGES),
                                              default=FILTER_DEFAULTS["f_stage"], key="f_stage",
                                              format_func=lambda s: STAGE_LABEL[s])
                ctype_filter = st.multiselect("Customer type", list(CUSTOMER_TYPES),
                                              default=FILTER_DEFAULTS["f_ctype"], key="f_ctype",
                                              format_func=lambda c: CUSTOMER_TYPE_LABEL[c],
                                              help="B2B vs B2C lens the classifier applied. "
                                                   "Unclassified leads are never hidden by this.")
                min_score = st.slider("Minimum score", 0, 100, 0, key="f_minscore")
                min_fit = st.slider("Minimum thesis fit", 0, 100, 0, format="%d%%", key="f_minfit")
                hide_passed = st.toggle("Hide passed", value=True, key="f_hidepassed")
                if n_active and st.button("Reset filters"):
                    st.session_state["filters_reset"] = True
                    st.rerun()
        with f4:
            pass

        HIDE_LABELS = {"type": "type filter", "stage": "stage filter",
                       "ctype": "customer type", "score": "min score",
                       "fit": "min fit", "passed": "passed", "search": "search"}

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
            # Same unknown-is-never-hidden semantics for the B2B/B2C lens.
            if (verdict is not None and verdict.customer_type is not None
                    and verdict.customer_type not in ctype_filter):
                return "ctype"
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
        elif sort_by == "Quality":
            shown.sort(key=lambda p: -(qs if (qs := quality_score(p[0].llm, thesis)) is not None
                                       else -1.0))
        elif sort_by == lift_sort:
            shown.sort(key=lambda p: -(p[0].llm.value_add_fit
                                       if p[0].llm and p[0].llm.value_add_fit is not None else -1))
        elif sort_by == "Followers":
            shown.sort(key=lambda p: -p[0].account.followers)
        elif sort_by == "Score change":
            shown.sort(key=lambda p: (p[1].score_delta
                                      if p[1] and p[1].score_delta is not None
                                      else float("-inf")), reverse=True)
        else:  # Score — explicit, so manual adjustments re-rank the view
            shown.sort(key=lambda p: -p[0].score)

        # Startup-first in EVERY track: fold accounts attributed to the same
        # startup into one card; unnamed founders are singleton stealth
        # startups (companies.startup_identity).
        display = group_by_company(shown)

        hidden_note = ""
        if hidden_counts:
            parts = [f"{count} by {HIDE_LABELS[reason]}"
                     for reason, count in sorted(hidden_counts.items(), key=lambda kv: -kv[1])]
            hidden_note = " · hidden: " + ", ".join(parts)
        if len(display) != len(shown):
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
                         tuple(ctype_filter), min_score, min_fit, hide_passed, query, sort_by))
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


# ============================================================ LONGLIST · SHORTLIST


def _pick_label(h: str) -> str:
    picked = lead_by_handle.get(h)
    ident = startup_identity(picked) if picked else None
    return f"{ident[0]} — @{h}" if ident else f"@{h}"


def _ranked(handles: list[str]) -> list[str]:
    return sorted(handles,
                  key=lambda h: -(lead_by_handle[h].score if h in lead_by_handle else 0))


def _dimension_cards(handles: list[str], key_ns: str) -> None:
    """Score-ranked lead cards with Claude's per-dimension scoring open on
    every card: signal bars, step-by-step score math, thesis fit, and the
    firm value-add levers."""
    with_leads = [h for h in handles if h in lead_by_handle]
    view_max = max((lead_by_handle[h].score for h in with_leads), default=100.0)
    for h in handles:
        lead = lead_by_handle.get(h)
        if lead is None:
            st.markdown(
                f'<div class="subtle">@{_e(h)} — no lead data in the store '
                '(cleared or never scored); manage it in the table above.</div>',
                unsafe_allow_html=True,
            )
            continue
        _lead_card(lead, entry_by_handle.get(h),
                   fresh=h in latest_handles, view_max=view_max,
                   details_open=True, key_ns=key_ns)


if nav == "Longlist":
    longlist = _ranked([h for h, p in pipeline.items()
                        if (p.get("status") or "") == "longlisted"])
    if not longlist:
        st.markdown(
            '<div class="section-title">Nothing longlisted yet</div>'
            '<div class="section-sub">Review the <b>Startups</b> feed and longlist the promising '
            'ones — they collect here for a closer look before the shortlist cut.</div>',
            unsafe_allow_html=True,
        )
    else:
        ll_fits = [lead_by_handle[h].llm.thesis_fit for h in longlist
                   if h in lead_by_handle and lead_by_handle[h].llm
                   and lead_by_handle[h].llm.thesis_fit is not None]
        ll_scores = [lead_by_handle[h].score for h in longlist if h in lead_by_handle]
        t1, t2, t3 = st.columns(3)
        t1.markdown(_tile("Longlisted", str(len(longlist))), unsafe_allow_html=True)
        t2.markdown(_tile("Strong fit", str(sum(1 for f in ll_fits if f >= 0.7)),
                          "thesis fit ≥ 70%"), unsafe_allow_html=True)
        t3.markdown(_tile("Top score", f"{max(ll_scores):.0f}" if ll_scores else "—"),
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub" style="margin-top:14px">Claude\'s scoring is open on every '
            'card — signals, score math, thesis fit, value-add levers. Promote the best to the '
            'shortlist.</div>',
            unsafe_allow_html=True,
        )
        _dimension_cards(longlist, key_ns="ll")


if nav == "Shortlist":
    shortlist = _ranked([h for h, p in pipeline.items()
                         if (p.get("status") or "") in WIN_STAGES])
    if not shortlist:
        st.markdown(
            '<div class="section-title">Nothing shortlisted yet</div>'
            '<div class="section-sub">Promote startups from the <b>Longlist</b> — they land here '
            'as “Shortlisted”, ready to work toward allocation.</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(len(WIN_STAGES))
        for col, stage_key in zip(cols, WIN_STAGES):
            col.markdown(_tile(STATUS_LABELS[stage_key], str(counts.get(stage_key, 0))),
                         unsafe_allow_html=True)
        st.write("")

        editor_rows = []
        for handle_key in shortlist:
            row = pipeline[handle_key]
            lead = lead_by_handle.get(handle_key)
            verdict = lead.llm if lead else None
            identity = startup_identity(lead) if lead else None
            editor_rows.append({
                "handle": handle_key,
                "Startup": identity[0] if identity else "—",
                "Lead": lead.account.url if lead else f"https://x.com/{handle_key}",
                "Fit": (f"{verdict.thesis_fit:.0%}" if verdict and verdict.thesis_fit is not None else "—"),
                "Score": lead.score if lead else 0,
                "Status": STATUS_LABELS.get(row.get("status") or "shortlisted", "Shortlisted"),
                "Notes": row.get("notes") or "",
            })
        edited = st.data_editor(
            editor_rows, hide_index=True, use_container_width=True, key="pipeline_editor",
            column_order=["Startup", "Lead", "Fit", "Score", "Status", "Notes"],
            column_config={
                "handle": None,
                "Startup": st.column_config.TextColumn("Startup", disabled=True,
                                                       width="medium"),
                "Lead": st.column_config.LinkColumn("Lead", display_text=r"x\.com/(.+)",
                                                    disabled=True, width="small"),
                "Fit": st.column_config.TextColumn("Fit", disabled=True, width="small"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                         format="%d", width="medium"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=([STATUS_LABELS["longlisted"]]
                             + [STATUS_LABELS[s] for s in WIN_STAGES]
                             + [STATUS_LABELS["passed"]]),
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
        st.markdown('<div class="section-title">How Claude scored them</div>'
                    '<div class="section-sub">Per-dimension detail for every shortlisted startup — '
                    'signals, score math, thesis fit, value-add levers.</div>',
                    unsafe_allow_html=True)
        _dimension_cards(shortlist, key_ns="sl")

        st.write("")
        export_rows = pipeline_rows(store, thesis)
        if export_rows:
            export_path = write_pipeline_csv(export_rows, PROJECT_ROOT / settings.out_dir)
            st.download_button(
                f"Export pipeline CSV ({len(export_rows)} leads)",
                export_path.read_bytes(), file_name=export_path.name,
            )


# ============================================================ MEMO


def _slug(name: str) -> str:
    """Filesystem-safe filename fragment from a startup name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "memo"


@st.cache_data(show_spinner=False)
def _memo_pdf_cached(memo_md: str, title: str, subtitle: str) -> bytes:
    return memo_pdf_bytes(memo_md, title, subtitle)


def _generate_memo(handle: str) -> None:
    """Assemble the evidence dossier (cached website text — fetched live if
    missing — plus recent tweets and the investor's notes), run the memo
    agent, store the result, and rerun with a startup-named toast."""
    lead = lead_by_handle.get(handle)
    if lead is None:
        st.error("No lead data in the store for this startup — run discovery first.")
        return
    row = pipeline.get(handle, {})
    verdict = lead.llm
    site_text = ""
    url = normalize_site_url(
        ((verdict.company_url if verdict else None) or lead.account.website)
    )
    if url:
        page = store.cached_site(url, settings.website_ttl_days)
        if page is None:
            try:
                page = asyncio.run(fetch_sites([url], settings, store)).get(url)
            except Exception:
                page = None  # a dead site must never block the memo
        if page is not None and page.usable:
            site_text = page.text
    tweets = store.get_tweets(lead.account.id, limit=12)
    name = display_name(lead)
    with st.spinner(f"Writing the investment memo for {name} — Claude is working "
                    "through the full dossier (this takes a minute)…"):
        memo, is_ai = investment_memo(
            lead, thesis, settings, site_text=site_text, tweets=tweets,
            notes=row.get("notes") or "",
        )
    store.set_pipeline(handle, brief=memo)
    st.session_state["toast"] = (
        f"Memo ready — {name}" if is_ai
        else f"Data-only memo skeleton saved for {name} — add an Anthropic key "
             "and regenerate for the full analysis."
    )
    st.rerun()


if nav == "Memos":
    # Arriving from a card's Memo button: select that startup here and, when
    # it has no memo yet, write one on arrival.
    memo_target = st.session_state.pop("memo_target", None)
    if memo_target:
        st.session_state["memo_pick"] = memo_target
        st.session_state.pop("memo_editing", None)
        if not pipeline.get(memo_target, {}).get("brief"):
            st.session_state["memo_autogen"] = True

    funnel_handles = [h for h, p in pipeline.items()
                      if (p.get("status") or "") in FUNNEL_STAGES]
    briefed = [h for h, p in pipeline.items() if p.get("brief")]
    memo_pool = _ranked(list(dict.fromkeys(funnel_handles + briefed)))
    if memo_target and memo_target not in memo_pool:
        memo_pool.insert(0, memo_target)

    if not memo_pool:
        st.markdown(
            '<div class="section-title">No memos yet</div>'
            '<div class="section-sub">Longlist or shortlist startups first — or hit '
            '<b>Memo</b> on any card in the feed — and the full investment memo '
            '(product, tech, competition, market, acquisition dynamics, '
            'recommendation) gets written here.</div>',
            unsafe_allow_html=True,
        )
    else:
        n_briefed = len(briefed)
        st.markdown(
            '<div class="section-title">Investment memos</div>'
            f'<div class="section-sub">{n_briefed or "No"} memo'
            f'{"s" if n_briefed != 1 else ""} on file across {len(memo_pool)} '
            'startups in the funnel. Generate, edit in place, export as PDF — '
            'the outreach draft lives below the memo.</div>',
            unsafe_allow_html=True,
        )
        if "memo_pick" in st.session_state and st.session_state["memo_pick"] not in memo_pool:
            st.session_state["memo_pick"] = memo_pool[0]
        pick = st.selectbox("Startup", memo_pool, format_func=_pick_label,
                            key="memo_pick", label_visibility="collapsed")
        picked_lead = lead_by_handle.get(pick)
        picked_row = pipeline.get(pick, {})
        picked_name = display_name(picked_lead) if picked_lead else f"@{pick}"

        bits = []
        if picked_lead is not None:
            v = picked_lead.llm
            bits.append(f"score {picked_lead.score:.0f}")
            if v and v.thesis_fit is not None:
                bits.append(f"thesis fit {v.thesis_fit:.0%}")
            bits.append(STATUS_LABELS.get(_status_of(picked_lead), ""))
        if picked_row.get("brief_at"):
            bits.append(f"generated {_ago(picked_row['brief_at'])}")
        if picked_row.get("brief_edited_at"):
            bits.append(f"edited {_ago(picked_row['brief_edited_at'])}")
        if bits:
            st.markdown(f'<div class="subtle">{_e(" · ".join(b for b in bits if b))}</div>',
                        unsafe_allow_html=True)

        if st.session_state.pop("memo_autogen", False) and not picked_row.get("brief"):
            _generate_memo(pick)  # ends in st.rerun()

        existing_memo = picked_row.get("brief") or ""
        editing = st.session_state.get("memo_editing") == pick

        a1, a2, a3, a4, _asp = st.columns([1.5, 0.9, 1.3, 1.05, 1.25])
        regen_help = ("Regenerating rewrites the memo from the current evidence — "
                      "your manual edits are overwritten." if existing_memo else None)
        if a1.button("Regenerate memo" if existing_memo else "Generate memo",
                     type="primary", disabled=picked_lead is None, help=regen_help,
                     key="memo_generate"):
            _generate_memo(pick)
        if existing_memo and not editing:
            if a2.button("Edit", key="memo_edit_btn"):
                st.session_state["memo_editing"] = pick
                st.rerun()
        if existing_memo:
            a3.download_button("Markdown", existing_memo.encode("utf-8"),
                               file_name=f"memo_{_slug(picked_name)}.md",
                               key="memo_dl_md")
            memo_date = (picked_row.get("brief_edited_at")
                         or picked_row.get("brief_at") or "")
            try:
                date_label = datetime.fromisoformat(memo_date).strftime("%B %d, %Y")
            except ValueError:
                date_label = datetime.now().strftime("%B %d, %Y")
            sub_bits = [f"{thesis.firm_name or 'Scout'} · {date_label}"]
            if picked_lead is not None:
                sub_bits.append(f"score {picked_lead.score:.0f}/100")
            sub_bits.append(f"@{pick} · internal — confidential")
            try:
                a4.download_button(
                    "PDF",
                    _memo_pdf_cached(existing_memo,
                                     f"{picked_name} — Investment memo",
                                     " · ".join(sub_bits)),
                    file_name=f"memo_{_slug(picked_name)}.pdf",
                    mime="application/pdf", key="memo_dl_pdf",
                )
            except RuntimeError as exc:
                a4.caption(str(exc))

        if editing:
            draft_md = st.text_area("Memo markdown", value=existing_memo, height=560,
                                    key=f"memo_editor_{pick}",
                                    label_visibility="collapsed")
            e1, e2, _esp = st.columns([1, 0.9, 4.1])
            if e1.button("Save memo", type="primary", key="memo_save"):
                store.set_pipeline(pick, brief=draft_md, brief_edited=True)
                st.session_state.pop("memo_editing", None)
                st.session_state["toast"] = f"Memo saved — {picked_name}"
                st.rerun()
            if e2.button("Cancel", key="memo_cancel"):
                st.session_state.pop("memo_editing", None)
                st.rerun()
        elif existing_memo:
            st.write("")
            with st.container(border=True):
                st.markdown(f"### {picked_name}")
                st.markdown(existing_memo)
        else:
            st.markdown(
                '<div class="subtle" style="margin-top:8px">No memo yet for '
                f'<b>{_e(picked_name)}</b> — generate the full investment memo: '
                'overview, product &amp; differentiation, technology, competitive '
                'landscape, market sizing, acquisition dynamics, recommendation.</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        with st.expander("Outreach draft"):
            channel = st.selectbox("Channel", list(CHANNELS))
            if st.button("Draft with AI", disabled=picked_lead is None):
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


# ============================================================ THESIS


if nav == "Thesis":
    # --- AI strategy designer -------------------------------------------------
    st.markdown('<div class="section-title">Define the thesis</div>'
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
    run_ready = not scan_active

    # Time estimate up front — empirical after the first completed run,
    # settings-derived before that.
    est_lo, est_hi, _per, est_label = _estimate_scan("run", max_accounts=int(max_accounts))
    st.markdown(
        f'<div class="rp-est">Estimated <b>{_fmt_dur(est_lo)}–{_fmt_dur(est_hi)}</b> '
        f'for this run · {_e(est_label)}. Runs happen in the background — you can '
        'keep browsing while the pipeline reports progress below.</div>',
        unsafe_allow_html=True,
    )
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
    run_col, preview_col, reclass_col, _sp = st.columns([1, 1.75, 1.75, 1.5])
    if run_col.button("Run scout", type="primary", disabled=not run_ready):
        _launch_scan(["run", "--source", "xapi" if paid_run else "twscrape",
                      "--max-accounts", str(int(max_accounts)),
                      "--min-score", str(int(min_score_run)), "--ttl-days", str(int(ttl))],
                     "run")
    if preview_col.button("Preview discovery (free, no scoring)", disabled=scan_active):
        _launch_scan(["source", "--max-accounts", str(int(max_accounts))], "source")
    if reclass_col.button(
        "Reclassify latest run", disabled=scan_active or not (leads or ledger),
        help="Re-run Claude classification + the adversarial audit on the latest "
             "run's leads — no discovery, cache-first, minutes not an hour. Use "
             "after editing the thesis, prompt, or weights.",
    ):
        _launch_scan(["reclassify"], "reclassify")

    # The run cockpit — phase stepper, progress bars, ETA, log tail. Renders
    # only while a scan runs (or just finished) and polls on its own.
    _run_panel()

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
    v_lo, v_hi, _vper, _vlabel = _estimate_scan("verify", n_verify=int(n_verify))
    st.markdown(
        f'<div class="subtle">Worst case ≈ <b>${v_est:.2f}</b> · '
        f'<b>${v_remaining:.2f}</b> left of the cap · estimated '
        f'<b>{_fmt_dur(v_lo)}–{_fmt_dur(v_hi)}</b>. Results land as a verify run '
        f'in the ledger.</div>',
        unsafe_allow_html=True,
    )
    v_ready = st.checkbox(f"Spend up to ${v_est:.2f} of the X API budget",
                          key="confirm_verify_spend")
    if st.button("Run precision pass", disabled=not v_ready or scan_active):
        _launch_scan(["verify", "--max", str(int(n_verify)), "--tweets", str(int(v_tweets))],
                     "verify")

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
            st.markdown('<div class="subtle">Triage at least 5 leads (longlist, shortlist, or '
                        'pass) and insights appear here: how your decisions cluster by sector and '
                        'signal, plus AI-suggested weight adjustments.</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="subtle">Score = blend of <b>company quality</b> (evidence-backed '
                    'rubric), <b>thesis fit</b>, and <b>X signals</b> — weighted 45/35/20 by '
                    'default, renormalized over what each lead evidences — then × Claude '
                    'confidence, × 0.2 if not-a-founder, × stage multiplier, × value-add '
                    'multiplier (off by default), × ungrounded multiplier. All editable.</div>',
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
            st.markdown("**Final score blend** — quality / fit / signals, renormalized "
                        "over whatever a lead evidences")
            params = thesis.signal_params
            b1, b2, b3 = st.columns(3)
            with b1:
                wq = st.number_input("Company quality weight", 0.0, 1.0,
                                     float(params.score_weight_quality), 0.05,
                                     help="Claude's evidence-backed quality rubric "
                                          "(team, tech, market, moat, traction, investors).")
            with b2:
                wf = st.number_input("Thesis fit weight", 0.0, 1.0,
                                     float(params.score_weight_fit), 0.05,
                                     help="How squarely the PRODUCT matches the thesis.")
            with b3:
                ws = st.number_input("X-signal weight", 0.0, 1.0,
                                     float(params.score_weight_signals), 0.05,
                                     help="Momentum: smart-money follows, launch traction, "
                                          "departures — the deterministic signal score.")
            st.markdown("**Quality dimensions** — relative weights inside the quality component")
            new_quality_weights: dict[str, float] = {}
            qcols = st.columns(3)
            for i, (dim_key, dim_help) in enumerate(QUALITY_DIMENSIONS.items()):
                with qcols[i % 3]:
                    new_quality_weights[dim_key] = float(st.slider(
                        dim_key.replace("_", " & ") if dim_key == "tech_product" else dim_key,
                        0, 50, int(thesis.quality_weights.get(dim_key, 0)),
                        help=dim_help, key=f"qw_{dim_key}"))
            st.divider()
            q1, q2, q3 = st.columns(3)
            with q1:
                tf = st.number_input("Traction floor (eng/followers)", 0.0, 1.0, float(params.traction_floor), 0.01)
                ts = st.number_input("Traction saturation", 0.01, 2.0, float(params.traction_saturation), 0.01)
            with q2:
                tw = st.number_input("Traction window (days)", 1, 180, int(params.traction_window_days))
                cf = st.number_input("Convergence full-credit follows", 1, 10, int(params.convergence_full_credit))
            with q3:
                sm = st.number_input("Off-target stage multiplier", 0.0, 1.0, float(params.stage_mismatch_multiplier), 0.05)
                um = st.number_input("Ungrounded multiplier", 0.0, 1.0,
                                     float(params.ungrounded_multiplier), 0.05,
                                     help="Score × this when the product claim never traced "
                                          "to evidence (audit 'unverifiable', or unaudited "
                                          "with grounding none/bio).")
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
                    "quality_weights": new_quality_weights,
                    # Every param must be listed — a missing one silently
                    # resets to its default on save.
                    "signal_params": SignalParams(
                        traction_floor=tf, traction_saturation=ts,
                        traction_window_days=int(tw), convergence_full_credit=int(cf),
                        stage_mismatch_multiplier=sm,
                        score_weight_quality=wq, score_weight_fit=wf,
                        score_weight_signals=ws,
                        value_add_weight=vw, ungrounded_multiplier=um),
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


# ============================================== STARTUPS · database sub-page


# Friendly presentation order + one-liners; unknown tables still appear after.
DB_TABLE_ORDER = [
    "leads", "accounts", "tweets", "llm_verdicts", "pipeline", "runs",
    "unlinked_leads", "follow_edges", "follow_meta", "bio_snapshots",
    "searches", "xapi_usage", "scan", "scan_history",
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
    "scan": "Live scan status (drives the banner + run cockpit).",
    "scan_history": "Completed scans with per-phase timings (powers time estimates).",
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


def _render_database() -> None:
    """The startup database: one dossier row per tracked startup — what it
    does, how it scored (Q/F/S) and why, funnel status, history. Selecting a
    row opens the full record. The raw SQLite browser lives behind an
    expander at the bottom."""
    st.markdown(
        '<div class="section-title">Startup database</div>'
        '<div class="section-sub">Every startup scout tracks, across all runs — the product, '
        'the score and its quality / fit / signals components, funnel status, and history. '
        'Select a row for the full dossier: evidence, per-dimension scoring with reasons, '
        'the score math, and your manual adjustments.</div>',
        unsafe_allow_html=True,
    )
    if not ledger:
        st.markdown(
            '<div class="subtle">Nothing tracked yet — run discovery on the '
            '<b>Thesis</b> page, or <code>./scout-cli demo</code> for an offline '
            'sample.</div>',
            unsafe_allow_html=True,
        )
    else:
        db_rows: list[dict] = []
        for e in ledger:
            lead = e.lead
            v = lead.llm
            handle = lead.account.handle.lower()
            comps = score_components(lead, thesis)
            status = pipeline.get(handle, {}).get("status") or "new"
            what = ((v.product_summary if v else "")
                    or (v.one_line_summary if v else "") or lead.account.bio or "")
            db_rows.append({
                "handle": handle,
                "Startup": display_name(lead),
                "Score": round(lead.score),
                "Q": None if comps["quality"] is None else round(comps["quality"]),
                "F": None if comps["fit"] is None else round(comps["fit"]),
                "S": None if comps["signals"] is None else round(comps["signals"]),
                "What they do": " ".join(what.split()),
                "Stage": (STAGE_LABEL.get(v.stage, v.stage) if v and v.stage else ""),
                "Sector": (" · ".join(x for x in [v.sector or "", v.subsector or ""] if x)
                           if v else ""),
                "Customers": (CUSTOMER_TYPE_LABEL.get(v.customer_type, "")
                              if v and v.customer_type else ""),
                "Status": (STATUS_LABELS.get(status, status) if status != "new" else ""),
                "Adjusted": "✎" if overrides.get(handle) else "",
                "Memo": "✓" if pipeline.get(handle, {}).get("brief") else "",
                "Followers": lead.account.followers,
                "First seen": (e.first_seen_at.date().isoformat()
                               if e.first_seen_at else ""),
                "Runs": e.times_seen,
                "Profile": lead.account.url,
            })

        f1, f2, f3, f4 = st.columns([2.5, 1.35, 1.35, 1.1])
        with f1:
            db_search = st.text_input(
                "Search startups", key="sdb_q", label_visibility="collapsed",
                placeholder="Search startup, product, sector, status…")
        with f2:
            stage_pick = st.multiselect(
                "Stage", [STAGE_LABEL[s] for s in STAGES], key="sdb_stage",
                label_visibility="collapsed", placeholder="All stages")
        with f3:
            status_pick = st.multiselect(
                "Status", list(STATUS_LABELS.values()), key="sdb_status",
                label_visibility="collapsed", placeholder="All statuses")
        with f4:
            sdb_min = st.number_input("Min score", 0, 100, 0, 5, key="sdb_min",
                                      label_visibility="collapsed",
                                      help="Minimum score")

        def _sdb_keep(r: dict) -> bool:
            if stage_pick and r["Stage"] not in stage_pick:
                return False
            if status_pick and (r["Status"] or "New") not in status_pick:
                return False
            if r["Score"] < sdb_min:
                return False
            if db_search.strip():
                hay = " ".join(str(x) for x in r.values()).lower()
                if db_search.strip().lower() not in hay:
                    return False
            return True

        view_rows = [r for r in db_rows if _sdb_keep(r)]
        st.markdown(
            f'<div class="subtle" style="margin:2px 0 8px">{len(view_rows)} of '
            f'{len(db_rows)} startups · select a row for the full dossier</div>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(view_rows)
        event = st.dataframe(
            df, use_container_width=True, hide_index=True, key="sdb_table",
            on_select="rerun", selection_mode="single-row",
            height=min(560, 37 * (len(df) + 1) + 5),
            column_order=["Startup", "Score", "Q", "F", "S", "What they do",
                          "Stage", "Sector", "Customers", "Status", "Adjusted",
                          "Memo", "Followers", "First seen", "Runs", "Profile"],
            column_config={
                "handle": None,
                "Startup": st.column_config.TextColumn("Startup", width="medium",
                                                       pinned=True),
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%d", width="small",
                    help="Final score — quality/fit/signals blend × trust multipliers"),
                "Q": st.column_config.NumberColumn(
                    "Q", width="small",
                    help="Company quality 0–100 — Claude's evidence-backed rubric "
                         "(team, tech, market, moat, traction, investors)"),
                "F": st.column_config.NumberColumn(
                    "F", width="small", help="Thesis fit 0–100"),
                "S": st.column_config.NumberColumn(
                    "S", width="small", help="X-signal momentum 0–100"),
                "What they do": st.column_config.TextColumn("What they do",
                                                            width="large"),
                "Adjusted": st.column_config.TextColumn(
                    "✎", width="small", help="Manually adjusted scoring"),
                "Memo": st.column_config.TextColumn(
                    "Memo", width="small", help="Investment memo on file"),
                "Followers": st.column_config.NumberColumn("Followers",
                                                           format="localized"),
                "Profile": st.column_config.LinkColumn(
                    "Profile", display_text=r"x\.com/(.+)", width="small"),
            },
        )
        st.download_button(
            f"Download view as CSV ({len(view_rows)} startups)",
            df.drop(columns=["handle"]).to_csv(index=False).encode("utf-8")
            if not df.empty else b"",
            file_name="startups_view.csv", key="sdb_csv",
        )

        picked_rows = (event.selection.rows or []) if event is not None else []
        if picked_rows and not df.empty:
            sel_handle = str(df.iloc[picked_rows[0]]["handle"])
            sel_lead = lead_by_handle.get(sel_handle)
            if sel_lead is not None:
                st.write("")
                st.markdown('<div class="section-title" style="font-size:1.15rem">Dossier</div>',
                            unsafe_allow_html=True)
                _lead_card(sel_lead, entry_by_handle.get(sel_handle),
                           fresh=sel_handle in latest_handles,
                           details_open=True, key_ns="db")

    st.write("")
    # A toggle, not an expander — the raw browser has its own SQL expander
    # inside, and Streamlit forbids nesting expanders.
    if st.toggle("Raw tables — the underlying SQLite store", key="sdb_raw"):
        _render_raw_tables()


def _render_raw_tables() -> None:
    """The raw store, browsable: any table, auto-generated filters, full-text
    search, CSV export of the view, and a read-only SQL console."""
    db = store.db
    known_tables = [t for t in DB_TABLE_ORDER if db[t].exists()]
    extra_tables = sorted(
        t for t in db.table_names()
        if t not in known_tables and not t.startswith("sqlite_")
    )
    db_tables = known_tables + extra_tables
    db_file = Path(store.db_path)

    st.markdown(
        f'<div class="subtle">Everything scout knows, raw — <code>{_e(str(db_file))}</code>. '
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


# The Startups page: the sourcing feed + the startup database, as sub-pages.
if nav == "Startups":
    sub_feed, sub_db = st.tabs(["Latest run", "Database"])
    with sub_feed:
        _render_startup_feed()
    with sub_db:
        _render_database()


# ============================================================ SETTINGS


if nav == "Settings":
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
