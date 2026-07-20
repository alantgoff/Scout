# AGENTS.md — orientation for AI agents working on `scout`

Read this first. It's the fast map: what the project is, how to run it, where
everything lives, the data flow, the invariants you must not break, and the
non-obvious gotchas that will otherwise waste your time. (`README.md` is the
user-facing version; this file is denser and aimed at contributors.)

---

## 1. What this is

`scout` sources **launched startups** (primary) and pre-launch founders-to-be
(secondary "watch" track) from Twitter/X, GitHub, and HN for a VC analyst at
Headline. The discovery unit is an X account; the *product* unit is a STARTUP —
the classifier extracts `company_name`/`company_url` and `scout/companies.py`
resolves every founder-like lead to a startup identity (real company name, or
a synthesized "Ada Lin's stealth startup" placeholder when unnamed) and folds
founder + company accounts into one entry, in every view and the report. It's
a Python 3.12 package with a **Typer CLI** and a **Streamlit UI** (Headline design
language, funnel-ordered: Thesis · Startups (Latest run / Database) · Longlist ·
Shortlist · Memos · Settings; session-state nav, so buttons route across
pages), managed by **uv**. A thesis
(`thesis.yaml`) drives all targeting; nothing is hardcoded.

The pipeline: **discover** candidate accounts (free) → run cheap deterministic
**heuristics** → gate + rank → **read each candidate's company website**
(scout/web.py: normalized root URL, cached in the `websites` table with TTL,
failures negative-cached) → **Claude** classification GROUNDED in the site
text (fine-grained: stage/sector/subsector/business model/tags + a 0–1
**thesis_fit** + a 0–1 **value_add_fit** with per-lever breakdown, plus
**product_summary** and a **grounding** claim; EVIDENCE_RULES forbid inferring
the product from founder pedigree, with an unknown-escape; verdicts cached in
the store) → **adversarial verification** of the top `VERIFY_TOP_N` verdicts
against their dossiers (corrections re-cached) → **score** 0–100 → **export**
CSV/Markdown → **triage & pursue** in the UI. `scout reclassify` re-runs
everything from classification down on the latest run's leads — cache-first,
no discovery — the fast loop for thesis/prompt iteration. Agents (`scout/agents.py`): the **strategy agent**
(plain-language thesis → full thesis.yaml + seeds.yaml proposal), the
**research-brief agent** (compact pre-call brief), and the **investment-memo
agent** (full multi-section memo — overview / product & differentiation /
tech & architecture / competitive landscape / market sizing / strategic
capital & acquisition dynamics / recommendation — grounded in website capture
+ tweets + the quality rubric; stored in the pipeline table, editable in the
UI, exportable as Markdown or PDF via `export.memo_pdf_bytes`). Scoring is
overridable: `score_overrides` (store) + `score.apply_override` layer the
investor's manual quality/fit/pinned-score numbers into the same math at load
time.

---

## 2. Run it / test it — always via `./scout-cli` or `uv run`

```bash
uv run pytest -q                 # ~115 tests, ~3s, no network (incl. an AppTest UI smoke test)
./scout-cli demo                 # $0 offline end-to-end run on sample founders — best smoke test
./scout-cli source --strategy github,hn   # free live discovery, no scoring
./scout-cli ui                   # Streamlit workspace on :8501
./start                          # user-facing launcher: sync → seed-if-empty → serve → open browser
```

**Never rely on the bare `scout` console script.** macOS + uv marks `.venv`
hidden; CPython's `site.py` skips hidden `.pth` files, so after any dependency
`uv sync` the editable install silently drops off `sys.path` and `uv run scout`
fails with `ModuleNotFoundError: No module named 'scout'`. Mitigations already
in place:
- `./scout-cli` runs `python -m scout.cli` from the repo root (immune).
- `conftest.py` puts the repo root on `sys.path` (pytest immune).
- `scout/ui.py` bootstraps `sys.path` at the top (Streamlit immune).
- If the bare command still breaks: `chflags -R nohidden .venv`.

**Streamlit + config edits:** the dev server hot-reloads the *page file* but not
already-imported modules. After editing `scout/config.py` (or any imported
module) you must **restart** the preview/server, or you'll see a stale
`ImportError`. Editing `scout/ui.py` alone hot-reloads fine.

Python is pinned to **3.12** via `.python-version` (3.14's `site.py` is stricter
about the hidden-`.pth` issue).

---

## 3. Repo map

```
scout/
  cli.py            Typer app — ALL orchestration. Commands: run, source, inspect,
                    verify, probe, demo, export, budget, strategy, ui. Pipeline
                    helpers: _run_pipeline, _enrich_accounts, _run_discovery,
                    _merge_accounts (fills Account.sources), _fetch_tweets
                    (parallel for free adapters).
  config.py         Pydantic Settings (.env) + Thesis/Seeds/SignalParams (yaml)
                    + save_thesis/save_seeds (shared by CLI + UI).
                    STAGE_* maps: stage → search categories / discovery sources.
  models.py         The ONLY data structures crossing module boundaries:
                    Account, Tweet, Signal, LLMVerdict, Lead, UnlinkedLead.
  store.py          SQLite (sqlite-utils) — cache, dedupe, TTL, budget ledger,
                    follow-edge/bio snapshots, unlinked leads, deal-flow pipeline,
                    llm_verdicts cache.
  score.py          Pure scoring. score_leads + score_breakdown (THE score math,
                    single source of truth; UI renders its steps) + apply_override
                    (manual quality/fit/pinned-score overrides re-entering that math).
  export.py         CSV + Markdown writers, rich terminal table, memo_pdf_bytes
                    (markdown-pdf/PyMuPDF investment-memo PDF), override-aware
                    pipeline_rows(store, thesis).
  outreach.py       Claude-drafted first-touch outreach (Memos page); template fallback.
  agents.py         Strategy agent (generate_strategy/parse_strategy/apply_strategy),
                    research-brief agent (research_brief), investment-memo agent
                    (investment_memo + MEMO_SECTIONS; dossier = brief context +
                    website text + tweets + notes; offline skeleton fallback),
                    weight-tuning agent (suggest_weights/parse_weight_proposal),
                    watchlist validation (validate_watchlist). Parse/apply helpers
                    are pure. All Claude clients carry explicit timeouts
                    (STRATEGY/BRIEF/WEIGHTS/MEMO_TIMEOUT_S); timeouts are excluded
                    from tenacity retry — fail fast in the UI.
  insights.py       Pure triage analytics: triage_stats/stats_prompt contrast
                    shortlisted-vs-passed leads per signal/sector/stage/fit;
                    feeds the UI insights panel and the weight-tuning agent.
  companies.py      Pure startup identity + grouping: startup_identity resolves
                    every founder-like lead to (name, synthesized) — real
                    company, or "Ada Lin's stealth startup" when unnamed;
                    founder_like gates it (verdict, else founder-evidence
                    signals); company_key/group_by_company fold accounts
                    sharing a company into one entry (primary = highest score).
  demo_data.py      8 synthetic sample founders for `scout demo` (obviously fake handles).
  ui.py             Streamlit app: Thesis / Startups (Latest run + Database) /
                    Longlist / Shortlist / Memos / Settings. Session-state nav
                    (nav_target routes across pages — the card Memo button lands
                    on Memos and auto-generates); startup database with dossier
                    row-select; per-card Q/F/S score breakout + Adjust-scoring
                    popover; Memos page with in-place editing and .md/.pdf
                    export. Headline design language. ~2600 lines.
  ingest/
    base.py         SourceAdapter ABC (X sources) + DiscoverySource ABC (github/hn).
    twscrape_src.py Primary free X adapter: query bank, bio search, list members,
                    follow-graph snapshotting.
    xapi_src.py     Paid X API v2 adapter — BUDGET-GUARDED. BudgetExceededError.
    github_src.py   GitHub discovery (repo search → owner → X-handle bridge).
    hn_src.py       Hacker News (Algolia) discovery.
    linkedin_src.py Stub (NotImplementedError) — LinkedIn automation is a dead end.
  signals/
    heuristics.py   8 deterministic signals + run_heuristics + intent_appeared.
    llm.py          Claude classification, GROUNDED: site text in the dossier,
                    EVIDENCE_RULES appended to every prompt (even custom ones),
                    unknown-escape, batches of CLASSIFY_BATCH_SIZE (5). Plus the
                    adversarial audit (verify_leads/apply_verification) — top-N
                    verdicts re-checked against evidence, corrections re-cached.
                    EVERY parsed verdict attaches (no MIN_CONFIDENCE drop) —
                    score ×= confidence does the sinking.
  web.py            Company-website evidence: normalize_site_url (root URL,
                    skip link farms/socials/IPs), extract_site_text (bs4,
                    title+meta first for SPAs), async fetch_sites (semaphore
                    fan-out, cache-first, negative caching).
tests/              pytest — test_heuristics, test_score, test_store,
                    test_sourcing_v2, test_web, test_grounding.
thesis.yaml         Targeting + weights + signal_params + firm value-add levers
                    + llm_prompt. User-owned.
seeds.yaml          Query bank, bio_searches, watchlist, github_topics. User-owned.
scout-cli           Bash wrapper → `uv run python -m scout.cli "$@"`.
start               One-command launcher: uv sync → .env → seed demo when empty
                    (only key-less, so guaranteed $0) → streamlit → open browser.
                    Idempotent; `--port N` overrides 8501.
Scout.command       macOS double-click wrapper → ./start.
conftest.py         sys.path shim for pytest.
.streamlit/config.toml   Headless config for the UI.
```

Future-hook stubs that are intentionally unbuilt: `linkedin_src.py`, star-velocity
diffing in `github_src.py`, `--watch` in `cli.run`.

---

## 4. Data flow (one `scout run`)

```
seeds.yaml + thesis.target_stages
      │
      ▼   (thesis.active_search_categories / active_discovery_sources decide what runs)
TwscrapeSource.fetch_accounts ──┐
GitHubSource.discover ──────────┤─►  merge (by handle)  ─►  _enrich_accounts
HackerNewsSource.discover ──────┘                              │  (store history:
                                                               │   bio_change, recent_followed_by)
      ▼
run_heuristics(account, tweets, thesis)  →  9 Signals + disqualified?
      ▼   (gate: ≥1 signal hit; ranked by pre-score, capped at LLM_MAX_CANDIDATES)
classify(candidates, thesis, settings, store)  →  {handle: LLMVerdict}
      (cache-first via llm_verdicts fingerprint; batches run concurrently)
      ▼
score_leads(leads, thesis)               →  score 0–100, ranked   (uses score_breakdown)
      ▼
store.save_leads + write_csv + write_markdown + print_top_table
```

The UI then reads `store.load_latest_leads()` for **Pick** (triage → longlist →
shortlist / pass, persisted in the `pipeline` table as statuses `longlisted` /
`shortlisted`…`won` / `passed`) and **Win** (stage/notes/outreach/memo).

---

## 5. The scoring model (know this cold)

`score_breakdown(lead, thesis)` in `score.py` is the **single source of truth**;
`score_leads` just takes its last value, and the UI renders every step. The math:

1. Components (each 0–100, absent when unevidenced):
   `signals = 100 × Σ(value_i × weight_i) / Σ(all thesis weights)`;
   `quality = 100 × Σ(dim_i × quality_weight_i) / Σ(weights of PRESENT dims)`
   (dims from `llm.quality` — the evidence-backed rubric, keys in
   `config.QUALITY_DIMENSIONS`, weights in `thesis.quality_weights`);
   `fit = 100 × llm.thesis_fit`.
2. `base = Σ(score_weight_c × component_c) / Σ(weights of PRESENT components)`
   — the 45/35/20 blend (`signal_params.score_weight_quality/_fit/_signals`),
   renormalized so a lead is never punished for a component nobody could
   evidence. No verdict → signals only (demo unchanged).
3. `× llm.confidence` when a verdict is attached.
4. `× 0.2` when `llm.is_founder` is false (kills corporate/commentator accounts).
5. `× signal_params.stage_mismatch_multiplier` (0.5) when `llm.stage` ∉ `target_stages`.
6. `× ((1 − w) + w × llm.value_add_fit)` (w = `signal_params.value_add_weight`,
   default **0** — opt-in). The value-add dimension is otherwise informational.
7. `× signal_params.ungrounded_multiplier` (0.6) when the product claim never
   traced to evidence: audit says "unverifiable", or the lead was never
   audited AND `llm.grounding` ∈ {None, "none", "bio"}. Audit-confirmed/
   corrected leads and evidence-grounded leads are exempt. NOTE: verdicts
   below MIN_CONFIDENCE are no longer dropped — they attach and sink via the
   confidence multiplier (dropping them used to RESTORE the full heuristic
   score, rewarding speculation over honest unknowns).

The quality rubric is scored through a **customer_type lens** (b2b / b2c /
b2b2c / mixed): B2B traction = logos/pilots/pipeline/SOC2, B2C = users/
growth/retention/virality. `team` is the one dimension where founder
background counts (as cited team evidence — never product evidence);
`investors` needs NAMED investors (watchlist follows cap it at 0.3).

The 9 signals (heuristics.py). Three read enrichment fields set by the pipeline
from store history, not by adapters — `recent_followed_by`, `bio_changed`,
`github_repo`; `source_corroboration` reads `Account.sources`, filled by
`cli._merge_accounts` and the twscrape adapter's internal dedupe:

| signal | fires on |
|---|---|
| bio_intent | thesis keyword in bio |
| departure_signal | target_bios marker in bio (literal substring) |
| bio_change | intent language *newly appeared* in bio vs prior snapshot |
| smart_money_follow | watchlist follows this account (any age; saturates at 3) |
| smart_money_convergence | watchlist follows that are NEW this window (2+ = full) |
| launch_traction | recent launch-y tweet with engagement/followers > floor |
| builder_evidence | github/personal-site link in bio |
| github_evidence | discovered via a recent starred repo (source=="github") |
| source_corroboration | 2+ distinct discovery strategies surfaced the account (full credit at 3) |

`signal_params` (traction floor/saturation/window, convergence threshold, stage
multiplier) are UI-editable, read by the heuristics — do not re-hardcode them.

---

## 6. Store tables (`~/.scout/scout.db`)

| table | purpose |
|---|---|
| accounts, tweets | fetch cache (incremental runs) |
| leads | saved scored runs (run_id keyed) |
| runs | run provenance: source + strategy_hash (thesis+seeds fingerprint) + config snapshot — groups runs into "strategies" |
| searches | per-query TTL cache (xapi mode: repeat runs free) |
| follow_edges, follow_meta | investor follow-graph snapshots + per-watcher baseline |
| bio_snapshots | bio history for bio_change detection |
| unlinked_leads | github/hn founders with no X handle (manual lookup) |
| pipeline | deal-flow state: status, notes, outreach, channel, brief |
| llm_verdicts | **verdict cache** — Claude verdict per handle, keyed by an input fingerprint (bio + tweets + rendered prompt + model + website URL + site-text hash + github + pinned); TTL `VERDICT_TTL_DAYS`. The audit pass re-records corrected verdicts under the same fingerprint |
| websites | **site cache** — extracted company-site text per normalized root URL; TTL `WEBSITE_TTL_DAYS` (7d), failures expire after 1 day |
| xapi_usage | **budget ledger** — every paid call, cumulative spend |
| scan, scan_history | live run status (drives the UI cockpit) + completed-scan phase timings (drives time estimates) |

DB path defaults to `~/.scout/scout.db` (not cwd) so the budget guard can't be
defeated by running from another directory; `DB_PATH` overrides. Handle lookups
are `COLLATE NOCASE`. `set_pipeline` is read-merge-write (partial updates don't
clobber other fields; outreach/brief writes also stamp `outreach_at`/`brief_at`).

**The lead ledger** (`store.load_lead_ledger`) is the person-centric read path:
one window-function query returns each handle's latest Lead + movement metadata
(prev_score, first/last seen, is_new) as `LedgerEntry` models. Invariants: the
partition key is `lower(handle)` (leads pk is case-sensitive, everything else is
NOCASE); ordering is `(created_at, run_id)`, never run_id alone (rows in a run
share one created_at; `verify-` > `demo-` lexicographically); `demo-` runs are
excluded unless `include_demo`; `verify-` runs are always included (real
re-scores). The UI's Longlist / Shortlist / Memos pages always resolve leads
through the ledger so a triaged lead missing from the latest run never degrades. `scout export` and
`scout verify` deliberately keep latest-run semantics (verify is paid — never
silently widen its input set to all-time).

---

## 7. Invariants you must not break

- **X API budget guard (highest priority).** The recruiter's token has a hard
  $25 budget; testing uses a personal key capped at $5. Every paid call in
  `xapi_src.py` must: (1) pre-check worst-case cost vs `XAPI_SPEND_CAP_USD` +
  cumulative `store.xapi_spend_usd()` BEFORE the request and raise
  `BudgetExceededError` if it'd exceed; (2) record actual usage AFTER a 2xx body.
  **No pagination. No retry that re-issues a request whose body was already
  received** (retries restricted to connect errors / 429 / 5xx). Free adapters
  (twscrape/github/hn) never touch the ledger. Corollary: paid adapters keep
  `parallel_safe = False` — tweet fetches fan out ONLY for free adapters, so
  concurrent calls can never race past the budget pre-check.
- **Pydantic models are the only cross-module data structures** (models.py).
- **Adapters only fetch.** Enrichment fields (`recent_followed_by`, `bio_changed`,
  `github_repo`) are set by `cli._enrich_accounts` from store history, never by
  adapters directly.
- **All network calls wrapped in tenacity** (3 attempts, jittered backoff),
  with the retry-safety rule above for paid calls.
- **Nothing hardcoded that belongs in thesis.yaml/seeds.yaml.** Signal *mechanics*
  (regexes for launch language, github detection) live in code; *targeting*
  (keywords, orgs, stages, weights, params, prompt) lives in the yaml.
- **Tests never make live network calls.** Adapter tests hit pure parser
  functions on fixture JSON.

---

## 8. External facts (verified; don't re-research)

- **X API v2 is pay-per-use since Feb 2026**, no free tier: ~$0.005/post read,
  ~$0.010/user read, 24h billing dedup. `/2/users/search` (bio search) returned
  **403 on pay-per-use** (confirmed via `scout probe`) → bio search is
  twscrape-only, permanently. `min_faves:`/engagement operators are web-only
  (twscrape); the paid API rejects them, so `xapi_src` strips them.
- **X exposes no follow timestamps** → the follow-graph "recent follows" signal
  is done by snapshot-diffing (first_seen proxy), cold-starting on run 2.
- **LinkedIn automation is dead** (Proxycurl sued & shut down 2025) — do not add
  a LinkedIn scraper.
- **Claude model:** `claude-sonnet-4-6` is the configured default (`CLAUDE_MODEL`),
  a real current model ID. Don't "correct" it.

---

## 9. How to extend (common changes)

- **Add a signal:** write `_my_signal(account, thesis) -> Signal` in
  heuristics.py, add it to the list in `run_heuristics`, add a default weight in
  thesis.yaml, add its `SIGNAL_HELP` entry in ui.py. If it needs data an adapter
  doesn't provide, add a field to `Account` and populate it in
  `cli._enrich_accounts`.
- **Add a discovery source:** subclass `DiscoverySource` (ingest/base.py),
  implement `discover(seeds, thesis) -> (accounts, unlinked_leads)`, register it
  in `cli._discovery_sources`, and add it to the relevant stage in
  `config.STAGE_DISCOVERY_SOURCES`. Keep pure parsing in module functions for
  testability (see github_src/hn_src).
- **Add a CLI command:** `@app.command()` in cli.py; wrap top-level in
  `try/except (BudgetExceededError, RuntimeError, Exception)` → clean message +
  `typer.Exit(1)`, never a traceback.
- **Change scoring:** edit `score_breakdown` only — everything else (score_leads,
  the UI's step display) flows from it.
- **Add a stage behavior:** edit the `STAGE_SEARCH_CATEGORIES` /
  `STAGE_DISCOVERY_SOURCES` / `STAGE_BIO_GRAPH` maps in config.py.

---

## 10. Current state / open threads

- Fully working: demo, source (all strategies), the whole scored pipeline, the
  Thesis/Startups/Longlist/Shortlist/Memos/Settings UI (Headline design
  language, lead cards),
  strategy agent (`scout strategy` + Thesis tab, validated live), investment
  memos (multi-section, editable, PDF export — validated live), manual score
  overrides, verdict cache, paid probe + verify (validated live, ~$0.83 spent).
- v4 additions (all validated live): person-centric lead ledger with score
  deltas + strategy grouping (runs table), paid-run + precision-pass cost
  confirmation gates in the UI, watchlist handle validation, card-face triage
  with expander state preservation (see the stateless-expander note in ui.py),
  triage insights + AI weight suggestions (insights.py + suggest_weights),
  pipeline CSV export, agent timeouts, staleness nudge.
- **Waiting on the user:** X account cookies (`TW_COOKIES`) to activate the X
  discovery legs (query bank, bio search, follow-graph); replacing the suggested
  default `watchlist` in seeds.yaml with Headline's own investors.
- `thesis.target_stages` is currently `[launched]` (the user is sourcing
  just-launched startups).
- Two secrets were pasted into chat historically (Anthropic + X bearer) — the
  user plans to regenerate them; treat any in-repo key as disposable.
