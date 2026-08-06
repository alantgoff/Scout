# AGENTS.md — orientation for AI agents working on `scout`

Read this first. It's the fast map: what the project is, how to run it, where
everything lives, the data flow, the invariants you must not break, and the
non-obvious gotchas that will otherwise waste your time. (`README.md` is the
user-facing version; this file is denser and aimed at contributors.)

**If you are short on time, read §7 (invariants), §11 (why the backtest's
statistical guards exist) and §12 (traps in this stack).** Those three are
where a well-intentioned change does real damage: §7 is how two members'
edits stay intact, §11 is why several "over-cautious" statistics must not be
simplified, and §12 is the library behaviour that has already cost hours.

---

## 1. What this is

`scout` sources **launched startups** (primary) and pre-launch founders-to-be
(secondary "watch" track) from Twitter/X, GitHub, and HN for a VC analyst at
Headline. The discovery unit is an X account; the *product* unit is a STARTUP —
the classifier extracts `company_name`/`company_url` and `scout/companies.py`
resolves every founder-like lead to a startup identity (real company name, or
a synthesized "Ada Lin's stealth startup" placeholder when unnamed) and folds
founder + company accounts into one entry, in every view and the report. It's
a Python 3.12 package with a **Typer CLI**, a **Streamlit UI** (Headline design
language, funnel-ordered: Thesis · Startups (Feed / Database) · Longlist ·
Shortlist · Memos · Activity · Evidence · Automation · Settings; session-state
nav, so buttons route across pages) and a **background worker**, managed by
**uv**. A thesis drives all targeting; nothing is hardcoded.

**Scout is a multi-member tool.** It runs as one shared instance for a firm —
Google sign-in, per-member votes and comments, an activity feed, a job worker
that sources on a schedule, and per-member theses over shared data. Everything
in §7's multiplayer invariants exists to keep two people editing the same
database from destroying each other's work. If you are tempted to simplify
something back to single-user, read that section first.

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
agent** (full multi-section memo — TL;DR / overview / product / tech /
competitive table / market sizing / acquisition dynamics / VERDICT
recommendation — at three depths: quick = dossier only, standard = + labeled
multi-page site crawl (web.fetch_site_bundle/bundle_text), deep = + live
Anthropic server-side web_search/web_fetch research, streamed with pause_turn
continuation, an on_event narration callback, and harvested citation Sources;
stored in the pipeline table with brief_meta_json, editable in the UI,
exportable as Markdown or PDF via `export.memo_pdf_bytes`). Hardened:
per-request transient-error retries with narration-counter rollback
(timeouts fail fast), continuations re-declare tools with only the REMAINING
search/fetch budget, sectionless/garbage output is rejected (never stored as
a memo), meta flags truncated/exhausted/missing_sections drive a UI warning,
prompts treat fetched pages as evidence-never-instructions, and the UI never
overwrites an existing memo with a fallback skeleton or a crashed run. Scoring is
overridable: `score_overrides` (store) + `score.apply_override` layer the
investor's manual quality/fit/pinned-score numbers into the same math at load
time.

---

## 2. Run it / test it — always via `./scout-cli` or `uv run`

```bash
uv run pytest -q                 # ~445 tests, ~25s, no network (incl. AppTest UI smoke tests)
./scout-cli demo                 # $0 offline end-to-end run on sample founders — best smoke test
./scout-cli source --strategy github,hn   # free live discovery, no scoring
./scout-cli ui                   # Streamlit workspace on :8501
./start                          # user-facing launcher: sync → seed-if-empty → serve → open browser

# multiplayer / background
./scout-cli migrate --owner you@firm.com  # adopt a single-user DB (idempotent)
./scout-cli worker --bootstrap --once     # create default schedules, drain, exit
./scout-cli worker                        # the loop (systemd in deploy/)
./scout-cli jobs                          # queue state;  --enqueue / --cancel
./scout-cli schedule --list               # recurring work;  --add / --enable / --delete
./scout-cli digest --window daily         # post the Slack digest now
./scout-cli memo <handle> --depth deep    # headless memo (what the worker shells to)
./scout-cli hindsight --outcomes outcomes.yaml --sweep 3 --suggest-weights
```

**Signing in locally.** The UI expects an authenticated member. Set
`SCOUT_DEV_USER=you@firm.com` to bypass Google sign-in in development; without
it and without OAuth configured, the actor falls back to `local@scout`.

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
  cli.py            Typer app — ALL orchestration. Commands: run, source,
                    inspect, verify, reclassify, probe, demo, export, budget,
                    strategy, thesis, publish, ui + the v9 additions: migrate,
                    worker, jobs, schedule, digest, memo, hindsight. Pipeline
                    helpers: _run_pipeline, _enrich_accounts, _run_discovery,
                    _merge_accounts (fills Account.sources), _fetch_tweets
                    (parallel for free adapters), _resolve_thesis_or_exit
                    (explicit --thesis-id → workspace default → file).
  config.py         Pydantic Settings (.env) + Thesis/Seeds/SignalParams (yaml)
                    + save_thesis/save_seeds (shared by CLI + UI).
                    STAGE_* maps: stage → search categories / discovery sources.
  models.py         The ONLY data structures crossing module boundaries:
                    Account, Tweet, Signal, LLMVerdict, Lead, UnlinkedLead.
  store.py          SQLite (sqlite-utils) — cache, dedupe, TTL, budget ledger,
                    follow-edge/bio snapshots, unlinked leads, deal-flow pipeline,
                    llm_verdicts cache, score_overrides, and the user-owned
                    startup data layer: startup_columns (schema: select/
                    multiselect/text/number/checkbox, options, ai_fill flag,
                    seed-once ensure_default_columns) + startup_attrs
                    (values_json merge store).
  dbfields.py       Pure grid helpers for the Database CRM: slugify_key,
                    canon, editor_changes (data_editor diffing), attr_display.
                    Kept out of ui.py so they unit-test without Streamlit.
  rubric.py         Pure scorecard registry: the B2B "Enterprise readiness" and
                    B2C "Consumer readiness" rubrics (sections → 1–3 criteria
                    with purposes, anchors, sub-weights), rubric_for routing,
                    band_for/BAND_LABELS, and prompt_block() (renders both
                    rubrics into the classifier prompt via {scorecard}).
  score.py          Pure scoring. score_leads + score_breakdown (THE score math,
                    single source of truth; UI renders its steps) +
                    scorecard_score (criteria → sections → 0–100 + band) +
                    company_quality (scorecard-vs-legacy dispatch) + apply_override
                    (manual section/fit/pinned-score overrides re-entering that math).
  export.py         CSV + Markdown writers, rich terminal table, memo_pdf_bytes
                    (markdown-pdf/PyMuPDF investment-memo PDF), override-aware
                    pipeline_rows(store, thesis).
  outreach.py       Claude-drafted first-touch outreach (Memos page); template fallback.
  agents.py         Strategy agent (generate_strategy/parse_strategy/apply_strategy),
                    research-brief agent (research_brief), investment-memo agent
                    (investment_memo + MEMO_SECTIONS; dossier = brief context +
                    website text + tweets + notes; offline skeleton fallback),
                    weight-tuning agent (suggest_weights/parse_weight_proposal),
                    watchlist validation (validate_watchlist), categorization agent
                    (categorize_startups + parse_categorization — batched,
                    strictly on-list, ai_fill-aware via the UI). Parse/apply
                    helpers are pure. All Claude clients carry explicit timeouts
                    (STRATEGY/BRIEF/WEIGHTS/CATEGORIZE_TIMEOUT_S + the per-depth
                    MEMO_TIMEOUTS_S map); timeouts are excluded from retry —
                    fail fast in the UI.
  insights.py       Pure triage analytics: triage_stats/stats_prompt contrast
                    shortlisted-vs-passed leads per signal/sector/stage/fit;
                    feeds the UI insights panel and the weight-tuning agent.
                    Plus actor_stats (the same contrast over ONE member's
                    votes — status is shared state, a vote is a named
                    judgment) and model_disagreements (where a partner and
                    the model parted ways, both directions).
  collab.py         Pure collaboration logic, no DB: STANCES (the four vote
                    weights — `pass` is deliberately −2, mirroring
                    strong_yes, so one holdout vs one champion reads as a
                    split rather than a mild positive), vote_summary,
                    contested_sort_key, disagreements (the partner-meeting
                    agenda), parse_mentions.
  theses.py         Which thesis a caller works against. Precedence: explicit
                    id → the member's own pointer (users.active_thesis_id) →
                    the workspace default (theses.is_active) → thesis.yaml.
                    The DB holds the config; YAML is a readable export.
                    sync_files_to_db adopts file-only installs, idempotently.
  memos.py          Headless memo generation, lifted out of ui.py so the
                    worker can write memos overnight. The UI keeps its live
                    deep-research narration and shares everything below it.
  jobs.py           Pure background-work logic: job kinds, deterministic
                    backoff, and ScheduleSpec/next_occurrence — deliberately
                    NOT cron ("weekdays at 07:00 Europe/London" is what
                    anyone wants, and it survives DST because the arithmetic
                    happens in the schedule's own zone).
  worker.py         The job loop: reap stale leases → materialize due
                    schedules → claim one job → run it. One job at a time on
                    purpose (the X budget guard, the SQLite writer and the
                    scraper rate limits each get exactly one contender).
                    Long jobs run as SUBPROCESSES so a scraper segfault kills
                    a child, not the scheduler.
  notify.py         Slack: mention/assignment pings (inline) and digests
                    (worker). Pure builders (digest_data → digest_blocks) so
                    message shape tests without network; post_slack swallows
                    its own failures — a Slack outage must never break a
                    triage click.
  hindsight.py      The backtest. Reconstructs public evidence as it stood on
                    a past date (HN Algolia archive + GitHub starring
                    TIMESTAMPS, never today's counts), scores it with the
                    SAME pipeline production uses, and compares against
                    controls. Read §11 before touching the methodology.
  signal_eval.py    Which individual signals predict outcomes: AUC with
                    bootstrap CIs, permutation p-values, Benjamini-Hochberg
                    correction, minimum detectable effect, marginal
                    contribution, correlation-based double-counting
                    detection, and shrunk weight suggestions. All pure,
                    seeded, heavily tested. Also read §11.
  companies.py      Pure startup identity + grouping: startup_identity resolves
                    every founder-like lead to (name, synthesized) — real
                    company, or "Ada Lin's stealth startup" when unnamed;
                    founder_like gates it (verdict, else founder-evidence
                    signals); company_key/group_by_company fold accounts
                    sharing a company into one entry (primary = highest score).
  demo_data.py      8 synthetic sample founders for `scout demo` (obviously fake handles).
  publish.py        Phone digest: renders the ledger to docs/index.html (mobile
                    page for GitHub Pages in the separate public DIGEST_REPO
                    checkout); `scout publish [--push]`.
  ui.py             Streamlit app, NINE pages: Thesis / Startups (Feed +
                    Database) / Longlist / Shortlist / Memos / Activity /
                    Evidence / Automation / Settings. Session-state nav
                    (nav_target routes across pages — the card Memo button lands
                    on Memos and auto-generates); Slack deep links arrive as
                    ?s=<handle>&p=<page> and are translated into that same
                    mechanism once, then cleared. Startup database with dossier
                    row-select; per-card Q/F/S score breakout + Adjust-scoring
                    popover; Memos page with in-place editing, version history
                    and .md/.pdf export. Headline design language. ~5200 lines.
                    Heavy reads (latest leads, ledger, pipeline, overrides,
                    attrs, stale handles, votes, comment counts, users) load
                    through st.cache_data keyed on the DB file stamp
                    (_db_stamp) — Streamlit reruns the whole script per click,
                    and re-parsing every stored lead's JSON dominated latency;
                    provenance backfills are session-gated
                    (st.session_state["thesis_synced"]). The backtest's
                    per-signal statistics are cached separately
                    (_signal_evaluation) because bootstrap + permutation cost
                    ~480ms and a stored backtest is immutable.
  ingest/
    base.py         SourceAdapter ABC (X sources) + DiscoverySource ABC (github/hn).
    twscrape_src.py Primary free X adapter: query bank, bio search, list members,
                    follow-graph snapshotting.
    xapi_src.py     Paid X API v2 adapter — BUDGET-GUARDED. BudgetExceededError.
    github_src.py   GitHub discovery (repo search → owner → X-handle bridge).
    hn_src.py       Hacker News (Algolia) discovery.
    arxiv_src.py    arXiv discovery — the EARLIEST founder signal available.
                    A researcher publishes, keeps publishing, then the
                    affiliation changes and a company follows months later.
                    Free, unauthenticated, and (unlike X) a complete
                    timestamped archive, so anything sourced here is
                    admissible in the hindsight backtest. Pure parsers over
                    the Atom feed; authors that cannot be bridged to an X
                    handle become UnlinkedLeads, which is most of them and
                    is the correct outcome rather than a guess.
    linkedin_src.py Stub (NotImplementedError) — LinkedIn automation is a dead end.
  signals/
    heuristics.py   10 deterministic signals + run_heuristics + intent_appeared.
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
                    fan-out, cache-first, negative caching), and the memo
                    crawl — bundle_urls/bundle_text (pure) + fetch_site_bundle
                    (root + about/product/pricing/… + extra candidate roots).
tests/              pytest, no network — test_agents (incl. mocked memo
                    stream/pause_turn/retry loop + categorization), test_score
                    (incl. apply_override), test_store (incl. columns/attrs/
                    overrides), test_dbfields, test_memo_export (PDF bytes +
                    attr columns), test_ui_smoke (AppTest: nav, memo flow,
                    overwrite guard, database modes), test_web (incl. site
                    bundle), test_heuristics, test_grounding, test_sourcing_v2,
                    test_companies, test_insights, test_value_add, test_publish,
                    test_cli_helpers, test_collab (stance maths, the events
                    atomicity invariant, memo version recovery, notes
                    conflict detection, single-user migration), test_jobs
                    (schedule arithmetic incl. DST, queue claim/lease/retry,
                    digests), test_theses (per-member resolution),
                    test_hindsight (point-in-time discipline, base rates,
                    fairness), test_signal_eval (the statistics, against
                    known answers).
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
diffing in `github_src.py`. (`--watch` in `cli.run` is no longer one of them —
it was removed rather than left as a lie, and now points at `scout worker`.)

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
   `quality = the readiness SCORECARD` (`score.scorecard_score`): criteria
   from `llm.scorecard` (key → 1–3, evidence-cited, omit-don't-guess; the
   rubric registry lives in `scout/rubric.py`) → per-section weighted average
   over PRESENT criteria mapped `(avg−1)/2×100` → total renormalized over
   present sections with weights from `thesis.scorecard_weights` (per rubric).
   `customer_type` routes the rubric: b2c → consumer scorecard; b2b / b2b2c /
   mixed / None → enterprise. The 0–100 total carries a band
   (`rubric.band_for`: ≥80 Strong · 60–79 Promising · <60 Too early).
   LEGACY verdicts (pre-scorecard cache, flat `llm.quality` dims over
   `config.QUALITY_DIMENSIONS` / `thesis.quality_weights`) still score via
   `score.quality_score`; `score.company_quality` is the dispatch point —
   use IT, never call either path directly from UI/export code.
   `fit = 100 × llm.thesis_fit`.
2. `base = Σ(score_weight_c × component_c) / Σ(weights of PRESENT components)`
   — the blend (`signal_params.score_weight_quality/_fit/_signals`, currently
   35/50/15), renormalized so a lead is never punished for a component nobody
   could evidence. No verdict → signals only (demo unchanged). Fit outweighs
   quality deliberately: a well-built off-thesis company used to outrank an
   on-thesis one, which is backwards for thesis-driven sourcing.
3. `× llm.confidence` when a verdict is attached. **Known bias, deliberately
   not "fixed":** confidence measures how legible an account was, and stealth
   companies are inherently less legible — in one run they averaged 0.57
   against 0.83–0.93 for launched/scaling while carrying the HIGHEST mean
   thesis fit. Flooring it was tried and reverted: it lifts every lead whose
   product claim never traced to evidence, which is exactly the Raindrop
   failure `test_stealth_pedigree_founder_sinks_despite_team_score` guards.
   A real fix must separate "stealthy" from "unevidenced", not blur both.
4. `× 0.2` when `llm.is_founder` is false (kills corporate/commentator accounts).
5. `× signal_params.stage_mismatch_multiplier` (0.3) when `llm.stage` ∉
   `target_stages` — an off-stage company must be ~3× better to rank
   alongside an on-stage one. Demoted, not deleted: it still reads as market
   signal.
6. `× ((1 − w) + w × llm.value_add_fit)` (w = `signal_params.value_add_weight`,
   default **0** — opt-in). The value-add dimension is otherwise informational.
7. `× signal_params.ungrounded_multiplier` (0.6) when the product claim never
   traced to evidence: audit says "unverifiable", or the lead was never
   audited AND `llm.grounding` ∈ {None, "none", "bio"}. Audit-confirmed/
   corrected leads and evidence-grounded leads are exempt. NOTE: verdicts
   below MIN_CONFIDENCE are no longer dropped — they attach and sink via the
   confidence multiplier (dropping them used to RESTORE the full heuristic
   score, rewarding speculation over honest unknowns).

The scorecard rubrics (`scout/rubric.py`, derived from the V5 "Enterprise
Readiness Evaluation Framework" spreadsheet, refined for sourcing-time
evidence): **B2B "Enterprise readiness"** — Founders & cap table 18 / Market
23 / Technology 23 / Product 18 / Traction & business model 18; **B2C
"Consumer readiness"** — Founders & cap table 18 / Market 23 / Product &
experience 23 / Growth & engagement 18 / Monetization & business model 18.
Criterion sub-weights are code-owned in the registry; section weights are
UI-editable (`thesis.scorecard_weights`). The founders & cap table criteria
are the ONLY place founder background counts (cited concretely — never
product evidence); the classifier prompt renders both rubrics from
`rubric.prompt_block()` via the `{scorecard}` placeholder, so prompt and
math can never drift.

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

**Manual overrides** layer on top at load time, not at run time:
`store.score_overrides` rows (per-handle scorecard sections 0–100 / legacy
quality dims / thesis fit / pinned score + note) are applied by
`score.apply_override` in ui.py and in `export.pipeline_rows(store, thesis)`
— adjusted sections land in `verdict.scorecard_manual` and replace the
computed section scores inside the rollup (legacy dim overrides only touch
pre-scorecard verdicts); fit is written into the verdict (with a "manual"
reason); everything re-enters `score_breakdown`; a pinned score wins
outright and shows as a final "manual score override" step. Stored leads
in the DB keep the model's numbers; overrides live only in their own table.

---

## 6. Store tables (`~/.scout/scout.db`)

| table | purpose |
|---|---|
| accounts, tweets | fetch cache (incremental runs) |
| leads | saved scored runs (run_id keyed) |
| runs | run provenance: source + strategy_hash + **thesis_id** (durable identity) + **thesis_version** (the tuning fingerprint) + config snapshot. The id/version split is the whole point: `strategy_fingerprint` hashes the entire config, so every weight tweak minted a new "strategy" and one thesis fragmented across all of them (the live DB had 7 hashes for 4 theses) |
| theses | the thesis registry: id, name, statement, current_version, is_active, archived_at. `backfill_thesis_ids()` recovers identity for pre-v8 runs by grouping on thesis_statement — the one field that stays put while weights churn |
| thesis_versions | one row per distinct tuning of a thesis, so "v3" is nameable and staleness is computable (`stale_handles`) |
| llm_verdict_history | **prior verdicts**, appended by `record_verdict` BEFORE it overwrites. `llm_verdicts` is one row per handle, so rescoring under a new thesis would otherwise destroy the old judgment — and "0.20 under Edge AI, 0.75 under Novel Architectures" is the most useful thing a thesis change produces |
| searches | per-query TTL cache (xapi mode: repeat runs free) |
| follow_edges, follow_meta | investor follow-graph snapshots + per-watcher baseline |
| bio_snapshots | bio history for bio_change detection |
| unlinked_leads | github/hn founders with no X handle (manual lookup) |
| pipeline | deal-flow state: status, notes, outreach, channel, brief (the investment memo) + brief_at / brief_edited_at / brief_meta_json (depth, sources, searches, honesty flags) |
| score_overrides | the investor's manual scoring per handle (scorecard sections 0–100 / legacy quality dims / fit / pinned score + note) — applied at load by `score.apply_override` |
| startup_columns | the Database CRM's user-owned column schema: key, label, type (select/multiselect/text/number/checkbox), options_json, builtin, ai_fill, position. Seeded ONCE by `ensure_default_columns` (table existence suppresses re-seeding, so deleting a builtin sticks) |
| startup_attrs | per-startup values for those columns (values_json merge store; None deletes a key) |
| llm_verdicts | **verdict cache** — Claude verdict per handle, keyed by an input fingerprint (bio + tweets + rendered prompt + model + website URL + site-text hash + github + pinned); TTL `VERDICT_TTL_DAYS`. The audit pass re-records corrected verdicts under the same fingerprint |
| websites | **site cache** — extracted company-site text per normalized root URL; TTL `WEBSITE_TTL_DAYS` (7d), failures expire after 1 day |
| xapi_usage | **budget ledger** — every paid call, cumulative spend |
| scan, scan_history | live run status (drives the UI cockpit) + completed-scan phase timings (drives time estimates) |
| users | firm members: id (email), name, role (admin/member), slack_member_id, active_thesis_id, last_seen_at. Role is settable ONLY via `set_user_role`, never via the generic `update_user` — folding an authorization decision into a profile update is how privilege escalation happens |
| settings | firm-shared runtime knobs (spend cap, model, run sizes, Slack webhook, app base URL, digest threshold) that OUTRANK .env, so every session and worker run agrees. `apply_settings_overrides` overlays them onto a loaded Settings |
| events | **the activity spine.** Append-only, written INSIDE the same transaction as the change it describes (see §7). Powers the Activity feed, unread counts, and digests |
| read_cursors | per-member last-seen event id — how "3 new since you looked" is computed without marking your own actions unread |
| votes | one row per (handle, actor): stance + rationale. Separate from `pipeline.status` on purpose — status is the firm's shared state, a vote is a named person's judgment, and two partners must be able to disagree without overwriting each other |
| comments | threaded discussion per startup, soft-deleted, with @mentions that ping Slack |
| memo_versions | **every generation and edit of a memo.** `pipeline.brief` stays the current text so all existing read paths are untouched; this accumulates the history that makes regeneration safe. Before it existed, regenerating destroyed a human's edits with no way back |
| jobs | the work queue: kind, payload, status, attempts, lease_expires_at, worker_id. Claimed atomically under BEGIN IMMEDIATE |
| schedules | recurring work: kind + payload + ScheduleSpec + next_run_at. Materialized into jobs by the worker, at most one per schedule per pass |
| papers, paper_authors | arXiv record + **per-author affiliation snapshots**. The affiliation HISTORY is the asset: `departure_signal` infers a lab exit from self-reported bio language, while a change of institution across a publication record is the same event observed at the source and typically far earlier (`authors_who_moved`). An absent affiliation is "unknown", never "unaffiliated" — arXiv's field is optional and reading a gap as a move would manufacture departures out of metadata sparsity |
| backtests | stored hindsight runs (report JSON + headline metrics), so "has this got better as we tuned?" is a series rather than one screenshot |

**Two table-creation styles, and the rule for choosing.** Most tables are born
from their first `upsert(..., alter=True)` — sqlite-utils infers the columns,
which is fine for string or composite primary keys (`users`, `settings`,
`votes`, `read_cursors`, `pipeline`, …). Tables that need an **auto-assigning
INTEGER PRIMARY KEY**, or that are written by **two different code paths**, must
be declared up front in `_ensure_collab_tables` / `_ensure_job_tables`
(`events`, `comments`, `memo_versions`, `jobs`, `schedules`, `backtests`,
`theses`) — see §12 for the two failure modes that forces. Every read path
guards with `db[...].exists()`, so a fresh database is never a crash.

DB path defaults to `~/.scout/scout.db` (not cwd) so the budget guard can't be
defeated by running from another directory; `DB_PATH` overrides. Handle lookups
are `COLLATE NOCASE`. `set_pipeline` is read-merge-write (partial updates don't
clobber other fields; outreach/brief writes also stamp `outreach_at`/`brief_at`).

**Performance contract:** `Store.__init__` creates secondary indexes
(`_ensure_indexes`) for the per-handle hot paths — every composite pk here
indexes the wrong prefix for them (leads is run_id-first, follow_edges
watcher-first, tweets by tweet id). Batched forms exist for what the pipeline
used to do as per-account query loops: `last_scored_map` (the TTL skip),
`recent_watchers_map` (enrichment + the twscrape graph leg), `upsert_accounts`
(adapters persist once per leg, not once per sighting). Prefer these over
calling the per-handle forms in a loop; the per-handle forms remain for
single-lead paths (`inspect`, the UI detail pane).

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
- **Never assert a figure the system did not read.** The whole product rests
  on this. A `funding_stage` without `funding_evidence` is downgraded to
  `unknown` by a model validator in `models.py` — enforced in code, not
  merely asked for in a prompt, because asking was tried and 3 of 20 tagged
  rounds came back unevidenced (one recalled Perplexity as "Series C+" from
  a bio reading "Everything is Computer."). The memo prompt carries the same
  ban at EVERY depth, including the tiers that cannot research: the format
  demands a competitor funding column, so without the ban the model fills it
  from memory. If you add a field that states a number, add the evidence
  field and the validator alongside it.
- **Two different "stage" concepts, never conflate them.** `stage` is
  lifecycle (idea/stealth/launched/scaling — has it shipped). `funding_stage`
  is the cap table (seed/series_a/… — who priced it). A launched company can
  be bootstrapped or Series B.
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

### Multiplayer invariants (added when Scout became a two-partner tool)

- **Every multi-statement write goes through `store.write_tx()`**, which opens
  `BEGIN IMMEDIATE`. SQLite's default deferred transaction upgrades to a write
  lock mid-transaction, which is exactly how two people editing the same
  startup produce a lost update. WAL is on so readers never block the writer.
- **An event exists if and only if the change it describes committed.**
  `_append_event` is called INSIDE the same `write_tx` as the state write —
  never after it, never in a second transaction. A feed that can disagree with
  the data is worse than no feed. `test_collab.py` pins this by rolling back
  mid-transaction and asserting neither survived.
- **A vote is not a status.** Don't "simplify" by deriving one from the other.
  Status is the firm's shared funnel position (whoever moved it last); a vote
  is one member's named judgment. The Split badge, the contested sort and the
  partner-meeting agenda all exist because those are different facts.
- **The thesis is resolved per member** (`theses.resolve`), never from a
  global. Reintroducing a single active thesis would mean one partner
  exploring a new space silently re-aims the other's workspace and the
  scheduled run. Unattended work (worker, digests) uses the workspace default
  precisely because it belongs to the firm, not to whoever clicked Switch last.
- **Judgment rows carry an actor.** pipeline, votes, comments, overrides,
  attrs and memo versions all record who. `Store.actor` is bound at login (UI)
  or from `SCOUT_ACTOR` (subprocesses the UI and worker spawn).
- **The worker runs one job at a time.** Serial execution is what leaves the
  budget guard, the SQLite writer and the scraper rate limits each with a
  single contender. Adding concurrency here buys nothing and costs those
  invariants.
- **A lease is not a lock.** A claimed job holds a lease with a heartbeat; a
  worker that dies has its job requeued once the lease lapses. If you add long
  work, heartbeat it (`_Heartbeat`), or the reaper will double-execute a job
  that was progressing fine.

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
- **Change the scorecard rubrics:** edit the registry in `scout/rubric.py`
  only — criteria, purposes, anchors, sub-weights, and the prompt block all
  live there; `test_rubric.py` pins the invariants (weights sum, unique keys,
  3 anchors each). Changing it re-renders the classifier prompt, which
  invalidates the verdict cache — `scout reclassify` backfills. Section
  weights are data (`thesis.scorecard_weights`), not code.
- **Add a stage behavior:** edit the `STAGE_SEARCH_CATEGORIES` /
  `STAGE_DISCOVERY_SOURCES` / `STAGE_BIO_GRAPH` maps in config.py.
- **Add a database column type:** extend `dbfields.ATTR_TYPE_LABELS`, the
  per-type branches in ui.py's `_render_db_editor` (column_config) and
  `_render_column_manager`, and `dbfields.canon`/`attr_display` if the value
  shape is new. Curated seed lists live in `config.CURATED_*`.
- **Change the memo:** the section contract is `agents.MEMO_SECTIONS` + the
  `_MEMO_SYSTEM` prompt; depths/timeouts in `MEMO_DEPTHS`/`MEMO_TIMEOUTS_S`;
  research budgets in `MEMO_MAX_SEARCHES/_FETCHES/_CONTINUATIONS`. The UI's
  verdict chip greps `VERDICT: (PURSUE|TRACK|PASS)`; keep that line format.
  **Write memos through `store.set_memo`, never `set_pipeline(brief=...)`** —
  set_memo appends the immutable version that makes regeneration safe.
- **Add a page:** append to `PAGES` in ui.py and add an `if nav == "…":` block.
  Sub-views within a page are a sidebar `segmented_control`, NOT `st.tabs` —
  st.tabs renders every tab server-side, which once leaked the feed's sidebar
  controls onto the Database view. Route to it from elsewhere by setting
  `st.session_state["nav_target"]` then `st.rerun()`.
- **Add a job kind:** add the constant + label to `jobs.JOB_KINDS/JOB_LABELS`,
  write `handle_x(store, settings, job) -> dict` in worker.py, register it in
  `worker.HANDLERS`. Long or crash-prone work should shell out via `_run_cli`
  so a child dies instead of the scheduler; short in-process work (digests)
  should not. Raise on failure — the loop turns that into retry-or-fail.
- **Add an event verb:** emit it from the store method inside the same
  `write_tx` as the state change, then add its sentence to `_VERB_TEXT` in
  ui.py (that map is presentation; the store records what happened, the UI
  says it in English). Add it to a `verb_groups` filter if it belongs in one.
- **Add a firm-wide setting:** `store.set_setting`/`get_setting`, and if it
  should override a `.env` field add it to `Store.RUNTIME_SETTING_FIELDS` so
  `apply_settings_overrides` picks it up for every session and worker run.
- **Add a signal to the backtest:** nothing to do — `score_evidence` records
  every signal's normalized value per company, so a new heuristic appears in
  the per-signal table automatically. It will show near-zero coverage until
  the reconstructed evidence can actually trigger it, which is honest.
- **Change the backtest methodology:** read §11 first. Every guard there was
  added because the naive version produced a confident wrong number, and the
  tests pin the behaviour, not the implementation.

---

## 10. Current state / open threads

- Fully working: demo, source (all strategies), the whole scored pipeline, the
  Thesis/Startups/Longlist/Shortlist/Memos/Settings UI (Headline design
  language, lead cards), strategy agent (`scout strategy` + Thesis tab,
  validated live), investment memos v2 (three depths, live web research with
  cited sources, in-place editing, PDF export, hardened stream loop — deep
  runs validated live on Walden Robotics and Cosmic Labs), manual score
  overrides, the Database CRM (Browse/Edit modes, curated + custom columns,
  per-field filters, AI auto-categorize with ai_fill guard — validated live),
  verdict cache, paid probe + verify (validated live, ~$0.83 X spend).
- v4 additions (all validated live): person-centric lead ledger with score
  deltas + strategy grouping (runs table), paid-run + precision-pass cost
  confirmation gates in the UI, watchlist handle validation, card-face triage
  with expander state preservation (see the stateless-expander note in ui.py),
  triage insights + AI weight suggestions (insights.py + suggest_weights),
  pipeline CSV export, agent timeouts, staleness nudge.
- v5 additions: the readiness SCORECARD replaced the flat v7 quality rubric as
  the quality component — B2B enterprise + B2C consumer rubrics in
  `scout/rubric.py` (derived from the V5 scorecard spreadsheet), criteria 1–3
  → weighted sections → 0–100 + band, routed by customer_type; section-level
  manual overrides (`sections_json`); Band column in the CRM; scorecard in
  memos/CSV/digest. Legacy flat-quality verdicts still render and score via
  `company_quality` dispatch until `scout reclassify` re-scores them (the
  prompt change auto-invalidated the verdict cache). The dead
  `thesis_fit_weight` key in thesis.yaml is ignored by pydantic and drops on
  the next Settings save.
- v8 additions (all validated live on the 942-startup ledger):
  - **Thesis as a first-class object.** Identity (`thesis.id`/`name`, durable)
    split from version (`thesis_version`, the tuning fingerprint, which
    deliberately EXCLUDES id/name — hashing them marked all 602 startups
    stale the moment a thesis was named). `theses` / `thesis_versions` /
    `llm_verdict_history` tables, `scout thesis list|show|new|use|clone|
    archive`, `reclassify --stale-only`, a thesis library in `theses/`,
    provenance rendered in `_score_detail_html` (shared by the detail pane
    and the Database dossier, so they cannot drift).
  - **Funding stage** (`funding_stage` + amount/investors/evidence), evidence-
    gated by a model validator. 17 of 942 tagged, every one cited.
  - **Memo v3:** 12 sections (added Why now, Team, Traction, Deal terms,
    Risks), deep-by-default, research budget 8/10 → 14/16 spent on founders
    BEFORE competitors, and the unverified-figure ban at every depth. A live
    run produced 65 citations across 11 sources.
  - **`_rank_candidates` now classifies signal-less SEARCH leads.** The
    deterministic signals all describe a PERSON (bio_intent reads a personal
    bio, departure_signal wants an ex-employer), so the company-account query
    bank was feeding the pipeline leads that fire none of them — 220 of 645
    were dropped before Claude. One (@BiggerMax19, no bio, 0 followers)
    classified at fit 0.70. github/hn bulk discovery stays excluded.
- **Known gaps / next threads:**
  - The confidence-multiplier bias against stealth (see §5 step 3) is real and
    unfixed. It needs a fix that distinguishes "stealthy" from "unevidenced".
  - Per-thesis verdicts (one company scored under several theses at once, for
    routing to the right partner) are deliberately out of scope — it needs an
    `llm_verdicts` PK migration and multiplies Claude spend per thesis.
    `llm_verdict_history` is the stepping stone.
  - Memos now run ~3,200 words against a 1,100–1,700 target. Denser and
    better-sourced, not padded — but narrow the target rather than cutting
    sections if that matters.
- **Waiting on the user:** X account cookies (`TW_COOKIES`) to activate the X
  discovery legs (query bank, bio search, follow-graph); replacing the suggested
  default `watchlist` in seeds.yaml with Headline's own investors.
- `thesis.target_stages` is currently `[launched]` (the user is sourcing
  just-launched startups).
- Two secrets were pasted into chat historically (Anthropic + X bearer) — the
  user plans to regenerate them; treat any in-repo key as disposable.

---

- **v9 — Scout became a multi-member tool** (this is the largest change since
  v8; every item validated live):
  - **Multiplayer foundation.** WAL + `write_tx()` (BEGIN IMMEDIATE) so two
    people editing the same startup merge rather than clobber; Google sign-in
    with a `SCOUT_DEV_USER` bypass for local work; `users` table with roles and
    a domain allowlist; actor attribution on every judgment row; firm-shared
    settings in the DB that outrank `.env`; the `deploy/` kit (systemd, Caddy,
    litestream).
  - **Collaboration core.** The `events` spine (append-only, transactional with
    the change it describes), per-member votes with a Split badge and a
    partner-meeting agenda, comment threads with @mentions that ping Slack,
    memo versioning with restore and review/approval, an Activity feed with
    unread counts, and per-partner taste analytics.
  - **Background work.** A SQLite job queue with atomic claim, leases and
    heartbeats, retry with deterministic backoff, and dedupe; DST-correct
    schedules; Slack digests ordered by what should change a partner's next
    hour. `scout worker`, `jobs`, `schedule`, `digest`, `memo`.
  - **Per-member theses.** The DB is the source of truth for thesis config
    (`theses.config_json`); YAML is a readable export. Three pointers with
    precedence: explicit id → member → workspace default → file.
  - **Evidence.** The hindsight backtest and per-signal evaluation, with the
    statistical guards in §11, surfaced as a top-level Evidence page
    (Results / Signals / Over time).
  - `run --watch` is GONE rather than left as a stub; scheduling is the worker.
  - **Not yet built:** CRM write-back (Affinity/Attio — deliberately deferred
    until a firm names theirs), warm-intro paths from the `follow_edges` data
    already collected, funnel/source-attribution analytics, and a self-host
    (Docker Compose) bundle for firms that will not put dealflow on a
    third-party server.

---

## 11. The backtest's methodological commitments (read before editing)

`hindsight.py` and `signal_eval.py` contain guards that look like excessive
caution and are not. Each exists because the naive version produces a
confident, wrong number — the worst possible output for a tool whose entire
purpose is evidence. Do not "simplify" any of these without reading why:

- **Ties count as half a win** in AUC. Without it a scorer that rates
  everything identically scores 1.0 instead of 0.5.
- **Perfect separation does not use the bootstrap.** Every resample
  reproduces the same ordering, so the percentile interval collapses to
  `[1.00, 1.00]` — certainty from three companies a side. `auc_ci` falls
  back to a bound derived from the number of pairwise comparisons made.
- **A zero observed rate is not a zero rate.** `bounded_rates` applies the
  rule of three (no events in n trials bounds the rate at ~3/n) before any
  Bayes projection. Feeding a literal FPR of 0.0 in produced "100%
  precision, 100× lift" from eighteen controls, which is how this guard was
  found.
- **Precision is restated at production base rates.** The backtest pool is
  ~40% companies that raised; the real funnel is 1–2%. Unadjusted precision
  overstates production by an order of magnitude, and a reader will hear
  "three in four companies this flags will raise". Lift is the honest
  headline because it survives the base-rate shift.
- **p-values are corrected for multiple comparisons** (Benjamini-Hochberg).
  Testing a dozen signals at p<0.05 yields a false positive more often than
  not; significance uses the corrected q-value.
- **Under-powered evaluations withhold results entirely** rather than
  showing them with a caveat. A caveat under a number still leaves the
  number on the slide.
- **The minimum detectable effect is stated**, so "we found nothing" is
  distinguishable from "we could never have found anything with this much
  data" — which changes the action from re-tune to collect more.
- **Decay requires disjoint confidence intervals**, not merely a lower point
  estimate. Otherwise the tool invents a decay story from noise.
- **Weight suggestions are a shrunk heuristic, not a fitted model.** At
  fifteen outcomes a logistic regression yields coefficients whose intervals
  span the plausible range; presenting those as "learned weights" would be
  the most misleading thing this code could do.
- **Redundancy is detected by correlation, not leave-one-out.** With other
  noisy signals present, dropping one of two duplicates lets the noise take
  more weight, so the duplicate looks uniquely valuable. Correlation answers
  it outright.

**Two limitations are structural and cannot be fixed in code.** Say so
rather than engineering around them:

1. **The thesis is not frozen at the cutoff.** Evidence is rewound; the
   weights and prompt are today's, written by people who already know who
   raised. The supportable claim is "would today's thesis have ranked these
   highly given only past evidence", NOT "would Scout have caught them at
   the time". Only forward measurement against a frozen thesis settles the
   second.
2. **Outcomes and controls are chosen by hand**, and memorable companies are
   the ones with loud public footprints. `evidence_symmetry` detects the
   worst version of this (comparing companies with GitHub orgs against
   companies without), but it cannot fix selection itself.

`default_limitations()` renders all of this into every report. If you add a
capability that could mislead, add its limitation there in the same commit.

---

## 12. Traps in this stack (each cost real debugging time)

**sqlite-utils**

- `db["table"]` builds a **new `Table` object on every access**. So
  `db["t"].insert(row)` followed by `db["t"].last_pk` reads a different
  object and returns `None`. Chain it: `db["t"].insert(row).last_pk`.
- `db.execute(...)` returns **raw tuples**, not dicts — `dict(row)` raises
  "cannot convert dictionary update sequence element #0". Use
  `table.rows_where(...)` when you need dicts; keep `db.execute` for
  aggregates read by index.
- A table whose rows are ordered by an **autoincrement `id` must be created
  explicitly** (`table.create({...}, pk="id", if_not_exists=True)`).
  sqlite-utils infers columns from the first inserted row, which never
  carries an id, so `order_by="id desc"` then fails with "no such column:
  id". This bit events, comments, memo_versions, jobs, schedules and
  backtests — all of them are created up front in `_ensure_*_tables`.
- A table written by **two different paths** needs its full shape declared
  up front too, or whichever writes first leaves the other's columns
  missing (`theses` is written by per-run metadata AND per-save config;
  `is_active` was read before either had run).

**Streamlit**

- The whole script re-executes on **every interaction**. Anything expensive
  must go through `st.cache_data` keyed on something that actually changes
  (`_db_stamp()` for store reads, the immutable report JSON for backtest
  statistics). A 480ms computation called twice per render is a full second
  of latency per click.
- `st.column_config.NumberColumn(format="%.0f%%")` expects the value
  **already scaled to 0–100**, not a 0–1 fraction.
- Widget keys must be unique across the whole app; in a 5,000-line module
  prefix them by surface (`feeddet_`, `ev_`, `sched_`).
- **Name collisions are silent.** `_initials(name, handle)` (startup avatar)
  and a new `_initials(actor)` (stance badge) simply shadowed each other at
  import. Grep before naming a helper here.

**AppTest (tests/test_ui_smoke.py)**

- `at.markdown` contains expander **contents but not expander labels**.
  Assert on text inside the expander, not its title.
- Buttons are found by `.label` or `.key`; a page that renders behind a rail
  switch needs its `session_state` view key set before `at.run()`.
- Setting `at.session_state["nav"]` is how you route to a page — the nav is
  session-state driven precisely so tests (and buttons) can route.
