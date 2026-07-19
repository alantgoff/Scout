# AGENTS.md — orientation for AI agents working on `scout`

Read this first. It's the fast map: what the project is, how to run it, where
everything lives, the data flow, the invariants you must not break, and the
non-obvious gotchas that will otherwise waste your time. (`README.md` is the
user-facing version; this file is denser and aimed at contributors.)

---

## 1. What this is

`scout` sources **launched startups** (primary) and pre-launch founders-to-be
(secondary "watch" track) from Twitter/X, GitHub, and HN for a VC analyst at
Headline. The discovery unit is an X account; the *product* unit is a startup —
the classifier extracts `company_name`/`company_url` and `scout/companies.py`
folds founder + company accounts into one entry (Startups track, report). It's
a Python 3.12 package with a **Typer CLI** and a **Streamlit UI** (Apple design
language: Leads · Pipeline · Sourcing · Settings), managed by **uv**. A thesis
(`thesis.yaml`) drives all targeting; nothing is hardcoded.

The screening pipeline: **discover** candidate accounts (free) → run cheap
deterministic **heuristics** → gate + rank to **Claude** classification
(fine-grained: stage/sector/subsector/business model/tags + a 0–1
**thesis_fit**; verdicts cached in the store) → **score** 0–100 → **export**
CSV/Markdown → **triage & pursue** in the UI. Two agents (`scout/agents.py`):
the **strategy agent** (plain-language thesis → full thesis.yaml + seeds.yaml
proposal) and the **research-brief agent** (per-lead pre-call memo, cached in
the pipeline table).

On top of that sit two v5 systems:

- **Diligence engine** (`scout/diligence/`, `scout analyze`): a manually
  triggered multi-agent deep analysis — recon + 8 web-researched dimensions
  on the configured model, cross-examination + memo synthesis on a premium
  model — producing a **Diligence Score** (0–100 composite, a DIFFERENT
  score from the screening score) plus a native investment memo, feeding a
  small knowledge graph so competitive analysis compounds across memos.
  ~$4–8/memo, hard-capped, fingerprint-cached.
- **Remote-control digest** (`scout publish`, `.github/workflows/`): the
  GitHub Pages phone page can triage/tag startups (decision files →
  `remote/inbox/` in the code repo) and dispatch headless runs via GitHub
  Actions, with the store accumulating at `data/scout.db` IN the repo.
  The page's only "backend" is the GitHub API + a user-held PAT.

---

## 2. Run it / test it — always via `./scout-cli` or `uv run`

```bash
uv run pytest -q                 # ~180 tests, ~5s, no network (incl. an AppTest UI smoke test)
./scout-cli demo                 # $0 offline end-to-end run on sample founders — best smoke test
./scout-cli source --strategy github,hn   # free live discovery, no scoring
./scout-cli ui                   # Streamlit workspace on :8501
./scout-cli analyze <name>       # PAID (~$4-8) deep analysis — never in tests/CI experiments
./scout-cli memo <name>          # print a stored memo ($0)
./scout-cli inbox                # list pending phone-triage decisions ($0)
./scout-cli publish              # render the phone digest to docs/ ($0; --push publishes)
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
                    verify, probe, demo, export, budget, strategy, analyze,
                    memo, inbox, publish, ui. Pipeline helpers: _run_pipeline,
                    _enrich_accounts, _run_discovery, _merge_accounts (fills
                    Account.sources), _fetch_tweets (parallel for free
                    adapters). `run` degrades to github/hn + cached data when
                    the twscrape adapter can't be built (headless CI).
  config.py         Pydantic Settings (.env) + Thesis/Seeds/SignalParams (yaml)
                    + save_thesis/save_seeds (shared by CLI + UI).
                    STAGE_* maps: stage → search categories / discovery sources.
  models.py         The ONLY data structures crossing module boundaries:
                    Account, Tweet, Signal, LLMVerdict, Lead, UnlinkedLead.
  llmcall.py        Shared Claude call plumbing — client factory (max_retries=1),
                    the transient-retry policy (timeouts fail fast), fence
                    stripping, call_with_parse_retry (JSON-in-text corrective
                    loop). agents/classifier/outreach/diligence all go through
                    here; NEVER hand-roll another client/retry/parse loop.
  store.py          SQLite (sqlite-utils) — cache, dedupe, TTL, budget ledger,
                    follow-edge/bio snapshots, unlinked leads, deal-flow pipeline,
                    llm_verdicts cache.
  score.py          Pure scoring. score_leads + score_breakdown (THE score math,
                    single source of truth; UI renders its steps).
  export.py         CSV + Markdown writers, rich terminal table.
  outreach.py       Claude-drafted first-touch outreach (Pipeline tab); template fallback.
  agents.py         Strategy agent (generate_strategy/parse_strategy/apply_strategy),
                    research-brief agent (research_brief), weight-tuning agent
                    (suggest_weights/parse_weight_proposal), watchlist validation
                    (validate_watchlist). Parse/apply helpers are pure. All Claude
                    clients carry explicit timeouts (STRATEGY/BRIEF/WEIGHTS_TIMEOUT_S);
                    timeouts are excluded from tenacity retry — fail fast in the UI.
  insights.py       Pure triage analytics: triage_stats/stats_prompt contrast
                    shortlisted-vs-passed leads per signal/sector/stage/fit;
                    feeds the UI insights panel and the weight-tuning agent.
  companies.py      Pure company grouping: company_key/group_by_company fold
                    accounts sharing a classifier-extracted company_name into
                    one startup entry (primary = highest score). startup_key
                    is the CANONICAL diligence identity (memos + KG nodes).
  inbox.py          Remote decision inbox: the phone digest commits one JSON
                    file per triage tap into remote/inbox/ of the code repo;
                    `scout inbox --apply` folds them into the pipeline table
                    (idempotent; malformed files skipped, never fatal).
  publish.py        Digest builder — now REMOTE-CAPABLE: the static page talks
                    to the GitHub API (token pasted once on the phone, stored
                    only in that browser's localStorage) to write inbox
                    decisions, dispatch the scout-run/scout-analyze workflows,
                    and poll run status. Read-only without a token.
  demo_data.py      8 synthetic sample founders for `scout demo` (obviously fake handles).
  ui.py             Streamlit app: Leads / Pipeline / Sourcing / Settings.
                    Apple design language; lead cards; agent flows; diligence
                    surfaces ("Worth a deep look" shelf, Analyze gate,
                    scorecard + memo, Landscape/KG block). ~1500 lines.
                    All diligence page-state (memo_by_handle, fresh keys,
                    suggestions, known company names) is derived ONCE per
                    rerun near the top over the FULL ledger grouping — never
                    derive memo identity/staleness from a filtered card view.
  diligence/        Multi-agent deep-analysis engine (`scout analyze`) — the
                    Diligence Score + investment memos. TWO-TIER SCORING: the
                    screening score (score.py) ranks the feed; the Diligence
                    Score (0-100 composite of 8 researched dimensions) attaches
                    only to analyzed startups. Never conflate them in UI copy.
    schema.py       Pydantic models (Evidence/DimensionFinding/ReconReport/Memo)
                    + pure parse helpers (JSON-in-text, clamp, payload validation).
    rubrics.py      The 8 dimension system prompts (stepped rubrics + the three
                    evidence laws) + recon/cross-exam/memo prompts + render().
    research.py     research_call(): one Messages API call with server web tools
                    (web_search/web_fetch _20260209, max_uses-capped), pause_turn
                    continuation loop, corrective parse retry, usage ledger.
    composite.py    diligence_breakdown() — pure weighted math mirroring
                    score_breakdown; thin_wrapper+weak-flywheel commodity cap.
    pipeline.py     run_analysis() orchestrator: evidence pack ($0) → recon →
                    8 dims (ThreadPoolExecutor, budget-gated) → cross-exam
                    (downgrades only) → composite → memo synthesis → KG ingest.
    graph.py        Knowledge-graph pure logic (node ids reuse the company_key
                    normalizer); persistence lives in store.py.
    priority.py     $0 "worth a deep look" ranking (the Startups-track shelf).
  ingest/
    base.py         SourceAdapter ABC (X sources) + DiscoverySource ABC (github/hn).
    twscrape_src.py Primary free X adapter: query bank, bio search, list members,
                    follow-graph snapshotting.
    xapi_src.py     Paid X API v2 adapter — BUDGET-GUARDED. BudgetExceededError.
    github_src.py   GitHub discovery (repo search → owner → X-handle bridge).
    hn_src.py       Hacker News (Algolia) discovery.
    linkedin_src.py Stub (NotImplementedError) — LinkedIn automation is a dead end.
  signals/
    heuristics.py   9 deterministic signals + run_heuristics + intent_appeared.
    llm.py          Claude classification. DEFAULT_PROMPT_TEMPLATE. Batches of 10.
tests/              pytest — offline only. test_diligence_* stub research_call;
                    test_inbox/test_publish cover the remote layer; test_ui_smoke
                    renders the whole UI via AppTest against a temp DB.
thesis.yaml         Targeting + weights + diligence_weights + signal_params +
                    llm_prompt. User-owned.
seeds.yaml          Query bank, bio_searches, watchlist, github_topics. User-owned.
scout-cli           Bash wrapper → `uv run python -m scout.cli "$@"`.
conftest.py         sys.path shim for pytest.
.streamlit/config.toml   Headless config for the UI.
.github/workflows/  Headless backend for the phone digest (all commit with
                    [skip ci] and serialize on the `scout-db` concurrency group):
  scout-run.yml       daily cron + dispatch: fold inbox → run → publish → commit DB
  scout-apply.yml     fires on remote/inbox pushes: apply + republish (~1 min loop)
  scout-analyze.yml   dispatch-only deep analysis (cost-confirmed on the page)
data/scout.db       The HEADLESS store (gitignore exception) — Actions runs use
                    DB_PATH=data/scout.db and commit it back; distinct from the
                    local ~/.scout/scout.db. Absent until the first Actions run.
remote/inbox/       Phone-triage decision files (one JSON per tap; .gitkeep only
                    when empty). Written by the digest page via the GitHub
                    Contents API; consumed by `scout inbox --apply --delete`.
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

The UI then reads `store.load_latest_leads()` for **Pick** (triage → shortlist/pass,
persisted in the `pipeline` table) and **Win** (status/notes/outreach).

### Data flow — one `scout analyze` (diligence)

```
build_evidence_pack ($0: ledger group, bios, cached tweets, verdict,
      │              pipeline notes, KG competitor seed)
      ▼   fingerprint hit? → serve cached Memo, done
recon agent (web tools) ──►  8 dimension agents (ThreadPoolExecutor,
      │                      each budget-gated BEFORE launch; a failure
      │                      becomes a gap finding, never aborts)
      ▼
cross-exam (synth model, no web; confidence DOWNGRADES only)
      ▼
diligence_breakdown (pure math, thin-wrapper cap) ──► memo synthesis
      ▼                                               (synth model, restate-only)
store.save_memo + graph.ingest → kg_nodes/kg_edges  (seeds the NEXT analysis)
```

Identity note: everything diligence-keyed (memos pk, KG company nodes, UI
lookups, priority exclusions) flows through `companies.startup_key(lead)` —
company key when the classifier named the company, else the NORMALIZED
handle. Never re-derive this rule inline.

### Data flow — the remote loop (phone → GitHub → phone)

```
digest page tap ──PUT──► remote/inbox/<ts>-<action>-<handle>-<rand>.json
      │                      (private code repo, via user-held PAT)
      ▼  push triggers scout-apply.yml   (or scout-run.yml cron/dispatch)
scout inbox --apply --delete  →  pipeline table (status/tags/notes)
      ▼
scout publish --push  →  digest repo (GitHub Pages)  →  page reflects triage
      ▼
commit data/scout.db back  [skip ci]
```

---

## 5. The scoring model (know this cold)

`score_breakdown(lead, thesis)` in `score.py` is the **single source of truth**;
`score_leads` just takes its last value, and the UI renders every step. The math:

1. `base = 100 × Σ(value_i × weight_i) / Σ(all thesis weights)` — weights relative.
2. `× llm.confidence` when a verdict is attached.
3. `× 0.2` when `llm.is_founder` is false (kills corporate/commentator accounts).
4. `× signal_params.stage_mismatch_multiplier` (0.5) when `llm.stage` ∉ `target_stages`.
5. `× ((1 − w) + w × llm.thesis_fit)` when the verdict carries `thesis_fit`
   (w = `signal_params.thesis_fit_weight`, default 0.5). Legacy verdicts without
   fit skip this step.

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

**The other tier — the Diligence Score** lives in
`diligence/composite.py::diligence_breakdown` (same single-source-of-truth
pattern: pipeline takes the last step's value, UI renders the steps). It is a
weighted mean of the 8 dimension scores (0–10 each, `score=None` excluded
from BOTH numerator and denominator) × 10, weights from
`thesis.yaml: diligence_weights` merged over `DEFAULT_WEIGHTS`, with one hard
rule: `thin_wrapper` classification + data-flywheel score < 4 caps the
composite at 40 (`commodity_risk`). Adding a dimension touches FIVE
registries — see §9; `test_prompt_registries_cover_all_dimensions` guards
them.

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
| pipeline | deal-flow state: status, notes, outreach, channel, brief, tags (JSON list; parse with store.pipeline_tags) |
| llm_verdicts | **verdict cache** — Claude verdict per handle, keyed by an input fingerprint (bio + tweets + thesis + model); TTL `VERDICT_TTL_DAYS` |
| xapi_usage | **budget ledger** — every paid call, cumulative spend |
| memos | deep-analysis memos, pk company_key; fingerprint-cached (bios + website + thesis + both model ids) — unchanged inputs re-serve for $0 |
| diligence_usage | diligence spend ledger — every research/synth call (tokens, searches, est $); `diligence_spend_usd()` |
| kg_nodes, kg_edges | knowledge graph: company/person/org/sector nodes, competes_with/founded_by/worked_at/in_sector edges. **Edge provenance "user" beats "agent": manual links are never overwritten or deleted by analysis runs** |

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
re-scores). The UI's Pipeline tab always resolves leads through the ledger so a
shortlisted lead missing from the latest run never degrades. `scout export` and
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
  with the retry-safety rule above for paid calls. For Claude calls the
  policy is centralized in `scout/llmcall.py` (transient errors retried,
  timeouts excluded, SDK retries capped at 1) — use it, don't copy it.
- **Nothing hardcoded that belongs in thesis.yaml/seeds.yaml.** Signal *mechanics*
  (regexes for launch language, github detection) live in code; *targeting*
  (keywords, orgs, stages, weights, params, prompt) lives in the yaml.
- **Tests never make live network calls.** Adapter tests hit pure parser
  functions on fixture JSON; diligence pipeline tests stub `research_call`.
- **Diligence cost cap.** Every diligence API call goes through
  `research.research_call` (usage → `diligence_usage` ledger); the pipeline's
  budget tracker checks spend-so-far before LAUNCHING each call and stops at
  `DILIGENCE_COST_CAP_USD`, keeping completed findings and flagging the gap
  (`budget_capped`). Web tools carry hard `max_uses` caps. Never add a
  diligence call path that bypasses research_call.
- **Two-tier scoring, never conflated.** The screening score (score.py) ranks
  the feed; the Diligence Score (diligence/composite.py) is a separate 0–100
  composite shown only on analyzed startups. UI copy must always name which
  one it means.
- **KG user edges are sacred.** `kg_upsert_edge` refuses agent writes over
  provenance="user" rows; nothing may delete user edges.
- **Memos never reach the public digest** (scout publish stays lead-cards
  only — memos are private work product).
- **The digest page never carries secrets.** The remote-control layer works
  by the USER pasting a fine-grained PAT into the page on their phone
  (localStorage only). Nothing token-shaped may ever be rendered into
  docs/index.html; the private repo's slug is embedded only when
  CONTROL_REPO is explicitly set. test_publish enforces this.
- **Headless DB home.** GitHub Actions runs use DB_PATH=data/scout.db,
  committed back after every run ([skip ci]) — that file is the accumulating
  system of record for headless operation; workflows serialize on the
  `scout-db` concurrency group. Never point a workflow at ~/.scout.
- **Inbox decisions are files, one per decision, applied idempotently.**
  The page writes unique filenames (no write races, no lost updates);
  `scout inbox --apply` must stay safe to re-run after partial failure.

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
  a real current model ID. Don't "correct" it. The diligence synth model
  default `claude-opus-4-8` (`DILIGENCE_SYNTH_MODEL`) is likewise real.
- **Server web tools (diligence):** `web_search_20260209` / `web_fetch_20260209`
  are the current API tool types and run on `claude-sonnet-4-6`. web_fetch only
  fetches URLs already present in the conversation — recon/dimension prompts
  must include candidate URLs verbatim. `stop_reason == "pause_turn"` means
  the server-side tool loop paused: re-send with the partial assistant turn
  appended (handled in research.py). Do NOT declare code_execution alongside
  these tools (dynamic filtering is built in; a second env confuses the model).
  Structured outputs are NOT guaranteed on sonnet-4-6 — all diligence agents
  keep the JSON-in-text + corrective-retry idiom.
- **GitHub API facts the remote layer relies on:** `api.github.com` allows
  CORS from any origin, so a GitHub Pages page can call it directly with a
  user-supplied PAT (`Authorization: Bearer`, `X-GitHub-Api-Version:
  2022-11-28`). `workflow_dispatch` is addressed by workflow FILE NAME
  (`/actions/workflows/scout-run.yml/dispatches`, returns 204). The Contents
  API `PUT` needs a `sha` only when UPDATING a file — creating a NEW uniquely
  named file needs none, which is why one-decision-one-file has no write
  races. Commits containing `[skip ci]` skip workflow triggers natively.
  PyYAML parses a workflow's `on:` key as boolean `True` (YAML 1.1) — tests
  that load workflow files must check `data.get("on") or data.get(True)`.
- **sqlite across threads:** sqlite3 connections must not cross threads —
  diligence dimension workers each open their OWN `Store(db_path)`; the
  usage-ledger write inside `research.py` is best-effort (a transient sqlite
  error must never discard a paid finding).

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
  the UI's step display) flows from it. Same rule for the Diligence Score:
  edit `diligence_breakdown` only.
- **Add a stage behavior:** edit the `STAGE_SEARCH_CATEGORIES` /
  `STAGE_DISCOVERY_SOURCES` / `STAGE_BIO_GRAPH` maps in config.py.
- **Add a diligence dimension:** five registries must stay in sync (guarded
  by `test_prompt_registries_cover_all_dimensions`): `DIMENSION_KEYS` +
  `DIMENSION_LABELS` (+ optionally `PAYLOAD_MODELS`/`CLASSIFIED_DIMENSIONS`)
  in diligence/schema.py, a stepped rubric in `RUBRICS` (diligence/rubrics.py
  — end it with THREE_LAWS + `_finding_schema`), a default weight in
  `composite.DEFAULT_WEIGHTS`, and a commented default in thesis.yaml's
  `diligence_weights`. The pipeline, UI scorecard, and memo template pick the
  new key up automatically. Note the user-facing agent count derives from
  `pipeline.AGENT_COUNT` — don't hardcode "11".
- **Add a phone (remote) action:** add the action name to `ACTIONS` +
  `apply_decision` in scout/inbox.py (keep it idempotent — absolute statuses,
  deduped tags), a button/`data-act` handler in `_REMOTE_JS`
  (scout/publish.py), and a test in test_inbox.py. The JS writes decisions;
  ONLY `apply_decision` interprets them.
- **Add or edit a workflow:** keep the `scout-db` concurrency group, the
  `[skip ci]` commit convention, `DB_PATH: data/scout.db`, and secrets via
  env-indirection (never interpolate `inputs.*` into `run:` shell — use an
  `env:` block; workflow inputs are attacker-influencable text). If the page
  must dispatch it, add the file name to publish.py's constants so
  `test_workflow_files_parse_and_match_dispatch_targets` covers it.

---

## 10. Current state / open threads

- Fully working: demo, source (all strategies), the whole scored pipeline, the
  Leads/Pipeline/Sourcing/Settings UI (Apple design language, lead cards),
  strategy agent (`scout strategy` + Sourcing tab, validated live), research
  briefs, verdict cache, paid probe + verify (validated live, ~$0.83 spent).
- v4 additions (all validated live): person-centric lead ledger with score
  deltas + strategy grouping (runs table), paid-run + precision-pass cost
  confirmation gates in the UI, watchlist handle validation, card-face triage
  with expander state preservation (see the stateless-expander note in ui.py),
  triage insights + AI weight suggestions (insights.py + suggest_weights),
  pipeline CSV export, agent timeouts, staleness nudge.
- v5 additions: the **diligence engine** (`scout/diligence/`) — `scout analyze`
  / `scout memo`, Diligence Score scorecard + native investment memos in the
  UI, knowledge graph with manual competitor linking, the "Worth a deep look"
  shelf, diligence spend ledger + per-memo cost cap. Fully covered by offline
  tests (stubbed research calls); **live end-to-end run still pending** — it
  needs `ANTHROPIC_API_KEY` plus a real tracked startup (e.g.
  `./scout-cli analyze tuva` after a discovery run), then a UI walkthrough:
  scorecard, memo render, Landscape chips, manual link persisting into the
  next analysis's competition seed.
- v5.1 additions: the **remote-control digest** — pipeline tags, the decision
  inbox (`scout/inbox.py` + `scout inbox`), the GitHub-API-driven page
  (`publish.py`), and the Actions backend (`.github/workflows/scout-run|
  scout-apply|scout-analyze.yml`) accumulating the store at `data/scout.db`.
  Headless `scout run` degrades gracefully without X cookies (github/hn legs
  + cached data). NOT yet exercised against live GitHub: the user still needs
  to add the Actions secrets, create the phone PAT, and tap through the flow
  (see README → Remote control) — the page↔API contract is covered by offline
  tests only.
- **Waiting on the user:** X account cookies (`TW_COOKIES` locally /
  `TW_COOKIES_B64` Actions secret) to activate the X discovery legs (query
  bank, bio search, follow-graph); replacing the suggested default
  `watchlist` in seeds.yaml with Headline's own investors; the remote-control
  first-run setup (Actions secrets, phone PAT — README → Remote control);
  the first live `scout analyze`.
- `thesis.target_stages` is currently `[launched]` (the user is sourcing
  just-launched startups).
- Two secrets were pasted into chat historically (Anthropic + X bearer) — the
  user plans to regenerate them; treat any in-repo key as disposable.
