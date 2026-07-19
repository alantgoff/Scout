# scout

Startup-sourcing engine for a VC analyst. **The primary output is real,
launched startups** — discovered through their founders' and their own X
accounts (launch announcements, GitHub, Hacker News, investor follow-graph),
classified fine-grained (stage, sector/subsector, business model, **company
name + URL**, tags, and an explicit 0–1 **thesis fit**), and presented as
companies: a founder account and the startup's account fold into one entry.
The secondary, completeness track is a **pre-launch watch**: people the
signals say are about to leave a lab, go stealth, or launch. Two built-in
agents do the heavy lifting — a **strategy agent** turns a plain-language
thesis into the full sourcing configuration, and a **research-brief agent**
writes a pre-call memo per lead. CLI + a Streamlit UI in Apple design
language. Internal tool — scrappy on purpose.

> **Agents:** read [`AGENTS.md`](AGENTS.md) first — it's the fast map of the
> codebase, data flow, invariants, and gotchas.

## Quickstart

```bash
git clone https://github.com/alantgoff/X-Sourcing-Tool.git scout && cd scout
uv sync
cp .env.example .env            # fill in TW_COOKIES (and optionally the keys)
$EDITOR thesis.yaml             # thesis: stages, keywords, orgs, weights
$EDITOR seeds.yaml              # seeds: query bank, watchlist, github topics
./scout-cli demo                # $0 offline end-to-end test on sample founders
./scout-cli ui                  # the Leads · Pipeline · Sourcing · Database workspace (localhost:8501)
./scout-cli run                 # full pipeline → ./out/leads_*.csv + report_*.md
```

**Use `./scout-cli <cmd>`, not the bare `scout` console script.** On macOS, uv
marks `.venv` hidden and CPython then skips the editable-install `.pth` after a
dependency sync, breaking `uv run scout` with `ModuleNotFoundError`. `./scout-cli`
runs `python -m scout.cli` from the repo root, which is immune. (If the bare
command ever breaks: `chflags -R nohidden .venv`.) No `uv`?
`python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .`.

Runs are incremental: everything fetched is cached in `~/.scout/scout.db`
(override the location with `DB_PATH` in `.env`), and accounts scored within
the last `--ttl-days` (default 7) are skipped, so re-running while you tune
the thesis is fast and (in xapi mode) free. A legacy `./scout.db` in the
working directory is migrated to the home location automatically on first run.

## The workspace: Leads · Pipeline · Sourcing · Database · Settings

The UI (`./scout-cli ui`) is a content-first workspace:

1. **Leads** — the STARTUP is the first-class object in every view: cards are
   titled by the company (founder + company accounts folded into one entry,
   founders as the byline). A founder whose company isn't named yet still
   renders as a startup with a synthesized identity tied to the person —
   "Ada Lin's stealth startup" (pre-launch) or "…'s unnamed startup" (launched,
   name unknown); only non-founder accounts keep a plain account card. Three
   tracks: **Startups** (default — launched companies), **Pre-launch watch**
   (expected to launch or found soon — departures, stealth language, bio
   changes), and **Everything**. Two time
   scopes: **All runs** (the ledger — every handle ever scored, best-known
   state, with **score-change arrows**, **New** chips, and a "seen N× since"
   history line) and **Latest run**. Runs made with identical settings group
   into a **strategy**; a strategy filter appears once you have more than one.
   Cards carry the thesis-fit chip, taxonomy chips, and triage buttons
   (Shortlist / Pass) on the face; details expand to per-signal bars,
   step-by-step score math, and a one-click **research brief**. Search, sort,
   and filters — with a hidden-by-which-filter caption — live in one toolbar.
   A quiet banner nudges you when the last real run is >24h old.
2. **Pipeline** — work the shortlist to allocation (To reach out → Contacted →
   Meeting → Diligence → Allocated) with inline status and notes, **AI-drafted
   outreach** and the research brief side by side (each stamped with freshness),
   and a one-click **pipeline CSV export** (CRM-import-ready).
3. **Sourcing** — the **strategy agent** front and center: describe the thesis
   in plain language, review the proposed targeting / query bank / watchlist
   (handles are **existence-checked** via twscrape; fabrications are struck
   through and dropped on apply). Below: run controls — the paid X API path
   shows a **worst-case cost estimate and requires an explicit confirmation**
   before the Run button enables — a **Precision pass** section (paid verify
   with the same confirm-the-cost gate), and every manual knob behind
   disclosure. The Signals & scoring panel opens with **triage insights**
   (how your shortlist/pass decisions cluster) and can ask Claude to
   **suggest weight adjustments**, reviewed before apply.
4. **Database** — the raw store, browsable: pick any table (row counts inline),
   full-text search across text columns, auto-generated filters (low-cardinality
   columns become value pickers, numeric columns become range sliders), a column
   chooser, sort controls, CSV export of the filtered view, and a read-only SQL
   console for anything else.
5. **Settings** — keys, the budget ledger, and defaults.

Deal-flow state (status, notes, outreach, briefs) persists in `scout.db`.

## Stage targeting

`thesis.target_stages` (e.g. `[launched]`) is the master switch — it decides
**which search strategies run** and **how leads are scored**:

| Stage | Runs these searches | Discovery | Bio search + follow-graph |
|---|---|---|---|
| idea | departure | — | ✓ |
| stealth | departure, stealth, hiring | github | ✓ |
| launched | launch, hiring | github, hn | ✓ |
| scaling | launch | hn | — |

A lead whose Claude-classified stage falls outside your target stages is scored
× `signal_params.stage_mismatch_multiplier` (default 0.5) — so targeting
"launched" pushes idea/stealth accounts down without hiding them. Toggle stages
in the UI's **Targeting** tab; the preview line shows what each activates.

Every score input is inspectable and editable: **Sourcing → Signals & scoring**
exposes the weights, the calculation parameters (traction floor/saturation/
window, convergence threshold, stage multiplier, thesis-fit weight), and the
full Claude classification prompt. Each lead card in **Leads** tags the account
**Founder / Startup / Other** and expands to the signal bars and the score
computed step by step.

## Sourcing architecture (v2)

scout finds founders with a **person-centric, staged funnel** — the same shape
commercial signal platforms (Harmonic, Specter) use, built from free parts:

```
WATCH ──▶ DISCOVER ──▶ FUSE ──▶ VERIFY ──▶ SCORE
(free)      (free)     (local)  (paid, tiny)  (Claude + heuristics)
```

| Stage | What runs | Cost |
|---|---|---|
| **Discover — X** | Query bank (departure / stealth-intent / hiring tweets), **bio/people search** (`bio_searches` — the paid API can't do this), list members | $0 (twscrape) |
| **Discover — GitHub** | Recent repos in `github_topics` (created <90d, ≥10 stars); owners bridged to X via their GitHub profile | $0 (`GITHUB_TOKEN` optional) |
| **Discover — HN** | Show HN launches + "Who wants to be hired?" comments matching your sectors | $0 (keyless) |
| **Watch** | Each run snapshots every `watchlist` investor's following list; **new follows** (vs the last snapshot) become candidates. 2+ watchers newly following the same account fires `smart_money_convergence` — the strongest X signal. Bios of known accounts are also snapshotted; intent language newly appearing fires `bio_change`. | $0 (twscrape) |
| **Verify** | `scout verify` re-fetches the top shortlist with the official X API (fresh profile + ~10 tweets each) | ~$0.01/profile + $0.005/tweet, hard-capped |

Key commands:

- `scout source` — **discovery preview**: raw accounts per strategy, no scoring.
  Use `--strategy searches,bio,graph,github,hn,lists` to test one at a time.
- `scout run` — full pipeline (discovery → signals → Claude → score → export).
- `scout verify --max 50` — paid hydration of the current shortlist (~$3).
- `scout probe` — one-time ~$0.50 check of X API capabilities on your tier.

Cadence: **one batched run per day** is the sweet spot — it keeps twscrape
usage human-ish, matches the X API's 24h billing dedup, and gives follow-diffing
a meaningful baseline. Note the graph strategy needs **two runs** before it
yields anything (run 1 records the baseline snapshots).

The `watchlist` in seeds.yaml ships with well-known AI-infra investors as
**suggested defaults** — replace them with your own tastemakers; the quality of
this list drives the quality of the convergence signal.

## Cookie setup (twscrape — the default, free source)

twscrape authenticates with cookies from a logged-in x.com browser session.

1. **Use a burner account.** Scraping violates X's ToS and accounts do get
   banned — don't risk one you care about.
2. Log into x.com with that account in your browser.
3. Export the x.com cookies:
   - **Easiest:** a cookies-export extension (e.g. Cookie-Editor or "Get
     cookies.txt LOCALLY") → export the x.com cookies as **JSON**.
   - **Manual:** DevTools → Application → Cookies → `https://x.com`, copy the
     `auth_token` and `ct0` values into a JSON file yourself.
4. Save as `cookies.json` and point `TW_COOKIES` in `.env` at it.

`TW_COOKIES` expects a JSON file: either the array-of-objects format the
extensions produce (`[{"name": "auth_token", "value": "...", ...}, ...]`) or a
flat object (`{"auth_token": "...", "ct0": "..."}`). Only `auth_token` and
`ct0` are actually required.

## Editing `thesis.yaml`

All targeting lives here — the code never hardcodes keywords.

| Key | What it does |
|---|---|
| `thesis` | One-line statement; passed to the LLM classifier as context. |
| `target_stages` | Which company stages to hunt (`idea`/`stealth`/`launched`/`scaling`). **The master switch** — decides which searches/sources run and applies a scoring penalty to off-target leads. See [Stage targeting](#stage-targeting). |
| `keywords` | Founder-intent phrases matched against bios ("stealth", "day 1"). Drives `bio_intent` (and `bio_change` when they newly appear). |
| `target_bios` | Departure markers matched as **literal substrings** of the bio (case-insensitive). Include the marker in the entry itself — `"ex-OpenAI"` matches "ex-OpenAI", but a bare `"OpenAI"` would also match current employees. Fires `departure_signal`. |
| `launch_phrases` | Launch-y tweet phrases ("launching", "waitlist", …) that gate `launch_traction`. Whole-word/phrase, case-insensitive ("day 1" won't match "day 10"). Omit the key to keep the stock list. |
| `sectors` | Sectors you care about; LLM context + HN search terms. |
| `disqualifiers` | Any of these in a bio → account dropped entirely (no score). |
| `weights` | Per-signal weights for the 0–100 score (relative). |
| `signal_params` | Tunable constants: traction floor/saturation/window, convergence full-credit threshold, off-target stage multiplier, thesis-fit weight, value-add weight. |
| `firm_name` | The firm whose value-add leads are scored against (default: Headline). |
| `firm_value_add` | The firm's strategic value-add levers (`key`/`label`/`description`), fed verbatim to the classifier. Ships with Headline's four levers; edit to re-target. |
| `llm_prompt` | Optional override of the Claude classification prompt (placeholders `{thesis}` `{sectors}` `{stages}` `{firm}` `{value_add}`). Empty = built-in default. |

**How the score works** (see it stepped out live in each lead card):
`base = 100 × Σ(weight_i × value_i) / Σ(all weights)` — weights are relative, so
`20/20/20/…` is identical to `2/2/2/…`. Then, when a Claude verdict is attached:
`× confidence`, `× 0.2` if classified not-a-founder, `× stage multiplier`
(default 0.5) when the classified stage is off-target, and
`× ((1 − w) + w × thesis_fit)` where `w = signal_params.thesis_fit_weight`
(default 0.5) — a textbook thesis match keeps its score, an off-thesis lead is
scaled down but never zeroed on fit alone. The same shape applies to
`value_add_fit` via `signal_params.value_add_weight`, but that weight defaults
to **0** — the value-add dimension is informational unless you opt it into the
ranking.

### Headline value-add fit — which startups benefit from what Headline offers

Beyond "does this startup match the thesis?", every classified lead also gets a
**value-add fit**: would *Headline's* specific strategic value-add accelerate
this particular startup? The firm's levers live in `thesis.yaml →
firm_value_add` (edit them there; the classifier receives them verbatim) and
ship pre-filled from Headline's own materials:

1. **Local-to-global expansion** — autonomous local funds (US, Europe, LatAm,
   Asia) on one global platform; benefits startups that must cross borders early.
2. **Multi-stage follow-on capital** — $1.5–15M early checks chaining into the
   $865M Global Growth IV fund ($20–70M from Series B); benefits
   capital-intensive trajectories.
3. **Data-driven growth benchmarking** — Headline's in-house systems (EVA
   sourcing, ATHENA analytics, Searchlight, the founder-facing Deepdive);
   benefits metrics-rich models the platform can benchmark and coach.
4. **Sector depth & portfolio network** — fintech, commerce/consumer, B2B SaaS,
   AI infra (Mistral AI, NGINX, Sonos, Bumble, Gopuff…); benefits startups in
   those lanes.

Each verdict returns a 0–1 `value_add_fit`, a per-lever breakdown, and a
one-line reason. It surfaces as a **"Headline lift" chip** on lead cards (with
per-lever bars and the reason under Details), a **sort option**, columns in the
leads/pipeline CSVs and the Markdown report, a line in AI research briefs, and
a chip on the phone digest. Judged independently of thesis fit: a lead can
match the thesis yet need nothing Headline uniquely offers — and vice versa.

The nine signals the heuristics emit (names must match in `weights`):
`bio_intent`, `departure_signal`, `bio_change`, `smart_money_follow`,
`smart_money_convergence`, `launch_traction`, `builder_evidence`,
`github_evidence`, `source_corroboration` (2+ independent discovery strategies
surfacing the same account).

## X API budget (`--source xapi`)

The official X API v2 is **pay-per-use since Feb 2026** — no free tier.
As of mid-2026: ~$0.005 per tweet/post read, ~$0.010 per user-profile read,
~2M reads/month cap. Our bearer token has a hard **$25 total** budget, so:

- **$25 buys roughly 5,000 tweet reads or 2,500 profile reads.** A single
  careless run can eat a big chunk of that. Treat xapi as the fallback, not
  the default.
- **Spend guard:** every API call is logged to a persistent ledger in
  `~/.scout/scout.db` (it survives across runs and working directories;
  override with `DB_PATH`). Before each call, scout checks cumulative
  estimated spend against `XAPI_SPEND_CAP_USD` (default `20.0`, deliberately
  under $25) and hard-stops with `BudgetExceededError` rather than exceed it.
  Check where you stand anytime with `scout budget` (it also prints the
  ledger path).
- **xapi mode is search-only:** it skips list ingestion and the tastemaker
  graph-hop (those endpoints aren't worth the spend). Only `seeds.yaml →
  searches` are used.
- Costs are *estimates* computed client-side from the per-read prices in
  `.env` / `config.py` — reconcile against the X developer console if you're
  pushing close to the cap.

## CLI reference

Invoke as `./scout-cli <command>` (see the Quickstart note on why).

```
strategy     AI strategy agent: thesis in plain language → full sourcing config
  "description"            the thesis, quoted
  --apply                  write the proposal to thesis.yaml + seeds.yaml

run          Full pipeline: discover → heuristics → Claude → score → export
  --source twscrape|xapi   default twscrape; xapi = paid official API, search-only
  --max-accounts N         cap accounts ingested per run (default 500)
  --min-score N            drop leads scoring below N (0–100)
  --ttl-days N             skip accounts scored within the last N days (default 7)

source       Discovery PREVIEW — raw accounts per strategy, no scoring/LLM/cost
  --strategy searches,bio,graph,github,hn,lists   subset to test (default: stage-aware)
  --max-accounts N

inspect <handle>   Score one account and print the per-signal breakdown (@ optional)

verify       Hydrate the current shortlist with FRESH paid X API data and re-score
  --max N            how many top leads to hydrate (default 50)
  --tweets N         tweets to pull per account (default 10)
  --discovered       hydrate recently-discovered accounts instead of the scored run

probe        One-time ~$0.50 empirical check of X API capabilities on your tier
demo         $0 offline end-to-end test on built-in sample founders
export       Re-export the last run from the cache DB (--format md|csv|both)
  --pipeline               export the deal flow (status, notes, outreach, briefs) instead
budget       Cumulative X API spend vs. XAPI_SPEND_CAP_USD
ui           Launch the Leads · Pipeline · Sourcing · Database · Settings workspace
```

**Efficiency:** free-source tweet fetches run concurrently
(`TWEET_FETCH_CONCURRENCY`, default 8; paid xapi stays sequential so the budget
guard can't be raced). Claude verdicts are cached in `scout.db` keyed by a
fingerprint of bio + tweets + thesis + model (`VERDICT_TTL_DAYS`, default 14),
so re-runs while tuning weights are free; classification runs
`LLM_CONCURRENCY` batches in parallel and is capped to the top
`LLM_MAX_CANDIDATES` (default 150) accounts by heuristic pre-score.
Sourcing itself has a hard wall-clock cap, `SOURCING_TIME_BUDGET_S`
(default 480 = 8 minutes): X rate-limits the follow-graph endpoint hard and
twscrape sleeps through 15-minute reset windows, so when the budget expires
the run simply continues with whatever was gathered (search legs run first,
the graph leg last, so the highest-yield legs get budget priority).

Outputs land in `./out/`: `leads_YYYYMMDD.csv` (full columns) and
`report_YYYYMMDD.md` (top-20 cards — the thing you paste into the one-pager).
Top 10 also prints to the terminal.

## Phone digest (GitHub Pages)

`./scout-cli publish --push` renders the deal flow into a single mobile-first
page (launched startups grouped by company, pre-launch watch, briefs, client-
side search) and pushes it to the public repo named in `DIGEST_REPO`, which
GitHub Pages serves. On your phone, open the page in Safari → Share →
**Add to Home Screen** — it installs like an app (Scout icon, full-screen)
and refreshes every time you publish after a scan.

Read-only by design: triage lives in the desktop app. The page contains only
lead data (never keys, config, or the watchlist), is `noindex`, and lives in
a separate repo so the code stays private. Remember the URL is still public —
anyone with the link can read your thesis statement and lead cards.

## Operational notes

Deliberate single-user, single-machine assumptions — fine for an internal
tool, worth knowing about:

- **Storage is one SQLite file** (`~/.scout/scout.db`): caches, the lead
  ledger, deal flow, and the spend ledger. Back it up if the history matters.
- **The spend ledger is per-machine, not per-token** — running scout on a
  second machine starts a fresh ledger against the same X API budget.
- **Secrets live in `.env`** in the project root, never in the DB or UI.
- **The UI commits to one light appearance** (set in `.streamlit/config.toml`):
  Streamlit pins its native widgets to a single theme once one is configured,
  so scout ships one deterministic look rather than a half-themed dark mode.
- **Agent calls run inline in the UI** with hard timeouts (strategy 120s,
  briefs/weights 60s); Streamlit's stop button is the cancel path.
- **Runs record provenance** (`runs` table: strategy fingerprint of
  thesis+seeds) — identical settings group as one strategy in the Leads view.

## Known limitations

- **ToS risk:** scraping X via twscrape is against X's Terms of Service.
  Accounts get banned. Use a burner, keep `--max-accounts` sane.
- **No graph-hop in xapi mode:** the `smart_money_follow` signal only works
  with `--source twscrape`; via the official API it always scores 0.
- **List ingestion is best-effort:** scout fetches the actual membership
  roll (twscrape's `list_members`); if that call fails it falls back to
  authors of recent tweets on the list timeline — a lossier proxy that
  misses members who haven't tweeted lately.
- **Engagement search operators are twscrape-only:** `min_faves:N`,
  `min_retweets:N`, `min_replies:N` in `seeds.yaml` searches are rejected by
  the X API v2 search endpoint, so xapi mode strips them from the query
  (a dim note shows what was actually sent).
- **Pinned tweets aren't always seen by Claude:** only a pinned tweet that
  happens to be among the cached recent tweets is included in the LLM
  context; older pinned tweets are not fetched separately.
- **xapi accounts are scored from search-matched tweets:** in xapi mode the
  tweets that matched the search double as the account's cached timeline
  (cache-first — no extra paid timeline call during runs), so
  `launch_traction` sees only those tweets.
- **`smart_money_follow` is coarse:** it saturates at 3 tastemaker follows,
  and "recent follows" is approximated by each tastemaker's ~100
  most-recent follows.
- **LLM is optional:** no `ANTHROPIC_API_KEY` → heuristics-only mode
  (scout says so at runtime). You lose stage/sector/summary/account-type
  enrichment and the confidence multiplier, but ranking still works. Outreach
  drafting also falls back to a fill-in template without a key.
- **Bio/people search is twscrape-only:** the paid X API has no bio-search
  endpoint (`/2/users/search` returned 403 on our tier — see `scout probe`),
  so `bio_searches` and the follow-graph only run under `--source twscrape`.
- **GitHub identity bridge is partial:** GitHub discovery only links an owner
  to X when they've filled in their profile's Twitter/X field; otherwise the
  founder is recorded as an *unlinked lead* (surfaced in `scout source`) for
  manual lookup. Star-velocity diffing is a noted future hook (needs scheduled
  runs). LinkedIn adapter and `--watch` remain unbuilt stubs.
- **Pricing drift:** the X API figures above are as of mid-2026 and will
  change; the per-read prices are configurable in `.env` if they do.
