# Scout

A thesis-driven sourcing engine for early-stage VC. Describe an investment
thesis in plain English; Scout searches X, GitHub, Hacker News and arXiv for
companies that match it, scores them against the thesis, and drafts the
investment memo.

Every figure it reports traces to something it actually read. Where it has no
evidence it says so rather than estimating — that constraint is enforced in
code, not just asked for in a prompt.

| | |
|---|---|
| Startups in the database | **942** across 13 runs |
| Classified with a thesis-fit score | **643** |
| Strong fit (≥ 70%) | **28** |
| Adversarially audited | 43 — every one returned corrections |
| Funding rounds tagged | 17, all evidence-cited |
| Theses tracked in parallel | 4, with version history |
| X API spend | **$3.21** of a $25 grant |

```bash
git clone https://github.com/alantgoff/Scout.git scout && cd scout
./start
```

No API keys needed to look around — `./start` seeds a sample dataset and opens
the workspace at `localhost:8501`. On macOS, double-click **`Scout.command`**
in Finder instead.

---

## The nine pages

**Thesis** — the control room. Write your thesis in plain language and the
strategy agent generates the whole sourcing configuration: X query bank, bio
searches, GitHub topics, an investor watchlist (handles are existence-checked;
fabrications are dropped), and scoring weights. You review before anything
saves. This page also holds the thesis library — switch between theses without
losing either, and see how many startups each one found. Runs launch from here,
with a worst-case cost estimate and an explicit confirmation before any paid
run.

**Startups** — everything sourcing found, newest run or all runs, as a triage
cockpit: a dense scannable list on the left, full dossier on the right. Each row
shows thesis fit, score, funding round and B2B/B2C at a glance. The dossier
gives you the product summary, the sector line, "How it scored" with every point
attributed, and one-click Longlist / Pass. Filter by round, stage, customer
type, fit or score. A **Database** view of the same data adds editable CRM
columns (vertical, use case, priority, your own custom fields) with AI
auto-categorisation.

**Longlist** — companies you've marked worth a closer look. Same cockpit,
narrower set, with the next action being shortlist or pass.

**Shortlist** — the ones you're seriously considering. This is where memos get
written and outreach gets drafted.

**Memos** — a 12-section first-draft investment memo per company: *Overview ·
Why now · Team · Product & differentiation · Technology & architecture ·
Traction & metrics · Competitive landscape · Market sizing · Strategic capital
& acquisition dynamics · Deal terms & ownership · Risks · Recommendation*,
opening with a TL;DR and closing with a **VERDICT: PURSUE / TRACK / PASS**,
tripwires, and first-call questions. Deep research is the default: it finds the
real company site, researches founders by name, verifies funding, and cites its
sources. Memos are editable in place and export as Markdown or styled PDF.
AI-drafted outreach sits below each one.

**Activity** — the firm's shared memory. Every vote, comment, triage move and
memo edit, newest first, filterable by member and kind, with an unread count in
the nav. With partners in different timezones this is the first thing you open:
what happened while you were asleep.

**Evidence** — whether any of this actually works. A backtest scores companies
using only evidence that was public on a past date, then checks where the ones
that went on to raise actually ranked. Three views: **Results** (recall,
separation, lead time, and every company plotted against the controls),
**Signals** (which individual signals carried the information, with confidence
intervals and a check for two signals measuring the same thing), and **Over
time** (whether a signal is wearing out as the world catches on). Every report
carries its own limitations — including the ones that undercut it.

**Automation** — schedules, the job queue, and whether the worker is alive.
Momentum signals decay in days, so the run that matters is the one nobody had
to remember. Sourcing runs, digests and memos can all be scheduled or queued.

**Settings** — API keys, the X spend ledger, Slack notifications, your own
taste profile, and firm-wide defaults.

## How it works

```
DISCOVER            SCORE                      DECIDE
X search queries    10 deterministic signals    Longlist / Shortlist / Pass
bio search       →  + Claude classification  → Memo
GitHub topics       + adversarial audit        Outreach
Hacker News         + thesis fit
arXiv papers
investor follows
```

A run sources candidate accounts, folds a founder and their company account
into one entry, and scores each on three components:

- **Quality** — a readiness scorecard (B2B or B2C rubric), criteria scored 1–3
  from cited evidence only, rolled up 0–100
- **Fit** — how squarely the *product* matches your thesis, 0–1
- **Signal** — momentum: investor follow-graph convergence, bio changes,
  departure language, launch traction, and a change of published affiliation
  — a researcher whose next paper carries a different lab has usually left
  months before any bio says so

These blend (currently 35 / 50 / 15 — fit outweighs quality deliberately, so a
well-built off-thesis company can't outrank an on-thesis one), then pass
through multipliers for classifier confidence, evidence grounding, and stage
match. A second adversarial pass audits the top verdicts against their own
evidence and corrects what the first pass overstated.

Runs are incremental: everything fetched is cached, and accounts scored within
the last 7 days are skipped, so re-running while you tune a thesis is fast and
mostly free.

## Why it's built this way

**Evidence or nothing.** An LLM asked for a competitor's funding round will
produce one — plausible, specific, often a year stale, and indistinguishable
from a real one. A live run tagged Perplexity "Series C+" off a bio reading
*"Everything is Computer."* Rounds without a cited source are now discarded by
a model validator, and the memo prompt bans recalled figures and
citation-shaped phrases (*"per analyst estimates"*) at every research depth.

**A thesis is an object, not a config file.** Identity is kept separate from
version, so editing weights bumps the version while the thesis stays itself.
Scores name their origin ("Scored against Novel Architectures · v3"), changing
a thesis marks old scores stale rather than silently reinterpreting them, and
previous verdicts are archived — so "0.20 under Edge AI, 0.75 under Novel
Architectures" stays answerable.

**Cost is a correctness problem.** The X API is pay-per-use against a fixed
grant. The guard pre-checks worst-case cost before each request, never retries
anything that may already have been billed, and buys breadth through many
narrow queries rather than one deep page — worst case per run fell from $30 to
$3.60, measured actual $1.55.

## Configuration

Two files, both editable in the UI:

- **`thesis.yaml`** — the thesis statement, target stages, keywords, sectors,
  disqualifiers, signal weights, scorecard weights, and scoring parameters
- **`seeds.yaml`** — the X query bank (departure / stealth-intent / hiring /
  launch), bio searches, investor watchlist, GitHub topics

Optional keys in `.env`: `ANTHROPIC_API_KEY` (classification and memos —
without it, heuristics-only), `TW_COOKIES` (free X scraping),
`X_BEARER_TOKEN` + `XAPI_SPEND_CAP_USD` (paid X API), `GITHUB_TOKEN`
(raises GitHub's rate limit from 60/hr to 5,000).

## CLI

```
run          Full pipeline → out/leads_*.csv + report_*.md
  --source twscrape|xapi   free scraping (default) or the paid X API
  --max-accounts N · --min-score N · --ttl-days N

reclassify   Re-score without discovery — no X spend, cache-first
  --all                    every startup in the ledger
  --stale-only             only those scored under an older thesis version

thesis       list · show <id> · new <name> · use <id> · clone <id> <name> · archive <id>
source       Discovery preview — raw accounts per strategy, no scoring or cost
inspect <handle>   Score one account, print the per-signal breakdown
verify       Hydrate the shortlist with fresh paid X data and re-score
budget       X API spend against the cap
demo         $0 offline end-to-end test
ui           Launch the workspace

migrate      Adopt a single-user database into the multi-member schema
  --owner you@firm.com     attribute existing judgments, snapshot memos,
                           import past triage as votes. Idempotent.

worker       Run the background worker: schedules fire, queued jobs execute
  --bootstrap              create the default weekday-run + morning-digest
  --once                   drain the queue and exit (for cron)
jobs         Queue state · --enqueue <kind> · --cancel <id>
schedule     --list · --add <kind> --at 06:00 --weekdays --tz Europe/London
digest       Post the Slack digest now (the scheduled one runs via the worker)
memo <handle>  Write one memo headlessly · --depth quick|standard|deep

hindsight    Backtest the scorer against companies that went on to raise
  --outcomes outcomes.yaml   companies that raised, plus controls that did not
  --cutoff 2025-02-01        score only evidence public on that date
  --sweep 3                  three cutoffs six months apart, for signal decay
  --suggest-weights          propose weights from the measured predictive power
  --blinded                  re-score with names redacted, to measure how much
                             the model is recognising rather than judging
```

Use `./scout-cli <cmd>`. On macOS, uv marks `.venv` hidden and CPython then
skips the editable-install `.pth`, which breaks the bare `scout` command;
`./scout-cli` runs `python -m scout.cli` and is immune.

## Running it for a firm

Scout is one shared instance, not a copy per person: partners sign in with
Google, vote and comment on the same startups, and a worker sources on a
schedule so momentum signals get caught while they are still fresh.

```bash
scout migrate --owner you@firm.com      # adopt an existing single-user database
scout worker --bootstrap --once         # create the default schedules
scout worker                            # run them (systemd unit in deploy/)
```

[`deploy/`](deploy/) has the whole kit — systemd units, Caddy for TLS, and
litestream for continuous backup to object storage. One small VM is enough;
SQLite in WAL mode handles a firm's concurrency comfortably, and the database
is a single file you can copy.

To run the backtest, describe the companies you want to test against in an
outcomes file — see [`outcomes.example.yaml`](outcomes.example.yaml). It needs
both companies that raised **and** controls that did not: recall on its own
proves nothing, because a scorer that rates everyone 90 recalls everything.
The most persuasive version is a firm's own history — the ones you passed on,
and a few you correctly skipped.

## Known limitations

- **ToS risk:** scraping X via twscrape violates X's Terms of Service. Use a
  burner account.
- **arXiv affiliation coverage is partial.** `lab_departure` reads an
  author's stated affiliation across papers, but arXiv makes that field
  optional and it is often absent. A gap is treated as unknown, so the
  signal under-fires: real departures are missed, but the ones it reports
  are evidenced.
- **Paper authors are matched by name**, the only identifier arXiv gives.
  The key deliberately under-merges, so a researcher who changes how their
  name is written splits into two records rather than two people fusing into
  one. The signal quotes its evidence so either case is visible.
- **The follow-graph signal is twscrape-only** — via the official API it always
  scores 0, as engagement operators (`min_faves:`) are also unsupported there
  and get stripped.
- **Confidence is biased against stealth.** Classifier confidence multiplies the
  score, and stealth companies are inherently less legible (0.57 average vs
  0.83–0.93) despite carrying the highest thesis fit. Flooring it was tried and
  reverted — it also lifts companies whose product claims never traced to
  evidence. Documented in `AGENTS.md` as an open thread.
- **X API cost constants are unverified.** Spend figures derive from hardcoded
  per-read prices, never reconciled against X's actual rate card.
- **Without `ANTHROPIC_API_KEY`** you get heuristics-only ranking: no
  stage/sector/summary enrichment, and memos fall back to a data-only skeleton.
- **The backtest cannot prove what people will read it as proving.** Evidence
  is rewound to the cutoff, but the thesis and weights applied are today's —
  written by people who already know which companies raised. So it supports
  "would today's thesis have ranked these highly given only past evidence",
  not "would Scout have caught them at the time". Only running a frozen thesis
  forward settles the second. Outcomes and controls are also chosen by hand,
  and X history cannot be reconstructed at all, so every figure is a floor
  with a selection bias on top. Each report states all of this itself; the
  Evidence page will not show a number without them.

---

> **Working on the code?** [`AGENTS.md`](AGENTS.md) is the map — architecture,
> data flow, invariants, and the gotchas that will bite you.
