"""Settings from .env plus thesis.yaml / seeds.yaml loaders.

All targeting lives in thesis.yaml — never hardcode keywords or weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from scout import rubric

# The budget ledger + cache DB lives in the user's home dir so the $25 spend
# guard is enforced no matter which directory scout is run from. Override
# with DB_PATH in .env / the environment.
DEFAULT_DB_PATH = Path.home() / ".scout" / "scout.db"

# Launch-y tweet language for the launch_traction signal — overridable via
# `launch_phrases` in thesis.yaml.
DEFAULT_LAUNCH_PHRASES = [
    "launch",
    "launching",
    "launched",
    "introducing",
    "shipped",
    "we built",
    "day 1",
    "beta",
    "waitlist",
]


STAGES = ("idea", "stealth", "launched", "scaling")

# Who the company sells to — picks the quality-rubric lens in classification
# and backs the UI's customer-type filter.
CUSTOMER_TYPES = ("b2b", "b2c", "b2b2c", "mixed")

# Curated taxonomy defaults for the startup database's user-owned columns
# (store.ensure_default_columns seeds them once; afterwards the option lists
# are edited in the UI's column manager — these are just the starting sets).
CURATED_VERTICALS = [
    "AI infrastructure", "Developer tools", "Fintech", "Healthcare / bio",
    "Robotics / hardware", "Security", "Commerce / marketplaces", "Consumer",
    "Enterprise SaaS", "Data / analytics", "Climate / energy", "Industrial",
    "Media / creative", "Education", "Real estate / proptech",
    "Legal / compliance", "Gov / defense", "Logistics / supply chain",
]
CURATED_USE_CASES = [
    "Agent infrastructure", "Evals / observability", "Inference / serving",
    "Model training / silicon", "Data infrastructure", "Search / retrieval",
    "Copilots / assistants", "Workflow automation", "Vertical SaaS",
    "API platform", "Developer productivity", "Edge / on-device",
    "Networking / orchestration", "Physical automation", "Consumer app",
    "Marketplace", "Payments / banking", "Risk / underwriting",
]
CURATED_PRIORITIES = ["High", "Medium", "Low"]

# Anthropic pricing, $ per million tokens (input, output) — the numbers the
# LLM spend ledger converts usage into. Matched by model-id prefix so dated
# snapshots ("claude-haiku-4-5-20251001") price like their family. Cache
# reads bill at ~0.1x the input rate, cache writes at ~1.25x; web_search is
# $10 per 1,000 searches. These are ledger estimates for the budget gate,
# not an invoice — when Anthropic reprices, edit here.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4": (5.0, 25.0),  # 4.6 / 4.7 / 4.8
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Unknown model → assume Opus-tier. Overcounting an unknown model throttles
# the scan early; undercounting blows the envelope — only one is safe.
DEFAULT_PRICING_USD_PER_MTOK: tuple[float, float] = (5.0, 25.0)
WEB_SEARCH_COST_USD = 0.01
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def llm_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    searches: int = 0,
) -> float:
    """Estimated cost of one Claude call (or an accumulated batch of them).

    `input_tokens` is the UNCACHED input (usage.input_tokens is already net
    of the cache fields — do not subtract them again)."""
    rate_in, rate_out = DEFAULT_PRICING_USD_PER_MTOK
    for prefix, rates in MODEL_PRICING_USD_PER_MTOK.items():
        if model.startswith(prefix):
            rate_in, rate_out = rates
            break
    tokens_cost = (
        input_tokens * rate_in
        + cache_read_tokens * rate_in * CACHE_READ_MULTIPLIER
        + cache_write_tokens * rate_in * CACHE_WRITE_MULTIPLIER
        + output_tokens * rate_out
    ) / 1_000_000
    return tokens_cost + searches * WEB_SEARCH_COST_USD


# LEGACY (v7) flat quality-rubric dimensions — superseded by the scorecard
# rubrics in scout.rubric. Kept so pre-scorecard cached verdicts (which carry
# these keys in LLMVerdict.quality) still render and score via the legacy
# path in score.quality_score.
QUALITY_DIMENSIONS: dict[str, str] = {
    "team": "Founder/team strength as builders of THIS company — exits, "
            "senior domain experience, notable shipped work",
    "tech_product": "Evidence of real working technology — live product, "
                    "demos, docs, shipping cadence",
    "market": "Size and urgency of the problem — clear buyer/user, growing category",
    "defensibility": "Moat evidence — proprietary data/hardware, deep tech, "
                     "network effects, lock-in",
    "traction": "Commercial/user proof — logos, pilots, revenue (B2B) or "
                "users, growth, retention (B2C)",
    "investors": "Named investors or rounds — logos on site, funding "
                 "announcements, accelerator badges",
}
DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "team": 20, "tech_product": 20, "market": 15,
    "defensibility": 15, "traction": 20, "investors": 10,
}


class ValueAddLever(BaseModel):
    """One lever of the firm's strategic value-add (thesis.yaml → firm_value_add).

    Fed to the classifier so every lead gets a value_add_fit: "which startups
    would benefit from what THIS firm specifically offers", independent of
    thesis fit. `key` is the stable id the verdict's per-lever map uses."""

    key: str
    label: str
    description: str


# Headline's value-add, researched from the firm's own materials (July 2026):
# regional early-stage funds (US VII $408M / EU VII €320M / Brazil III / Asia V)
# feeding an $865M Global Growth IV fund, plus the in-house data platforms
# (EVA sourcing, ATHENA analytics, Searchlight, founder-facing Deepdive).
# Overridable per-thesis via `firm_value_add` in thesis.yaml.
DEFAULT_VALUE_ADD_LEVERS = [
    ValueAddLever(
        key="global_expansion",
        label="Local-to-global expansion",
        description=(
            "Autonomous local funds in the US, Europe, Latin America, and Asia "
            "connected into one global platform — startups whose product or "
            "go-to-market must cross borders early (multi-region customers, "
            "US↔EU expansion, international marketplaces) get on-the-ground "
            "help in each market."
        ),
    ),
    ValueAddLever(
        key="follow_on_capital",
        label="Multi-stage follow-on capital",
        description=(
            "Early-stage checks of $1.5–15M chain into an $865M growth fund "
            "writing $20–70M from Series B — startups on capital-intensive "
            "trajectories (infra buildout, GPU spend, market-by-market "
            "expansion) that will need deep follow-on rounds benefit most."
        ),
    ),
    ValueAddLever(
        key="data_driven_growth",
        label="Data-driven growth benchmarking",
        description=(
            "Proprietary systems (EVA sourcing, ATHENA analytics, Searchlight, "
            "the founder-facing Deepdive) benchmark traction, retention, and "
            "product-market fit against one of the largest startup datasets in "
            "venture — startups with measurable usage or revenue metrics "
            "(consumer, marketplaces, usage-based SaaS, PLG devtools) can be "
            "benchmarked and coached with it."
        ),
    ),
    ValueAddLever(
        key="sector_playbooks",
        label="Sector depth & portfolio network",
        description=(
            "Operator networks and repeat playbooks in fintech, "
            "commerce/consumer, B2B SaaS, and AI infrastructure (portfolio "
            "includes Mistral AI, NGINX, Sonos, Bumble, Gopuff, Acorns, "
            "Raisin) — startups in these lanes tap portfolio intros, customer "
            "pipelines, and hiring networks."
        ),
    ),
]

# Which X-search query categories are worth running for each target stage.
STAGE_SEARCH_CATEGORIES: dict[str, set[str]] = {
    "idea": {"departure", "search"},
    "stealth": {"departure", "stealth_intent", "hiring", "search"},
    "launched": {"launch", "hiring", "search"},
    "scaling": {"launch", "search"},
}

# Which supplementary discovery sources fit each stage.
STAGE_DISCOVERY_SOURCES: dict[str, set[str]] = {
    # arXiv is the earliest instrument here: it surfaces researchers before
    # there is a company to find, which is exactly the idea/stealth window
    # and useless by the time a company is scaling.
    "idea": {"arxiv"},
    # RSS is a post-launch instrument: launch feeds and funding coverage
    # only carry a company once it has something to announce, which is
    # exactly why it is useless for idea/stealth and valuable after.
    # A Form D is filed within 15 days of the first sale — before the launch
    # post and the press. It reaches back into stealth: a stealth raise IS
    # a Form D, and nothing else public says so.
    "stealth": {"github", "arxiv", "sec"},
    # YC's directory lists a batch from the day it starts, website and
    # one-liner included: launched by definition, seed by definition.
    "launched": {"github", "hn", "rss", "sec", "yc"},
    "scaling": {"hn", "rss", "sec", "yc"},
}

# Bio search & watchlist graph-hop are early-stage instruments.
STAGE_BIO_GRAPH = {"idea", "stealth", "launched"}


class SignalParams(BaseModel):
    """Tunable constants behind the deterministic signals (thesis.yaml →
    `signal_params`). Every value is editable in the UI's Signals tab."""

    traction_floor: float = 0.05  # min engagement/followers ratio to count
    traction_saturation: float = 0.25  # ratio at which launch_traction = 1.0
    traction_window_days: int = 30  # how recent a launch tweet must be
    convergence_full_credit: int = 2  # recent watcher follows for value 1.0
    # star_velocity: stars the discovery repo gained over the window for
    # full credit. 100 in a week is a launch that landed; the baseline is
    # the daily snapshot history, so it needs scheduled runs to exist.
    star_velocity_full: int = 100
    star_velocity_window_days: int = 7
    stage_mismatch_multiplier: float = 0.5  # score × this when stage off-target
    # Final-score blend: base = Σ(weight × component) / Σ(weights of PRESENT
    # components), then the multiplier chain. Components: company QUALITY
    # (Claude's evidence-backed rubric), thesis FIT, X-SIGNAL momentum.
    # Weights are relative and renormalize over what a lead actually has —
    # a lead is never punished for a component nobody could evidence.
    score_weight_quality: float = 0.45
    score_weight_fit: float = 0.35
    score_weight_signals: float = 0.20
    # value_add_fit multiplier weight (would the firm's value-add accelerate
    # this startup?). Defaults to 0 — an informational dimension that never
    # moves the score unless you opt in.
    value_add_weight: float = 0.0
    # Score × this when the verdict's product claim never traced to real
    # evidence: the audit said "unverifiable", or the lead was never audited
    # AND its grounding is none/bio. Grounded-in-evidence or audit-confirmed
    # leads are exempt.
    ungrounded_multiplier: float = 0.6


class Thesis(BaseModel):
    """The investment thesis that drives all targeting (thesis.yaml)."""

    # IDENTITY — durable across tuning. `strategy_fingerprint` changes whenever
    # any weight or query moves, which makes it a version, not an identity: the
    # edge-AI thesis fragmented into three "strategies" purely from weight
    # edits. `id` is what survives, so "which thesis scored this" stays
    # answerable while the thesis is being refined. Both default empty, so a
    # thesis.yaml written before this existed still validates (see
    # `ensure_thesis_id` for how those are backfilled).
    id: str = ""  # stable slug, e.g. "novel-architectures"
    name: str = ""  # display name, e.g. "Novel Architectures"
    thesis: str = ""
    keywords: list[str] = Field(default_factory=list)
    target_bios: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    # Disqualifiers for what the CLASSIFIER learned about the company, checked
    # after classification against the product summary / sector rather than
    # the X bio. Deliberately a separate list: `disqualifiers` is full of
    # person markers ("angel investor", "PhD student", "opinions my own") that
    # describe an ACCOUNT, and matching those against product text would drop
    # a fintech whose summary happens to mention investing. Keep this list to
    # domains the thesis excludes outright.
    product_disqualifiers: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    # LEGACY (v7) per-dimension weights behind score.quality_score — still
    # applied to pre-scorecard cached verdicts; new verdicts score via
    # scorecard_weights below.
    quality_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS)
    )
    # Section weights behind the scorecard quality component
    # (score.scorecard_score) — {"b2b": {section: weight}, "b2c": {...}},
    # relative, renormalized over the sections a verdict actually evidences.
    # UI-editable in Signals & scoring; criterion sub-weights are code-owned
    # in scout.rubric.
    scorecard_weights: dict[str, dict[str, float]] = Field(
        default_factory=rubric.default_section_weights
    )
    launch_phrases: list[str] = Field(
        default_factory=lambda: list(DEFAULT_LAUNCH_PHRASES)
    )
    # Which company stages to hunt (drives search strategy AND scoring fit).
    target_stages: list[str] = Field(default_factory=lambda: list(STAGES))
    signal_params: SignalParams = Field(default_factory=SignalParams)
    # The firm whose value-add the classifier scores leads against.
    firm_name: str = "Headline"
    firm_value_add: list[ValueAddLever] = Field(
        default_factory=lambda: [x.model_copy() for x in DEFAULT_VALUE_ADD_LEVERS]
    )
    # Optional override of the Claude classification system prompt.
    # Placeholders: {thesis} {sectors} {stages} {firm} {value_add}.
    # Empty = built-in default.
    llm_prompt: str = ""

    @property
    def active_search_categories(self) -> set[str]:
        cats: set[str] = set()
        for stage in self.target_stages:
            cats |= STAGE_SEARCH_CATEGORIES.get(stage, set())
        return cats or {"search"}

    @property
    def active_discovery_sources(self) -> set[str]:
        srcs: set[str] = set()
        for stage in self.target_stages:
            srcs |= STAGE_DISCOVERY_SOURCES.get(stage, set())
        return srcs

    @property
    def bio_graph_active(self) -> bool:
        return any(stage in STAGE_BIO_GRAPH for stage in self.target_stages)


DEFAULT_SEC_INDUSTRIES = [
    "Other Technology", "Computers", "Telecommunications", "Business Services",
]


class Seeds(BaseModel):
    """Seed strategies (seeds.yaml).

    v2 splits searches into a labeled query bank (departure / stealth-intent /
    hiring) and adds bio/people search, a follow-graph watchlist, and GitHub
    topics. Legacy keys (`searches`, `tastemakers`) still work.
    """

    lists: list[str] = Field(default_factory=list)  # public X List IDs
    searches: list[str] = Field(default_factory=list)  # legacy catch-all queries
    searches_departure: list[str] = Field(default_factory=list)
    searches_stealth_intent: list[str] = Field(default_factory=list)
    searches_hiring: list[str] = Field(default_factory=list)
    searches_launch: list[str] = Field(default_factory=list)  # just-launched language
    bio_searches: list[str] = Field(default_factory=list)  # twscrape people search
    # arXiv categories to sweep (cs.LG, cs.AI, cs.CL, …). Targeting,
    # so it lives here; which affiliations count as a top lab is
    # signal mechanics and stays in arxiv_src.TOP_LABS.
    arxiv_categories: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)  # investors/operators to follow-diff
    tastemakers: list[str] = Field(default_factory=list)  # legacy alias for watchlist
    github_topics: list[str] = Field(default_factory=list)  # GitHub repo topics
    # RSS/Atom feeds read by the rss discovery source: launch feeds (YC,
    # Product Hunt), funding coverage, portfolio announcements, company
    # blogs. Free and keyless, so this is the cheapest channel to widen.
    rss_feeds: list[str] = Field(default_factory=list)
    # Form D industry groups worth resolving (the form's own vocabulary:
    # "Other Technology", "Computers", "Telecommunications", "Business
    # Services", "Manufacturing", "Biotechnology", …). Thesis-specific — a
    # health-care thesis wants different groups — so it lives with the seeds.
    sec_industries: list[str] = Field(default_factory=lambda: list(DEFAULT_SEC_INDUSTRIES))

    @property
    def all_searches(self) -> list[tuple[str, str]]:
        """(category, query) pairs across the query bank, legacy included."""
        return (
            [("departure", q) for q in self.searches_departure]
            + [("stealth_intent", q) for q in self.searches_stealth_intent]
            + [("hiring", q) for q in self.searches_hiring]
            + [("launch", q) for q in self.searches_launch]
            + [("search", q) for q in self.searches]
        )

    @property
    def watchers(self) -> list[str]:
        """Deduped watchlist ∪ tastemakers, @-stripped, order-preserving."""
        seen: dict[str, str] = {}
        for handle in self.watchlist + self.tastemakers:
            cleaned = handle.lstrip("@").strip()
            if cleaned and cleaned.lower() not in seen:
                seen[cleaned.lower()] = cleaned
        return list(seen.values())


class Settings(BaseSettings):
    """Runtime settings from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # twscrape (primary, free)
    tw_cookies: Path | None = None  # TW_COOKIES: path to cookies file (see README)

    # X API v2 (fallback, PAID — pay-per-use since Feb 2026)
    x_bearer_token: str | None = None
    # Hard lifetime spend cap across ALL runs, persisted in scout.db.
    # Default leaves buffer under a $25 grant.
    xapi_spend_cap_usd: float = 20.0
    xapi_cost_per_post_read: float = 0.005  # $ per tweet returned
    xapi_cost_per_user_read: float = 0.010  # $ per user profile returned
    # Results requested per paid search. A result can bill a post read AND its
    # author's profile read, so each one costs up to
    # cost_per_post + cost_per_user = $0.015 and this is the main budget lever:
    # the endpoint's 100 ceiling would make a single query cost $1.50, where 20
    # costs $0.30. Endpoint accepts 10-100.
    xapi_search_page_size: int = 20
    # Paid timeline fetches allowed per run. A search response already seeds
    # the tweet cache for every account the query bank found, so a cache miss
    # in the tweet phase means a discovery-sourced (github/hn) account — the
    # most numerous and least qualified group in a run. At an id resolve plus
    # a timeline read each, enriching them quietly outspends the entire search
    # phase, so it is opt-in rather than default.
    xapi_max_timeline_fetches: int = 0

    # Claude classification (omit key to run heuristics-only)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-6"

    # Daily spend envelope — the total the system may spend in one UTC day
    # across BOTH ledgers (Claude tokens + X API), enforced at the spend
    # sites (classify, verify, research, memo). 0 disables the cap. This is
    # what makes an unattended daily scan safe: the cheap work (heuristics,
    # caches, free scraping) always runs; paid calls stop when the envelope
    # is spent and resume tomorrow.
    daily_spend_cap_usd: float = 1.0
    # Tracked-company refresh (`scout refresh`): how many longlisted+
    # companies get a live re-research per day, and how recently-refreshed a
    # company must be to be skipped. 3/day at 7-day spacing keeps ~21
    # companies continuously watched inside a ~$1 envelope.
    scan_refresh_per_day: int = 3
    refresh_min_age_days: int = 7
    # Unlinked-lead resolver (`scout resolve`): how many headlines and posts
    # with no company key get a live lookup per day. Each costs at most one
    # small web-search call plus the same research `scout add` runs, inside
    # the daily envelope — the cap keeps a busy news day from spending
    # tomorrow's classification budget on today's coverage.
    scan_resolve_per_day: int = 3

    # GitHub discovery (optional; unauthenticated works at lower rate limits)
    github_token: str | None = None

    # SEC EDGAR (Form D discovery). SEC's fair-access policy asks every
    # automated reader to identify itself with a descriptive User-Agent that
    # includes a contact — set SEC_USER_AGENT to "<firm> <email>" in .env.
    # Requests over 10/s are throttled; the source stays well under.
    sec_user_agent: str = "scout/0.1 (+https://github.com/alantgoff/Scout)"
    # Filings above this total (sold, else offered) are not seed-stage and
    # are dropped before they can reach the resolver.
    sec_max_offering_usd: int = 15_000_000
    # YC directory (yc discovery source): how many of the most recent batches
    # to read each run. Two covers the current batch and the one just before
    # it — the companies still raising seed rounds.
    yc_batches: int = 2

    # Vercel (the phone app's other host). A token lets the worker deploy
    # headlessly (`scout publish --vercel`); interactive use needs only a
    # one-time `vercel link` inside docs/. The digest password is NOT here:
    # it is the DIGEST_PASSWORD env var of the Vercel project.
    vercel_token: str | None = None

    # Pipeline knobs
    max_accounts: int = 500  # cap accounts ingested per run
    ttl_days: int = 7  # skip accounts scored within the last N days
    tweets_per_account: int = 20
    recent_follow_days: int = 30  # window for "new follow" / convergence signals

    # Efficiency knobs
    tweet_fetch_concurrency: int = 8  # parallel tweet fetches (free adapters only)
    llm_max_candidates: int = 150  # top-N by heuristic pre-score sent to Claude
    llm_concurrency: int = 4  # Claude classification batches in flight
    verdict_ttl_days: int = 14  # reuse a cached verdict if inputs unchanged
    classify_batch_size: int = 5  # accounts per Claude classification call

    # Grounded classification — the classifier reads each candidate's website
    website_ttl_days: int = 7  # company-site text cache TTL (failures: 1 day)
    web_fetch_concurrency: int = 12  # parallel company-site fetches
    web_fetch_timeout_s: float = 8.0  # per-site fetch cap
    web_text_max_chars: int = 6000  # site text sent to Claude (~1.5k tokens)
    verify_top_n: int = 25  # adversarial verdict audit on the top N leads (0 = off)
    # Hard wall-clock cap on the twscrape sourcing phase. X rate limits (the
    # follow-graph endpoint especially) make twscrape sleep through 15-minute
    # windows; when the budget expires the run continues with what it has.
    sourcing_time_budget_s: int = 480  # 8 minutes

    # Paths
    db_path: Path = DEFAULT_DB_PATH
    out_dir: Path = Path("out")

    # Phone digest (`scout publish --push`): a PUBLIC repo that serves the
    # rendered docs/ page via GitHub Pages. Kept separate from the code repo
    # so only the digest — never code, config, or the watchlist — is public.
    digest_repo: str | None = None


def load_thesis(path: Path = Path("thesis.yaml")) -> Thesis:
    with open(path, encoding="utf-8") as f:
        return Thesis.model_validate(yaml.safe_load(f) or {})


def load_seeds(path: Path = Path("seeds.yaml")) -> Seeds:
    with open(path, encoding="utf-8") as f:
        return Seeds.model_validate(yaml.safe_load(f) or {})


_THESIS_HEADER = (
    "# Managed by scout (UI / `scout strategy`). target_stages steers search + scoring;\n"
    "# signal_params and llm_prompt are editable in the UI.\n"
)
_SEEDS_HEADER = (
    "# Managed by scout (UI / `scout strategy`). Query bank + bio search + "
    "watchlist + github topics.\n"
)


def _save_yaml(path: Path, data: dict, header: str) -> None:
    path.write_text(
        header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def strategy_fingerprint(thesis: Thesis, seeds: Seeds) -> str:
    """Stable hash of the full sourcing configuration (thesis + seeds).

    Runs that share a fingerprint were produced by identical settings and are
    grouped together as one "strategy" in the UI's ledger view.
    """
    payload = json.dumps(
        {"thesis": thesis.model_dump(mode="json"), "seeds": seeds.model_dump(mode="json")},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# The thesis library. thesis.yaml stays THE active thesis — every existing
# code path (`load_thesis(Path("thesis.yaml"))`, `--thesis PATH`) is unchanged —
# and this directory holds the saved copy of each thesis so switching away from
# one does not lose it.
THESES_DIR = Path("theses")


def slugify(text: str, max_len: int = 48) -> str:
    """Lowercase, hyphenated, filesystem- and URL-safe id fragment."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return cleaned[:max_len].strip("-")


def thesis_version(thesis: Thesis, seeds: Seeds) -> str:
    """The TUNING fingerprint: what a lead was actually scored under.

    Deliberately blind to `id` and `name`. Those are identity, not targeting —
    renaming "Edge AI" changes nothing about how a startup scores, and hashing
    them would mark every lead stale the moment a thesis is given a proper
    name, which is the opposite of the point. Everything that does move a
    score (weights, disqualifiers, prompt, queries) is included.
    """
    payload = json.dumps(
        {
            "thesis": thesis.model_dump(mode="json", exclude={"id", "name"}),
            "seeds": seeds.model_dump(mode="json"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_thesis_id(thesis: Thesis) -> str:
    """This thesis's stable id, derived when it was never given one.

    Falls back through name, then the statement's opening words. The
    statement fallback is also the backfill rule for runs recorded before
    identity existed: grouping historical runs by `thesis_statement` recovers
    exactly the theses a human would name, because the statement is the one
    field that stays put while weights and queries churn.
    """
    for candidate in (thesis.id, thesis.name, thesis.thesis):
        slug = slugify(candidate)
        if slug:
            return slug
    return "untitled-thesis"


def thesis_path(thesis_id: str, active_path: Path = Path("thesis.yaml")) -> Path:
    """Library file for a thesis, kept beside the active thesis.yaml."""
    return active_path.parent / THESES_DIR.name / f"{slugify(thesis_id)}.yaml"


def save_thesis(thesis: Thesis, path: Path = Path("thesis.yaml")) -> None:
    """Write the active thesis, and mirror it into the library.

    The mirror is what makes switching non-destructive: the library copy is
    always current, so moving to another thesis and back returns the same
    configuration rather than whatever thesis.yaml last happened to hold.
    """
    thesis.id = thesis.id or ensure_thesis_id(thesis)
    _save_yaml(path, thesis.model_dump(), _THESIS_HEADER)
    library = thesis_path(thesis.id, path)
    library.parent.mkdir(parents=True, exist_ok=True)
    _save_yaml(library, thesis.model_dump(), _THESIS_HEADER)


def switch_thesis(thesis_id: str, path: Path = Path("thesis.yaml")) -> Thesis:
    """Make a library thesis active, preserving the one being left.

    Saves the outgoing thesis back to its own library file first, so an
    unsaved tweak made while it was active is not lost by the switch.
    """
    library = thesis_path(thesis_id, path)
    if not library.exists():
        raise FileNotFoundError(f"no thesis {thesis_id!r} in {library.parent}/")
    if path.exists():
        outgoing = load_thesis(path)
        if ensure_thesis_id(outgoing) != slugify(thesis_id):
            save_thesis(outgoing, path)
    incoming = load_thesis(library)
    incoming.id = incoming.id or slugify(thesis_id)
    _save_yaml(path, incoming.model_dump(), _THESIS_HEADER)
    return incoming


def save_seeds(seeds: Seeds, path: Path = Path("seeds.yaml")) -> None:
    _save_yaml(path, seeds.model_dump(), _SEEDS_HEADER)
