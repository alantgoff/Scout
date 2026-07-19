"""Settings from .env plus thesis.yaml / seeds.yaml loaders.

All targeting lives in thesis.yaml — never hardcode keywords or weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    "idea": set(),
    "stealth": {"github"},
    "launched": {"github", "hn"},
    "scaling": {"hn"},
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
    stage_mismatch_multiplier: float = 0.5  # score × this when stage off-target
    # How much Claude's thesis_fit (0..1) sways the final score:
    # multiplier = (1 - w) + w × fit. 0 = ignore fit, 1 = fit scales fully.
    thesis_fit_weight: float = 0.5
    # Same shape for value_add_fit (would the firm's value-add accelerate this
    # startup?). Defaults to 0 — an informational dimension that never moves
    # the score unless you opt in.
    value_add_weight: float = 0.0


class Thesis(BaseModel):
    """The investment thesis that drives all targeting (thesis.yaml)."""

    thesis: str = ""
    keywords: list[str] = Field(default_factory=list)
    target_bios: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
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
    watchlist: list[str] = Field(default_factory=list)  # investors/operators to follow-diff
    tastemakers: list[str] = Field(default_factory=list)  # legacy alias for watchlist
    github_topics: list[str] = Field(default_factory=list)  # GitHub repo topics

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

    # Claude classification (omit key to run heuristics-only)
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-6"

    # GitHub discovery (optional; unauthenticated works at lower rate limits)
    github_token: str | None = None

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


def save_thesis(thesis: Thesis, path: Path = Path("thesis.yaml")) -> None:
    _save_yaml(path, thesis.model_dump(), _THESIS_HEADER)


def save_seeds(seeds: Seeds, path: Path = Path("seeds.yaml")) -> None:
    _save_yaml(path, seeds.model_dump(), _SEEDS_HEADER)
