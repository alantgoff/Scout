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
    # Relative weights for the 8 Diligence Score dimensions (deep analysis).
    # Empty = built-in defaults (scout.diligence.composite.DEFAULT_WEIGHTS).
    # Distinct from `weights` (the screening score) — never conflate the two.
    diligence_weights: dict[str, float] = Field(default_factory=dict)
    # Optional override of the Claude classification system prompt.
    # Placeholders: {thesis} {sectors} {stages}. Empty = built-in default.
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

    # Diligence engine (deep analysis memos — scout analyze / UI Analyze).
    # Research dimensions run on claude_model with web tools; cross-exam and
    # memo synthesis run on the synth model below (premium depth).
    diligence_synth_model: str = "claude-opus-4-8"  # DILIGENCE_SYNTH_MODEL
    diligence_cost_cap_usd: float = 8.0  # hard per-memo spend cap
    diligence_concurrency: int = 4  # dimension agents in flight
    # Cost ESTIMATES (like the xapi_cost_* knobs): per-MTok token rates for the
    # research + synth models and a per-web-search charge. Tune via .env if
    # Anthropic pricing changes; the ledger records estimates, not invoices.
    diligence_cost_per_search: float = 0.01
    diligence_research_cost_per_mtok_in: float = 3.0  # claude-sonnet-4-6
    diligence_research_cost_per_mtok_out: float = 15.0
    diligence_synth_cost_per_mtok_in: float = 5.0  # claude-opus-4-8
    diligence_synth_cost_per_mtok_out: float = 25.0

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
    # Remote control: "owner/repo" of the PRIVATE code repo. When set, the
    # digest page pre-fills its remote setup panel with it, enabling phone
    # triage (decisions → remote/inbox/) and run dispatch (GitHub Actions).
    # The user's fine-grained PAT is never embedded — it lives only in the
    # phone browser's localStorage. Leaving this unset keeps even the private
    # repo's NAME off the public page (it can be typed in the panel instead).
    control_repo: str | None = None


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
