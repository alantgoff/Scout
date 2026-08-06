"""Pydantic models — the only data structures passed between modules."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Stage = Literal["idea", "stealth", "launched", "scaling"]

# The FUNDING round, orthogonal to Stage above. Stage is lifecycle (has it
# shipped?); this is the cap table (who has already priced it?). A launched
# company can be bootstrapped or Series B, and the two answer different
# questions — "is this too early to matter" vs. "is this too late to enter".
# "unknown" is the honest and most common answer: most seed companies never
# announce, and inferring a round from headcount or traction is guesswork.
FundingStage = Literal[
    "bootstrapped", "pre_seed", "seed", "series_a", "series_b",
    "series_c_plus", "unknown",
]
FUNDING_STAGE_LABELS: dict[str, str] = {
    "bootstrapped": "Bootstrapped",
    "pre_seed": "Pre-seed",
    "seed": "Seed",
    "series_a": "Series A",
    "series_b": "Series B",
    "series_c_plus": "Series C+",
    "unknown": "Unknown",
}
# Rank for sorting/filtering — "unknown" sorts last, never in the middle.
FUNDING_STAGE_ORDER: dict[str, int] = {
    "bootstrapped": 0, "pre_seed": 1, "seed": 2, "series_a": 3,
    "series_b": 4, "series_c_plus": 5, "unknown": 6,
}


class Account(BaseModel):
    """A Twitter/X account under evaluation."""

    id: str
    handle: str  # without the leading @
    name: str = ""
    bio: str = ""
    website: str | None = None
    followers: int = 0
    following: int = 0
    pinned_tweet_id: str | None = None
    source: str = ""  # "list" | "search" | "bio_search" | "graph" | "github" | "hn" | "manual"
    # Every strategy that independently surfaced this account (set by the
    # pipeline's merge step; `source` stays the first/primary one). Two or
    # more distinct sources fire the source_corroboration signal.
    sources: list[str] = Field(default_factory=list)
    followed_by: list[str] = Field(default_factory=list)  # watcher handles (any age)
    # Enrichment fields set by the pipeline from store history, not by adapters:
    recent_followed_by: list[str] = Field(default_factory=list)  # watchers whose follow is new
    bio_changed: bool = False  # stealth/intent language newly appeared in bio
    # "google deepmind → Sparse Labs" when this person's PUBLISHED affiliation
    # moved. Filled by cli._enrich_accounts from paper history, never by an
    # adapter (Account carries no store).
    lab_move: str = ""
    github_repo: str | None = None  # evidence repo URL when discovered via GitHub
    fetched_at: datetime | None = None

    @property
    def url(self) -> str:
        return f"https://x.com/{self.handle}"


class SitePage(BaseModel):
    """One fetched company website — the classifier's ground truth for what
    a startup actually does. Cached in the store's `websites` table
    (failures too: negative caching keeps dead domains from being re-tried
    every run)."""

    url: str  # normalized root URL — the cache key ("https://raindrop.ai/")
    final_url: str = ""  # after redirects
    status: str = ""  # "ok" | "thin" | "non-html" | "too-large" | "error:*"
    text: str = ""  # extracted text ("" on failure)
    fetched_at: datetime | None = None

    @property
    def usable(self) -> bool:
        """Text worth showing to the classifier."""
        return self.status in ("ok", "thin") and bool(self.text)


class UnlinkedLead(BaseModel):
    """A founder signal from a non-X source with no X handle to bridge to.

    Surfaced in `scout source` output and report appendices for manual lookup.
    """

    source: str  # "github" | "hn"
    ref: str  # github login / hn username
    name: str = ""
    bio: str = ""
    url: str = ""  # repo / story URL
    found_at: datetime | None = None


class Tweet(BaseModel):
    id: str
    account_id: str
    text: str
    created_at: datetime | None = None
    likes: int = 0
    retweets: int = 0
    replies: int = 0

    @property
    def engagement(self) -> int:
        return self.likes + self.retweets + self.replies


class Signal(BaseModel):
    """One scored signal for an account.

    Heuristics emit (name, value, detail); score.py fills in `weight`
    from thesis.yaml and computes the weighted contribution.
    """

    name: str
    value: float = 0.0  # normalized 0..1
    weight: float = 0.0
    detail: str = ""

    @property
    def contribution(self) -> float:
        return self.value * self.weight


AccountType = Literal["founder", "startup", "other"]


class LLMVerdict(BaseModel):
    """Claude's classification of one account.

    v3 adds a fine-grained taxonomy (subsector, business_model, tags) and an
    explicit thesis_fit score that feeds the ranking — all optional so cached
    v2 verdicts still validate.
    """

    handle: str
    account_type: AccountType | None = None  # founder (person) vs startup (company)
    is_founder: bool = False
    stage: Stage | None = None
    # v8 — the cap table, separate from the lifecycle `stage` above. Only ever
    # set from an announcement in evidence; never inferred from headcount,
    # traction or polish, because a wrong round is worse than no round when
    # deciding whether a company is already past your entry point.
    funding_stage: FundingStage | None = None
    funding_amount: str | None = None  # as announced, e.g. "$10M"
    funding_investors: list[str] = Field(default_factory=list)  # named leads
    funding_evidence: str | None = None  # where the round was stated
    sector: str | None = None
    subsector: str | None = None  # finer slice, e.g. "agent evals" under "ai infra"
    business_model: str | None = None  # "b2b saas" | "devtools" | "infra" | "consumer" | ...
    # The startup behind the account, when identifiable — lets the UI present
    # STARTUPS (and fold a founder + the company account into one entry)
    # rather than raw X accounts.
    company_name: str | None = None
    company_url: str | None = None
    one_line_summary: str = ""
    why_interesting: str = ""
    thesis_fit: float | None = None  # 0..1 — how squarely this matches the thesis
    fit_reason: str = ""  # one line explaining the fit score
    # v5: would the firm's strategic value-add (thesis.firm_value_add levers)
    # specifically accelerate this startup? Independent of thesis_fit — a lead
    # can match the thesis yet need nothing the firm uniquely offers.
    value_add_fit: float | None = None  # 0..1 aggregate
    value_add_levers: dict[str, float] = Field(default_factory=dict)  # lever key → 0..1
    value_add_reason: str = ""  # one line naming the lever(s) that apply
    tags: list[str] = Field(default_factory=list)  # fine-grained descriptors
    confidence: float = 0.0
    # v6 — grounded classification: the product claim must trace to evidence.
    # All optional so cached v5 verdicts still validate.
    product_summary: str | None = None  # what the company ACTUALLY does, evidence-cited
    # Strongest evidence that established the product:
    # "website" | "pinned_tweet" | "tweets" | "github" | "bio" | "none"
    grounding: str | None = None
    # Adversarial audit outcome (top-N leads): "confirmed" | "corrected" |
    # "unverifiable"; None = not audited.
    verification: str | None = None
    verification_note: str = ""
    # v7 — company-quality rubric. customer_type picks the scoring lens
    # ("b2b" | "b2c" | "b2b2c" | "mixed"; None when the product itself is
    # not established). quality holds evidence-backed 0..1 scores per
    # dimension (config.QUALITY_DIMENSIONS keys); a dimension with no
    # evidence is OMITTED, never guessed. quality_score is computed OUR
    # side (score.quality_score) over the present dims only.
    customer_type: str | None = None
    quality: dict[str, float] = Field(default_factory=dict)
    quality_reasons: dict[str, str] = Field(default_factory=dict)  # dim → ≤15-word citation
    # v8 — readiness scorecard (scout.rubric). customer_type routes the
    # rubric (b2c → consumer; b2b/b2b2c/mixed/None → enterprise). scorecard
    # holds criterion key → 1..3 for ONLY the criteria the evidence supports
    # (omit, never guess); scorecard_reasons a ≤12-word citation per scored
    # criterion. Section/total/band math is computed OUR side
    # (score.scorecard_score). quality/quality_reasons above are the legacy
    # v7 flat rubric — old cached verdicts still validate; new verdicts
    # leave them empty.
    scorecard: dict[str, float] = Field(default_factory=dict)
    scorecard_reasons: dict[str, str] = Field(default_factory=dict)
    # Section-level manual overrides (section key → 0..100) — written only
    # by score.apply_override at load time, never by the classifier.
    scorecard_manual: dict[str, float] = Field(default_factory=dict)

    @field_validator("funding_stage", mode="before")
    @classmethod
    def _norm_funding_stage(cls, value):  # noqa: ANN001 — pydantic hook
        """Normalize the round; unrecognised spellings become "unknown".

        Same reasoning as customer_type below: a bare Literal would throw
        inside _parse_verdicts on one off-vocabulary answer and burn the
        batch. "Series A" and "seriesA" are the same round as "series_a".
        """
        if not isinstance(value, str):
            return None
        cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "preseed": "pre_seed", "pre_seed_round": "pre_seed",
            "seriesa": "series_a", "seriesb": "series_b",
            "series_c": "series_c_plus", "series_d": "series_c_plus",
            "series_e": "series_c_plus", "seriesc": "series_c_plus",
            "growth": "series_c_plus", "late_stage": "series_c_plus",
            "self_funded": "bootstrapped", "none": "unknown", "": "unknown",
        }
        cleaned = aliases.get(cleaned, cleaned)
        return cleaned if cleaned in FUNDING_STAGE_LABELS else "unknown"

    @model_validator(mode="after")
    def _round_requires_evidence(self):
        """A round without a source is downgraded to "unknown".

        The prompt asks for funding_evidence, but asking is not enforcing:
        the first live run tagged Perplexity "series_c_plus" off a bio
        reading "Everything is Computer." — recalled from model memory, with
        no evidence string, exactly the failure the field exists to prevent.
        A stated round decides whether a company looks past the fund's entry
        point, so it has to be checkable, and the cheapest check is that the
        classifier had to name where it read it.

        This also drops the occasional CORRECT tag whose evidence line was
        simply not filled in. That trade is deliberate: an unknown round
        costs one reclassify to recover, while a confident wrong one is
        never questioned.
        """
        if self.funding_stage and self.funding_stage != "unknown":
            if not (self.funding_evidence or "").strip():
                self.funding_stage = "unknown"
                self.funding_amount = None
                self.funding_investors = []
        return self

    @field_validator("customer_type", mode="before")
    @classmethod
    def _norm_customer_type(cls, value):  # noqa: ANN001 — pydantic hook
        """Normalize instead of rejecting: a Literal here would make one
        off-vocabulary value from Claude throw inside _parse_verdicts and
        burn the whole batch's corrective retries."""
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        aliases = {"consumer": "b2c", "enterprise": "b2b", "smb": "b2b",
                   "b2b/b2c": "mixed", "b2c2b": "b2b2c"}
        value = aliases.get(value, value)
        return value if value in ("b2b", "b2c", "b2b2c", "mixed") else None


class Lead(BaseModel):
    """An account plus everything we learned about it."""

    account: Account
    signals: list[Signal] = Field(default_factory=list)
    llm: LLMVerdict | None = None
    score: float = 0.0  # 0..100 after aggregation
    rank: int | None = None
    disqualified: bool = False

    @property
    def signals_hit(self) -> list[str]:
        return [s.name for s in self.signals if s.value > 0]

    @property
    def evidence_links(self) -> list[str]:
        links = [self.account.url]
        if self.account.website:
            links.append(self.account.website)
        return links


class LedgerEntry(BaseModel):
    """One handle's best-known state across ALL runs (store.load_lead_ledger).

    `lead` is the handle's most recent scored Lead; the metadata tracks how it
    moved: prev_score is from the second-most-recent appearance, is_new means
    the handle first appeared in the newest run.
    """

    lead: Lead
    prev_score: float | None = None
    first_seen_run: str = ""
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    times_seen: int = 1
    is_new: bool = False

    @property
    def score_delta(self) -> float | None:
        if self.prev_score is None:
            return None
        return round(self.lead.score - self.prev_score, 1)


# --------------------------------------------------------------- collaboration
# Human judgments carry the same provenance grammar as model verdicts:
# an author, a thesis lens, a rationale, and a timestamp.


class Vote(BaseModel):
    """One partner's stance on one startup — the human counterpart of a
    verdict. Updating a stance replaces the row; history lives in events."""

    handle: str
    actor: str
    stance: str  # strong_yes | yes | unsure | pass
    rationale: str = ""
    thesis_id: str = ""  # the lens the stance was taken under
    created_at: datetime | None = None
    updated_at: datetime | None = None


class VoteSummary(BaseModel):
    """A startup's stance tally (collab.vote_summary)."""

    handle: str
    n_votes: int = 0
    mean: float | None = None  # mean stance value over STANCES
    spread: float = 0.0  # max - min stance value; disagreement measure
    contested: bool = False  # 2+ votes spanning yes-ish vs pass
    by_actor: dict[str, str] = Field(default_factory=dict)  # actor -> stance
    latest_at: datetime | None = None


class Comment(BaseModel):
    """One comment on a startup (optionally anchored to a memo version)."""

    id: int | None = None
    handle: str
    actor: str
    body: str
    mentions: list[str] = Field(default_factory=list)
    memo_version_id: int | None = None
    created_at: datetime | None = None
    edited_at: datetime | None = None
    deleted_at: datetime | None = None


class Event(BaseModel):
    """One row of the append-only activity spine (store._append_event)."""

    id: int | None = None
    at: datetime | None = None
    actor: str
    verb: str
    handle: str | None = None
    thesis_id: str | None = None
    payload: dict = Field(default_factory=dict)


class MemoVersion(BaseModel):
    """An immutable memo snapshot. pipeline.brief stays the CURRENT memo;
    every generation and every edit appends a version, so regeneration can
    never destroy a human's edits again."""

    id: int | None = None
    handle: str
    version_no: int
    body: str
    meta: dict = Field(default_factory=dict)
    author: str = ""  # "agent:memo" or a member's email
    kind: str = "generated"  # generated | edited
    created_at: datetime | None = None
