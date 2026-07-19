"""Pydantic models — the only data structures passed between modules."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["idea", "stealth", "launched", "scaling"]


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
