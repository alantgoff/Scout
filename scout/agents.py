"""Claude-powered workflow agents for the VC loop.

Two agents:

- **Strategy agent** (`generate_strategy`): natural-language thesis in, a
  complete sourcing configuration out — keywords, departure markers, the whole
  X query bank, bio searches, GitHub topics, watchlist suggestions, and stage
  targeting, each with the reasoning. `apply_strategy` merges a proposal into
  Thesis/Seeds objects (pure — callers persist the yaml).

- **Research-brief agent** (`research_brief`): one scored lead in, a compact
  diligence memo out (what they're building, evidence, thesis fit, risks,
  questions for a first call). Cached by the caller in the pipeline table.

- **Weight-tuning agent** (`suggest_weights`): triage statistics in
  (scout.insights), a reviewed-before-apply weight proposal out — the
  feedback loop from shortlist/pass decisions back into scoring.

All degrade gracefully without an Anthropic key. Every call carries an
explicit client timeout (the UI runs these inline — a hung call must not
wedge the app; Streamlit's stop button is the cancel path).
"""

from __future__ import annotations

import asyncio
import json

import anthropic
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import STAGES, Seeds, Settings, Thesis
from scout.models import Lead

PARSE_ATTEMPTS = 3

_CORRECTIVE_NOTE = (
    "\n\nIMPORTANT: your previous reply could not be parsed. Respond with ONLY "
    "a valid JSON object exactly as specified — no prose, no markdown fences."
)


class StrategyProposal(BaseModel):
    """A complete sourcing configuration proposed by the strategy agent."""

    thesis: str = ""
    rationale: str = ""  # the agent's reasoning, shown before applying
    target_stages: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    target_bios: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    launch_phrases: list[str] = Field(default_factory=list)
    searches_departure: list[str] = Field(default_factory=list)
    searches_stealth_intent: list[str] = Field(default_factory=list)
    searches_hiring: list[str] = Field(default_factory=list)
    searches_launch: list[str] = Field(default_factory=list)
    bio_searches: list[str] = Field(default_factory=list)
    github_topics: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)  # suggested investor handles


_STRATEGY_SYSTEM = """You are a venture sourcing strategist configuring "scout", \
a person-centric founder-sourcing engine for Twitter/X, GitHub, and Hacker News.

scout's signals and what each config list drives:
- keywords -> bio_intent: founder-intent phrases matched against X bios ("stealth", "building something new")
- target_bios -> departure_signal: literal bio substrings marking departures from target orgs ("ex-OpenAI", "prev @"). Include the marker in the entry itself.
- launch_phrases -> launch_traction: launch language matched in recent tweets
- disqualifiers: any of these in a bio drops the account entirely
- sectors: context for the LLM classifier and Hacker News search terms
- target_stages: subset of ["idea", "stealth", "launched", "scaling"] — decides which query categories run and penalizes off-stage leads
- searches_*: X search queries, run daily. Syntax: quote exact phrases; OR must be uppercase; () group; min_faves:N filters engagement. Write high-precision queries a founder would actually tweet — generic queries drown the pipeline in noise.
  - searches_departure: people announcing they left a target org (earliest signal)
  - searches_stealth_intent: new-venture intent language
  - searches_hiring: founding-team hiring posts (a founder hiring reveals themselves)
  - searches_launch: just-launched announcements
- bio_searches: short bio keyword phrases for X people-search ("ex-openai", "stealth ai")
- github_topics: GitHub repo topics to scan for fresh, starred repos (lowercase-hyphenated existing topic names)
- watchlist: X handles of investors/operators whose NEW follows are treated as signal (follow-graph diffing). Suggest well-known, real, active accounts relevant to the thesis; never invent handles.

Given the user's thesis, produce the complete configuration. Be surgical: 4-8 entries per search list, 5-10 keywords, 5-12 target_bios, 3-6 github_topics, 8-15 watchlist handles. Also write:
- thesis: the thesis restated in one crisp sentence
- rationale: 3-5 sentences explaining the strategy — what the highest-precision signals for this thesis are and why these queries/watchlist capture them

Respond with ONLY a JSON object:
{"thesis": str, "rationale": str, "target_stages": [str], "keywords": [str], "target_bios": [str], "sectors": [str], "disqualifiers": [str], "launch_phrases": [str], "searches_departure": [str], "searches_stealth_intent": [str], "searches_hiring": [str], "searches_launch": [str], "bio_searches": [str], "github_topics": [str], "watchlist": [str]}"""


_BRIEF_SYSTEM = """You are a venture analyst writing an internal pre-call brief on a \
startup lead sourced from X. Using ONLY the provided data (do not invent facts — \
say "unknown" where the data is silent), write a tight markdown memo:

**What they're building** — 1-2 sentences, specific.
**Evidence** — bullet the concrete signals in the data (bio language, launch tweets + engagement, investor follows, GitHub, multi-source discovery) and what each implies.
**Thesis fit** — 2-3 sentences judging fit against the fund's thesis; name the strongest match and the biggest gap.
**Risks / open questions** — 2-4 bullets of what could make this uninvestable or is unknown.
**First-call questions** — 3 sharp questions an investor should ask.

Under 250 words. No preamble, no title. Section labels must be bold text
(**like this**) exactly as shown — never markdown # headings."""


# Explicit ceilings per agent — these run inline in the Streamlit process.
STRATEGY_TIMEOUT_S = 120.0
BRIEF_TIMEOUT_S = 60.0
WEIGHTS_TIMEOUT_S = 60.0


def _client(settings: Settings, timeout: float) -> anthropic.Anthropic:
    # max_retries=1: tenacity is the retry layer here — stacking the SDK's
    # default retries on top multiplies worst-case latency.
    return anthropic.Anthropic(
        api_key=settings.anthropic_api_key, timeout=timeout, max_retries=1
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=30),
    # Timeouts are excluded from retry: with a 60-120s client timeout, three
    # attempts would wedge the UI for many minutes. Fail fast instead.
    retry=(
        retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError))
        | (
            retry_if_exception_type(anthropic.APIConnectionError)
            & retry_if_not_exception_type(anthropic.APITimeoutError)
        )
    ),
    reraise=True,
)
def _call(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
) -> str:
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


def parse_strategy(text: str) -> StrategyProposal:
    """Parse the agent's JSON into a proposal. Pure — unit-testable.

    Unknown stages are dropped (never let a hallucinated stage into
    thesis.yaml); an empty result falls back to all stages at apply time.
    """
    data = json.loads(_strip_code_fences(text))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    proposal = StrategyProposal.model_validate(data)
    proposal.target_stages = [s for s in proposal.target_stages if s in STAGES]
    proposal.watchlist = [h.lstrip("@").strip() for h in proposal.watchlist if h.strip()]
    return proposal


def generate_strategy(
    description: str, thesis: Thesis, seeds: Seeds, settings: Settings
) -> StrategyProposal:
    """Turn a natural-language thesis into a full sourcing configuration.

    Raises RuntimeError when no Anthropic key is configured or the reply
    can't be parsed — the strategy agent has no meaningful offline fallback.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — the strategy agent needs Claude."
        )
    client = _client(settings, STRATEGY_TIMEOUT_S)
    context = f"Thesis: {description.strip()}"
    if thesis.thesis and thesis.thesis.strip() != description.strip():
        context += f"\n\n(Current thesis statement, for reference: {thesis.thesis})"
    if seeds.watchers:
        context += (
            "\n\nCurrent watchlist (keep entries that still fit, drop ones that "
            "don't, add better ones): " + ", ".join(seeds.watchers)
        )

    prompt = context
    last_error: Exception | None = None
    try:
        for _ in range(PARSE_ATTEMPTS):
            text = _call(client, settings.claude_model, _STRATEGY_SYSTEM, prompt)
            try:
                return parse_strategy(text)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                prompt = context + _CORRECTIVE_NOTE
    except anthropic.APITimeoutError:
        raise RuntimeError(
            f"The strategy agent timed out after {STRATEGY_TIMEOUT_S:.0f}s — try again."
        ) from None
    raise RuntimeError(f"strategy agent returned unparseable output: {last_error}")


def apply_strategy(
    proposal: StrategyProposal, thesis: Thesis, seeds: Seeds
) -> tuple[Thesis, Seeds]:
    """Merge a proposal into fresh Thesis/Seeds copies (pure; caller persists).

    Weights, signal_params, and llm_prompt are deliberately untouched — the
    agent proposes *targeting*, not scoring calibration. Empty proposal lists
    keep the current value rather than wiping it.
    """
    def keep(new: list[str], current: list[str]) -> list[str]:
        return new if new else current

    new_thesis = thesis.model_copy(
        update={
            "thesis": proposal.thesis or thesis.thesis,
            "target_stages": proposal.target_stages or thesis.target_stages or list(STAGES),
            "keywords": keep(proposal.keywords, thesis.keywords),
            "target_bios": keep(proposal.target_bios, thesis.target_bios),
            "sectors": keep(proposal.sectors, thesis.sectors),
            "disqualifiers": keep(proposal.disqualifiers, thesis.disqualifiers),
            "launch_phrases": keep(proposal.launch_phrases, thesis.launch_phrases),
        }
    )
    new_seeds = seeds.model_copy(
        update={
            "searches_departure": keep(proposal.searches_departure, seeds.searches_departure),
            "searches_stealth_intent": keep(
                proposal.searches_stealth_intent, seeds.searches_stealth_intent
            ),
            "searches_hiring": keep(proposal.searches_hiring, seeds.searches_hiring),
            "searches_launch": keep(proposal.searches_launch, seeds.searches_launch),
            "bio_searches": keep(proposal.bio_searches, seeds.bio_searches),
            "github_topics": keep(proposal.github_topics, seeds.github_topics),
            "watchlist": keep(proposal.watchlist, seeds.watchers),
            "tastemakers": [],
        }
    )
    return new_thesis, new_seeds


def validate_watchlist(
    handles: list[str], settings: Settings, store
) -> tuple[list[str], list[str], bool]:
    """Check proposed watchlist handles actually exist on X (free, twscrape).

    Returns (keep, invalid, validated):
    - keep     — handles that resolved, plus ones whose lookup errored
                 (benefit of the doubt: never drop a handle on a flaky call)
    - invalid  — handles that definitively do not exist (the agent's failure
                 mode is plausible-looking fabrications, e.g. "@anthrpic_fund")
    - validated — False when validation couldn't run at all (no twscrape
                 cookies, or every lookup errored); keep is then all handles.
    """
    cleaned = [h.lstrip("@").strip() for h in handles if h.strip()]
    if not cleaned:
        return [], [], True
    try:
        from scout.ingest.twscrape_src import TwscrapeSource

        adapter = TwscrapeSource(settings, store)
    except RuntimeError:
        return cleaned, [], False  # no cookies configured — can't validate

    async def _check_all() -> list[tuple[str, str]]:
        semaphore = asyncio.Semaphore(5)

        async def one(handle: str) -> tuple[str, str]:
            async with semaphore:
                try:
                    account = await adapter.fetch_account(handle)
                except Exception:
                    return handle, "error"
                return handle, "found" if account is not None else "missing"

        return await asyncio.gather(*(one(h) for h in cleaned))

    results = asyncio.run(_check_all())
    if all(status == "error" for _, status in results):
        return cleaned, [], False  # systemic failure (bad cookies, network)
    keep = [h for h, status in results if status in ("found", "error")]
    invalid = [h for h, status in results if status == "missing"]
    return keep, invalid, True


def _brief_context(lead: Lead, thesis: Thesis) -> str:
    account = lead.account
    verdict = lead.llm
    lines = [
        f"Fund thesis: {thesis.thesis or '(none set)'}",
        "",
        f"Handle: @{account.handle} ({account.name or 'name unknown'})",
        f"Bio: {account.bio or '(empty)'}",
        f"Followers: {account.followers:,} · Following: {account.following:,}",
        f"Website: {account.website or 'none'}",
        f"Discovered via: {', '.join(account.sources or [account.source]) or 'unknown'}",
        f"Score: {lead.score:.0f}/100",
    ]
    if account.followed_by:
        lines.append(f"Followed by watchlist investors: {', '.join(account.followed_by)}")
    if account.recent_followed_by:
        lines.append(
            f"NEWLY followed (this window) by: {', '.join(account.recent_followed_by)}"
        )
    if account.github_repo:
        lines.append(f"GitHub evidence: {account.github_repo}")
    hits = [s for s in lead.signals if s.value > 0]
    if hits:
        lines.append(
            "Signals: "
            + "; ".join(f"{s.name}={s.value:.2f} ({s.detail})" if s.detail else f"{s.name}={s.value:.2f}" for s in hits)
        )
    if verdict:
        lines += [
            f"Classifier: {verdict.account_type or '?'} · stage {verdict.stage or '?'} · "
            f"{verdict.sector or '?'} / {verdict.subsector or '?'} · {verdict.business_model or '?'}",
            f"Summary: {verdict.one_line_summary or '(none)'}",
            f"Why interesting: {verdict.why_interesting or '(none)'}",
        ]
        if verdict.thesis_fit is not None:
            lines.append(f"Thesis fit: {verdict.thesis_fit:.2f} — {verdict.fit_reason}")
        if verdict.tags:
            lines.append(f"Tags: {', '.join(verdict.tags)}")
    return "\n".join(lines)


def _brief_template(lead: Lead) -> str:
    account = lead.account
    verdict = lead.llm
    what = (verdict.one_line_summary if verdict else "") or account.bio or "unknown"
    hits = ", ".join(s.name for s in lead.signals if s.value > 0) or "none"
    return (
        f"**What they're building** — {what}\n\n"
        f"**Evidence** — signals hit: {hits}. "
        f"{account.followers:,} followers. {account.url}\n\n"
        f"**Thesis fit** — no AI verdict available (add ANTHROPIC_API_KEY for a full brief).\n\n"
        f"**First-call questions** — What are you building and why now? "
        f"What's the unique data advantage? Who is the first customer?"
    )


class WeightProposal(BaseModel):
    """Adjusted signal weights proposed by the weight-tuning agent."""

    weights: dict[str, float]
    rationale: str = ""


_WEIGHTS_SYSTEM = """You are calibrating the signal weights of a founder-sourcing engine for a VC.
The lead score is 100 × Σ(signal value × weight) / Σ(all weights) — weights are RELATIVE, each 0-50.
You get the current weights plus statistics contrasting the leads the investor SHORTLISTED
against the leads they PASSED on (mean points per signal, sector/stage/business-model mix,
thesis-fit averages).

Propose adjusted weights that better rank what this investor actually shortlists:
- keep every existing signal name exactly as given; do not invent new ones
- values 0-50; move a weight only where the statistics support it
- when a signal contributes more points to passes than shortlists, lower it; the reverse, raise it
- prefer small moves (±5-10) unless the contrast is stark
- rationale: 2-4 sentences citing the specific statistics that drove each change

Respond with ONLY a JSON object, no other text:
{"weights": {"signal_name": number, ...}, "rationale": str}"""


def parse_weight_proposal(text: str, current: dict[str, float]) -> WeightProposal:
    """Parse + sanitize the agent's JSON. Pure — unit-testable.

    Unknown signal names are dropped, values clamped to 0-50, and any signal
    the model forgot keeps its current weight (a proposal must never silently
    delete a signal from thesis.yaml).
    """
    data = json.loads(_strip_code_fences(text))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    proposal = WeightProposal.model_validate(data)
    cleaned: dict[str, float] = {}
    for name, value in proposal.weights.items():
        if name in current:
            cleaned[name] = min(max(float(value), 0.0), 50.0)
    if not cleaned:
        raise ValueError("no known signal names in the proposal")
    for name, value in current.items():
        cleaned.setdefault(name, value)
    proposal.weights = cleaned
    return proposal


def suggest_weights(
    stats_text: str, thesis: Thesis, settings: Settings
) -> WeightProposal:
    """Turn triage statistics into a reviewed-before-apply weight proposal.

    Raises RuntimeError when no Anthropic key is configured, on timeout, or
    when the reply can't be parsed.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — weight suggestions need Claude."
        )
    client = _client(settings, WEIGHTS_TIMEOUT_S)
    context = (
        f"Current weights: {json.dumps(thesis.weights, sort_keys=True)}\n"
        f"Thesis: {thesis.thesis}\n\nTriage statistics:\n{stats_text}"
    )
    prompt = context
    last_error: Exception | None = None
    try:
        for _ in range(PARSE_ATTEMPTS):
            text = _call(client, settings.claude_model, _WEIGHTS_SYSTEM, prompt,
                         max_tokens=1000)
            try:
                return parse_weight_proposal(text, thesis.weights)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                prompt = context + _CORRECTIVE_NOTE
    except anthropic.APITimeoutError:
        raise RuntimeError(
            f"The weight agent timed out after {WEIGHTS_TIMEOUT_S:.0f}s — try again."
        ) from None
    raise RuntimeError(f"weight agent returned unparseable output: {last_error}")


def research_brief(
    lead: Lead, thesis: Thesis, settings: Settings
) -> tuple[str, bool]:
    """Return (markdown_brief, is_ai). Falls back to a data-only template
    when no Anthropic key is configured or the API call fails."""
    if not settings.anthropic_api_key:
        return _brief_template(lead), False
    try:
        client = _client(settings, BRIEF_TIMEOUT_S)
        brief = _call(
            client,
            settings.claude_model,
            _BRIEF_SYSTEM,
            _brief_context(lead, thesis),
            max_tokens=1500,
        )
        return brief, True
    except anthropic.APIError:
        return _brief_template(lead), False
