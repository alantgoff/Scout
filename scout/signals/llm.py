"""Claude classification of heuristic-passing candidates.

The caller gates on >=1 heuristic hit; this module classifies whatever it is
given, in batches of 10, with batches running concurrently (the Anthropic
client is thread-safe). Verdicts are cached in the store keyed by a
fingerprint of the inputs (bio + tweets + thesis + model), so re-runs while
tuning weights cost nothing. Missing API key degrades gracefully to {}.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from urllib.parse import urlparse

from scout import rubric
from scout.config import Settings, Thesis
from scout.models import Account, Lead, LLMVerdict, SitePage, Tweet
from scout.store import Store

console = Console()

BATCH_SIZE = 5  # default accounts per call; settings.classify_batch_size wins
# Verdicts below this confidence used to be DROPPED — which handed the lead
# its full heuristic score back and made honest uncertainty score HIGHER
# than a grounded verdict. Now every verdict attaches and score ×= confidence
# does the sinking; the constant remains only as the verification-pass floor.
MIN_CONFIDENCE = 0.4
RECENT_TWEETS = 10
TWEET_MAX_CHARS = 400
PARSE_ATTEMPTS = 3  # initial call + 2 corrective retries

_CORRECTIVE_NOTE = (
    "\n\nIMPORTANT: your previous reply could not be parsed. Respond with ONLY a "
    "valid JSON array of account objects exactly as specified — no prose, no "
    "markdown code fences, no trailing commentary."
)


# Default classification prompt. Editable per-thesis via thesis.llm_prompt
# (UI: Thesis → Signals & scoring); placeholders {thesis} {sectors} {stages}
# {firm} {value_add} are substituted.
DEFAULT_PROMPT_TEMPLATE = """You are a venture analyst at {firm} screening Twitter/X accounts for startup leads.
Investment thesis: {thesis}
Sectors of interest: {sectors}
Target stages: {stages}

{firm}'s strategic value-add — what the firm specifically offers portfolio companies:
{value_add}

For each account in the user message decide:
- account_type: "founder" (a person building a company), "startup" (the company's own account), or "other" (corporate account, investor, commentator, hobbyist)
- is_founder: true only for "founder" or "startup" accounts worth a VC conversation
- stage: how far along they are ("idea", "stealth", "launched", "scaling")
- company_name: the STARTUP behind the account, when identifiable from the bio/tweets/website — the product or company name a VC would put in a memo. null when genuinely unknown (deep stealth). Never invent one.
- company_url: the company/product website if visible in the data, else null.
- product_summary: 1-2 sentences stating what the company ACTUALLY does, naming the evidence source (e.g. website: "AI agent monitoring…"). null when the product cannot be established from the evidence.
- grounding: the strongest evidence that established the product — "website" | "pinned_tweet" | "tweets" | "github" | "bio" | "none"
- sector: the broad sector, and subsector: the finest slice you can name (e.g. sector "ai infra", subsector "agent evals"). null when the product itself is not established.
- business_model: one of "b2b saas", "devtools", "infra", "consumer", "marketplace", "api", "open source", "hardware", "services", "other" — or null when the product is not established
- thesis_fit: 0.0-1.0 — how squarely the PRODUCT matches the investment thesis above. 1.0 = textbook match on space and stage; 0.5 = adjacent; 0.0 = unrelated. Judge against the thesis, not general quality.
- fit_reason: one short sentence justifying thesis_fit, citing the evidence
- value_add_levers: an object mapping EACH lever key listed above to 0.0-1.0 — how much that specific lever would accelerate this startup (0.0 = irrelevant to them, 1.0 = exactly what they need next)
- value_add_fit: 0.0-1.0 — overall, how much {firm}'s value-add above would accelerate this startup. Judge independently of thesis_fit: a lead can match the thesis yet need nothing the firm uniquely offers, or vice versa.
- value_add_reason: one short sentence naming the strongest lever(s) and why they apply
- customer_type: who the company sells to — "b2b" (businesses), "b2c" (consumers), "b2b2c" (businesses that serve consumers), "mixed", or null when the product itself is not established. This routes WHICH scorecard below you fill in.

{scorecard}

- tags: 2-5 lowercase descriptors a VC would filter on (e.g. "rl environments", "ex-deepmind", "open source", "seed stage")
- one_line_summary: what they are building, specifically
- why_interesting: why (or why not) worth a VC conversation
- confidence: 0-1 confidence in this classification overall

Respond with ONLY a JSON array, one object per account, no other text:
[{{"handle": str, "account_type": "founder"|"startup"|"other", "is_founder": bool, "stage": "idea"|"stealth"|"launched"|"scaling", "company_name": str|null, "company_url": str|null, "product_summary": str|null, "grounding": "website"|"pinned_tweet"|"tweets"|"github"|"bio"|"none", "customer_type": "b2b"|"b2c"|"b2b2c"|"mixed"|null, "scorecard": {{"criterion_key": 1|2|3}} (OMIT criteria without evidence), "scorecard_reasons": {{"criterion_key": str}}, "sector": str|null, "subsector": str|null, "business_model": str|null, "thesis_fit": 0-1, "fit_reason": str, "value_add_levers": {{"lever_key": 0-1}}, "value_add_fit": 0-1, "value_add_reason": str, "tags": [str], "one_line_summary": str, "why_interesting": str, "confidence": 0-1}}]"""


# Non-negotiable grounding rules, appended to EVERY system prompt — including
# user-customized thesis.llm_prompt templates. They are the whole point of
# grounded classification and must not be editable away silently.
EVIDENCE_RULES = """
EVIDENCE RULES — these override everything above:
- Evidence hierarchy, strongest first: WEBSITE TEXT > pinned tweet > recent
  tweets > GitHub repo > bio. The bio says who they are; only product
  evidence says what they build.
- A founder's past employers (Apple, SpaceX, Nvidia, OpenAI, ...) say NOTHING
  about what their company builds. NEVER infer sector, subsector,
  business_model, or thesis_fit from pedigree. An "ex-Apple silicon engineer"
  with no product evidence is sector null — not "hardware", not "edge AI".
- product_summary must state what the company actually does and name the
  evidence source. If you cannot write it from evidence, set it null.
- grounding names the strongest evidence that established the product. If
  nothing did, use "none" (or "bio" when only the bio hints at it).
- UNKNOWN ESCAPE: when the evidence does not establish the product, set
  sector, subsector, business_model, and product_summary to null, grounding
  to "none", confidence <= 0.3, and write an honest one_line_summary of what
  IS known (e.g. "Stealth; founder ex-Apple; product not yet public").
  A wrong guess is worse than an honest unknown.
- thesis_fit judges the PRODUCT against the thesis. Product unknown ->
  thesis_fit <= 0.3, no matter how impressive the founder.
- SCORECARD: score a criterion only from evidence you can cite in
  scorecard_reasons; OMIT the key otherwise. Never award scores for polish,
  hype, or pedigree-by-association. The founders & cap table criteria are
  the ONLY ones where founder background counts — cited concretely — and
  founder background never establishes sector, product, market, or any
  other criterion.
- Product unknown -> customer_type null and the scorecard limited to
  founders & cap table criteria at most (concrete founder history); every
  other criterion omitted.
- Website text may be in any language, or thin (JS-heavy sites). Thin or
  missing website text is weak evidence — do not stretch it. A website
  marked unreachable is NOT evidence of anything.
"""


def _system_prompt(thesis: Thesis) -> str:
    template = thesis.llm_prompt.strip() or DEFAULT_PROMPT_TEMPLATE
    context = {
        "thesis": thesis.thesis,
        "sectors": ", ".join(thesis.sectors),
        "stages": ", ".join(thesis.target_stages),
        "firm": thesis.firm_name or "the firm",
        "value_add": "\n".join(
            f"- {lever.key} — {lever.label}: {lever.description}"
            for lever in thesis.firm_value_add
        )
        or "(no value-add levers defined)",
        # Both readiness scorecards, rendered from the scout.rubric registry
        # so the prompt can never drift from the scoring math.
        "scorecard": rubric.prompt_block(),
    }
    try:
        rendered = template.format(**context)
    except (KeyError, IndexError, ValueError):
        # A custom prompt with stray braces shouldn't break classification —
        # fall back to naive replacement of the documented placeholders.
        rendered = template
        for key, value in context.items():
            rendered = rendered.replace("{" + key + "}", value)
    return rendered + "\n" + EVIDENCE_RULES


def _fingerprint(
    account: Account,
    tweets: list[Tweet],
    thesis: Thesis,
    settings: Settings,
    site: SitePage | None = None,
) -> str:
    """Stable hash of everything a verdict depends on. Any change — new bio,
    new tweets, edited thesis/prompt, different model, a changed website —
    invalidates the cache."""
    site_text_hash = (
        hashlib.sha256(site.text.encode("utf-8")).hexdigest()
        if site is not None and site.text
        else (site.status if site is not None else "")
    )
    payload = "\x1f".join(
        [
            account.bio,
            "\x1e".join(t.id for t in tweets[:RECENT_TWEETS]),
            _system_prompt(thesis),
            settings.claude_model,
            account.website or "",
            site_text_hash,
            account.github_repo or "",
            account.pinned_tweet_id or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _site_domain(site: SitePage) -> str:
    return urlparse(site.final_url or site.url).netloc or site.url


def _format_account(
    account: Account,
    tweets: list[Tweet],
    site: SitePage | None = None,
    site_text_chars: int = 6000,
) -> str:
    """One candidate's evidence dossier, strongest product evidence first:
    fetched WEBSITE TEXT (when available), then github, pinned, tweets."""
    lines = [
        f"handle: @{account.handle}",
        f"bio: {account.bio or '(empty)'}",
        f"website: {account.website or '(none listed)'}",
        f"followers: {account.followers}",
    ]
    # Investor-attention evidence for the quality rubric. Like `followers`,
    # follow lists are NOT part of _fingerprint — bounded staleness under the
    # verdict TTL is accepted.
    if account.followed_by:
        lines.append("followed by watchlist investors: "
                     + ", ".join(account.followed_by[:8]))
    if account.recent_followed_by:
        lines.append("newly followed this window by: "
                     + ", ".join(account.recent_followed_by[:8]))
    if account.github_repo:
        lines.append(f"github repo: {account.github_repo}")
    if site is not None and site.usable:
        note = " (thin — JS-heavy site, weak evidence)" if site.status == "thin" else ""
        lines.append(
            f"website text (from {_site_domain(site)}, extracted{note}):\n"
            f"{site.text[:site_text_chars]}"
        )
    elif site is not None and account.website:
        lines.append(
            f"website: listed but unreachable ({site.status}) — "
            "treat the product as UNVERIFIED"
        )
    pinned = None
    if account.pinned_tweet_id:
        pinned = next((t for t in tweets if t.id == account.pinned_tweet_id), None)
    if pinned is not None:
        lines.append(f"pinned tweet: {pinned.text[:TWEET_MAX_CHARS]}")
    recent = [t for t in tweets if pinned is None or t.id != pinned.id]
    for tweet in recent[:RECENT_TWEETS]:
        lines.append(f"recent tweet: {tweet.text[:TWEET_MAX_CHARS]}")
    return "\n".join(lines)


def _user_prompt(
    batch: list[tuple[Account, list[Tweet]]],
    sites: dict[str, SitePage] | None = None,
    site_text_chars: int = 6000,
) -> str:
    sites = sites or {}
    blocks = [
        f"--- account {i} ---\n"
        + _format_account(account, tweets,
                          sites.get(account.handle.lower()), site_text_chars)
        for i, (account, tweets) in enumerate(batch, start=1)
    ]
    return (
        f"Classify these {len(batch)} accounts:\n\n" + "\n\n".join(blocks)
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type(
        (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        )
    ),
    reraise=True,
)
def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        # The system prompt (thesis + both scorecards, ~4k tokens) is
        # identical across every batch of a run — cache it server-side.
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return next(b.text for b in response.content if b.type == "text")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


def _parse_verdicts(text: str) -> list[LLMVerdict]:
    data = json.loads(_strip_code_fences(text))
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of account objects")
    verdicts = []
    for obj in data:
        verdict = LLMVerdict.model_validate(obj)
        verdict.handle = verdict.handle.lstrip("@")
        verdicts.append(verdict)
    return verdicts


def _classify_batch(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    batch: list[tuple[Account, list[Tweet]]],
    sites: dict[str, SitePage] | None = None,
    site_text_chars: int = 6000,
) -> list[LLMVerdict]:
    base_prompt = _user_prompt(batch, sites, site_text_chars)
    prompt = base_prompt
    # Richer dossiers (site text, 10 tweets) plus the per-criterion scorecard
    # need real output headroom per account — a truncated JSON array is a
    # wasted corrective retry.
    max_tokens = min(16384, max(2048, 2500 * len(batch)))
    last_error: Exception | None = None
    for _ in range(PARSE_ATTEMPTS):
        text = _call_claude(client, model, system_prompt, prompt, max_tokens)
        try:
            return _parse_verdicts(text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            prompt = base_prompt + _CORRECTIVE_NOTE
    console.print(
        f"[yellow]Skipping batch of {len(batch)} accounts: Claude returned "
        f"unparseable output after {PARSE_ATTEMPTS} attempts ({last_error})[/yellow]"
    )
    return []


def _classify_batch_safe(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    batch: list[tuple[Account, list[Tweet]]],
    sites: dict[str, SitePage] | None = None,
    site_text_chars: int = 6000,
) -> list[LLMVerdict]:
    """One batch, all API failure modes reduced to 'skip with a warning' —
    a single bad batch must never sink the run (or its sibling batches)."""
    try:
        return _classify_batch(client, model, system_prompt, batch, sites,
                               site_text_chars)
    except anthropic.RateLimitError:
        console.print(
            "[yellow]Claude rate limit persisted after retries — "
            f"skipping batch of {len(batch)} accounts[/yellow]"
        )
    except anthropic.APIStatusError as exc:
        console.print(
            f"[yellow]Claude API error ({exc.status_code}) — "
            f"skipping batch of {len(batch)} accounts: {exc.message}[/yellow]"
        )
    except anthropic.APIConnectionError:
        console.print(
            "[yellow]Could not reach the Claude API after retries — "
            f"skipping batch of {len(batch)} accounts[/yellow]"
        )
    return []


def classify(
    candidates: list[tuple[Account, list[Tweet]]],
    thesis: Thesis,
    settings: Settings,
    store: Store | None = None,
    progress: Callable[[int, int], None] | None = None,
    sites: dict[str, SitePage] | None = None,
) -> dict[str, LLMVerdict]:
    """Classify candidates with Claude; returns verdicts keyed by lowercased handle.

    `sites` (lowercased handle → SitePage) grounds each dossier in the
    company's actual website text — the strongest product evidence.

    Cache-first when a store is provided: an account whose bio/tweets/site/
    thesis/model are unchanged since the last classification reuses its cached
    verdict (TTL settings.verdict_ttl_days). Batches of the remainder run
    concurrently (settings.llm_concurrency). EVERY parsed verdict attaches —
    including low-confidence ones: scoring multiplies by confidence, so an
    honest "unknown" sinks the lead instead of silently restoring its full
    heuristic score. Returns {} (with a warning) when no API key is
    configured — the pipeline runs heuristics-only.

    `progress(done, total)` (optional) fires from the caller's thread as
    accounts finish — cache hits count immediately, then once per completed
    batch — so the UI can draw a live progress bar.
    """
    if not settings.anthropic_api_key:
        console.print(
            "[bold yellow]ANTHROPIC_API_KEY not set — running heuristics-only[/bold yellow]"
        )
        return {}
    if not candidates:
        return {}

    sites = sites or {}
    results: dict[str, LLMVerdict] = {}
    fresh: list[tuple[Account, list[Tweet]]] = []
    fingerprints: dict[str, str] = {}
    cache_hits = 0
    for account, tweets in candidates:
        key = account.handle.lstrip("@").lower()
        fingerprints[key] = _fingerprint(account, tweets, thesis, settings,
                                         sites.get(key))
        cached = (
            store.cached_verdict(key, fingerprints[key], settings.verdict_ttl_days)
            if store is not None
            else None
        )
        if cached is not None:
            cache_hits += 1
            results[key] = cached
        else:
            fresh.append((account, tweets))
    if cache_hits:
        console.print(f"[dim]{cache_hits} verdicts served from cache.[/dim]")
    n_total = len(candidates)
    n_done = cache_hits
    if progress is not None:
        progress(n_done, n_total)
    if not fresh:
        return results

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system_prompt = _system_prompt(thesis)
    batch_size = max(1, settings.classify_batch_size)
    batches = [fresh[i : i + batch_size] for i in range(0, len(fresh), batch_size)]
    workers = max(1, min(settings.llm_concurrency, len(batches)))
    batch_results: list[list[LLMVerdict]] = []
    if workers > 1:
        # submit + as_completed (not pool.map) so progress can fire from THIS
        # thread as each batch lands — the store below is not thread-safe.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _classify_batch_safe,
                    client, settings.claude_model, system_prompt, b,
                    sites, settings.web_text_max_chars,
                ): len(b)
                for b in batches
            }
            for future in as_completed(futures):
                batch_results.append(future.result())
                n_done += futures[future]
                if progress is not None:
                    progress(n_done, n_total)
    else:
        for b in batches:
            batch_results.append(
                _classify_batch_safe(client, settings.claude_model, system_prompt,
                                     b, sites, settings.web_text_max_chars)
            )
            n_done += len(b)
            if progress is not None:
                progress(n_done, n_total)

    for verdicts in batch_results:
        for verdict in verdicts:
            # Lowercased key so call sites match regardless of the casing
            # Claude echoed back. Every parsed verdict attaches — see
            # docstring; confidence does its work in scoring, not here.
            key = verdict.handle.lstrip("@").lower()
            results[key] = verdict
            if store is not None and key in fingerprints:
                store.record_verdict(key, fingerprints[key], verdict)
    return results


# ------------------------------------------------------------- verification
# The adversarial audit: a second Claude pass over each TOP lead's dossier +
# verdict, prompted to catch exactly one failure mode — product claims that
# came from the founder's resume instead of product evidence. (Distinct from
# the `scout verify` CLI command, which is paid X-API re-hydration.)

VERIFY_SYSTEM_TEMPLATE = """You are an adversarial reviewer at {firm} auditing a junior analyst's classification of one startup lead. Your job is to catch speculation.
Investment thesis: {thesis}

You get the lead's evidence dossier (bio, website text, tweets, github) and the analyst's verdict. Check every claim against the evidence:
- Do sector / subsector / business_model come from PRODUCT evidence, or from the founder's resume? Past employers are NOT product evidence.
- Does the website text contradict the verdict? The website wins.
- Is thesis_fit justified by the PRODUCT?
- Is product_summary actually supported by the evidence it cites?
- Is every scorecard criterion score (1-3) backed by the evidence its scorecard_reasons cites? The founders & cap table criteria are the ONLY ones where founder background counts. When correcting the scorecard, return the COMPLETE corrected "scorecard" and "scorecard_reasons" objects (they replace wholesale) and drop any criterion whose evidence does not exist in the dossier. If you correct customer_type across the b2b/b2c boundary, re-issue the scorecard in the OTHER rubric's criterion keys. For "unverifiable", scorecard should be {{}} or founders-criteria-only. (Legacy verdicts carry a flat "quality" dict instead — audit and correct it the same way, with team as the only founder-background dimension.)

Respond with ONLY a JSON object, no other text:
{{"handle": str, "verification": "confirmed"|"corrected"|"unverifiable", "corrections": {{}} or any of {{"sector": str|null, "subsector": str|null, "business_model": str|null, "stage": "idea"|"stealth"|"launched"|"scaling", "thesis_fit": 0-1, "product_summary": str|null, "one_line_summary": str, "grounding": "website"|"pinned_tweet"|"tweets"|"github"|"bio"|"none", "customer_type": "b2b"|"b2c"|"b2b2c"|"mixed"|null, "scorecard": {{"criterion_key": 1|2|3}}, "scorecard_reasons": {{"criterion_key": str}}, "quality": {{"dim": 0-1}}, "quality_reasons": {{"dim": str}}, "confidence": 0-1}}, "note": str}}

"confirmed": the evidence supports the verdict — leave corrections empty.
"corrected": the evidence contradicts part of it — put ONLY the fixed fields in corrections, and say what was wrong in note.
"unverifiable": the evidence cannot establish what the company does — null out speculated fields and include "confidence" <= 0.3 in corrections."""

# Only these verdict fields may be overwritten by an audit. quality/
# quality_reasons stay whitelisted so audits of legacy cached verdicts
# still apply.
_CORRECTION_FIELDS = {
    "sector", "subsector", "business_model", "stage", "thesis_fit",
    "product_summary", "one_line_summary", "grounding", "confidence",
    "customer_type", "quality", "quality_reasons",
    "scorecard", "scorecard_reasons",
}


class VerificationResult(BaseModel):
    """Parsed audit output for one lead."""

    handle: str
    verification: str  # "confirmed" | "corrected" | "unverifiable"
    corrections: dict = Field(default_factory=dict)
    note: str = ""


def parse_verification(text: str) -> VerificationResult:
    """Parse one audit reply (pure — unit-tested)."""
    data = json.loads(_strip_code_fences(text))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    result = VerificationResult.model_validate(data)
    if result.verification not in ("confirmed", "corrected", "unverifiable"):
        raise ValueError(f"bad verification value: {result.verification!r}")
    return result


def apply_verification(verdict: LLMVerdict, result: VerificationResult) -> LLMVerdict:
    """Corrected copy of a verdict (pure — unit-tested). Only whitelisted
    fields apply, re-validated through the model; 'unverifiable' caps
    confidence at MIN_CONFIDENCE × 0.75 even without an explicit correction."""
    updates = {k: v for k, v in result.corrections.items() if k in _CORRECTION_FIELDS}
    updates["verification"] = result.verification
    updates["verification_note"] = result.note
    fixed = LLMVerdict.model_validate({**verdict.model_dump(), **updates})
    if result.verification == "unverifiable":
        fixed.confidence = min(fixed.confidence, 0.3)
    return fixed


def _verify_system(thesis: Thesis) -> str:
    return VERIFY_SYSTEM_TEMPLATE.format(
        firm=thesis.firm_name or "the firm", thesis=thesis.thesis
    )


def _verify_user(lead: Lead, tweets: list[Tweet], site: SitePage | None,
                 site_text_chars: int) -> str:
    verdict = lead.llm
    assert verdict is not None
    claim = verdict.model_dump(include={
        "handle", "account_type", "stage", "company_name", "company_url",
        "product_summary", "grounding", "customer_type", "quality",
        "quality_reasons", "scorecard", "scorecard_reasons", "sector",
        "subsector", "business_model", "thesis_fit", "fit_reason",
        "one_line_summary", "confidence",
    })
    return (
        "DOSSIER:\n"
        + _format_account(lead.account, tweets, site, site_text_chars)
        + "\n\nANALYST VERDICT:\n"
        + json.dumps(claim, ensure_ascii=False)
    )


def _verify_one(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    lead: Lead,
    tweets: list[Tweet],
    site: SitePage | None,
    site_text_chars: int,
) -> VerificationResult | None:
    base = _verify_user(lead, tweets, site, site_text_chars)
    prompt = base
    last_error: Exception | None = None
    for _ in range(PARSE_ATTEMPTS):
        # 3000: a wholesale scorecard correction (all criteria + citations)
        # would not fit in the old 1500.
        text = _call_claude(client, model, system, prompt, max_tokens=3000)
        try:
            return parse_verification(text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            prompt = base + (
                "\n\nIMPORTANT: your previous reply could not be parsed. "
                "Respond with ONLY the JSON object exactly as specified."
            )
    console.print(
        f"[yellow]Audit unparseable for @{lead.account.handle} "
        f"({last_error}) — leaving the verdict as-is.[/yellow]"
    )
    return None


def verify_leads(
    leads: list[Lead],
    tweets_by_handle: dict[str, list[Tweet]],
    sites: dict[str, SitePage] | None,
    thesis: Thesis,
    settings: Settings,
    store: Store | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Audit each lead's verdict against its dossier; mutate lead.llm with the
    outcome and RE-RECORD the corrected verdict under the same fingerprint —
    so the cache serves the corrected verdict on every later run.

    `tweets_by_handle` and `sites` are keyed by lowercased handle. Failures
    (API errors, unparseable output) leave the verdict untouched."""
    targets = [x for x in leads if x.llm is not None]
    if not targets or not settings.anthropic_api_key:
        return
    sites = sites or {}
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system = _verify_system(thesis)
    if progress is not None:
        progress(0, len(targets))

    def audit(lead: Lead) -> tuple[Lead, VerificationResult | None]:
        key = lead.account.handle.lower()
        try:
            return lead, _verify_one(
                client, settings.claude_model, system, lead,
                tweets_by_handle.get(key, []), sites.get(key),
                settings.web_text_max_chars,
            )
        except (anthropic.APIError, anthropic.APIConnectionError) as exc:
            console.print(
                f"[yellow]Audit failed for @{lead.account.handle} "
                f"({type(exc).__name__}) — leaving the verdict as-is.[/yellow]"
            )
            return lead, None

    n_done = 0
    workers = max(1, min(settings.llm_concurrency, len(targets)))
    # as_completed on the caller's thread — the store is not thread-safe.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit, lead) for lead in targets]
        for future in as_completed(futures):
            lead, result = future.result()
            n_done += 1
            if result is not None and lead.llm is not None:
                lead.llm = apply_verification(lead.llm, result)
                if store is not None:
                    key = lead.account.handle.lower()
                    fingerprint = _fingerprint(
                        lead.account, tweets_by_handle.get(key, []),
                        thesis, settings, sites.get(key),
                    )
                    store.record_verdict(key, fingerprint, lead.llm)
            if progress is not None:
                progress(n_done, len(targets))
