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
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import ValidationError
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Settings, Thesis
from scout.models import Account, LLMVerdict, Tweet
from scout.store import Store

console = Console()

BATCH_SIZE = 10
MIN_CONFIDENCE = 0.4
RECENT_TWEETS = 5
TWEET_MAX_CHARS = 280
PARSE_ATTEMPTS = 3  # initial call + 2 corrective retries

_CORRECTIVE_NOTE = (
    "\n\nIMPORTANT: your previous reply could not be parsed. Respond with ONLY a "
    "valid JSON array of account objects exactly as specified — no prose, no "
    "markdown code fences, no trailing commentary."
)


# Default classification prompt. Editable per-thesis via thesis.llm_prompt
# (UI: Sourcing → Signals & scoring); placeholders {thesis} {sectors} {stages}
# are substituted.
DEFAULT_PROMPT_TEMPLATE = """You are a venture analyst screening Twitter/X accounts for startup leads.
Investment thesis: {thesis}
Sectors of interest: {sectors}
Target stages: {stages}

For each account in the user message decide:
- account_type: "founder" (a person building a company), "startup" (the company's own account), or "other" (corporate account, investor, commentator, hobbyist)
- is_founder: true only for "founder" or "startup" accounts worth a VC conversation
- stage: how far along they are ("idea", "stealth", "launched", "scaling")
- company_name: the STARTUP behind the account, when identifiable from the bio/tweets/website — the product or company name a VC would put in a memo. null when genuinely unknown (deep stealth). Never invent one.
- company_url: the company/product website if visible in the data, else null.
- sector: the broad sector, and subsector: the finest slice you can name (e.g. sector "ai infra", subsector "agent evals")
- business_model: one of "b2b saas", "devtools", "infra", "consumer", "marketplace", "api", "open source", "hardware", "services", "other"
- thesis_fit: 0.0-1.0 — how squarely this account matches the investment thesis above. 1.0 = textbook match on space, stage, and founder profile; 0.5 = adjacent; 0.0 = unrelated. Judge against the thesis, not general quality.
- fit_reason: one short sentence justifying thesis_fit
- tags: 2-5 lowercase descriptors a VC would filter on (e.g. "rl environments", "ex-deepmind", "open source", "seed stage")
- one_line_summary: what they are building, specifically
- why_interesting: why (or why not) worth a VC conversation
- confidence: 0-1 confidence in this classification overall

Respond with ONLY a JSON array, one object per account, no other text:
[{{"handle": str, "account_type": "founder"|"startup"|"other", "is_founder": bool, "stage": "idea"|"stealth"|"launched"|"scaling", "company_name": str|null, "company_url": str|null, "sector": str, "subsector": str, "business_model": str, "thesis_fit": 0-1, "fit_reason": str, "tags": [str], "one_line_summary": str, "why_interesting": str, "confidence": 0-1}}]"""


def _system_prompt(thesis: Thesis) -> str:
    template = thesis.llm_prompt.strip() or DEFAULT_PROMPT_TEMPLATE
    context = {
        "thesis": thesis.thesis,
        "sectors": ", ".join(thesis.sectors),
        "stages": ", ".join(thesis.target_stages),
    }
    try:
        return template.format(**context)
    except (KeyError, IndexError, ValueError):
        # A custom prompt with stray braces shouldn't break classification —
        # fall back to naive replacement of the documented placeholders.
        rendered = template
        for key, value in context.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered


def _fingerprint(
    account: Account, tweets: list[Tweet], thesis: Thesis, settings: Settings
) -> str:
    """Stable hash of everything a verdict depends on. Any change — new bio,
    new tweets, edited thesis/prompt, different model — invalidates the cache."""
    payload = "\x1f".join(
        [
            account.bio,
            "\x1e".join(t.id for t in tweets[:RECENT_TWEETS]),
            _system_prompt(thesis),
            settings.claude_model,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_account(account: Account, tweets: list[Tweet]) -> str:
    lines = [
        f"handle: @{account.handle}",
        f"bio: {account.bio or '(empty)'}",
        f"website: {account.website or '(none)'}",
        f"followers: {account.followers}",
    ]
    pinned = None
    if account.pinned_tweet_id:
        pinned = next((t for t in tweets if t.id == account.pinned_tweet_id), None)
    if pinned is not None:
        lines.append(f"pinned tweet: {pinned.text[:TWEET_MAX_CHARS]}")
    recent = [t for t in tweets if pinned is None or t.id != pinned.id]
    for tweet in recent[:RECENT_TWEETS]:
        lines.append(f"recent tweet: {tweet.text[:TWEET_MAX_CHARS]}")
    return "\n".join(lines)


def _user_prompt(batch: list[tuple[Account, list[Tweet]]]) -> str:
    blocks = [
        f"--- account {i} ---\n{_format_account(account, tweets)}"
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
    client: anthropic.Anthropic, model: str, system_prompt: str, user_prompt: str
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
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
) -> list[LLMVerdict]:
    base_prompt = _user_prompt(batch)
    prompt = base_prompt
    last_error: Exception | None = None
    for _ in range(PARSE_ATTEMPTS):
        text = _call_claude(client, model, system_prompt, prompt)
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
) -> list[LLMVerdict]:
    """One batch, all API failure modes reduced to 'skip with a warning' —
    a single bad batch must never sink the run (or its sibling batches)."""
    try:
        return _classify_batch(client, model, system_prompt, batch)
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
) -> dict[str, LLMVerdict]:
    """Classify candidates with Claude; returns verdicts keyed by lowercased handle.

    Cache-first when a store is provided: an account whose bio/tweets/thesis/
    model are unchanged since the last classification reuses its cached verdict
    (TTL settings.verdict_ttl_days). Batches of the remainder run concurrently
    (settings.llm_concurrency). Verdicts with confidence < 0.4 are dropped.
    Returns {} (with a warning) when no API key is configured — the pipeline
    runs heuristics-only.
    """
    if not settings.anthropic_api_key:
        console.print(
            "[bold yellow]ANTHROPIC_API_KEY not set — running heuristics-only[/bold yellow]"
        )
        return {}
    if not candidates:
        return {}

    results: dict[str, LLMVerdict] = {}
    fresh: list[tuple[Account, list[Tweet]]] = []
    fingerprints: dict[str, str] = {}
    cache_hits = 0
    for account, tweets in candidates:
        key = account.handle.lstrip("@").lower()
        fingerprints[key] = _fingerprint(account, tweets, thesis, settings)
        cached = (
            store.cached_verdict(key, fingerprints[key], settings.verdict_ttl_days)
            if store is not None
            else None
        )
        if cached is not None:
            cache_hits += 1
            if cached.confidence >= MIN_CONFIDENCE:
                results[key] = cached
        else:
            fresh.append((account, tweets))
    if cache_hits:
        console.print(f"[dim]{cache_hits} verdicts served from cache.[/dim]")
    if not fresh:
        return results

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system_prompt = _system_prompt(thesis)
    batches = [fresh[i : i + BATCH_SIZE] for i in range(0, len(fresh), BATCH_SIZE)]
    workers = max(1, min(settings.llm_concurrency, len(batches)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(
                pool.map(
                    lambda b: _classify_batch_safe(
                        client, settings.claude_model, system_prompt, b
                    ),
                    batches,
                )
            )
    else:
        batch_results = [
            _classify_batch_safe(client, settings.claude_model, system_prompt, b)
            for b in batches
        ]

    for verdicts in batch_results:
        for verdict in verdicts:
            key = verdict.handle.lstrip("@").lower()
            if verdict.confidence >= MIN_CONFIDENCE:
                # Lowercased key so call sites match regardless of the casing
                # Claude echoed back.
                results[key] = verdict
            if store is not None and key in fingerprints:
                # Cache regardless of confidence so low-confidence accounts
                # aren't re-billed every run while inputs are unchanged.
                store.record_verdict(key, fingerprints[key], verdict)
    return results
