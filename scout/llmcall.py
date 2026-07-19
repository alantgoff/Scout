"""Shared Claude call plumbing — the ONE home for idioms every LLM-facing
module used to copy (agents, classifier, outreach, diligence): fence
stripping, client construction, the transient-error retry policy, and the
JSON-in-text corrective parse retry.

Policy decisions live here so they can't drift between call paths:

- Clients run **max_retries=1** — tenacity is the retry layer; stacking the
  SDK's own retries on top multiplies worst-case latency.
- Transient retries cover rate limits, 5xx, and connection drops but
  **exclude timeouts** — with explicit client timeouts, three attempts would
  wedge an inline UI caller for many minutes. Fail fast instead.
- Structured outputs are not guaranteed on the configured models, so JSON
  rides in text: parse the reply, and on failure re-ask (up to
  PARSE_ATTEMPTS total) with a corrective note appended.
"""

from __future__ import annotations

import json
from typing import Callable, TypeVar

import anthropic
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Settings

T = TypeVar("T")

PARSE_ATTEMPTS = 3  # initial call + 2 corrective retries

CORRECTIVE_OBJECT_NOTE = (
    "\n\nIMPORTANT: your previous reply could not be parsed. Respond with ONLY "
    "a valid JSON object exactly as specified — no prose, no markdown fences."
)


class UnparseableReplyError(RuntimeError):
    """The model's reply stayed unparseable after every corrective retry."""


def client(settings: Settings, timeout: float | None = None) -> anthropic.Anthropic:
    """An Anthropic client with the house policy applied. `timeout` in
    seconds; None keeps the SDK default (10 minutes)."""
    kwargs: dict = {"api_key": settings.anthropic_api_key, "max_retries": 1}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return anthropic.Anthropic(**kwargs)


# The transient-retry policy, importable as a decorator for call shapes this
# module doesn't cover (e.g. diligence research calls that pass web tools).
retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=(
        retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError))
        | (
            retry_if_exception_type(anthropic.APIConnectionError)
            & retry_if_not_exception_type(anthropic.APITimeoutError)
        )
    ),
    reraise=True,
)


@retry_transient
def call_text(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4096,
) -> str:
    """One plain text-in/text-out Messages call under the retry policy."""
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def call_with_parse_retry(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    parse: Callable[[str], T],
    *,
    max_tokens: int = 4096,
    corrective_note: str = CORRECTIVE_OBJECT_NOTE,
) -> T:
    """call_text + parse, re-asking with a corrective note on parse failure.

    Raises UnparseableReplyError after PARSE_ATTEMPTS; API errors (including
    timeouts, which are never retried) propagate for the caller's policy.
    """
    prompt = user
    last_error: Exception | None = None
    for _ in range(PARSE_ATTEMPTS):
        text = call_text(client, model, system, prompt, max_tokens)
        try:
            return parse(text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            prompt = user + corrective_note
    raise UnparseableReplyError(str(last_error))


def strip_code_fences(text: str) -> str:
    """Remove a wrapping ``` fence (with optional language tag) if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()
