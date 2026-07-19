"""One Messages API call with server-side web tools, done safely.

`research_call` is the single entry point every diligence agent goes through:

- server web tools (`web_search_20260209` / `web_fetch_20260209`) with hard
  `max_uses` caps — costs are bounded per agent, not per whim. No
  code_execution is declared alongside: the _20260209 tools embed dynamic
  filtering, and a second execution environment confuses the model.
- `stop_reason == "pause_turn"` continuation loop (the server-side tool loop
  pauses on long turns; re-sending the partial assistant turn resumes it).
- JSON-in-text parsing with corrective retry (mirrors scout.signals.llm).
- usage accounting: every individual API call lands in the diligence_usage
  ledger (tokens + web searches -> estimated $), and the accumulated cost is
  returned to the pipeline's budget tracker — even on failure, via
  ResearchError.cost_usd.

Timeouts are excluded from tenacity retry (a 240s research call retried
three times would wedge the UI); the client runs max_retries=1 for the same
reason as agents.py.
"""

from __future__ import annotations

import json
from typing import Callable, TypeVar

import anthropic
from pydantic import ValidationError

from scout.config import Settings
from scout.llmcall import client as make_client
from scout.llmcall import retry_transient
from scout.store import Store

T = TypeVar("T")

RESEARCH_TIMEOUT_S = 240.0  # web-tool turns are slow; fail fast beyond this
MAX_CONTINUATIONS = 5  # pause_turn resumptions per turn
PARSE_ATTEMPTS = 3  # initial call + 2 corrective retries
MAX_WEB_SEARCHES = 4  # per agent call
MAX_WEB_FETCHES = 3
MAX_FETCH_CONTENT_TOKENS = 20_000
DEFAULT_MAX_TOKENS = 8192

_CORRECTIVE_NOTE = (
    "\n\nIMPORTANT: your previous reply could not be parsed. End your reply "
    "with ONLY a valid JSON object exactly as specified — no prose after it, "
    "no markdown fences."
)


class ResearchError(RuntimeError):
    """A research call that failed after spending money. `cost_usd` carries
    what was already spent so the pipeline's budget tracker stays accurate
    (the diligence_usage ledger has the per-call detail regardless)."""

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


def web_tools() -> list[dict]:
    """The server-side tool declarations for research agents."""
    return [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": MAX_WEB_SEARCHES,
        },
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": MAX_WEB_FETCHES,
            "max_content_tokens": MAX_FETCH_CONTENT_TOKENS,
        },
    ]


def _rates(settings: Settings, model: str) -> tuple[float, float]:
    """($/MTok in, $/MTok out) for the model — synth vs research rate pair."""
    if model == settings.diligence_synth_model:
        return (
            settings.diligence_synth_cost_per_mtok_in,
            settings.diligence_synth_cost_per_mtok_out,
        )
    return (
        settings.diligence_research_cost_per_mtok_in,
        settings.diligence_research_cost_per_mtok_out,
    )


@retry_transient  # scout.llmcall policy: transient errors retried, timeouts fail fast
def _api_call(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
):
    kwargs: dict = {}
    if tools:
        kwargs["tools"] = tools
    return client.messages.create(
        model=model, max_tokens=max_tokens, system=system, messages=messages, **kwargs
    )


def _record_usage(
    settings: Settings,
    store: Store | None,
    company_key: str,
    stage: str,
    model: str,
    response,
) -> float:
    """Ledger one API response's usage; return its estimated cost."""
    usage = response.usage
    searches = sum(
        1
        for block in response.content
        if getattr(block, "type", "") == "server_tool_use"
        and getattr(block, "name", "") == "web_search"
    )
    rate_in, rate_out = _rates(settings, model)
    cost = (
        usage.input_tokens / 1e6 * rate_in
        + usage.output_tokens / 1e6 * rate_out
        + searches * settings.diligence_cost_per_search
    )
    if store is not None:
        try:
            store.record_diligence_usage(
                company_key=company_key,
                stage=stage,
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                searches=searches,
                est_cost_usd=cost,
            )
        except Exception:
            # The ledger is best-effort accounting: a transient sqlite error
            # (e.g. concurrent dimension threads racing table creation) must
            # never discard an already-paid finding. The cost still reaches
            # the budget tracker via the return value.
            pass
    return cost


def _run_turn(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    tools: list[dict],
    max_tokens: int,
    settings: Settings,
    store: Store | None,
    company_key: str,
    stage: str,
) -> tuple[str, float]:
    """One full agent turn: pause_turn continuation loop + usage ledger.

    Returns (text of the final response, total estimated cost). Server-tool
    errors (e.g. max_uses_exceeded) arrive as HTTP-200 result blocks whose
    content is an error object, not a list — we only read text blocks, so
    they degrade gracefully instead of crashing the turn.
    """
    messages: list[dict] = [{"role": "user", "content": user}]
    cost = 0.0
    response = None
    for _ in range(MAX_CONTINUATIONS + 1):
        response = _api_call(client, model, system, messages, tools, max_tokens)
        cost += _record_usage(settings, store, company_key, stage, model, response)
        if response.stop_reason != "pause_turn":
            break
        # Server-side tool loop paused mid-turn: append the partial assistant
        # turn and re-send — the API resumes where it left off. No extra user
        # message ("Continue") — the trailing server_tool_use block signals it.
        messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason == "pause_turn":
        # Still paused after every continuation: the text is partial, and
        # letting the parse-retry loop re-run the whole (paid) web turn would
        # multiply spend. Fail this call outright with its cost attached.
        raise ResearchError(
            f"{stage or 'research'} turn still paused after "
            f"{MAX_CONTINUATIONS} continuations",
            cost_usd=cost,
        )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text.strip(), cost


def research_call(
    system: str,
    user: str,
    parse: Callable[[str], T],
    settings: Settings,
    store: Store | None = None,
    *,
    model: str | None = None,
    company_key: str = "",
    stage: str = "",
    web: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[T, float]:
    """Run one diligence agent: call (with web tools unless web=False), parse
    with corrective retry, account usage. Returns (parsed, est_cost_usd).

    Raises ResearchError (carrying the cost already spent) when the key is
    missing, the call times out, the API fails after retries, or the reply
    stays unparseable after PARSE_ATTEMPTS.
    """
    if not settings.anthropic_api_key:
        raise ResearchError("ANTHROPIC_API_KEY is not set — deep analysis needs Claude.")
    client = make_client(settings, RESEARCH_TIMEOUT_S)
    chosen_model = model or settings.claude_model
    tools = web_tools() if web else []
    total = 0.0
    prompt = user
    last_error: Exception | None = None
    try:
        for _ in range(PARSE_ATTEMPTS):
            try:
                text, cost = _run_turn(
                    client, chosen_model, system, prompt, tools, max_tokens,
                    settings, store, company_key, stage,
                )
            except ResearchError as exc:  # exhausted pause_turn — add prior spend
                raise ResearchError(str(exc), cost_usd=total + exc.cost_usd) from None
            total += cost
            try:
                return parse(text), total
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                prompt = user + _CORRECTIVE_NOTE
    except ResearchError:
        raise
    except anthropic.APITimeoutError:
        raise ResearchError(
            f"{stage or 'research'} call timed out after {RESEARCH_TIMEOUT_S:.0f}s",
            cost_usd=total,
        ) from None
    except anthropic.APIError as exc:
        raise ResearchError(
            f"{stage or 'research'} call failed: {exc}", cost_usd=total
        ) from exc
    except Exception as exc:
        # Anything unexpected (parse infrastructure, response shape) still
        # carries the spend so the pipeline's budget tracker stays truthful.
        raise ResearchError(
            f"{stage or 'research'} call failed unexpectedly: "
            f"{type(exc).__name__}: {exc}",
            cost_usd=total,
        ) from exc
    raise ResearchError(
        f"{stage or 'research'} agent returned unparseable output: {last_error}",
        cost_usd=total,
    )
