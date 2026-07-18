"""Claude-drafted personalized outreach for the Win stage.

Given a scored lead, draft a short, specific first-touch message. Degrades to a
fill-in template when no Anthropic key is set.
"""

from __future__ import annotations

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Settings, Thesis
from scout.models import Lead

CHANNELS = ("X DM", "Email", "Warm intro ask")

_SYSTEM = (
    "You are an analyst at a top-tier VC firm doing founder outreach. Write a "
    "first-touch {channel} to the founder below. Requirements: 2-4 sentences, "
    "specific to what they are building (reference a concrete detail), warm but "
    "not fawning, no buzzwords, no emoji unless it reads naturally. The goal is a "
    "short intro call. For 'Warm intro ask', instead write a note to a mutual "
    "connection asking to be introduced. Output ONLY the message text — no "
    "subject line unless it's Email, no preamble, no quotes."
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)
    ),
    reraise=True,
)
def _call(client: anthropic.Anthropic, model: str, system: str, user: str) -> str:
    resp = client.messages.create(
        model=model, max_tokens=500, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


def _context(lead: Lead, thesis: Thesis) -> str:
    v = lead.llm
    lines = [
        f"Handle: @{lead.account.handle}",
        f"Name: {lead.account.name or lead.account.handle}",
        f"Bio: {lead.account.bio or '(none)'}",
    ]
    if v:
        if v.one_line_summary:
            lines.append(f"What they're building: {v.one_line_summary}")
        if v.stage:
            lines.append(f"Stage: {v.stage}")
        if v.why_interesting:
            lines.append(f"Why interesting: {v.why_interesting}")
    if thesis.thesis:
        lines.append(f"Our thesis: {thesis.thesis}")
    return "\n".join(lines)


def _template(lead: Lead, channel: str) -> str:
    name = (lead.account.name or f"@{lead.account.handle}").split()[0]
    what = lead.llm.one_line_summary if lead.llm and lead.llm.one_line_summary else "what you're building"
    if channel == "Warm intro ask":
        return (
            f"Would you be able to introduce me to @{lead.account.handle}? "
            f"I'm at Headline and we're excited about {what.lower()} — would love a quick chat."
        )
    return (
        f"Hi {name} — I'm an analyst at Headline. Came across {what.lower()} and "
        f"thought it was genuinely interesting. Would you be open to a short call "
        f"to hear more about the direction? No agenda beyond learning."
    )


def draft_outreach(
    lead: Lead, thesis: Thesis, settings: Settings, channel: str = "X DM"
) -> tuple[str, bool]:
    """Return (message, is_ai). Falls back to a template (is_ai=False) when no
    Anthropic key is configured or the API call fails."""
    if not settings.anthropic_api_key:
        return _template(lead, channel), False
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        system = _SYSTEM.format(channel=channel)
        message = _call(client, settings.claude_model, system, _context(lead, thesis))
        return message, True
    except anthropic.APIError:
        return _template(lead, channel), False
