"""Agent-layer tests — pure parse/apply functions on fixture JSON (no network)."""

from __future__ import annotations

import json

import pytest

from scout.agents import StrategyProposal, apply_strategy, parse_strategy
from scout.config import Seeds, Thesis

PROPOSAL_JSON = json.dumps(
    {
        "thesis": "Ex-lab founders building vertical agents on proprietary data",
        "rationale": "Departure announcements are the earliest signal.",
        "target_stages": ["stealth", "launched", "warp-speed"],  # last is invalid
        "keywords": ["stealth", "building something new"],
        "target_bios": ["ex-OpenAI", "ex-Anthropic"],
        "sectors": ["vertical agents"],
        "disqualifiers": ["crypto airdrop"],
        "launch_phrases": ["launching", "waitlist"],
        "searches_departure": ['"my last day at" (@OpenAI OR @AnthropicAI)'],
        "searches_stealth_intent": ['"building something new" agents'],
        "searches_hiring": ['"founding engineer" agents'],
        "searches_launch": ['"we just launched" agents'],
        "bio_searches": ["ex-openai stealth"],
        "github_topics": ["ai-agents"],
        "watchlist": ["@eladgil", "saranormous", "  "],
    }
)


def test_parse_strategy_validates_and_cleans() -> None:
    proposal = parse_strategy(PROPOSAL_JSON)
    assert proposal.target_stages == ["stealth", "launched"]  # invalid stage dropped
    assert proposal.watchlist == ["eladgil", "saranormous"]  # @ and blanks stripped
    assert proposal.searches_departure[0].startswith('"my last day')


def test_parse_strategy_handles_code_fences() -> None:
    fenced = f"```json\n{PROPOSAL_JSON}\n```"
    assert parse_strategy(fenced).thesis.startswith("Ex-lab founders")


def test_parse_strategy_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_strategy("[1, 2, 3]")


def test_apply_strategy_merges_and_preserves_scoring() -> None:
    thesis = Thesis(
        thesis="old thesis",
        keywords=["old keyword"],
        weights={"bio_intent": 20.0},
        llm_prompt="CUSTOM PROMPT",
    )
    seeds = Seeds(watchlist=["old_watcher"], github_topics=["old-topic"])
    proposal = parse_strategy(PROPOSAL_JSON)

    new_thesis, new_seeds = apply_strategy(proposal, thesis, seeds)

    assert new_thesis.thesis.startswith("Ex-lab founders")
    assert new_thesis.keywords == ["stealth", "building something new"]
    # Scoring calibration is never touched by the strategy agent:
    assert new_thesis.weights == {"bio_intent": 20.0}
    assert new_thesis.llm_prompt == "CUSTOM PROMPT"
    assert new_seeds.watchlist == ["eladgil", "saranormous"]
    assert new_seeds.github_topics == ["ai-agents"]
    # Originals untouched (pure function):
    assert thesis.keywords == ["old keyword"]
    assert seeds.watchlist == ["old_watcher"]


def test_apply_strategy_empty_lists_keep_current_config() -> None:
    thesis = Thesis(keywords=["keep me"], target_stages=["launched"])
    seeds = Seeds(bio_searches=["keep-search"], watchlist=["keeper"])
    empty = StrategyProposal(thesis="new statement")

    new_thesis, new_seeds = apply_strategy(empty, thesis, seeds)

    assert new_thesis.thesis == "new statement"
    assert new_thesis.keywords == ["keep me"]
    assert new_thesis.target_stages == ["launched"]
    assert new_seeds.bio_searches == ["keep-search"]
    assert new_seeds.watchlist == ["keeper"]


def test_validate_watchlist_without_cookies_returns_unvalidated(tmp_path) -> None:
    from scout.agents import validate_watchlist
    from scout.config import Settings
    from scout.store import Store

    settings = Settings(tw_cookies=None, _env_file=None)
    store = Store(tmp_path / "t.db")
    keep, invalid, validated = validate_watchlist(
        ["@eladgil", "  ", "saranormous"], settings, store
    )
    assert keep == ["eladgil", "saranormous"]  # cleaned, nothing dropped
    assert invalid == []
    assert validated is False  # no cookies -> validation skipped, not failed


def test_brief_template_fallback_needs_no_key() -> None:
    from scout.agents import research_brief
    from scout.config import Settings
    from scout.models import Account, Lead, Signal

    lead = Lead(
        account=Account(id="1", handle="ada", bio="ex-OpenAI, stealth"),
        signals=[Signal(name="bio_intent", value=1.0)],
        score=72.0,
    )
    settings = Settings(anthropic_api_key=None, _env_file=None)
    brief, is_ai = research_brief(lead, Thesis(), settings)
    assert is_ai is False
    assert "What they're building" in brief
