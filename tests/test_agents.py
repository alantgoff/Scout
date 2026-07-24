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


# --- the investment-memo agent (offline paths) ----------------------------------


def _memo_lead():
    from scout.models import Account, Lead, Signal

    return Lead(
        account=Account(id="1", handle="ada", name="Ada Lin",
                        bio="ex-OpenAI, building evals", followers=2100,
                        website="https://evalhq.ai"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=None,
        score=61.0,
    )


def test_investment_memo_template_has_every_section_without_key() -> None:
    from scout.agents import MEMO_SECTIONS, investment_memo
    from scout.config import Settings

    memo, is_ai, meta = investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key=None, _env_file=None),
        depth="deep",
    )
    assert is_ai is False
    assert meta == {"depth": "deep", "sources": [], "searches": 0, "fetches": 0}
    for section in MEMO_SECTIONS:
        assert f"## {section}" in memo, section
    # The template carries the v2 format markers too (verdict chip contract).
    assert "**TL;DR**" in memo
    assert "**VERDICT: TRACK" in memo


def test_memo_context_carries_site_bundle_focus_and_provenance() -> None:
    from datetime import datetime, timezone

    from scout.agents import _memo_context
    from scout.models import LLMVerdict, Tweet

    lead = _memo_lead()
    lead.llm = LLMVerdict(handle="ada", company_name="EvalHQ",
                          product_summary="Evals for agent teams",
                          grounding="website", thesis_fit=0.8, confidence=0.9)
    tweets = [Tweet(id="1", account_id="1", text="We just launched EvalHQ!",
                    created_at=datetime.now(timezone.utc), likes=120)]
    ctx = _memo_context(
        lead, Thesis(),
        site_text="### Page: /home\nEvalHQ scores agent runs.",
        tweets=tweets, notes="met at the eval summit",
        site_note="captured from adriengaidon.com — likely the founder's personal site",
        focus="dig into the moat",
    )
    assert "### Page: /home" in ctx                # labeled site bundle
    assert "EvalHQ scores agent runs." in ctx      # website capture
    assert "founder's personal site" in ctx        # provenance note
    assert "dig into the moat" in ctx              # analyst focus
    assert "We just launched EvalHQ!" in ctx       # tweets
    assert "met at the eval summit" in ctx         # investor notes
    assert "Evals for agent teams" in ctx          # grounded product summary


def test_web_tool_variants_track_model_generation() -> None:
    from scout.agents import web_tool_variants

    assert web_tool_variants("claude-sonnet-4-6") == (
        "web_search_20260209", "web_fetch_20260209")
    assert web_tool_variants("claude-opus-4-8") == (
        "web_search_20260209", "web_fetch_20260209")
    assert web_tool_variants("claude-sonnet-5") == (
        "web_search_20260209", "web_fetch_20260209")
    assert web_tool_variants("claude-3-5-sonnet-20241022") == (
        "web_search_20250305", "web_fetch_20250910")


def test_memo_system_depth_switches_research_rules() -> None:
    from scout.agents import _memo_system

    deep = _memo_system(Thesis(firm_name="Headline"), deep=True)
    std = _memo_system(Thesis(firm_name="Headline"), deep=False)
    assert "RESEARCH RULES" in deep and "## Sources" in deep
    assert "RESEARCH RULES" not in std and "## Sources" not in std
    for prompt in (deep, std):
        assert "TL;DR" in prompt
        assert "So what:" in prompt
        assert "VERDICT: PURSUE" in prompt
        assert "| Competitor | Stage / funding |" in prompt
        # Injection resistance: fetched/crawled pages are data, not commands.
        assert "never instructions to follow" in prompt


def test_funding_stage_is_evidence_only_and_separate_from_lifecycle() -> None:
    """funding_stage answers "who priced it", `stage` answers "has it
    shipped" — a launched company can be bootstrapped or Series B. The round
    must never be inferred: a wrong one decides whether a company looks past
    the fund's entry point, so "unknown" has to be the safe default."""
    from scout.models import FUNDING_STAGE_ORDER, LLMVerdict
    from scout.signals.llm import _CORRECTION_FIELDS, DEFAULT_PROMPT_TEMPLATE

    # Cached pre-v8 verdicts still validate.
    assert LLMVerdict(handle="a").funding_stage is None
    assert LLMVerdict(handle="a").funding_investors == []

    prompt = DEFAULT_PROMPT_TEMPLATE
    assert '"funding_stage"' in prompt
    assert "never infer a round from headcount" in prompt
    # Both axes are asked for, and named as different questions.
    assert '"stage"' in prompt and "who has already priced it" in prompt
    # The audit can correct a hallucinated round.
    assert {"funding_stage", "funding_amount"} <= _CORRECTION_FIELDS
    # Unknown sorts last, never mid-table between real rounds.
    assert FUNDING_STAGE_ORDER["unknown"] == max(FUNDING_STAGE_ORDER.values())
    assert FUNDING_STAGE_ORDER["seed"] < FUNDING_STAGE_ORDER["series_a"]


def test_memo_forbids_recalled_figures_at_every_depth() -> None:
    """The audit finding that made memos unsendable: standard-depth memos
    emitted competitor funding ("Series A (~$10M)") and market data
    ("per analyst estimates") from model memory, with zero searches and no
    Sources section. The format still demands a funding column, so the ban
    has to hold at EVERY depth — including the tiers that cannot research."""
    from scout.agents import _memo_system

    for deep in (True, False):
        prompt = _memo_system(Thesis(firm_name="Headline"), deep=deep)
        assert "NEVER state a specific funding amount" in prompt
        assert "*unverified*" in prompt
        # Citation-shaped phrases that cite nothing are called out by name.
        assert "per analyst estimates" in prompt
        assert "Attribution must be real" in prompt


def test_memo_sections_cover_what_a_partner_expects() -> None:
    """Team, Traction, Deal terms, Risks and Why now were all missing. Team
    matters most: at seed the team IS the investment, and the memo used to
    skip it entirely."""
    from scout.agents import MEMO_SECTIONS, _memo_system

    for section in ("Why now", "Team", "Traction & metrics",
                    "Deal terms & ownership", "Risks"):
        assert section in MEMO_SECTIONS
    prompt = _memo_system(Thesis(firm_name="Headline"), deep=True)
    for section in MEMO_SECTIONS:
        assert f"## {section}" in prompt, section
    # Deep research spends its budget on founders before competitors.
    assert "research the FOUNDERS by name" in prompt


# --- the deep-research stream loop, fully mocked (no network) -------------------


from types import SimpleNamespace as _NS


class _FakeStream:
    def __init__(self, message, events=()):
        self._message = message
        self._events = list(events)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._message


class _FakeClient:
    """Stands in for anthropic.Anthropic: pops one scripted turn per
    stream() call — either an (events, message) pair, or an Exception to
    raise — and records the request kwargs."""

    def __init__(self, turns):
        self.calls: list[dict] = []
        self._turns = list(turns)
        client = self

        class _Messages:
            def stream(self, **kwargs):
                client.calls.append(kwargs)
                turn = client._turns.pop(0)
                if isinstance(turn, Exception):
                    raise turn
                events, message = turn
                return _FakeStream(message, events)

        self.messages = _Messages()


def _api_request():
    import httpx

    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit_error():
    import anthropic
    import httpx

    return anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_api_request()),
        body=None)


def _timeout_error():
    import anthropic

    return anthropic.APITimeoutError(request=_api_request())


def _search_events(n: int, start_index: int = 0):
    """n web_search server-tool blocks with the input delivered fully-formed
    on block_start (the no-deltas variant) — doubles as its regression test."""
    events = []
    for i in range(n):
        idx = start_index + i
        events += [
            _NS(type="content_block_start", index=idx,
                content_block=_NS(type="server_tool_use", name="web_search",
                                  input={"query": f"query {i}"})),
            _NS(type="content_block_stop", index=idx),
        ]
    return events


def test_investment_memo_deep_loop_pauses_narrates_and_cites(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    search_events = [
        _NS(type="content_block_start", index=0,
            content_block=_NS(type="server_tool_use", name="web_search")),
        _NS(type="content_block_delta", index=0,
            delta=_NS(type="input_json_delta",
                      partial_json='{"query": "walden robotics site"}')),
        _NS(type="content_block_stop", index=0),
        _NS(type="content_block_start", index=1,
            content_block=_NS(type="server_tool_use", name="web_fetch")),
        _NS(type="content_block_delta", index=1,
            delta=_NS(type="input_json_delta",
                      partial_json='{"url": "https://waldenrobotics.com"}')),
        _NS(type="content_block_stop", index=1),
    ]
    paused = _NS(stop_reason="pause_turn",
                 content=[_NS(type="text", text="partial…", citations=None)])
    final = _NS(
        stop_reason="end_turn",
        content=[_NS(
            type="text",
            text="**TL;DR**\n- ok\n\n## Overview\nreal memo",
            citations=[_NS(url="https://waldenrobotics.com"),
                       _NS(url="https://techcrunch.com/walden")],
        )],
    )
    fake = _FakeClient([(search_events, paused), ([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    seen_events: list[tuple[str, str]] = []
    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(),
        Settings(anthropic_api_key="k", _env_file=None),
        depth="deep", on_event=lambda kind, detail: seen_events.append((kind, detail)),
    )

    assert is_ai is True
    # pause_turn → a second request whose messages resume the paused turn.
    assert len(fake.calls) == 2
    assert fake.calls[1]["messages"][1]["role"] == "assistant"
    assert fake.calls[1]["messages"][1]["content"] is paused.content
    # Server tools were requested with capped uses.
    tools = fake.calls[0]["tools"]
    assert {t["name"] for t in tools} == {"web_search", "web_fetch"}
    assert all(t["max_uses"] for t in tools)
    # Narration surfaced the real query/URL, and meta counted them.
    assert ("search", "walden robotics site") in seen_events
    assert ("fetch", "https://waldenrobotics.com") in seen_events
    assert meta["searches"] == 1 and meta["fetches"] == 1
    # Citations harvested in order, and the missing Sources section appended.
    assert meta["sources"] == ["https://waldenrobotics.com",
                               "https://techcrunch.com/walden"]
    assert "## Sources" in memo and "techcrunch.com/walden" in memo


def test_investment_memo_standard_sends_no_tools(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    final = _NS(stop_reason="end_turn",
                content=[_NS(type="text", text="## Overview\nok", citations=None)])
    fake = _FakeClient([([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is True and memo.startswith("## Overview")
    assert "tools" not in fake.calls[0]
    assert meta["sources"] == [] and meta["searches"] == 0


# --- failure modes: retries, budgets, truncation, garbage -----------------------


def test_memo_retries_transient_errors_then_succeeds(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    final = _NS(stop_reason="end_turn",
                content=[_NS(type="text", text="## Overview\nok", citations=None)])
    fake = _FakeClient([_rate_limit_error(), ([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)
    monkeypatch.setattr(agents, "MEMO_STREAM_BACKOFF_S", 0.0)

    events: list[tuple[str, str]] = []
    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard", on_event=lambda k, d: events.append((k, d)),
    )
    assert is_ai is True and len(fake.calls) == 2
    assert any(k == "retry" for k, _ in events)


def test_memo_timeout_fails_fast_to_fallback(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    fake = _FakeClient([_timeout_error(), ([], None)])  # 2nd turn must not run
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)
    monkeypatch.setattr(agents, "MEMO_STREAM_BACKOFF_S", 0.0)

    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is False            # fell back to the template…
    assert len(fake.calls) == 1      # …without retrying a multi-minute wait
    assert "## Overview" in memo


def test_memo_gives_up_after_retry_budget(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    fake = _FakeClient([_rate_limit_error(), _rate_limit_error(),
                        _rate_limit_error()])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)
    monkeypatch.setattr(agents, "MEMO_STREAM_BACKOFF_S", 0.0)

    memo, is_ai, _meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is False
    assert len(fake.calls) == agents.MEMO_STREAM_RETRIES


def test_deep_continuations_shrink_the_tool_budget(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    paused = _NS(stop_reason="pause_turn",
                 content=[_NS(type="text", text="## Overview\npartial",
                              citations=None)])
    final = _NS(stop_reason="end_turn",
                content=[_NS(type="text", text="## Overview\ndone",
                             citations=None)])
    fake = _FakeClient([(_search_events(7), paused), ([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    events: list[tuple[str, str]] = []
    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="deep", on_event=lambda k, d: events.append((k, d)),
    )
    assert is_ai is True and meta["searches"] == 7
    # block_start-delivered inputs narrate the real query (no deltas needed).
    assert ("search", "query 0") in events
    first, second = fake.calls
    assert first["tools"][0]["max_uses"] == agents.MEMO_MAX_SEARCHES
    # The continuation reopens only what's left of the budget (floor 1).
    assert second["tools"][0]["max_uses"] == max(
        agents.MEMO_MAX_SEARCHES - 7, 1)
    assert second["tools"][1]["max_uses"] == agents.MEMO_MAX_FETCHES


def test_memo_flags_truncation_and_missing_sections(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    cut = _NS(stop_reason="max_tokens",
              content=[_NS(type="text",
                           text="**TL;DR**\n- x\n\n## Overview\nonly this",
                           citations=None)])
    fake = _FakeClient([([], cut)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is True
    assert meta["truncated"] is True
    assert "Recommendation" in meta["missing_sections"]


def test_memo_flags_exhausted_research_budget(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    paused = _NS(stop_reason="pause_turn",
                 content=[_NS(type="text", text="## Overview\npartial",
                              citations=None)])
    fake = _FakeClient([([], paused)] * agents.MEMO_MAX_CONTINUATIONS)
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="deep",
    )
    assert is_ai is True
    assert meta["exhausted"] is True
    assert len(fake.calls) == agents.MEMO_MAX_CONTINUATIONS


def test_memo_rejects_sectionless_garbage_output(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    refusal = _NS(stop_reason="end_turn",
                  content=[_NS(type="text",
                               text="I cannot analyze this company.",
                               citations=None)])
    fake = _FakeClient([([], refusal)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is False              # never presented as a real memo
    assert "## Overview" in memo       # the honest skeleton instead


def test_trim_memo_cuts_research_narration() -> None:
    from scout.agents import _trim_memo

    raw = ("I'll start by researching the company.\nHere is the memo:\n---\n\n"
           "**TL;DR**\n- one\n\n## Overview\nbody")
    assert _trim_memo(raw).startswith("**TL;DR**")
    # No TL;DR → cut at the first section heading instead.
    raw2 = "Let me compile.\n## Overview\nbody"
    assert _trim_memo(raw2).startswith("## Overview")
    # Already clean → unchanged.
    assert _trim_memo("**TL;DR**\n- x").startswith("**TL;DR**")
    assert _trim_memo("plain text") == "plain text"


def test_sources_from_text_parses_the_sources_section() -> None:
    from scout.agents import _sources_from_text

    memo = ("## Recommendation\nsee https://ignored.com in body\n\n"
            "## Sources\n\n"
            "[1] \"Launch\" — https://www.waldenrobotics.com/news/launch\n"
            "[2] Team — https://www.waldenrobotics.com/company\n"
            "[3] dupe — https://www.waldenrobotics.com/company\n")
    assert _sources_from_text(memo) == [
        "https://www.waldenrobotics.com/news/launch",
        "https://www.waldenrobotics.com/company",
    ]
    assert _sources_from_text("no sources here") == []


# --- startup categorization -----------------------------------------------------


_CAT_COLUMNS = [
    {"key": "vertical", "label": "Vertical", "type": "select",
     "options": ["Fintech", "Security"]},
    {"key": "use_case", "label": "Use case", "type": "multiselect",
     "options": ["Payments / banking", "Agent infrastructure"]},
]


def test_parse_categorization_validates_strictly() -> None:
    from scout.agents import parse_categorization

    text = json.dumps({
        "ada": {"vertical": "Fintech", "use_case": ["Payments / banking", "Yachts"]},
        "@BO": {"vertical": ["Security", "Fintech"]},      # list → first valid
        "ada2": {"vertical": "Crypto"},                    # off-list → dropped
        "ghost": {"vertical": "Fintech"},                  # unknown handle
        "casey": {"use_case": "Agent infrastructure"},     # str → 1-list
        "dana": "not an object",
    })
    out = parse_categorization(text, _CAT_COLUMNS, {"ada", "bo", "ada2", "casey", "dana"})
    assert out == {
        "ada": {"vertical": "Fintech", "use_case": ["Payments / banking"]},
        "bo": {"vertical": "Security"},
        "casey": {"use_case": ["Agent infrastructure"]},
    }


def test_categorize_startups_batches_and_retries(monkeypatch) -> None:
    from scout import agents
    from scout.config import Settings

    calls: list[str] = []
    replies = iter([
        "not json at all",                                   # batch 1, attempt 1
        json.dumps({"h0": {"vertical": "Fintech"}}),         # batch 1, attempt 2
        json.dumps({"h2": {"vertical": "Security"}}),        # batch 2
    ])

    def fake_call(client, model, system, user, max_tokens=4096):
        calls.append(user)
        return next(replies)

    monkeypatch.setattr(agents, "_client", lambda settings, timeout: object())
    monkeypatch.setattr(agents, "_call", fake_call)

    rows = [{"handle": f"h{i}", "summary": f"startup {i}"} for i in range(3)]
    out = agents.categorize_startups(
        rows, _CAT_COLUMNS, Settings(anthropic_api_key="k", _env_file=None),
        batch_size=2,
    )
    assert out == {"h0": {"vertical": "Fintech"}, "h2": {"vertical": "Security"}}
    assert len(calls) == 3                       # 2 batches + 1 corrective retry
    assert "could not be parsed" in calls[1]     # corrective note appended


def test_categorize_startups_requires_key_and_skips_thin_rows() -> None:
    import pytest as _pytest

    from scout.agents import categorize_startups
    from scout.config import Settings

    with _pytest.raises(RuntimeError):
        categorize_startups([{"handle": "a", "summary": "x"}], _CAT_COLUMNS,
                            Settings(anthropic_api_key=None, _env_file=None))
    # No summaries / no select columns → no-op without a client call.
    assert categorize_startups([{"handle": "a", "summary": "  "}], _CAT_COLUMNS,
                               Settings(anthropic_api_key="k", _env_file=None)) == {}
    assert categorize_startups([{"handle": "a", "summary": "x"}], [],
                               Settings(anthropic_api_key="k", _env_file=None)) == {}


def test_memo_section_check_is_case_insensitive(monkeypatch) -> None:
    """Title-cased headings ("Product & Differentiation") must not be
    flagged missing — the Cosmic Labs false-incomplete regression."""
    from scout import agents
    from scout.config import Settings

    text = "**TL;DR**\n- x\n\n" + "\n\n".join(
        f"## {s.title()}\nbody" for s in agents.MEMO_SECTIONS)
    final = _NS(stop_reason="end_turn",
                content=[_NS(type="text", text=text, citations=None)])
    fake = _FakeClient([([], final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    _memo, is_ai, meta = agents.investment_memo(
        _memo_lead(), Thesis(), Settings(anthropic_api_key="k", _env_file=None),
        depth="standard",
    )
    assert is_ai is True
    assert "missing_sections" not in meta


def test_call_stream_reports_progress_per_config_section() -> None:
    """The streamed strategy call fires on_progress once per top-level config
    key, in schema order — and a bare mention of a key inside the rationale
    prose must NOT prematurely fire that stage."""
    from scout import agents

    payload = json.dumps({
        "thesis": "t",
        # "watchlist" and "keywords" appear here as prose, NOT as JSON members.
        "rationale": "the watchlist and keywords are discussed here as prose",
        "keywords": ["a"],
        "watchlist": ["x"],
    })

    class _Block:
        type = "text"
        text = payload

    class _Final:
        content = [_Block()]

    class _Stream:
        text_stream = list(payload)  # char-by-char to stress incremental match

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return _Final()

    class _Client:
        class messages:  # noqa: N801 — mimics the SDK surface
            @staticmethod
            def stream(**kwargs):
                return _Stream()

    events: list[str] = []
    text = agents._call_stream(_Client(), "m", "sys", "user",
                               on_progress=events.append)
    assert text == payload
    # thesis → rationale → keywords → watchlist, each exactly once, in order.
    # The prose mentions of watchlist/keywords never fire early.
    assert events == ["Restating the thesis", "Writing the rationale",
                      "Bio-intent keywords", "Investor watchlist"]
