"""The by-domain research agent: `scout add <domain>` fills in the rest.

The parser and the overlay are pure and tested directly; the agent loop runs
against the same scripted fake client the memo tests use. No network.

The through-line of these tests is one failure: a company that is no longer
its own company. Its site is live, its X account posts, its press is glowing —
every input the classifier reads says "healthy startup" — and only a search
nobody thought to run says otherwise. Most of what is pinned below exists to
make sure that finding survives from the research call to the card.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as _NS

import pytest

from scout import agents
from scout.agents import CompanyProfile, apply_research, parse_company_profile
from scout.config import Settings
from scout.models import LLMVerdict

from tests.test_agents import _FakeClient


# --- parse_company_profile ----------------------------------------------------


def test_parser_normalizes_the_shapes_models_actually_return() -> None:
    profile = parse_company_profile(json.dumps({
        "is_company": True,
        "company_name": "  Pollen   Robotics ",
        "x_handle": "https://x.com/pollenrobotics?lang=en",
        "github_org": "pollen-robotics",
        "founded_year": "2016",
        "founders": "Matthieu Lapeyre — co-founder",   # lone string for a list
        "tags": ["Robotics", "OPEN SOURCE"],
        "funding_stage": "Series A",                    # off-vocabulary spelling
        "customer_type": "enterprise",                  # not in CUSTOMER_TYPES
        "stage": "Launched",
    }))
    assert profile.company_name == "Pollen Robotics"
    assert profile.x_handle == "pollenrobotics"
    assert profile.founded_year == 2016
    assert profile.founders == ["Matthieu Lapeyre — co-founder"]
    assert profile.tags == ["robotics", "open source"]
    assert profile.stage == "launched"
    # Strict about vocabulary: an unrecognised value becomes the safe default,
    # never a value the models would reject downstream.
    assert profile.funding_stage == "unknown"
    assert profile.customer_type is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("@pollenrobotics", "pollenrobotics"),
        ("x.com/pollenrobotics/", "pollenrobotics"),
        ("https://twitter.com/pollenrobotics", "pollenrobotics"),
        ("pollen robotics", None),        # a name, not a handle
        ("pollen-robotics", None),        # hyphen — not a legal X handle
        ("averyveryverylonghandle", None),  # over 15 chars
        ("", None),
        (None, None),
    ],
)
def test_handle_cleaning_refuses_anything_that_is_not_a_handle(raw, expected) -> None:
    """A wrong handle is the one error here that silently merges this
    company's record into a stranger's, so the bar is 'saw it', not 'plausible'."""
    assert agents._clean_handle(raw) == expected


def test_parser_rejects_non_objects() -> None:
    with pytest.raises(ValueError):
        parse_company_profile("[1, 2, 3]")


def test_founded_year_rejects_impossible_values() -> None:
    assert parse_company_profile('{"founded_year": 12016}').founded_year is None
    assert parse_company_profile('{"founded_year": "soon"}').founded_year is None


# --- apply_research: who wins -------------------------------------------------


def _classified() -> LLMVerdict:
    """A verdict the classifier produced from real site evidence."""
    return LLMVerdict(
        handle="pollenrobotics", account_type="startup",
        company_name="Pollen Robotics", company_url="https://pollen-robotics.com/",
        product_summary="website: open-source humanoid robots for AI research",
        sector="robotics", grounding="website", thesis_fit=0.55,
        scorecard={"technology": 3}, confidence=0.8,
    )


def test_research_never_overwrites_a_judgment() -> None:
    """The classifier owns thesis fit and the scorecard for every startup in
    the database. If research could move them, a hand-added company would be
    scored by different code than a discovered one, and the two would stop
    being comparable."""
    before = _classified()
    after = apply_research(before, CompanyProfile(
        sector="something else", product_summary="a different product",
        sources=["https://example.com"],
    ))
    assert after.thesis_fit == 0.55
    assert after.scorecard == {"technology": 3}
    assert after.sector == "robotics"
    assert after.product_summary == before.product_summary
    assert after.grounding == "website"


def test_research_fills_what_the_classifier_left_null() -> None:
    thin = LLMVerdict(handle="p", grounding="none")
    after = apply_research(thin, CompanyProfile(
        company_name="Pollen Robotics", website="https://pollen-robotics.com/",
        product_summary="website: open-source humanoid robots",
        sector="robotics", subsector="humanoids", customer_type="b2b",
        stage="launched", tags=["robotics"], sources=["https://x.example/a"],
    ))
    assert after.company_name == "Pollen Robotics"
    assert after.sector == "robotics"
    assert after.customer_type == "b2b"
    assert after.stage == "launched"
    # Cited live research IS product evidence — the strongest kind available
    # for a company with no X presence for the classifier to read.
    assert after.grounding == "research"


def test_grounding_is_only_claimed_when_something_was_actually_cited() -> None:
    thin = LLMVerdict(handle="p", grounding="none")
    assert apply_research(thin, CompanyProfile(
        product_summary="robots", sources=[])).grounding == "none"
    assert apply_research(thin, CompanyProfile(
        product_summary=None, sources=["https://a.example"])).grounding == "none"


def test_cited_rounds_fill_unknowns_and_progressions_but_never_conflicts() -> None:
    """"unknown" is what the classifier is instructed to say when the dossier
    is silent, and the dossier is silent about almost every round."""
    filled = apply_research(
        LLMVerdict(handle="p", funding_stage="unknown"),
        CompanyProfile(funding_stage="seed", funding_amount="€2.5M",
                       funding_investors=["Bpifrance"],
                       funding_evidence="techcrunch 2025-04-14"),
    )
    assert (filled.funding_stage, filled.funding_amount) == ("seed", "€2.5M")
    assert filled.funding_investors == ["Bpifrance"]

    # A round research cannot source never lands.
    unsourced = apply_research(
        LLMVerdict(handle="p", funding_stage="unknown"),
        CompanyProfile(funding_stage="series_a", funding_evidence=None),
    )
    assert unsourced.funding_stage == "unknown"

    # A cited LATER round is a progression — the company raised — and lands.
    raised = apply_research(
        LLMVerdict(handle="p", funding_stage="seed", funding_evidence="site: press page"),
        CompanyProfile(funding_stage="series_a", funding_amount="$14M",
                       funding_evidence="techcrunch 2026-08-01"),
    )
    assert raised.funding_stage == "series_a"
    assert raised.funding_amount == "$14M"

    # A cited round equal or EARLIER than the current one is a conflict, not
    # a raise — the verdict's own evidence stands.
    for conflicting in ("seed", "pre_seed"):
        kept = apply_research(
            LLMVerdict(handle="p", funding_stage="seed",
                       funding_evidence="site: press page"),
            CompanyProfile(funding_stage=conflicting, funding_evidence="a blog"),
        )
        assert kept.funding_stage == "seed"
        assert kept.funding_evidence == "site: press page"


def test_acquisition_survives_the_overlay_with_its_source() -> None:
    after = apply_research(_classified(), CompanyProfile(
        company_status="acquired",
        company_status_note="Hugging Face, April 2025",
        company_status_evidence="techcrunch.com, 2025-04-14",
        hq="Bordeaux, France", founded_year=2016,
        founders=["Matthieu Lapeyre — co-founder, ex-INRIA Flowers"],
        sources=["https://techcrunch.com/2025/04/14/x"],
    ))
    assert after.company_status == "acquired"
    assert after.company_status_note == "Hugging Face, April 2025"
    assert after.hq == "Bordeaux, France"
    assert after.founded_year == 2016
    assert after.founders == ["Matthieu Lapeyre — co-founder, ex-INRIA Flowers"]
    assert after.research_sources == ["https://techcrunch.com/2025/04/14/x"]


def test_an_unsourced_acquisition_is_discarded_by_the_model() -> None:
    """The mirror of the funding rule. There, a fabricated round makes a
    company look past your entry point; here, a fabricated acquisition kills a
    live company in your pipeline on a half-remembered headline."""
    after = apply_research(_classified(), CompanyProfile(
        company_status="acquired", company_status_note="by someone, probably",
        company_status_evidence=None,
    ))
    assert after.company_status != "acquired"
    assert after.company_status_note == ""


# --- research_company: the agent loop ----------------------------------------


def _settings() -> Settings:
    return Settings(anthropic_api_key="k", _env_file=None)


def _profile_json(**overrides) -> str:
    payload = {
        "is_company": True, "company_name": "Pollen Robotics",
        "x_handle": "@pollenrobotics", "one_line_summary": "open-source robots",
        "product_summary": "website: open-source humanoid robots",
        "company_status": "acquired", "company_status_note": "Hugging Face, Apr 2025",
        "company_status_evidence": "techcrunch", "sources": ["https://model.example/a"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _final(text: str, citations=None):
    return _NS(stop_reason="end_turn",
               content=[_NS(type="text", text=text, citations=citations)])


def test_research_company_narrates_counts_and_prefers_harvested_sources(monkeypatch) -> None:
    events = [
        _NS(type="content_block_start", index=0,
            content_block=_NS(type="server_tool_use", name="web_search",
                              input={"query": "pollen robotics acquired"})),
        _NS(type="content_block_stop", index=0),
        _NS(type="content_block_start", index=1,
            content_block=_NS(type="server_tool_use", name="web_fetch",
                              input={"url": "https://techcrunch.com/x"})),
        _NS(type="content_block_stop", index=1),
    ]
    final = _final(_profile_json(),
                   citations=[_NS(url="https://techcrunch.com/x")])
    fake = _FakeClient([(events, final)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    seen: list[tuple[str, str]] = []
    profile, meta = agents.research_company(
        "https://pollen-robotics.com/", _settings(),
        site_text="Reachy 2 — open source humanoid",
        on_event=lambda kind, detail: seen.append((kind, detail)),
    )

    assert profile.company_name == "Pollen Robotics"
    assert profile.x_handle == "pollenrobotics"
    assert profile.company_status == "acquired"
    assert meta["researched"] is True
    assert (meta["searches"], meta["fetches"]) == (1, 1)
    assert ("search", "pollen robotics acquired") in seen
    # What the tools actually returned beats the model's own list — the model
    # can embellish its `sources` array; it cannot embellish a citation block.
    assert meta["sources"] == ["https://techcrunch.com/x"]
    assert profile.sources == ["https://techcrunch.com/x"]
    # Research tools were declared, and the crawled text was handed over.
    assert {t["name"] for t in fake.calls[0]["tools"]} == {"web_search", "web_fetch"}
    assert "Reachy 2" in fake.calls[0]["messages"][0]["content"]


def test_research_company_retries_unparseable_output_then_succeeds(monkeypatch) -> None:
    fake = _FakeClient([([], _final("sorry, here's what I found:")),
                        ([], _final(_profile_json()))])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    profile, meta = agents.research_company("https://pollen-robotics.com/", _settings())
    assert profile.company_name == "Pollen Robotics"
    assert meta["researched"] is True
    assert agents._CORRECTIVE_NOTE in fake.calls[1]["messages"][0]["content"]


def test_research_company_gives_up_loudly_on_junk(monkeypatch) -> None:
    fake = _FakeClient([([], _final("nope")) for _ in range(agents.PARSE_ATTEMPTS)])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)
    with pytest.raises(RuntimeError, match="unparseable"):
        agents.research_company("https://pollen-robotics.com/", _settings())


def test_research_company_continues_a_paused_turn_on_a_shrinking_budget(monkeypatch) -> None:
    paused = _NS(stop_reason="pause_turn",
                 content=[_NS(type="text", text="", citations=None)])
    searches = [
        _NS(type="content_block_start", index=0,
            content_block=_NS(type="server_tool_use", name="web_search",
                              input={"query": "q"})),
        _NS(type="content_block_stop", index=0),
    ]
    fake = _FakeClient([(searches, paused), ([], _final(_profile_json()))])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)

    _profile, meta = agents.research_company("https://x.example/", _settings())
    assert len(fake.calls) == 2
    assert fake.calls[1]["messages"][1]["role"] == "assistant"
    # max_uses is per-request: the continuation must reopen only what is left,
    # or every pause would hand back the full allowance.
    budget = {t["name"]: t["max_uses"] for t in fake.calls[1]["tools"]}
    assert budget["web_search"] == agents.RESEARCH_MAX_SEARCHES - meta["searches"]


def test_research_company_without_a_key_returns_no_findings() -> None:
    profile, meta = agents.research_company(
        "https://pollen-robotics.com/", Settings(anthropic_api_key="", _env_file=None)
    )
    assert meta["researched"] is False
    assert profile == CompanyProfile()


def test_not_a_company_is_reported_not_invented(monkeypatch) -> None:
    fake = _FakeClient([([], _final(json.dumps({
        "is_company": False,
        "not_company_reason": "parked domain, no company behind it",
        "company_name": None,
    })))])
    monkeypatch.setattr(agents, "_client", lambda settings, timeout: fake)
    profile, _meta = agents.research_company("https://parked.example/", _settings())
    assert profile.is_company is False
    assert "parked" in profile.not_company_reason
