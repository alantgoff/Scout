"""Orchestration tests with a stubbed research_call: failure -> gap finding,
cost-cap stop mid-run, fingerprint cache hit, cross-exam downgrade
application. No network — the stub returns canned per-stage responses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scout.config import Settings, Thesis
from scout.diligence import pipeline
from scout.diligence.research import ResearchError
from scout.diligence.schema import DIMENSION_KEYS
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.store import Store


def make_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        anthropic_api_key="test-key",
        diligence_concurrency=1,  # deterministic budget-gating order
        diligence_cost_cap_usd=8.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_thesis() -> Thesis:
    return Thesis(thesis="vertical agents on proprietary data")


def seed_company(store: Store) -> None:
    lead = Lead(
        account=Account(
            id="1", handle="tuva_founder", name="Tuva Founder",
            bio="building Tuva AI", website="https://tuva.ai", followers=900,
            source="search",
        ),
        signals=[Signal(name="launch_traction", value=1.0, weight=10.0)],
        llm=LLMVerdict(
            handle="tuva_founder", account_type="founder", is_founder=True,
            stage="launched", sector="health ai", subsector="clinical agents",
            business_model="b2b saas", company_name="Tuva AI",
            company_url="https://tuva.ai", one_line_summary="Health data agents.",
            thesis_fit=0.72, confidence=0.9,
        ),
        score=64.0, rank=1,
    )
    store.save_leads("run-1", [lead])


def finding_json(key: str, score: float = 7.0, **overrides) -> str:
    data = {
        "key": key, "score": score, "confidence": 0.8,
        "summary": f"{key} looks fine", "detail_md": "## steps",
        "evidence": [{"claim": "c", "url": "https://tuva.ai", "source": "web"}],
        "unknowns": [], "flags": [], "payload": {},
    }
    if key == "competition":
        data["payload"] = {
            "competitors": [{"name": "RivalCo", "url": "https://rival.co"}],
            "positioning_axis": "vertical", "whitespace": "on-prem",
        }
    if key == "wrapper_vs_proprietary":
        data["payload"] = {"classification": "fine_tuned", "model_dependency": "claude",
                           "evidence_trail": []}
    data.update(overrides)
    return json.dumps(data)


RECON_JSON = json.dumps(
    {
        "company_name": "Tuva AI", "website": "https://tuva.ai",
        "product_summary": "health data agents",
        "founders": [{"name": "Jane Doe", "role": "CEO", "background_hints": "ex-OpenAI"}],
        "funding_events": [], "github_org": "",
        "key_sources": [{"url": "https://tuva.ai/docs", "title": "Docs", "what": "product"}],
    }
)
CROSS_EXAM_JSON = json.dumps({"downgrades": {}, "contradictions": []})
MEMO_MD = "## Recommendation\nPursue.\n\n## Company snapshot\nTuva AI.\n\n### Sources\n[1] https://tuva.ai"


class StubResearch:
    """research_call stand-in: canned response per stage, call log, optional
    per-stage failures, fixed cost per call."""

    def __init__(self, cost_per_call: float = 0.2,
                 fail_stages: set[str] | None = None) -> None:
        self.cost_per_call = cost_per_call
        self.fail_stages = fail_stages or set()
        self.calls: list[str] = []

    def responses(self, stage: str) -> str:
        if stage == "recon":
            return RECON_JSON
        if stage == "cross_exam":
            return CROSS_EXAM_JSON
        if stage == "memo":
            return MEMO_MD
        return finding_json(stage)

    def __call__(self, system, user, parse, settings, store=None, *, model=None,
                 company_key="", stage="", web=True, max_tokens=8192):
        self.calls.append(stage)
        if stage in self.fail_stages:
            raise ResearchError(f"{stage} exploded", cost_usd=self.cost_per_call)
        return parse(self.responses(stage)), self.cost_per_call


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "dg.db")
    seed_company(store)
    return store


def test_full_run_persists_memo_and_kg(store, monkeypatch) -> None:
    stub = StubResearch()
    monkeypatch.setattr(pipeline, "research_call", stub)
    memo = pipeline.run_analysis("Tuva AI", store, make_thesis(), make_settings())

    assert memo.company_key == "tuvaai"
    assert [f.key for f in memo.findings] == list(DIMENSION_KEYS)
    assert memo.composite == pytest.approx(70.0)
    assert memo.memo_md == MEMO_MD
    assert memo.cost_usd == pytest.approx(0.2 * 11)  # recon + 8 dims + exam + memo
    assert stub.calls[0] == "recon"
    assert stub.calls[-1] == "memo"
    # Persisted + KG ingested.
    assert store.load_memo("tuvaai") is not None
    competitor_names = [n["name"] for n in store.kg_neighbors("tuvaai")
                        if n["rel"] == "competes_with"]
    assert "RivalCo" in competitor_names
    # Usage: the stub bypasses the ledger, but the spend is on the memo.
    assert memo.model == make_settings().claude_model


def test_resolves_by_handle_too(store, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "research_call", StubResearch())
    memo = pipeline.run_analysis("@Tuva_Founder", store, make_thesis(), make_settings())
    assert memo.company_key == "tuvaai"


def test_dimension_failure_becomes_gap_finding(store, monkeypatch) -> None:
    stub = StubResearch(fail_stages={"data_flywheel"})
    monkeypatch.setattr(pipeline, "research_call", stub)
    memo = pipeline.run_analysis("tuva ai", store, make_thesis(), make_settings())

    gap = memo.finding("data_flywheel")
    assert gap is not None and gap.score is None
    assert gap.flags == ["analysis_failed"]
    # The other seven dimensions scored; composite excludes the gap.
    assert sum(1 for f in memo.findings if f.score is not None) == 7
    assert memo.composite == pytest.approx(70.0)
    # The failed call's spend still counted against the budget.
    assert memo.cost_usd == pytest.approx(0.2 * 11)


def test_cost_cap_stops_launching_and_flags_gap(store, monkeypatch) -> None:
    stub = StubResearch(cost_per_call=0.2)
    monkeypatch.setattr(pipeline, "research_call", stub)
    # Cap allows recon (0.2) + two dimensions (0.6 > 0.5 stops the rest).
    memo = pipeline.run_analysis(
        "tuvaai", store, make_thesis(), make_settings(diligence_cost_cap_usd=0.5)
    )
    skipped = [f for f in memo.findings if "skipped_budget_cap" in f.flags]
    scored = [f for f in memo.findings if f.score is not None]
    assert len(skipped) == 6 and len(scored) == 2
    assert "budget_capped" in memo.flags
    # Cross-exam and memo synthesis never launched; fallback memo renders.
    assert "cross_exam" not in stub.calls and "memo" not in stub.calls
    assert "memo_synthesis_unavailable" in memo.flags
    assert "Memo synthesis unavailable" in memo.memo_md
    assert memo.composite is not None  # completed findings kept


def test_fingerprint_cache_hit_skips_research(store, monkeypatch) -> None:
    stub = StubResearch()
    monkeypatch.setattr(pipeline, "research_call", stub)
    settings, thesis = make_settings(), make_thesis()
    first = pipeline.run_analysis("tuvaai", store, thesis, settings)
    calls_after_first = len(stub.calls)

    second = pipeline.run_analysis("tuvaai", store, thesis, settings)
    assert len(stub.calls) == calls_after_first  # served from cache, $0
    assert second.created_at == first.created_at

    # --refresh forces a re-run; so does a changed thesis (fingerprint miss).
    pipeline.run_analysis("tuvaai", store, thesis, settings, refresh=True)
    assert len(stub.calls) > calls_after_first


def test_changed_thesis_invalidates_cache(store, monkeypatch) -> None:
    stub = StubResearch()
    monkeypatch.setattr(pipeline, "research_call", stub)
    settings = make_settings()
    pipeline.run_analysis("tuvaai", store, make_thesis(), settings)
    n = len(stub.calls)
    pipeline.run_analysis(
        "tuvaai", store, Thesis(thesis="a completely different thesis"), settings
    )
    assert len(stub.calls) > n


def test_cross_exam_downgrade_applied_before_persist(store, monkeypatch) -> None:
    stub = StubResearch()
    downgrades = json.dumps(
        {"downgrades": {"market_size": 0.2}, "contradictions": ["TAM math off"]}
    )
    stub.responses = lambda stage: (  # type: ignore[method-assign]
        RECON_JSON if stage == "recon"
        else downgrades if stage == "cross_exam"
        else MEMO_MD if stage == "memo"
        else finding_json(stage)
    )
    monkeypatch.setattr(pipeline, "research_call", stub)
    memo = pipeline.run_analysis("tuvaai", store, make_thesis(), make_settings())
    market = memo.finding("market_size")
    assert market is not None
    assert market.confidence == 0.2
    assert "cross_exam_downgrade" in market.flags
    # Persisted memo carries the downgrade too.
    assert store.load_memo("tuvaai").finding("market_size").confidence == 0.2


def test_unknown_company_raises(store) -> None:
    with pytest.raises(RuntimeError, match="no tracked startup"):
        pipeline.run_analysis("nonexistentco", store, make_thesis(), make_settings())


def test_missing_api_key_raises(store) -> None:
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pipeline.run_analysis(
            "tuvaai", store, make_thesis(), make_settings(anthropic_api_key=None)
        )


def test_demo_lead_refused(tmp_path: Path) -> None:
    """Mirrors `scout verify`'s guard: never spend real money on synthetic
    sample founders (it would also pollute the knowledge graph)."""
    store = Store(tmp_path / "demo.db")
    lead = Lead(
        account=Account(id="d1", handle="demo_maya", bio="demo bio", source="demo"),
        llm=LLMVerdict(handle="demo_maya", account_type="founder", is_founder=True,
                       stage="launched", company_name="FakeCo", confidence=0.9),
        score=50.0,
    )
    store.save_leads("demo-1", [lead])
    with pytest.raises(RuntimeError, match="demo lead"):
        pipeline.run_analysis("FakeCo", store, make_thesis(), make_settings())


def test_evidence_pack_contents(store) -> None:
    pack = pipeline.build_evidence_pack(store, "Tuva AI")
    assert pack is not None
    assert pack.company_key == "tuvaai"
    assert pack.website == "https://tuva.ai"
    assert "https://tuva.ai" in pack.digest  # verbatim URL for web_fetch
    assert "launch_traction" in pack.digest
    assert pack.sectors == ["health ai", "clinical agents"]


def test_kg_seeding_reaches_competition_prompt(store, monkeypatch) -> None:
    # A previously mapped company in the same sector shows up in the pack.
    store.kg_upsert_node("company:oldco", "company", "OldCo")
    store.kg_upsert_node("sector:healthai", "sector", "health ai")
    store.kg_upsert_edge("company:oldco", "in_sector", "sector:healthai")
    pack = pipeline.build_evidence_pack(store, "tuvaai")
    assert "OldCo" in pack.kg_competitors
    assert "previously mapped" in pack.digest.lower()
