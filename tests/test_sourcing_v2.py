"""Sourcing v2 tests: follow-edge diffing, bio snapshots, unlinked leads,
Seeds v2 parsing, and the GitHub/HN pure parsers (fixture JSON, no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scout.config import Seeds
from scout.ingest.github_src import parse_repo_owners, profile_to_x_handle
from scout.ingest.hn_src import extract_x_handle, parse_hn_hits
from scout.models import UnlinkedLead
from scout.store import Store


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


# --- follow-edge snapshots -----------------------------------------------------


def test_first_snapshot_is_baseline_not_recent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_follow_snapshot("eladgil", ["alice", "bob"])
    # Cold start: baseline edges have unknown real age — never "recent".
    assert store.recent_watchers_for("alice", days=30) == []


def test_new_edge_after_baseline_is_recent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_follow_snapshot("eladgil", ["alice"])
    store.record_follow_snapshot("eladgil", ["alice", "newfounder"])
    assert store.recent_watchers_for("newfounder", days=30) == ["eladgil"]
    # And the pre-existing edge still doesn't count.
    assert store.recent_watchers_for("alice", days=30) == []


def test_convergence_two_watchers_same_followee(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_follow_snapshot("eladgil", ["x"])
    store.record_follow_snapshot("saranormous", ["y"])
    store.record_follow_snapshot("eladgil", ["x", "hotfounder"])
    store.record_follow_snapshot("saranormous", ["y", "hotfounder"])
    recent = store.recent_watchers_for("hotfounder", days=30)
    assert sorted(recent) == ["eladgil", "saranormous"]


def test_follow_edges_case_insensitive_and_at_stripped(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_follow_snapshot("EladGil", ["seed"])
    store.record_follow_snapshot("@eladgil", ["seed", "NewFounder"])
    assert store.recent_watchers_for("newfounder", days=30) == ["eladgil"]


def test_reobserved_edge_keeps_first_seen(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_follow_snapshot("w", ["base"])
    store.record_follow_snapshot("w", ["base", "f"])  # f first seen here
    store.record_follow_snapshot("w", ["base", "f"])  # re-observed
    rows = list(store.db["follow_edges"].rows_where("followee = ?", ["f"]))
    assert len(rows) == 1
    assert rows[0]["first_seen"] <= rows[0]["last_seen"]


# --- bio snapshots --------------------------------------------------------------


def test_record_bio_returns_previous_on_change(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.record_bio("alice", "ml engineer") is None  # first sighting
    assert store.record_bio("alice", "stealth — building") == "ml engineer"


def test_record_bio_unchanged_returns_same(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.record_bio("alice", "same bio")
    assert store.record_bio("alice", "same bio") == "same bio"


# --- unlinked leads --------------------------------------------------------------


def test_unlinked_leads_roundtrip_and_upsert(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    now = datetime.now(timezone.utc)
    lead = UnlinkedLead(source="github", ref="coolhacker", bio="agents", url="u", found_at=now)
    store.upsert_unlinked_leads([lead, lead])  # duplicate upsert is fine
    loaded = store.load_unlinked_leads(days=7)
    assert len(loaded) == 1
    assert loaded[0].ref == "coolhacker"


# --- Seeds v2 --------------------------------------------------------------------


def test_seeds_all_searches_categories() -> None:
    seeds = Seeds(
        searches=["legacy q"],
        searches_departure=["dep q"],
        searches_stealth_intent=["stealth q"],
        searches_hiring=["hire q"],
    )
    pairs = seeds.all_searches
    assert ("departure", "dep q") in pairs
    assert ("stealth_intent", "stealth q") in pairs
    assert ("hiring", "hire q") in pairs
    assert ("search", "legacy q") in pairs


def test_seeds_watchers_merges_and_dedupes() -> None:
    seeds = Seeds(watchlist=["@EladGil", "swyx"], tastemakers=["eladgil", "karpathy"])
    assert seeds.watchers == ["EladGil", "swyx", "karpathy"]


# --- stage targeting -------------------------------------------------------------


def test_launched_stage_activates_launch_and_hn_not_departure() -> None:
    from scout.config import Thesis

    thesis = Thesis(target_stages=["launched"])
    assert "launch" in thesis.active_search_categories
    assert "hiring" in thesis.active_search_categories
    assert "departure" not in thesis.active_search_categories
    assert "hn" in thesis.active_discovery_sources
    assert thesis.bio_graph_active is True  # launched still uses bio search


def test_scaling_stage_drops_bio_graph() -> None:
    from scout.config import Thesis

    thesis = Thesis(target_stages=["scaling"])
    assert thesis.bio_graph_active is False
    assert thesis.active_search_categories == {"launch", "search"}


def test_all_searches_includes_launch_category() -> None:
    seeds = Seeds(searches_launch=["we just launched"])
    assert ("launch", "we just launched") in seeds.all_searches


# --- GitHub parsers ---------------------------------------------------------------


def test_parse_repo_owners_dedupes_by_login() -> None:
    response = {
        "items": [
            {
                "html_url": "https://github.com/ada/agents",
                "full_name": "ada/agents",
                "stargazers_count": 42,
                "owner": {"login": "ada", "type": "User"},
            },
            {
                "html_url": "https://github.com/ada/other",
                "full_name": "ada/other",
                "stargazers_count": 7,
                "owner": {"login": "Ada", "type": "User"},
            },
        ]
    }
    owners = parse_repo_owners(response)
    assert len(owners) == 1
    assert owners[0]["repo_name"] == "ada/agents"


def test_profile_to_x_handle_prefers_twitter_username() -> None:
    assert profile_to_x_handle({"twitter_username": "@ada_infra"}, []) == "ada_infra"


def test_profile_to_x_handle_falls_back_to_socials() -> None:
    socials = [
        {"provider": "generic", "url": "https://ada.dev"},
        {"provider": "twitter", "url": "https://x.com/ada_builds/"},
    ]
    assert profile_to_x_handle({"twitter_username": None}, socials) == "ada_builds"
    assert profile_to_x_handle({}, []) is None


# --- HN parsers --------------------------------------------------------------------


def test_extract_x_handle_skips_reserved_paths() -> None:
    assert extract_x_handle("see https://x.com/intent/post?text=hi and x.com/realfounder") == "realfounder"
    assert extract_x_handle("no links here") is None


def test_parse_hn_hits_bridges_and_unlinks() -> None:
    now = datetime.now(timezone.utc)
    hits = [
        {
            "author": "builder1",
            "title": "Show HN: Agent evals platform",
            "story_text": "By https://twitter.com/eval_erin",
            "url": "https://evals.dev",
            "objectID": "1",
        },
        {
            "author": "quietdev",
            "title": "Show HN: LLM inference cache",
            "story_text": "no socials",
            "url": "https://cache.dev",
            "objectID": "2",
        },
    ]
    accounts, unlinked = parse_hn_hits(hits, now=now)
    assert [a.handle for a in accounts] == ["eval_erin"]
    assert accounts[0].source == "hn"
    assert [u.ref for u in unlinked] == ["quietdev"]


# --- deal-flow pipeline ----------------------------------------------------------


def test_pipeline_set_get_and_partial_merge(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("@Ada_Infra", status="shortlisted")
    store.set_pipeline("ada_infra", notes="strong ex-OpenAI")  # partial update, different casing
    row = store.get_pipeline("ada_infra")
    assert row["status"] == "shortlisted"       # preserved
    assert row["notes"] == "strong ex-OpenAI"   # merged in
    assert store.pipeline_counts()["shortlisted"] == 1


def test_pipeline_outreach_survives_status_change(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_pipeline("f", outreach="hi there", channel="X DM")
    store.set_pipeline("f", status="contacted")
    row = store.get_pipeline("f")
    assert row["outreach"] == "hi there"
    assert row["status"] == "contacted"


def test_outreach_template_without_key() -> None:
    from scout.config import Settings, Thesis
    from scout.models import Account, Lead, LLMVerdict
    from scout.outreach import draft_outreach

    lead = Lead(account=Account(id="1", handle="ada_infra", name="Ada Lin"),
                llm=LLMVerdict(handle="ada_infra", one_line_summary="an eval platform for agents"))
    settings = Settings(anthropic_api_key=None)
    msg, is_ai = draft_outreach(lead, Thesis(thesis="AI infra"), settings, "X DM")
    assert is_ai is False
    assert "eval platform" in msg.lower()
