"""Knowledge-graph tests: node-id normalization (reuses the company_key
normalizer), ingest from findings, edge merge with user-provenance
protection, and the sector seeding query."""

from __future__ import annotations

from pathlib import Path

from scout.diligence.graph import ingest, node_id, normalize_name
from scout.diligence.schema import DimensionFinding, Memo, ReconReport
from scout.store import Store


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "kg.db")


# --- node ids ----------------------------------------------------------------


def test_company_node_id_reuses_company_key_normalizer() -> None:
    assert node_id("company", "Eval-HQ.ai") == "company:evalhq"
    assert node_id("company", "EvalHQ") == "company:evalhq"
    assert node_id("company", "eval hq") == "company:evalhq"


def test_person_and_sector_ids() -> None:
    assert node_id("person", "Jane Doe") == "person:janedoe"
    assert node_id("sector", "AI Infra") == "sector:aiinfra"
    # No TLD stripping outside company nodes.
    assert node_id("org", "x.ai") == "org:xai"


def test_unusable_names_return_none() -> None:
    assert node_id("company", "X") is None
    assert node_id("person", "!") is None
    assert normalize_name("") is None


# --- ingest from findings -----------------------------------------------------


def make_memo() -> Memo:
    return Memo(
        company_key="tuvaai",
        company_name="Tuva AI",
        recon=ReconReport(website="https://tuva.ai"),
        findings=[
            DimensionFinding(
                key="competition",
                payload={
                    "competitors": [
                        {"name": "RivalCo", "url": "https://rival.co", "stage": "seed", "note": ""},
                        {"name": "", "url": "https://ignored.com"},  # nameless — dropped
                        {"name": "Tuva AI"},  # self — dropped
                    ],
                    "positioning_axis": "vertical",
                    "whitespace": "on-prem",
                },
            ),
            DimensionFinding(
                key="founder_pedigree",
                payload={
                    "founders": [
                        {
                            "name": "Jane Doe",
                            "verified_background": ["OpenAI — research engineer, 2021-2024"],
                            "prior_exits": [],
                        }
                    ]
                },
            ),
        ],
    )


def test_ingest_builds_competitors_founders_sectors() -> None:
    nodes, edges = ingest(make_memo(), sectors=["health ai"])
    ids = {n["id"] for n in nodes}
    assert {"company:tuvaai", "company:rivalco", "person:janedoe",
            "org:openai", "sector:healthai"} <= ids
    rels = {(e["src"], e["rel"], e["dst"]) for e in edges}
    assert ("company:tuvaai", "competes_with", "company:rivalco") in rels
    assert ("company:rivalco", "competes_with", "company:tuvaai") in rels  # both ways
    assert ("company:tuvaai", "founded_by", "person:janedoe") in rels
    assert ("person:janedoe", "worked_at", "org:openai") in rels
    assert ("company:tuvaai", "in_sector", "sector:healthai") in rels
    assert all(e["provenance"] == "agent" for e in edges)
    assert all(e["memo_ref"] == "tuvaai" for e in edges)
    # Nameless and self-referential competitors were dropped.
    assert "company:ignoredcom" not in ids


# --- store persistence --------------------------------------------------------


def test_kg_roundtrip_and_neighbors(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    nodes, edges = ingest(make_memo(), sectors=["health ai"])
    for node in nodes:
        store.kg_upsert_node(**node)
    for edge in edges:
        store.kg_upsert_edge(**edge)
    neighbors = store.kg_neighbors("tuvaai")
    by_rel = {}
    for n in neighbors:
        by_rel.setdefault(n["rel"], []).append(n["name"])
    assert "RivalCo" in by_rel["competes_with"]
    assert "Jane Doe" in by_rel["founded_by"]
    assert "health ai" in by_rel["in_sector"]


def test_user_edges_never_overwritten_by_agents(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.kg_link_competitors("Tuva AI", "HandPicked Co") is True
    # Agent re-analysis tries to write the same edge with its own metadata.
    store.kg_upsert_edge(
        "company:tuvaai", "competes_with", "company:handpickedco",
        provenance="agent", memo_ref="tuvaai",
    )
    rows = list(store.db["kg_edges"].rows_where(
        "src = ? and dst = ?", ["company:tuvaai", "company:handpickedco"]
    ))
    assert rows[0]["provenance"] == "user"  # survived the agent write
    # But a user write CAN update an agent edge.
    store.kg_upsert_edge("company:tuvaai", "competes_with", "company:rivalco",
                         provenance="agent")
    store.kg_upsert_edge("company:tuvaai", "competes_with", "company:rivalco",
                         provenance="user")
    rows = list(store.db["kg_edges"].rows_where(
        "src = ? and dst = ?", ["company:tuvaai", "company:rivalco"]
    ))
    assert rows[0]["provenance"] == "user"


def test_kg_link_competitors_writes_both_directions(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.kg_link_competitors("Alpha Corp", "Beta Inc") is True
    rels = {(r["src"], r["dst"]) for r in store.db["kg_edges"].rows}
    assert ("company:alphacorp", "company:betainc") in rels
    assert ("company:betainc", "company:alphacorp") in rels
    # Degenerate names refuse cleanly.
    assert store.kg_link_competitors("X", "Beta Inc") is False
    assert store.kg_link_competitors("Alpha Corp", "Alpha-Corp") is False  # self


def test_kg_competitors_for_sectors_seeds_next_analysis(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for node in ({"id": "company:oldco", "type": "company", "name": "OldCo"},
                 {"id": "company:tuvaai", "type": "company", "name": "Tuva AI"},
                 {"id": "sector:healthai", "type": "sector", "name": "health ai"}):
        store.kg_upsert_node(**node)
    store.kg_upsert_edge("company:oldco", "in_sector", "sector:healthai")
    store.kg_upsert_edge("company:tuvaai", "in_sector", "sector:healthai")
    # Analysis #2 in the same sector starts from the map built by #1 —
    # excluding the company being analyzed.
    names = store.kg_competitors_for_sectors(["healthai"], exclude_company_key="tuvaai")
    assert names == ["OldCo"]
    assert store.kg_competitors_for_sectors([]) == []


def test_kg_node_merge_keeps_richer_data(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.kg_upsert_node("company:rivalco", "company", "RivalCo",
                         {"url": "https://rival.co"})
    store.kg_upsert_node("company:rivalco", "company", "", {})  # sparse re-ingest
    row = list(store.db["kg_nodes"].rows_where("id = ?", ["company:rivalco"]))[0]
    assert row["name"] == "RivalCo"
    assert "rival.co" in row["meta"]
