"""The knowledge graph — evidence turned sideways.

The extraction is where the correctness lives: every edge must trace to a
field the pipeline sourced, node identity must merge spellings (or the
graph fragments into one-off nodes and answers nothing), and junk names
must be dropped rather than becoming the best-connected "investor" in the
database. Persistence is a full rebuild, pinned here as the property that
matters: a corrected verdict RETRACTS the edges it once implied.
"""

from __future__ import annotations

from pathlib import Path

from scout.graph import (
    Edge,
    _acquirer_from_note,
    dedupe,
    edges_for_lead,
    node_key,
)
from scout.models import Account, Lead, LLMVerdict
from scout.store import Store


def _lead(**verdict_kwargs) -> Lead:
    account = verdict_kwargs.pop("account", None) or Account(
        id="1", handle="pollenrobotics", name="Pollen Robotics")
    return Lead(
        account=account,
        llm=LLMVerdict(handle=account.handle, account_type="startup",
                       company_name="Pollen Robotics", **verdict_kwargs),
    )


def _rels(lead: Lead) -> set[tuple[str, str, str]]:
    return {(e.src_label, e.rel, e.dst_label) for e in edges_for_lead(lead)}


# --- node identity ------------------------------------------------------------


def test_aliases_and_spellings_land_on_one_node() -> None:
    """"a16z" and "Andreessen Horowitz" as two nodes would each look like a
    minor backer; as one node they are the pattern the graph exists to show."""
    assert node_key("a16z") == node_key("Andreessen Horowitz")
    assert node_key("Y Combinator") == node_key("YC")
    assert node_key("Eval-HQ.ai é") == node_key("EvalHQ ai")


def test_junk_investor_names_never_become_nodes() -> None:
    lead = _lead(funding_investors=["Bpifrance", "Undisclosed investors",
                                    "angels", "led by Sequoia Capital"],
                 funding_evidence="press release")
    labels = {e.src_label for e in edges_for_lead(lead)
              if e.rel == "invested_in"}
    assert labels == {"Bpifrance", "Sequoia Capital"}  # lead-phrase stripped too


# --- extraction ---------------------------------------------------------------


def test_full_dossier_produces_every_edge_type() -> None:
    account = Account(id="1", handle="pollenrobotics", name="Pollen Robotics",
                      followed_by=["karpathy"])
    lead = Lead(account=account, llm=LLMVerdict(
        handle="pollenrobotics", account_type="startup",
        company_name="Pollen Robotics",
        funding_investors=["Bpifrance"], funding_evidence="techcrunch",
        founders=["Matthieu Lapeyre — co-founder, ex-OpenAI"],
        company_status="acquired",
        company_status_note="Hugging Face, April 2025",
        company_status_evidence="techcrunch 2025-04-14",
    ))
    rels = _rels(lead)
    assert ("Bpifrance", "invested_in", "Pollen Robotics") in rels
    assert ("Matthieu Lapeyre", "founded", "Pollen Robotics") in rels
    assert ("Matthieu Lapeyre", "alum_of", "openai") in rels
    assert ("Pollen Robotics", "acquired_by", "Hugging Face") in rels
    assert ("@karpathy", "follows", "Pollen Robotics") in rels
    # Evidence travels with the edge.
    invest = next(e for e in edges_for_lead(lead) if e.rel == "invested_in")
    assert invest.evidence == "techcrunch"


def test_founder_account_and_bio_lab_edges() -> None:
    """A classified founder account links the PERSON to the company, and
    their bio's past lab to them — the graph works pre-research too."""
    account = Account(id="2", handle="ada_infra", name="Ada Lin",
                      bio="ex-DeepMind. building EvalHQ.")
    lead = Lead(account=account, llm=LLMVerdict(
        handle="ada_infra", account_type="founder", is_founder=True,
        company_name="EvalHQ"))
    rels = _rels(lead)
    assert ("Ada Lin", "founded", "EvalHQ") in rels
    assert any(src == "Ada Lin" and rel == "alum_of" and "deepmind" in dst
               for src, rel, dst in rels)


def test_lab_move_beats_nothing_and_names_the_published_record() -> None:
    account = Account(id="3", handle="quiet_q", name="Quinn A",
                      bio="building something", lab_move="google deepmind → Sparse Labs")
    lead = Lead(account=account, llm=LLMVerdict(
        handle="quiet_q", account_type="founder", is_founder=True))
    edge = next(e for e in edges_for_lead(lead) if e.rel == "alum_of")
    assert edge.dst_label == "google deepmind"
    assert "arXiv" in edge.evidence


def test_acquirer_parsing_is_conservative() -> None:
    assert _acquirer_from_note("Hugging Face, April 2025") == "Hugging Face"
    assert _acquirer_from_note("by BigCo (all-stock)") == "BigCo"
    assert _acquirer_from_note("April 2025") is None  # a date is not a company
    assert _acquirer_from_note("") is None
    # And an acquisition without a parseable counterparty emits no edge.
    lead = _lead(company_status="acquired", company_status_note="2025",
                 company_status_evidence="press")
    assert not any(e.rel == "acquired_by" for e in edges_for_lead(lead))


def test_corporate_account_person_stays_out_of_the_graph() -> None:
    """account_type "startup" means the account IS the company — inventing a
    founded edge from the account's display name would link the company to
    itself under a second name."""
    lead = _lead()
    assert not any(e.rel == "founded" for e in edges_for_lead(lead))


def test_dedupe_keeps_the_first_edge_per_triple() -> None:
    a = Edge(src_type="investor", src_key="x", src_label="X", rel="invested_in",
             dst_type="company", dst_key="y", dst_label="Y", evidence="fresh")
    b = a.model_copy(update={"evidence": "stale"})
    assert [e.evidence for e in dedupe([a, b])] == ["fresh"]


# --- persistence: the rebuild retracts ----------------------------------------


def _save(store: Store, run_id: str, lead: Lead) -> None:
    store.save_leads(run_id, [lead])


def test_rebuild_reflects_the_ledger_and_retracts_corrections(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    _save(store, "run-1", _lead(funding_investors=["Bpifrance"],
                                funding_evidence="press"))
    n = store.rebuild_graph(store.load_lead_ledger())
    assert n >= 1
    assert store.graph_edges(node_key("Bpifrance"))

    # The audit later removes the fabricated investor; the rebuild must
    # forget the edge — incremental upserts never would.
    _save(store, "run-2", _lead(funding_investors=[]))
    store.rebuild_graph(store.load_lead_ledger())
    assert store.graph_edges(node_key("Bpifrance")) == []


def test_hubs_and_related_answer_the_sideways_questions(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")

    def company(handle: str, name: str, investors: list[str]) -> Lead:
        return Lead(
            account=Account(id=handle, handle=handle, name=name),
            llm=LLMVerdict(handle=handle, account_type="startup",
                           company_name=name, funding_investors=investors,
                           funding_evidence="press"),
        )

    _save(store, "r1", company("acme", "Acme", ["Sequoia", "Bpifrance"]))
    _save(store, "r2", company("beta", "Beta", ["Sequoia"]))
    _save(store, "r3", company("gamma", "Gamma", ["Lone VC"]))
    store.rebuild_graph(store.load_lead_ledger())

    hubs = store.graph_hubs("invested_in", end="src")
    assert hubs[0][1] == "Sequoia" and hubs[0][2] == 2

    related = store.graph_related(node_key("Acme"))
    assert [(r["company_label"], r["via_label"]) for r in related] == [
        ("Beta", "Sequoia")]


def test_watchlist_candidates_names_multi_company_connectors_not_handles(tmp_path: Path) -> None:
    """Investors/labs touching 2+ tracked companies, minus what the
    watchlist covers — and NAMES only: guessing an X handle from a firm name
    is the error the graph refuses everywhere else."""
    from scout.graph import watchlist_candidates

    edges = [
        # Sequoia backs two companies; Bpifrance one; openai has two alumni.
        {"src_type": "investor", "src_key": "sequoia", "src_label": "Sequoia",
         "rel": "invested_in", "dst_type": "company", "dst_key": "a",
         "dst_label": "A", "evidence": ""},
        {"src_type": "investor", "src_key": "sequoia", "src_label": "Sequoia",
         "rel": "invested_in", "dst_type": "company", "dst_key": "b",
         "dst_label": "B", "evidence": ""},
        {"src_type": "investor", "src_key": "bpifrance", "src_label": "Bpifrance",
         "rel": "invested_in", "dst_type": "company", "dst_key": "a",
         "dst_label": "A", "evidence": ""},
        {"src_type": "person", "src_key": "ada", "src_label": "Ada",
         "rel": "alum_of", "dst_type": "lab", "dst_key": "openai",
         "dst_label": "openai", "evidence": ""},
        {"src_type": "person", "src_key": "sam", "src_label": "Sam",
         "rel": "alum_of", "dst_type": "lab", "dst_key": "openai",
         "dst_label": "openai", "evidence": ""},
    ]
    ranked = watchlist_candidates(edges, current_watchers=[])
    assert ranked == [("openai", 2), ("Sequoia", 2)]  # count desc, then name
    # Already watched (any spelling) → not suggested again.
    assert watchlist_candidates(edges, ["sequoia"]) == [("openai", 2)]
