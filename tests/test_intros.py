"""Warm-intro paths — who in the firm's network connects to a company.

A path is only worth showing if the partner reading it would act on it,
and only honest if two CITED facts meet. So these tests pin the grading
(who you'd actually ask), that aliases land on one node (a16z and
Andreessen Horowitz are one co-investor), that nothing paths to itself,
and that portfolio companies get no paths — they are already yours.
Edges come from the production derivation (graph.edges_for_lead), not
hand-built dicts, so the tests break if the graph's naming changes.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from scout.cli import app
from scout.graph import dedupe, edges_for_lead
from scout.intros import network_from, parse_list, warm_paths, warmth
from scout.models import Account, Lead, LedgerEntry, LLMVerdict
from scout.store import Store

runner = CliRunner()


def _lead(handle: str, company: str, *, founders=(), investors=(), followed_by=()) -> Lead:
    return Lead(
        account=Account(id=handle, handle=handle, name=company,
                        followed_by=list(followed_by)),
        llm=LLMVerdict(handle=handle, account_type="startup", company_name=company,
                       founders=list(founders), funding_investors=list(investors),
                       funding_stage="seed" if investors else "unknown",
                       funding_evidence="TechCrunch, 2026-09" if investors else None,
                       grounding="website"),
    )


LEADS = [
    _lead("beta", "Beta", founders=["Maya Chen — CEO, ex-DeepMind"],
          investors=["Accel", "a16z"]),                                   # portfolio
    _lead("gamma", "Gamma", founders=["Raj Patel — CTO, ex-OpenAI"],
          investors=["Sequoia"]),                                         # met
    _lead("acme", "Acme", founders=["Sam Lee — CEO, ex-DeepMind", "Maya Chen — cofounder"],
          investors=["Andreessen Horowitz", "Index Ventures"]),
    _lead("delta", "Delta", investors=["Headline"]),
    _lead("eps", "Epsilon", founders=["Raj Patel — CEO"], investors=["Index Ventures"]),
    _lead("zeta", "Zeta", followed_by=["eladgil"]),
    _lead("omega", "Omega", founders=["Nobody Known — CEO"], investors=["Unknown Fund"]),
]
PIPELINE = {"beta": {"status": "won"}, "gamma": {"status": "meeting"},
            "acme": {"status": "longlisted"}}


def _edges() -> list[dict]:
    return [e.model_dump() for e in dedupe([e for lead in LEADS for e in edges_for_lead(lead)])]


def _paths(**network_kw):
    ledger = [LedgerEntry(lead=lead) for lead in LEADS]
    net = network_from(ledger, PIPELINE, firm_name="Headline", **network_kw)
    return net, warm_paths(_edges(), net)


def _kinds(paths) -> list[tuple[str, int, str]]:
    return [(p.kind, p.strength, p.via) for p in paths]


# --- the network ------------------------------------------------------------------


def test_network_comes_from_statuses_and_the_firms_lists() -> None:
    net, _ = _paths(investors_text="Index Ventures, Lux Capital",
                    people_text="Elad Gil (@eladgil)\n@sarahguo\nJane Advisor")
    assert net.portfolio == {"beta": "Beta"}
    assert net.met == {"gamma": "Gamma"}
    assert set(net.investors) == {"indexventures", "luxcapital"}
    assert set(net.people) == {"eladgil", "janeadvisor"}
    assert net.handles == {"eladgil": "@eladgil", "sarahguo": "@sarahguo"}
    assert parse_list(" a,\n\n b , ") == ["a", "b"]


# --- the paths --------------------------------------------------------------------


def test_a_company_gets_every_path_graded_strongest_first() -> None:
    _, paths = _paths()
    acme = _kinds(paths["acme"])
    # Maya founded Acme AND your portfolio company Beta: the strongest tie.
    assert acme[0] == ("founder_known", 3, "Maya Chen")
    # "Andreessen Horowitz" on Acme and "a16z" on Beta are one co-investor.
    assert ("co_investor", 2, "Andreessen Horowitz") in acme
    # Sam and Maya (known via Beta) are both DeepMind alumni: a weak path.
    assert ("shared_lab", 1, "Maya Chen") in acme
    text = {p.kind: p.text for p in paths["acme"]}
    assert text["co_investor"] == ("Andreessen Horowitz backs Acme — and co-invested "
                                   "with you in Beta")
    assert "deepmind alumni" in text["shared_lab"]
    assert warmth(paths["acme"]) == 3


def test_the_firm_already_on_the_cap_table_is_the_first_thing_said() -> None:
    _, paths = _paths()
    assert _kinds(paths["delta"]) == [("already_backed", 3, "Headline")]


def test_a_founder_you_have_met_and_a_fund_you_listed() -> None:
    _, paths = _paths(investors_text="Index Ventures")
    # Graph nodes are keyed by COMPANY name, not handle: "eps" → "epsilon".
    eps = _kinds(paths["epsilon"])
    assert ("founder_known", 2, "Raj Patel") in eps       # met on Gamma
    assert ("co_investor", 2, "Index Ventures") in eps    # a fund you listed
    assert any("you've met them on Gamma" in p.text for p in paths["epsilon"])


def test_a_listed_person_following_on_x_is_the_weakest_path() -> None:
    _, paths = _paths(people_text="Elad Gil (@eladgil)")
    assert _kinds(paths["zeta"]) == [("watcher", 1, "@eladgil")]


def test_no_company_paths_to_itself_and_portfolio_gets_none() -> None:
    _, paths = _paths()
    assert "beta" not in paths          # already yours
    # Gamma is met via Raj; Raj founding Gamma is not a path TO Gamma.
    assert "gamma" not in paths
    # Nothing in the network touches Omega.
    assert "omega" not in paths


def test_no_network_means_no_paths_and_no_work() -> None:
    ledger = [LedgerEntry(lead=lead) for lead in LEADS]
    net = network_from(ledger, {}, firm_name="")
    assert warm_paths(_edges(), net) == {}
    assert warmth(None) == 0


# --- the commands -----------------------------------------------------------------


def _seed(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    store = Store(db, actor="partner:alan")
    store.save_leads("20260930-090000-000000", LEADS)
    for handle, row in PIPELINE.items():
        store.set_pipeline(handle, status=row["status"])
    store.rebuild_graph(store.load_lead_ledger())
    return db


def test_scout_network_sets_the_lists_and_scout_intros_uses_them(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    env = {"DB_PATH": str(db)}
    result = runner.invoke(app, ["network", "--investors", "Index Ventures",
                                 "--people", "Elad Gil (@eladgil)"], env=env)
    assert result.exit_code == 0, result.output
    assert "Index Ventures" in result.output and "@eladgil" in result.output
    assert "Beta" in result.output  # portfolio, derived from status

    listing = runner.invoke(app, ["intros"], env=env)
    assert listing.exit_code == 0, listing.output
    assert "Acme" in listing.output and "Epsilon" in listing.output
    # Beta is named inside Acme's path ("your portfolio company Beta") but is
    # never a ROW: a portfolio company is not a target.
    import re
    assert re.search(r"[●○]{3}\s+Acme\s", listing.output)
    assert not re.search(r"[●○]{3}\s+Beta\s", listing.output)

    one = runner.invoke(app, ["intros", "acme"], env=env)
    assert one.exit_code == 0, one.output
    assert "Maya Chen" in one.output and "Andreessen Horowitz" in one.output

    missing = runner.invoke(app, ["intros", "nothing-here"], env=env)
    assert missing.exit_code == 1
