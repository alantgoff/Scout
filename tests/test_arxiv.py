"""arXiv discovery: parsing, bridging, and affiliation history.

No network. The parsers run against a fixture in arXiv's documented Atom
format; the store half uses a tmp DB. The property that matters most is the
one about missing data: arXiv's affiliation field is optional and often
absent, and a gap must never be read as a move.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scout.ingest.arxiv_src import (
    build_query,
    extract_github_repo,
    extract_x_handle,
    is_top_lab,
    normalize_author,
    paper_links,
    parse_entries,
    to_unlinked,
)
from scout.store import Store, _author_key

# Two authors with affiliations, one without — the realistic mix.
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2026-07-02T18:00:00Z</published>
    <updated>2026-07-02T18:00:00Z</updated>
    <title>Sparse   Routing for
      Agentic Evaluation</title>
    <summary>We present a router. Code and weights at
      https://github.com/janechen/sparse-router and updates at
      https://x.com/janechen_ai for the curious.</summary>
    <author><name>Jane Chen</name><arxiv:affiliation>Google DeepMind</arxiv:affiliation></author>
    <author><name>Wei Zhang</name><arxiv:affiliation>Stanford AI Lab</arxiv:affiliation></author>
    <author><name>Sam Okafor</name></author>
    <arxiv:comment>NeurIPS 2026 camera ready</arxiv:comment>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.99999v2</id>
    <published>2026-07-20T09:00:00Z</published>
    <title>A Second Paper</title>
    <summary>No links here.</summary>
    <author><name>Jane Chen</name><arxiv:affiliation>Sparse Labs</arxiv:affiliation></author>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


# --- parsing --------------------------------------------------------------------


def test_parse_entries_reads_papers_authors_and_affiliations() -> None:
    papers = parse_entries(FEED)
    assert len(papers) == 2
    first = papers[0]
    assert first["arxiv_id"] == "2401.12345v1"
    # Atom wraps titles across lines; whitespace is collapsed.
    assert first["title"] == "Sparse Routing for Agentic Evaluation"
    assert first["categories"] == ["cs.LG", "cs.AI"]
    assert first["comment"] == "NeurIPS 2026 camera ready"
    assert [a["name"] for a in first["authors"]] == [
        "Jane Chen", "Wei Zhang", "Sam Okafor"]
    assert first["authors"][0]["affiliation"] == "Google DeepMind"
    # The optional field simply is not there for the third author.
    assert first["authors"][2]["affiliation"] == ""


def test_parse_entries_survives_malformed_and_empty_input() -> None:
    assert parse_entries("not xml at all") == []
    assert parse_entries("") == []
    # A feed whose entry has no authors is skipped rather than half-stored.
    assert parse_entries(
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>http://arxiv.org/abs/1</id></entry></feed>") == []


def test_links_are_extracted_from_abstract_and_comment() -> None:
    links = paper_links(parse_entries(FEED)[0])
    assert links["x"] == "janechen_ai"
    assert links["github_repo"] == "janechen/sparse-router"
    # The second paper links nothing.
    assert paper_links(parse_entries(FEED)[1]) == {}


def test_handle_and_repo_extraction_reject_non_identities() -> None:
    assert extract_x_handle("see https://x.com/i/status/123") is None
    assert extract_x_handle("https://twitter.com/@ada_infra now") == "ada_infra"
    assert extract_github_repo("https://github.com/features/copilot") is None
    assert extract_github_repo("github.com/torvalds") is None  # user, not repo
    assert extract_github_repo("code: github.com/openai/gym.") == "openai/gym"


def test_author_keys_merge_the_same_person_conservatively() -> None:
    assert normalize_author("Yann  LeCun ") == normalize_author("yann lecun")
    assert normalize_author("J. Chen-Smith") == "j chen-smith"
    # Deliberately under-merges: initials are NOT expanded, because fusing
    # two researchers into one record is worse than keeping them apart.
    assert normalize_author("J. Chen") != normalize_author("Jane Chen")


def test_store_and_adapter_agree_on_the_author_key() -> None:
    """They are separate implementations — the store must not import an
    ingest adapter — so a test pins them together."""
    for name in ("Jane Chen", "J. Chen-Smith", " Yann  LeCun ", "José Álvarez"):
        assert _author_key(name) == normalize_author(name)


def test_top_lab_recognition() -> None:
    assert is_top_lab("Google DeepMind") is True
    assert is_top_lab("OpenAI, San Francisco") is True
    assert is_top_lab("Dept. of Physics, Random University") is False
    assert is_top_lab("") is False


def test_query_ands_categories_against_thesis_terms() -> None:
    query = build_query(["cs.LG", "cs.AI"], ["agent evals", "inference"])
    assert query == '(cat:cs.LG OR cat:cs.AI) AND (all:"agent evals" OR all:"inference")'
    # Either half alone still produces a usable query.
    assert build_query(["cs.LG"], []) == "cat:cs.LG"
    assert build_query([], ["agents"]) == 'all:"agents"'
    assert build_query([], []) == ""


def test_unbridged_authors_become_manual_lookup_leads() -> None:
    paper = parse_entries(FEED)[0]
    lead = to_unlinked(paper, paper["authors"][1])
    assert lead.source == "arxiv"
    assert lead.ref == "wei zhang"
    assert lead.name == "Wei Zhang"
    assert lead.bio.startswith("Stanford AI Lab — Sparse Routing")
    assert lead.url == "http://arxiv.org/abs/2401.12345v1"
    assert lead.found_at is not None


# --- affiliation history: the differentiated part -------------------------------


def _seeded(tmp_path: Path) -> Store:
    store = Store(tmp_path / "arxiv.db")
    for paper in parse_entries(FEED):
        store.record_paper(paper)
    return store


def test_papers_and_authors_are_recorded(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    history = store.author_affiliations("jane chen")
    assert [row["affiliation"] for row in history] == [
        "Google DeepMind", "Sparse Labs"]
    papers = store.papers_by_author("jane chen")
    assert len(papers) == 2
    assert papers[0]["title"] == "A Second Paper"  # newest first
    # Re-recording the same paper does not duplicate rows.
    for paper in parse_entries(FEED):
        store.record_paper(paper)
    assert len(store.author_affiliations("jane chen")) == 2


def test_an_affiliation_change_is_a_departure_observed_at_the_source(
    tmp_path: Path,
) -> None:
    """The whole reason to integrate arXiv: `departure_signal` infers this
    from self-reported bio language, months later."""
    store = _seeded(tmp_path)
    moves = store.authors_who_moved()
    assert len(moves) == 1
    move = moves[0]
    assert move["name"] == "Jane Chen"
    assert move["from"] == "google deepmind"
    assert move["to"] == "Sparse Labs"
    assert move["at"].startswith("2026-07-20")


def test_a_missing_affiliation_is_not_a_move(tmp_path: Path) -> None:
    """arXiv's affiliation field is optional and frequently absent. Reading
    a gap as a departure would manufacture signal out of metadata
    sparsity — the most likely way this integration could lie."""
    store = Store(tmp_path / "gaps.db")
    store.record_paper({
        "arxiv_id": "1", "title": "First", "published": "2026-01-01T00:00:00Z",
        "authors": [{"name": "Alex Kim", "affiliation": "OpenAI"}],
    })
    store.record_paper({  # same person, affiliation simply not supplied
        "arxiv_id": "2", "title": "Second", "published": "2026-03-01T00:00:00Z",
        "authors": [{"name": "Alex Kim", "affiliation": ""}],
    })
    assert store.authors_who_moved() == []
    assert [r["affiliation"] for r in store.author_affiliations("alex kim")] == [
        "OpenAI"]


def test_moves_can_be_windowed_and_report_only_the_latest(tmp_path: Path) -> None:
    store = Store(tmp_path / "moves.db")
    for n, (affiliation, when) in enumerate([
        ("OpenAI", "2024-01-01T00:00:00Z"),
        ("Anthropic", "2025-01-01T00:00:00Z"),
        ("Vector Labs", "2026-06-01T00:00:00Z"),
    ], start=1):
        store.record_paper({
            "arxiv_id": str(n), "title": f"P{n}", "published": when,
            "authors": [{"name": "Sam Ito", "affiliation": affiliation}],
        })
    latest = store.authors_who_moved()[0]
    assert (latest["from"], latest["to"]) == ("anthropic", "Vector Labs")
    # Windowing drops a move that predates the cutoff.
    assert store.authors_who_moved(
        since=datetime(2026, 1, 1, tzinfo=timezone.utc)) != []
    assert store.authors_who_moved(
        since=datetime(2027, 1, 1, tzinfo=timezone.utc)) == []


def test_a_fresh_database_has_no_paper_history(tmp_path: Path) -> None:
    store = Store(tmp_path / "empty.db")
    assert store.author_affiliations("nobody") == []
    assert store.authors_who_moved() == []
    assert store.papers_by_author("nobody") == []


# --- the scoring signal ---------------------------------------------------------


def test_a_recorded_move_becomes_a_scored_signal(tmp_path: Path) -> None:
    """Recording affiliation history is only half the job: without this the
    arXiv data sits in the store and never reaches the ranking."""
    from scout.config import Thesis
    from scout.models import Account
    from scout.signals.heuristics import run_heuristics

    thesis = Thesis(thesis="AI infra")
    store = _seeded(tmp_path)
    move = store.authors_who_moved()[0]

    plain = Account(id="1", handle="janechen", name="Jane Chen")
    assert {s.name: s for s in run_heuristics(plain, [], thesis)[0]}[
        "lab_departure"].value == 0.0

    enriched = plain.model_copy(
        update={"lab_move": f"{move['from']} → {move['to']}"})
    signal = {s.name: s for s in run_heuristics(enriched, [], thesis)[0]}[
        "lab_departure"]
    assert signal.value == 1.0
    # The evidence travels with the signal, so a name-collision false
    # positive is visible rather than silent.
    assert signal.detail == "google deepmind → Sparse Labs"


def test_the_enrichment_key_matches_what_the_store_returns(tmp_path: Path) -> None:
    """The join is name → normalize_author → author_key. If those two ever
    diverge the signal silently never fires, which is why this is pinned."""
    store = _seeded(tmp_path)
    moved = {m["author_key"] for m in store.authors_who_moved()}
    assert normalize_author("Jane Chen") in moved
