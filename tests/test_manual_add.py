"""`scout add` — the hand-added company path.

Sourcing is the normal way in; this is the other one (a founder emails you, a
partner forwards a link). The tests pin the three properties that make a
hand-added company safe to sit next to a discovered one: its profile link is
real, a person typing a name never earns discovery credit, and re-adding never
downgrades what was already read.

No network and no API key — classification is skipped, which is also the path a
user without ANTHROPIC_API_KEY takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scout.cli import _parse_add_target, app
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.store import Store

runner = CliRunner()


def add(tmp_path: Path, *args: str):
    """Invoke `scout add` against a throwaway DB in tmp_path."""
    return runner.invoke(
        app,
        ["add", *args, "--no-classify", "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(tmp_path / "scout.db")},
    )


# --- target parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("@pollenrobotics", ("pollenrobotics", None, None)),
        ("pollenrobotics", ("pollenrobotics", None, None)),
        ("https://x.com/pollenrobotics", ("pollenrobotics", None, None)),
        ("x.com/pollenrobotics/", ("pollenrobotics", None, None)),
        ("twitter.com/pollenrobotics", ("pollenrobotics", None, None)),
        (
            "https://pollen-robotics.com/",
            ("pollen-robotics", "https://pollen-robotics.com/", "https://pollen-robotics.com/"),
        ),
        (
            "pollen-robotics.com",
            ("pollen-robotics", "https://pollen-robotics.com/", "https://pollen-robotics.com/"),
        ),
    ],
)
def test_parse_add_target_shapes(target: str, expected: tuple) -> None:
    assert _parse_add_target(target) == expected


@pytest.mark.parametrize("target", ["", "   ", "mailto:hi@pollen-robotics.com"])
def test_parse_add_target_rejects_junk(target: str) -> None:
    with pytest.raises(ValueError):
        _parse_add_target(target)


# --- the honest profile link --------------------------------------------------


def test_site_keyed_add_never_links_to_a_fabricated_x_profile(tmp_path: Path) -> None:
    """A company with no X account is keyed by a slug we invented. Building
    x.com/<slug> from it would render a 404 in the UI, the CSV and the memo as
    if it were the company's profile."""
    result = add(tmp_path, "https://pollen-robotics.com/")
    assert result.exit_code == 0, result.output

    store = Store(tmp_path / "scout.db")
    account = store.get_account("pollen-robotics")
    assert account is not None
    assert account.url == "https://pollen-robotics.com/"


def test_handle_keyed_add_still_links_to_x(tmp_path: Path) -> None:
    result = add(tmp_path, "@pollenrobotics", "--url", "pollen-robotics.com")
    assert result.exit_code == 0, result.output

    account = Store(tmp_path / "scout.db").get_account("pollenrobotics")
    assert account is not None
    assert account.url == "https://x.com/pollenrobotics"
    assert account.website == "https://pollen-robotics.com/"


# --- what an add writes -------------------------------------------------------


def test_add_lands_in_the_ledger_and_the_pipeline(tmp_path: Path) -> None:
    result = add(
        tmp_path, "@pollenrobotics", "--name", "Pollen Robotics",
        "--note", "acquired by Hugging Face, Apr 2025",
    )
    assert result.exit_code == 0, result.output

    store = Store(tmp_path / "scout.db")
    ledger = {e.lead.account.handle.lower(): e for e in store.load_lead_ledger()}
    assert "pollenrobotics" in ledger

    row = store.get_pipeline("pollenrobotics")
    assert row["status"] == "longlisted"
    assert row["notes"] == "acquired by Hugging Face, Apr 2025"


def test_unclassified_add_is_marked_unverified_not_scored_as_evidence(tmp_path: Path) -> None:
    """What a person typed is an assertion, not something we read. It has to
    carry a grounding the scorer already penalizes, or a hand-typed company
    outranks one that was actually researched."""
    assert add(tmp_path, "@pollenrobotics", "--name", "Pollen Robotics").exit_code == 0

    lead = Store(tmp_path / "scout.db").latest_lead("pollenrobotics")
    assert lead is not None and lead.llm is not None
    assert lead.llm.grounding == "manual"
    assert lead.llm.confidence == 0.0


def test_add_rejects_an_unknown_status(tmp_path: Path) -> None:
    result = add(tmp_path, "@pollenrobotics", "--status", "interesting")
    assert result.exit_code == 1
    assert "Unknown status" in result.output


# --- re-adding an account that is already known -------------------------------


def _seed_discovered(tmp_path: Path) -> Store:
    """A company sourcing already found, with real discovery provenance."""
    store = Store(tmp_path / "scout.db")
    store.upsert_account(
        Account(
            id="42", handle="pollenrobotics", name="Pollen Robotics",
            bio="open source robots", website="https://pollen-robotics.com/",
            followers=9000, source="search", sources=["search", "github"],
            followed_by=["karpathy"],
        )
    )
    store.save_leads(
        "20250101-000000-000000",
        [Lead(
            account=store.get_account("pollenrobotics"),
            signals=[Signal(name="bio_intent", value=1.0)],
            llm=LLMVerdict(
                handle="pollenrobotics", account_type="startup",
                company_name="Pollen Robotics", grounding="website",
                one_line_summary="open-source humanoid robots", confidence=0.9,
            ),
            score=61.0,
        )],
    )
    return store


def test_re_adding_does_not_manufacture_source_corroboration(tmp_path: Path) -> None:
    """`sources` feeds the source_corroboration signal — two independent
    strategies agreeing is evidence. A human typing the name is not a third
    strategy, and must not buy the account free credit."""
    store = _seed_discovered(tmp_path)
    assert add(tmp_path, "@pollenrobotics").exit_code == 0

    account = Store(tmp_path / "scout.db").get_account("pollenrobotics")
    assert account is not None
    assert account.sources == ["search", "github"]
    assert account.source == "search"
    assert account.followers == 9000  # the discovered row is merged, not replaced
    assert account.followed_by == ["karpathy"]
    del store


def test_re_adding_keeps_the_classification_that_was_actually_read(tmp_path: Path) -> None:
    """Without this, `scout add` on an already-classified company replaces a
    website-grounded verdict with a stub — a silent downgrade."""
    _seed_discovered(tmp_path)
    assert add(tmp_path, "@pollenrobotics", "--status", "shortlisted").exit_code == 0

    store = Store(tmp_path / "scout.db")
    lead = store.latest_lead("pollenrobotics")
    assert lead is not None and lead.llm is not None
    assert lead.llm.grounding == "website"
    assert lead.llm.one_line_summary == "open-source humanoid robots"
    assert store.get_pipeline("pollenrobotics")["status"] == "shortlisted"


# --- Store.latest_lead --------------------------------------------------------


def test_latest_lead_is_the_newest_row_and_case_insensitive(tmp_path: Path) -> None:
    store = _seed_discovered(tmp_path)
    store.save_leads(
        "20250601-000000-000000",
        [Lead(account=store.get_account("pollenrobotics"), score=72.0)],
    )
    lead = store.latest_lead("@PollenRobotics")
    assert lead is not None and lead.score == 72.0
    assert store.latest_lead("nobody") is None


# --- adding by domain: the system fills in the rest ---------------------------


def _stub_research(monkeypatch, profile, *, researched: bool = True):
    """Replace the live research + crawl + classify calls. The command's own
    wiring is what these tests are about, not the agents behind it."""
    from scout import agents, cli, web

    async def no_crawl(*_a, **_kw):
        return []

    monkeypatch.setattr(web, "fetch_site_bundle", no_crawl)
    monkeypatch.setattr(
        agents, "research_company",
        lambda *a, **kw: (profile, {"researched": researched,
                                    "searches": 2, "fetches": 1, "sources": []}),
    )
    monkeypatch.setattr(cli, "classify", lambda *a, **kw: {})


def researched_add(tmp_path: Path, *args: str):
    return runner.invoke(
        app, ["add", *args, "--thesis", "thesis.yaml"],
        env={"DB_PATH": str(tmp_path / "scout.db"), "ANTHROPIC_API_KEY": "k"},
    )


def test_a_domain_add_is_re_keyed_to_the_x_handle_research_found(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole point of finding the handle. Keyed by the domain slug, the
    next sourcing run that discovers @pollenrobotics creates a SECOND record
    for the same company and the database quietly double-counts it."""
    from scout.agents import CompanyProfile

    _stub_research(monkeypatch, CompanyProfile(
        company_name="Pollen Robotics", x_handle="pollenrobotics",
        github_org="pollen-robotics", one_line_summary="open-source robots",
        website="https://pollen-robotics.com/",
    ))
    result = researched_add(tmp_path, "pollen-robotics.com")
    assert result.exit_code == 0, result.output

    store = Store(tmp_path / "scout.db")
    assert store.get_account("pollen-robotics") is None  # the slug was not kept
    account = store.get_account("pollenrobotics")
    assert account is not None
    assert account.name == "Pollen Robotics"
    assert account.url == "https://x.com/pollenrobotics"
    assert account.github_repo == "https://github.com/pollen-robotics"
    # No X bio to read, so the researched one-liner backs the heuristics.
    assert account.bio == "open-source robots"


def test_a_typed_handle_is_the_persons_answer_and_is_not_re_keyed(
    tmp_path: Path, monkeypatch
) -> None:
    from scout.agents import CompanyProfile

    _stub_research(monkeypatch, CompanyProfile(
        company_name="Pollen Robotics", x_handle="someoneelse",
    ))
    result = researched_add(tmp_path, "@pollenrobotics", "--url", "pollen-robotics.com")
    assert result.exit_code == 0, result.output

    store = Store(tmp_path / "scout.db")
    assert store.get_account("pollenrobotics") is not None
    assert store.get_account("someoneelse") is None


def test_a_domain_with_no_company_behind_it_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    from scout.agents import CompanyProfile

    _stub_research(monkeypatch, CompanyProfile(
        is_company=False, not_company_reason="parked domain",
    ))
    result = researched_add(tmp_path, "parked.example")
    assert result.exit_code == 1
    assert "parked domain" in result.output
    assert Store(tmp_path / "scout.db").get_account("parked") is None

    forced = researched_add(tmp_path, "parked.example", "--force")
    assert forced.exit_code == 0, forced.output
    assert Store(tmp_path / "scout.db").get_account("parked") is not None


def test_research_findings_reach_the_stored_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end: what the agent established has to survive the classifier
    fallback, the overlay, scoring and the save — or the add is cosmetic."""
    from scout.agents import CompanyProfile

    _stub_research(monkeypatch, CompanyProfile(
        company_name="Pollen Robotics", x_handle="pollenrobotics",
        product_summary="website: open-source humanoid robots",
        hq="Bordeaux, France", founded_year=2016,
        founders=["Matthieu Lapeyre — co-founder, ex-INRIA Flowers"],
        company_status="acquired", company_status_note="Hugging Face, April 2025",
        company_status_evidence="techcrunch.com 2025-04-14",
        sources=["https://techcrunch.com/2025/04/14/x"],
    ))
    result = researched_add(tmp_path, "pollen-robotics.com")
    assert result.exit_code == 0, result.output
    # The one finding that changes what the company IS gets said out loud.
    assert "Acquired" in result.output and "Hugging Face" in result.output

    lead = Store(tmp_path / "scout.db").latest_lead("pollenrobotics")
    assert lead is not None and lead.llm is not None
    assert lead.llm.hq == "Bordeaux, France"
    assert lead.llm.founded_year == 2016
    assert lead.llm.founders == ["Matthieu Lapeyre — co-founder, ex-INRIA Flowers"]
    assert lead.llm.company_status == "acquired"
    assert lead.llm.research_sources == ["https://techcrunch.com/2025/04/14/x"]
    assert lead.llm.grounding == "research"


def test_no_research_flag_skips_it_entirely(tmp_path: Path, monkeypatch) -> None:
    from scout import agents

    def boom(*_a, **_kw):
        raise AssertionError("research ran despite --no-research")

    monkeypatch.setattr(agents, "research_company", boom)
    result = researched_add(tmp_path, "pollen-robotics.com",
                            "--no-research", "--no-classify")
    assert result.exit_code == 0, result.output
