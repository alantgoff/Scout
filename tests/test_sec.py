"""SEC Form D source — the government record of every US private raise.

What matters is what a filing is allowed to BECOME. Most Form Ds are funds
and vehicles; letting one through spends the resolver's budget on a
non-company and writes a wrong row. So the tests are mostly about refusing:
the vehicle gate, the industry gate, the amount gate — and, on the other
side, that a filing matching a tracked company attaches to that row with
its facts intact and a real date for the backtest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scout.companies import normalize_company_name
from scout.hindsight import outcome_from_filing
from scout.ingest.sec_src import (
    SECSource,
    filing_bio,
    filing_lines,
    is_startup_filing,
    looks_like_vehicle,
    match_issuer,
    parse_form_d,
    parse_index,
    to_row,
)
from scout.models import Account, Lead, LLMVerdict, UnlinkedLead
from scout.store import Store

INDEX = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    September 11, 2026
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/



Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        Big Public Co                                                 1000001     20260911    edgar/data/1000001/0001000001-26-000010.txt
D           Acme Robotics, Inc.                                           2000001     20260911    edgar/data/2000001/0002000001-26-000001.txt
D           Sequoia Growth Fund XII, L.P.                                 2000002     20260911    edgar/data/2000002/0002000002-26-000002.txt
D/A         Beta Labs Inc                                                 2000003     20260911    edgar/data/2000003/0002000003-26-000003.txt
D           Weird  Spacing   Holdings LLC                                 2000004     20260911    edgar/data/2000004/0002000004-26-000004.txt
"""

FORM_D = """<?xml version="1.0"?>
<edgarSubmission>
  <schemaVersion>X0708</schemaVersion>
  <submissionType>D</submissionType>
  <testOrLive>LIVE</testOrLive>
  <primaryIssuer>
    <cik>0002000001</cik>
    <entityName>Acme Robotics, Inc.</entityName>
    <issuerAddress>
      <street1>1 Market St</street1>
      <city>San Francisco</city>
      <stateOrCountry>CA</stateOrCountry>
      <stateOrCountryDescription>CALIFORNIA</stateOrCountryDescription>
      <zipCode>94105</zipCode>
    </issuerAddress>
    <issuerPhoneNumber>4155551234</issuerPhoneNumber>
    <jurisdictionOfInc>DELAWARE</jurisdictionOfInc>
    <entityType>Corporation</entityType>
    <yearOfInc><withinFiveYears>true</withinFiveYears></yearOfInc>
  </primaryIssuer>
  <relatedPersonsList>
    <relatedPersonInfo>
      <relatedPersonName><firstName>Jane</firstName><lastName>Doe</lastName></relatedPersonName>
      <relatedPersonRelationshipList>
        <relationship>Executive Officer</relationship>
        <relationship>Director</relationship>
      </relatedPersonRelationshipList>
    </relatedPersonInfo>
    <relatedPersonInfo>
      <relatedPersonName><firstName>Sam</firstName><lastName>Lee</lastName></relatedPersonName>
      <relatedPersonRelationshipList><relationship>Executive Officer</relationship></relatedPersonRelationshipList>
    </relatedPersonInfo>
  </relatedPersonsList>
  <offeringData>
    <industryGroup><industryGroupType>Other Technology</industryGroupType></industryGroup>
    <issuerSize><revenueRange>No Revenues</revenueRange></issuerSize>
    <federalExemptionsExclusions><item>06b</item></federalExemptionsExclusions>
    <typeOfFiling>
      <newOrAmendment><isAmendment>false</isAmendment></newOrAmendment>
      <dateOfFirstSale><value>2026-09-01</value></dateOfFirstSale>
    </typeOfFiling>
    <durationOfOffering><moreThanOneYear>false</moreThanOneYear></durationOfOffering>
    <typesOfSecuritiesOffered><isEquityType>true</isEquityType></typesOfSecuritiesOffered>
    <minimumInvestmentAccepted>0</minimumInvestmentAccepted>
    <offeringSalesAmounts>
      <totalOfferingAmount>3000000</totalOfferingAmount>
      <totalAmountSold>2500000</totalAmountSold>
      <totalRemaining>500000</totalRemaining>
    </offeringSalesAmounts>
    <investors>
      <hasNonAccreditedInvestors>false</hasNonAccreditedInvestors>
      <totalNumberAlreadyInvested>4</totalNumberAlreadyInvested>
    </investors>
  </offeringData>
</edgarSubmission>
"""

FUND_D = """<edgarSubmission><submissionType>D</submissionType>
<primaryIssuer><cik>2000002</cik><entityName>Quiet Growth XII</entityName>
<entityType>Limited Partnership</entityType></primaryIssuer>
<offeringData>
  <industryGroup><industryGroupType>Pooled Investment Fund</industryGroupType>
    <investmentFundInfo><investmentFundType>Venture Capital Fund</investmentFundType></investmentFundInfo>
  </industryGroup>
  <typeOfFiling><dateOfFirstSale><yetToOccur>true</yetToOccur></dateOfFirstSale></typeOfFiling>
  <typesOfSecuritiesOffered><isPooledInvestmentFundType>true</isPooledInvestmentFundType></typesOfSecuritiesOffered>
  <offeringSalesAmounts><totalOfferingAmount>Indefinite</totalOfferingAmount><totalAmountSold>0</totalAmountSold></offeringSalesAmounts>
</offeringData></edgarSubmission>"""

MINIMAL_D = "<edgarSubmission><primaryIssuer><entityName>Bare Co</entityName></primaryIssuer></edgarSubmission>"

NOW = datetime(2026, 9, 12, 6, 0, tzinfo=timezone.utc)
INDUSTRIES = ["Other Technology", "Computers"]


# --- the daily index ----------------------------------------------------------


def test_parse_index_reads_columns_off_the_header_and_keeps_every_form() -> None:
    rows = parse_index(INDEX)
    assert [r.form for r in rows] == ["10-K", "D", "D", "D/A", "D"]
    acme = rows[1]
    assert acme.company == "Acme Robotics, Inc."
    assert acme.cik == "2000001"
    assert acme.filed == "2026-09-11"
    assert acme.accession == "0002000001-26-000001"
    assert acme.primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/2000001/000200000126000001/primary_doc.xml")
    assert acme.filing_url.endswith("/2000001/000200000126000001/")
    assert rows[4].company == "Weird  Spacing   Holdings LLC"  # inner spacing preserved


def test_parse_index_without_a_header_yields_nothing() -> None:
    assert parse_index("") == []
    assert parse_index("Description: something else\n\nno table here\n") == []


def test_index_url_lands_in_the_right_quarter() -> None:
    from datetime import date

    assert SECSource.index_url(date(2026, 9, 11)).endswith(
        "/daily-index/2026/QTR3/form.20260911.idx")
    assert SECSource.index_url(date(2026, 1, 2)).endswith("/2026/QTR1/form.20260102.idx")
    assert SECSource.index_url(date(2026, 12, 31)).endswith("/2026/QTR4/form.20261231.idx")


# --- the filing ---------------------------------------------------------------


def test_parse_form_d_reads_every_fact_the_filing_states() -> None:
    filing = parse_form_d(FORM_D, accession="0002000001-26-000001",
                          url="https://www.sec.gov/x/", filed_at="2026-09-11")
    assert filing is not None
    assert filing.issuer == "Acme Robotics, Inc."
    assert filing.cik == "0002000001"
    assert filing.entity_type == "Corporation"
    assert filing.year_inc == "within5"
    assert filing.jurisdiction == "DELAWARE"
    assert (filing.city, filing.state) == ("San Francisco", "CA")
    assert filing.industry == "Other Technology"
    assert filing.is_pooled_fund is False
    assert filing.is_amendment is False
    assert filing.first_sale == "2026-09-01"
    assert filing.amount_offered == 3_000_000
    assert filing.amount_sold == 2_500_000
    assert filing.amount == 2_500_000  # "raised" means sold when stated
    assert filing.investors == 4
    assert filing.officers == ["Jane Doe (Executive Officer, Director)", "Sam Lee (Executive Officer)"]
    assert filing.filed_at == "2026-09-11"


def test_parse_form_d_tolerates_missing_fields_and_refuses_non_form_d() -> None:
    bare = parse_form_d(MINIMAL_D)
    assert bare is not None
    assert bare.issuer == "Bare Co"
    assert bare.amount is None and bare.first_sale == "" and bare.officers == []
    assert parse_form_d("<notASubmission/>") is None
    assert parse_form_d("<unclosed") is None
    # A namespaced document still parses (the schema carries none, but a
    # future revision might).
    namespaced = FORM_D.replace("<edgarSubmission>",
                                '<edgarSubmission xmlns="http://www.sec.gov/edgar/formd">')
    assert parse_form_d(namespaced).issuer == "Acme Robotics, Inc."


def test_a_fund_is_recognised_three_ways() -> None:
    fund = parse_form_d(FUND_D)
    assert fund.is_pooled_fund is True
    assert fund.amount_offered is None  # Indefinite
    assert fund.first_sale == ""  # yet to occur
    keep, reason = is_startup_filing(fund, INDUSTRIES, 15_000_000)
    assert not keep and "fund" in reason
    # Name alone is enough, before any XML is fetched.
    assert looks_like_vehicle("Sequoia Growth Fund XII, L.P.")
    assert looks_like_vehicle("Acme Capital Partners")
    assert looks_like_vehicle("Blue Ventures LLC")
    assert looks_like_vehicle("Main Street Real Estate Holdings")
    assert not looks_like_vehicle("Acme Robotics, Inc.")
    assert not looks_like_vehicle("Beta Labs Inc")


def test_is_startup_filing_gates_on_industry_amount_and_age() -> None:
    acme = parse_form_d(FORM_D)
    assert is_startup_filing(acme, INDUSTRIES, 15_000_000) == (True, "startup-sized")
    assert is_startup_filing(acme, [], 15_000_000)[0] is True  # no industries = any
    keep, reason = is_startup_filing(acme, ["Biotechnology"], 15_000_000)
    assert not keep and "industry" in reason
    keep, reason = is_startup_filing(acme, INDUSTRIES, 1_000_000)
    assert not keep and "exceeds" in reason
    old = acme.model_copy(update={"year_inc": "over5"})
    assert not is_startup_filing(old, INDUSTRIES, 15_000_000)[0]
    indefinite = acme.model_copy(update={"amount_offered": None, "amount_sold": 0})
    keep, reason = is_startup_filing(indefinite, INDUSTRIES, 15_000_000)
    assert not keep and "indefinite" in reason


def test_filing_bio_carries_the_facts_and_invents_no_round() -> None:
    bio = filing_bio(parse_form_d(FORM_D))
    assert bio.startswith("SEC Form D · Other Technology · $2.5M sold of $3M")
    assert "first sale 2026-09-01" in bio
    assert "4 investors" in bio
    assert "San Francisco, CA" in bio
    assert "incorporated within 5 years" in bio
    assert "officers: Jane Doe (Executive Officer, Director); Sam Lee (Executive Officer)" in bio
    for word in ("seed", "pre-seed", "series"):
        assert word not in bio.lower()
    assert len(bio) <= 280


# --- the name join ------------------------------------------------------------


def test_normalize_company_name_drops_legal_suffixes_not_words() -> None:
    assert normalize_company_name("Acme Robotics, Inc.") == "acmerobotics"
    assert normalize_company_name("acme robotics") == "acmerobotics"
    assert normalize_company_name("Acme Robotics Holdings LLC") == "acmeroboticsholdings"
    assert normalize_company_name("The Company Store Co.") == "thecompanystore"
    assert normalize_company_name("") == ""


def test_match_issuer_is_exact_on_the_normalized_name() -> None:
    index = {"acmerobotics": "acmerobots", "betalabs": "beta"}
    assert match_issuer("Acme Robotics, Inc.", index) == "acmerobots"
    assert match_issuer("ACME ROBOTICS", index) == "acmerobots"
    assert match_issuer("Acme Capital Partners LP", index) is None
    assert match_issuer("Acme", index) is None  # a prefix is not a match
    assert match_issuer("", index) is None


def test_company_name_index_joins_verdict_names_account_names_and_cache(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots", name="Acme"))
    store.save_leads("run-1", [Lead(
        account=store.get_account("acmerobots"),
        llm=LLMVerdict(handle="acmerobots", account_type="startup",
                       company_name="Acme Robotics, Inc.", grounding="website"),
    )])
    store.upsert_account(Account(id="x:beta", handle="betabuilds", name="Beta Labs Inc"))
    index = store.company_name_index()
    assert index["acmerobotics"] == "acmerobots"
    assert index["acme"] == "acmerobots"
    assert index["betalabs"] == "betabuilds"


# --- the store ----------------------------------------------------------------


def test_pending_days_skip_read_days_and_never_today(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    days = store.sec_days_pending(5, now=NOW)
    assert [d.isoformat() for d in days] == [
        "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]
    store.mark_sec_day("2026-09-10", 0)  # a Sunday-shaped 404
    store.mark_sec_day("2026-09-11", 12)
    assert [d.isoformat() for d in store.sec_days_pending(5, now=NOW)] == [
        "2026-09-07", "2026-09-08", "2026-09-09"]


def test_matched_filing_is_stored_announced_once_and_dated_for_the_backtest(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    store.actor = "agent:sec"
    filing = parse_form_d(FORM_D, accession="0002000001-26-000001",
                          url="https://www.sec.gov/Archives/edgar/data/2000001/x/",
                          filed_at="2026-09-11")
    row = to_row(filing, matched_handle="AcmeRobots")
    store.record_filings([row])
    store.note_filing_matched("acmerobots", row)
    store.note_filing_matched("acmerobots", row)  # a re-read of the same day

    rows = store.filings_for("acmerobots")
    assert len(rows) == 1
    assert rows[0]["officers"] == filing.officers  # JSON round-trips as a list
    assert rows[0]["amount_sold"] == 2_500_000
    assert store.filings_for_cik("0002000001")[0]["accession"] == "0002000001-26-000001"
    events = [e for e in store.events(since=datetime(2000, 1, 1, tzinfo=timezone.utc), limit=20)
              if e.verb == "filing_matched"]
    assert len(events) == 1
    assert events[0].payload["first_sale"] == "2026-09-01"

    outcome = outcome_from_filing(store.matched_filings()[0], domain="acme-robotics.com")
    assert outcome is not None
    assert outcome.round_date == datetime(2026, 9, 1, tzinfo=timezone.utc)  # first sale, not detection
    assert outcome.amount == "$2,500,000"
    assert outcome.round_stage == ""  # never invented
    assert outcome.x_handles == ["acmerobots"]
    assert "SEC Form D" in outcome.note
    assert outcome_from_filing({"issuer": "Nowhere", "first_sale": "", "filed_at": ""}) is None

    assert filing_lines(rows).startswith("- SEC Form D · Other Technology")


def test_resolver_queue_interleaves_sources_so_filings_cannot_starve_headlines(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "t.db")
    leads = [UnlinkedLead(source="sec", ref=f"https://sec/{i}", name=f"Issuer {i}",
                          bio="SEC Form D", url=f"https://sec/{i}", found_at=NOW)
             for i in range(5)]
    leads.append(UnlinkedLead(source="rss", ref="https://news/1", name="Acme raises",
                              bio="Acme raises $2M", url="https://news/1", found_at=NOW))
    leads.append(UnlinkedLead(source="hn", ref="poster", name="poster",
                              bio="Show HN: Beta", url="https://youtube.com/x", found_at=NOW))
    store.upsert_unlinked_leads(leads)
    queue = store.unresolved_leads(limit=3)
    assert [item.source for item in queue] == ["rss", "hn", "sec"]
    assert [item.source for item in store.unresolved_leads(limit=6)] == [
        "rss", "hn", "sec", "sec", "sec", "sec"]


# --- the source, end to end over stubbed HTTP ---------------------------------


class _Resp:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code, self.text = status, text


class _Client:
    """Answers the index for 2026-09-11, 404 for every other day, and the
    XML per accession folder; records every URL asked for."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url: str) -> _Resp:
        self.urls.append(url)
        if url.endswith("form.20260911.idx"):
            return _Resp(200, INDEX)
        if url.endswith(".idx"):
            return _Resp(404)
        if "/2000001/" in url:
            return _Resp(200, FORM_D)
        if "/2000003/" in url:  # the D/A: an amendment on an unknown issuer
            return _Resp(200, FORM_D.replace("Acme Robotics, Inc.", "Beta Labs Inc")
                         .replace("<submissionType>D</submissionType>",
                                  "<submissionType>D/A</submissionType>"))
        return _Resp(500)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


def test_discover_reads_pending_days_keeps_startups_and_matches_tracked(
    tmp_path: Path, monkeypatch
) -> None:
    import asyncio

    from scout import config
    from scout.ingest import sec_src

    store = Store(tmp_path / "t.db")
    store.actor = "agent:sec"
    store.upsert_account(Account(id="x:acmerobots", handle="acmerobots",
                                 name="Acme Robotics"))
    client = _Client()
    monkeypatch.setattr(sec_src.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(sec_src, "_REQUEST_GAP_S", 0)
    monkeypatch.setattr(sec_src, "datetime", _FrozenDatetime)
    settings = config.Settings(sec_max_offering_usd=15_000_000)
    seeds = config.Seeds(sec_industries=INDUSTRIES)

    accounts, unlinked = asyncio.run(SECSource(settings, store).discover(seeds, config.Thesis()))

    assert accounts == []
    # The fund never had its XML fetched: the name gate ran on the index.
    assert not any("/2000002/" in u for u in client.urls)
    assert not any("/2000004/" in u for u in client.urls)  # "Holdings"
    # Acme matched the tracked company → attached, not queued.
    assert store.filings_for("acmerobots")[0]["issuer"] == "Acme Robotics, Inc."
    assert unlinked == []  # Beta was an amendment on an issuer never seen → not a new lead
    # Every pending day is now read: the 404 days as empty, the real one as 2.
    assert store.sec_days_pending(5, now=NOW) == []
    days = {r["day"]: r["n_filings"] for r in store.db["sec_days"].rows}
    assert days["2026-09-11"] == 2 and days["2026-09-10"] == 0

    # A second run reads nothing — nothing is pending.
    client.urls.clear()
    asyncio.run(SECSource(settings, store).discover(seeds, config.Thesis()))
    assert client.urls == []


def test_discover_queues_an_unknown_startup_for_the_resolver(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from scout import config
    from scout.ingest import sec_src

    store = Store(tmp_path / "t.db")  # nothing tracked
    client = _Client()
    monkeypatch.setattr(sec_src.httpx, "AsyncClient", lambda **kw: client)
    monkeypatch.setattr(sec_src, "_REQUEST_GAP_S", 0)
    monkeypatch.setattr(sec_src, "datetime", _FrozenDatetime)
    _, unlinked = asyncio.run(SECSource(config.Settings(), store).discover(
        config.Seeds(sec_industries=INDUSTRIES), config.Thesis()))
    assert [u.name for u in unlinked] == ["Acme Robotics, Inc."]
    assert unlinked[0].source == "sec"
    assert unlinked[0].ref == unlinked[0].url  # the filing folder is the pk
    assert unlinked[0].bio.startswith("SEC Form D · Other Technology · $2.5M sold")


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)
