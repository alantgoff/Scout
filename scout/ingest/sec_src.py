"""SEC Form D discovery source — the government record of every US private raise.

Every US company that sells securities under Regulation D files a Form D
within 15 days of the first sale: issuer name and address, the amount sold,
the date of first sale, and the executive officers by name. Free, no key,
and — unlike a bio, a tweet or an article — a record with legal weight.
That makes it two things for a scouting engine:

- **Ground truth on companies already tracked.** A filing whose issuer name
  matches a company in the database is attached to that row (store.
  set_filing_match), announced in the activity feed and the digest, handed
  to the next research pass as context (so `funding_evidence` cites the
  filing), and gives the hindsight backtest a REAL round date (first sale)
  instead of the day a refresh noticed.
- **Discovery of companies nobody has written about yet.** An unmatched
  startup-shaped filing becomes an UnlinkedLead: `scout resolve` finds the
  domain behind the name and runs the normal add path. A Form D lands
  before the launch post and the press — often the first public trace of a
  stealth company's existence.

Most Form Ds are funds, SPVs and real-estate vehicles; `is_startup_filing`
is the gate, and it errs toward dropping (a fund resolved and classified is
budget spent on a non-company).

Two network shapes, both on www.sec.gov (fair-access rules: a descriptive
User-Agent with contact, no more than 10 requests/second):

- the daily form index, one small text file per business day
  (`Archives/edgar/daily-index/YYYY/QTRn/form.YYYYMMDD.idx`) — the
  enumeration; a missing file (404) is a weekend or holiday, not an error;
- `primary_doc.xml` per filing — the structured record, no XML namespace.

Pure parsing (`parse_index`, `parse_form_d`, `is_startup_filing`,
`match_issuer`, `filing_bio`) is separated from I/O and unit-tested on
recorded fixtures, the same split as every other source.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from typing import Any
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field
from rich.console import Console

from scout.companies import normalize_company_name
from scout.config import Seeds, Settings, Thesis
from scout.ingest.base import DiscoverySource
from scout.models import Account, UnlinkedLead
from scout.store import Store

_console = Console()

_ARCHIVES = "https://www.sec.gov/Archives"
_LOOKBACK_DAYS = 5  # a missed run or a long weekend heals itself
_FETCH_TIMEOUT_S = 15.0
_MAX_CONCURRENCY = 4
_REQUEST_GAP_S = 0.15  # 4 workers × ~6 req/s each stays under SEC's 10/s
_MAX_FILINGS_PER_DAY = 400  # a hard cap on XML fetches, whatever the index says
_STARTUP_FORMS = {"D", "D/A"}

# Issuer names that are vehicles, not companies. Deliberately broad: a fund
# that slips through costs a resolve call and a wrong row; a startup called
# "Acme Capital" that is dropped here costs one filing nobody sees, and it
# will still surface through every other source.
_VEHICLE_NAME = re.compile(
    r"\b(fund|funds|l\.?\s?p\.?|partners|spv|trust|reit|capital|ventures?|"
    r"investors|investments?|acquisition|holdings?|royalt(?:y|ies)|"
    r"real\s+estate|properties|equity|opportunit(?:y|ies)|portfolio|"
    r"feeder|master|series\s+[a-z0-9]+|llp|gp)\b",
    re.I,
)
POOLED_FUND_INDUSTRY = "Pooled Investment Fund"


class IndexRow(BaseModel):
    """One line of the EDGAR daily form index."""

    form: str
    company: str
    cik: str
    filed: str  # YYYY-MM-DD
    path: str  # edgar/data/<cik>/<accession>.txt

    @property
    def accession(self) -> str:
        return self.path.rsplit("/", 1)[-1].removesuffix(".txt")

    @property
    def primary_doc_url(self) -> str:
        return (f"{_ARCHIVES}/edgar/data/{int(self.cik)}/"
                f"{self.accession.replace('-', '')}/primary_doc.xml")

    @property
    def filing_url(self) -> str:
        return (f"{_ARCHIVES}/edgar/data/{int(self.cik)}/"
                f"{self.accession.replace('-', '')}/")


class Filing(BaseModel):
    """What one Form D says. Facts only, every field optional — a filing
    with a name and nothing else is still a filing."""

    accession: str = ""
    cik: str = ""
    issuer: str = ""
    entity_type: str = ""
    year_inc: str = ""  # "2024" | "within5" | "over5" | "" (unstated)
    jurisdiction: str = ""
    city: str = ""
    state: str = ""
    industry: str = ""
    is_pooled_fund: bool = False
    is_amendment: bool = False
    first_sale: str = ""  # YYYY-MM-DD, "" when yet to occur / unstated
    amount_offered: int | None = None  # None = "Indefinite"
    amount_sold: int = 0
    investors: int = 0
    officers: list[str] = Field(default_factory=list)  # "Jane Doe (Executive Officer)"
    url: str = ""
    filed_at: str = ""  # YYYY-MM-DD from the index

    @property
    def amount(self) -> int | None:
        """The number a person means by "raised": sold when stated, else
        offered; None when the offering is indefinite and nothing sold."""
        if self.amount_sold:
            return self.amount_sold
        return self.amount_offered


# --- pure parsers -------------------------------------------------------------


def parse_index(text: str) -> list[IndexRow]:
    """(pure, tested) The daily form index → rows. Column positions are read
    off the header line, not hard-coded, so a width change upstream does
    not silently shift every field."""
    lines = text.splitlines()
    header_at = next(
        (i for i, line in enumerate(lines)
         if "Form Type" in line and "Company Name" in line and "File Name" in line),
        None,
    )
    if header_at is None:
        return []
    header = lines[header_at]
    starts = [header.index(col) for col in
              ("Form Type", "Company Name", "CIK", "Date Filed", "File Name")]
    rows: list[IndexRow] = []
    for line in lines[header_at + 1:]:
        if not line.strip() or set(line.strip()) == {"-"}:
            continue
        fields = [line[a:b].strip() for a, b in zip(starts, starts[1:] + [None])]
        if len(fields) != 5 or not fields[2].isdigit() or len(fields[3]) != 8:
            continue
        form, company, cik, filed, path = fields
        rows.append(IndexRow(
            form=form, company=company, cik=cik,
            filed=f"{filed[:4]}-{filed[4:6]}-{filed[6:]}", path=path,
        ))
    return rows


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _strip_namespaces(root: ElementTree.Element) -> None:
    for el in root.iter():
        el.tag = _local(el.tag)


def _text(el: ElementTree.Element | None, path: str) -> str:
    if el is None:
        return ""
    found = el.find(path)
    return " ".join((found.text or "").split()) if found is not None else ""


def _flag(el: ElementTree.Element | None, path: str) -> bool:
    return _text(el, path).strip().lower() == "true"


def _int(value: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", value or "")
    return int(digits) if digits else None


def parse_form_d(xml_text: str, *, accession: str = "", cik: str = "",
                 url: str = "", filed_at: str = "") -> Filing | None:
    """(pure, tested) primary_doc.xml → Filing, or None when the document is
    not a Form D submission at all. Tolerant of every field being absent;
    strict about nothing, because the value of this source is the fields it
    does carry, and a missing one is "unstated", not "wrong"."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    _strip_namespaces(root)
    if root.tag != "edgarSubmission":
        return None
    issuer = root.find("primaryIssuer")
    offering = root.find("offeringData")
    submission = _text(root, "submissionType").upper()

    year_el = issuer.find("yearOfInc") if issuer is not None else None
    year_inc = ""
    if year_el is not None:
        if _text(year_el, "value"):
            year_inc = _text(year_el, "value")
        elif _flag(year_el, "withinFiveYears"):
            year_inc = "within5"
        elif _flag(year_el, "overFiveYears"):
            year_inc = "over5"

    first_sale = ""
    sale_el = offering.find("typeOfFiling/dateOfFirstSale") if offering is not None else None
    if sale_el is not None and not _flag(sale_el, "yetToOccur"):
        first_sale = _text(sale_el, "value")[:10]

    offered_raw = _text(offering, "offeringSalesAmounts/totalOfferingAmount")
    amount_offered = None if "indefinite" in offered_raw.lower() else _int(offered_raw)

    officers: list[str] = []
    for person in root.iterfind("relatedPersonsList/relatedPersonInfo"):
        name = " ".join(part for part in (
            _text(person, "relatedPersonName/firstName"),
            _text(person, "relatedPersonName/lastName"),
        ) if part)
        if not name:
            continue
        roles = [_local(r.text or "").strip()
                 for r in person.iterfind("relatedPersonRelationshipList/relationship")]
        roles = [r for r in roles if r]
        officers.append(f"{name} ({', '.join(roles)})" if roles else name)

    return Filing(
        accession=accession,
        cik=cik or _text(issuer, "cik"),
        issuer=_text(issuer, "entityName"),
        entity_type=_text(issuer, "entityType"),
        year_inc=year_inc,
        jurisdiction=_text(issuer, "jurisdictionOfInc"),
        city=_text(issuer, "issuerAddress/city"),
        state=_text(issuer, "issuerAddress/stateOrCountry"),
        industry=_text(offering, "industryGroup/industryGroupType"),
        is_pooled_fund=_flag(offering, "typesOfSecuritiesOffered/isPooledInvestmentFundType"),
        is_amendment=(submission == "D/A"
                      or _flag(offering, "typeOfFiling/newOrAmendment/isAmendment")),
        first_sale=first_sale,
        amount_offered=amount_offered,
        amount_sold=_int(_text(offering, "offeringSalesAmounts/totalAmountSold")) or 0,
        investors=_int(_text(offering, "investors/totalNumberAlreadyInvested")) or 0,
        officers=officers[:8],
        url=url,
        filed_at=filed_at,
    )


def looks_like_vehicle(name: str) -> bool:
    """(pure, tested) Fund / SPV / real-estate naming — the cheap pre-filter
    applied to the index BEFORE any XML is fetched."""
    return bool(_VEHICLE_NAME.search(name or ""))


def is_startup_filing(
    filing: Filing, industries: list[str], max_offering_usd: int,
) -> tuple[bool, str]:
    """(pure, tested) Whether a Form D describes an operating startup raising
    a seed-sized round. Returns (keep, reason) so a run log can say why a
    filing was dropped. Empty `industries` means every non-fund industry."""
    if filing.is_pooled_fund or filing.industry == POOLED_FUND_INDUSTRY:
        return False, "pooled investment fund"
    if looks_like_vehicle(filing.issuer):
        return False, "vehicle-shaped name"
    wanted = {i.strip().lower() for i in industries if i.strip()}
    if wanted and filing.industry.strip().lower() not in wanted:
        return False, f"industry {filing.industry or 'unstated'!r} not in seeds.sec_industries"
    amount = filing.amount
    if amount is None:
        return False, "indefinite offering, nothing sold"
    if amount > max_offering_usd:
        return False, f"${amount:,} exceeds SEC_MAX_OFFERING_USD"
    if filing.year_inc == "over5":
        return False, "incorporated over five years ago"
    return True, "startup-sized"


def money(value: int | None) -> str:
    if value is None:
        return "indefinite"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M".replace(".0M", "M")
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value}"


def filing_bio(filing: Filing) -> str:
    """(pure, tested) One line that carries every fact the filing states —
    the bio of the UnlinkedLead, and the context the resolver's research
    receives. Nothing inferred: no stage label, no round name."""
    parts = ["SEC Form D" + (" (amendment)" if filing.is_amendment else "")]
    if filing.industry:
        parts.append(filing.industry)
    if filing.amount_sold and filing.amount_offered:
        parts.append(f"{money(filing.amount_sold)} sold of {money(filing.amount_offered)}")
    elif filing.amount is not None:
        parts.append(f"{money(filing.amount)} {'sold' if filing.amount_sold else 'offered'}")
    if filing.first_sale:
        parts.append(f"first sale {filing.first_sale}")
    if filing.investors:
        parts.append(f"{filing.investors} investor{'s' if filing.investors != 1 else ''}")
    where = ", ".join(p for p in (filing.city, filing.state) if p)
    if where:
        parts.append(where)
    if filing.year_inc:
        parts.append({"within5": "incorporated within 5 years",
                      "over5": "incorporated over 5 years ago"}.get(
                          filing.year_inc, f"incorporated {filing.year_inc}"))
    if filing.officers:
        parts.append("officers: " + "; ".join(filing.officers[:4]))
    return " · ".join(parts)[:280]


def filing_lines(rows: list[dict[str, Any]]) -> str:
    """(pure) Stored filing rows → research context lines, newest first."""
    lines = []
    for row in rows:
        filing = Filing.model_validate({
            **{k: v for k, v in row.items() if k in Filing.model_fields},
            "officers": row.get("officers") if isinstance(row.get("officers"), list) else [],
        })
        lines.append(f"- {filing_bio(filing)} — {row.get('url', '')}")
    return "\n".join(lines)


def match_issuer(issuer: str, name_index: dict[str, str]) -> str | None:
    """(pure, tested) The tracked handle whose company name is this issuer,
    by normalized name (legal suffixes and punctuation dropped). Exact key
    only — "Acme Robotics" never matches "Acme Capital Partners"."""
    key = normalize_company_name(issuer)
    return name_index.get(key) if len(key) > 2 else None


def to_row(filing: Filing, *, matched_handle: str | None = None) -> dict[str, Any]:
    row = filing.model_dump()
    row["matched_handle"] = (matched_handle or "").lower() or None
    row["recorded_at"] = datetime.now(timezone.utc).isoformat()
    return row


# --- the source ---------------------------------------------------------------


class SECSource(DiscoverySource):
    """Reads new Form Ds daily. Free; the only cost is the resolver's later
    work on unmatched startup filings, which its own allowance bounds."""

    name = "sec"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate"}

    @staticmethod
    def index_url(day: date) -> str:
        quarter = (day.month - 1) // 3 + 1
        return (f"{_ARCHIVES}/edgar/daily-index/{day.year}/QTR{quarter}/"
                f"form.{day.strftime('%Y%m%d')}.idx")

    async def _get(self, client: httpx.AsyncClient, url: str) -> str | None:
        """None on 404 (no index for a non-business day) — anything else
        that is not a 200 is printed loudly, because a silent empty leg is
        exactly how a blocked host goes unnoticed for weeks."""
        try:
            resp = await client.get(url)
        except Exception as exc:  # a dead host must never sink discovery
            _console.print(f"[yellow]sec: {url} failed: {type(exc).__name__}[/]")
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            _console.print(f"[yellow]sec: {url} → HTTP {resp.status_code} "
                           f"(SEC fair-access requires a User-Agent with contact "
                           f"details: SEC_USER_AGENT)[/]")
            return None
        return resp.text

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        now = datetime.now(timezone.utc)
        days = self.store.sec_days_pending(_LOOKBACK_DAYS, now=now)
        if not days:
            return [], []
        unlinked: list[UnlinkedLead] = []
        name_index = self.store.company_name_index()
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
        kept = dropped = matched = 0

        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S, follow_redirects=True, headers=self._headers(),
        ) as client:
            for day in days:
                text = await self._get(client, self.index_url(day))
                if text is None:
                    self.store.mark_sec_day(day.isoformat(), 0)
                    continue
                rows = [r for r in parse_index(text)
                        if r.form in _STARTUP_FORMS and not looks_like_vehicle(r.company)]
                rows = rows[:_MAX_FILINGS_PER_DAY]

                async def fetch(row: IndexRow) -> Filing | None:
                    async with semaphore:
                        body = await self._get(client, row.primary_doc_url)
                        await asyncio.sleep(_REQUEST_GAP_S)
                    if body is None:
                        return None
                    return parse_form_d(body, accession=row.accession, cik=row.cik,
                                        url=row.filing_url, filed_at=row.filed)

                filings = [f for f in await asyncio.gather(*(fetch(r) for r in rows)) if f]
                to_store: list[dict[str, Any]] = []
                for filing in filings:
                    keep, reason = is_startup_filing(
                        filing, seeds.sec_industries, self.settings.sec_max_offering_usd)
                    if not keep:
                        dropped += 1
                        continue
                    kept += 1
                    handle = match_issuer(filing.issuer, name_index)
                    to_store.append(to_row(filing, matched_handle=handle))
                    if handle:
                        matched += 1
                        self.store.note_filing_matched(handle, to_store[-1])
                    elif not filing.is_amendment and not self.store.filings_for_cik(filing.cik):
                        unlinked.append(UnlinkedLead(
                            source="sec", ref=filing.url, name=filing.issuer,
                            bio=filing_bio(filing), url=filing.url, found_at=now,
                        ))
                self.store.record_filings(to_store)
                self.store.mark_sec_day(day.isoformat(), len(to_store))

        _console.print(
            f"[dim]sec: {kept} startup-shaped filing(s) kept, {dropped} dropped, "
            f"{matched} matched tracked companies, {len(unlinked)} new → resolve "
            f"({len(days)} day(s) read).[/dim]"
        )
        return [], unlinked
