"""arXiv discovery — catching researchers before they are founders.

The earliest public signal a technical founder emits is usually a paper, not
a repo and certainly not a stealth bio. Someone publishes from a top lab,
keeps publishing, then one day the affiliation changes and six months later
there is a company. GitHub sees that at the repo; X sees it when the bio
changes; arXiv saw it a year earlier.

Two properties make it worth the integration:

- **The archive is complete, free, and timestamped**, so unlike X history it
  can be rewound. Anything sourced here is admissible in the hindsight
  backtest, which is where a signal earns its weight.
- **Affiliation is a fact about a person, not a claim they made.** Scout's
  existing `departure_signal` reads self-reported bio language; a change of
  institution across a publication record is the same event observed at the
  source, and earlier.

Identity resolution is the honest difficulty. A paper yields "Jane Chen",
possibly an institution, and rarely a handle. Where the abstract or comment
field links X or GitHub we bridge (GitHub owners already resolve to handles
via `github_src.profile_to_x_handle`); otherwise the author becomes an
`UnlinkedLead`, exactly as HN and GitHub already do for the unbridgeable.
Most authors will land there, and that is the correct outcome rather than a
guess.

Everything that parses is a pure module-level function tested against
fixture XML, per the house pattern in `github_src`/`hn_src`.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree

import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.config import Seeds, Settings, Thesis
from scout.ingest.base import DiscoverySource
from scout.models import Account, UnlinkedLead
from scout.store import Store

_console = Console()

_API = "https://export.arxiv.org/api/query"
_WINDOW_DAYS = 45
_RESULTS_PER_QUERY = 60
# arXiv asks for one request every three seconds; being a good citizen here
# also keeps us well inside their informal rate limit.
_PAUSE_S = 3.0

_ATOM = {"a": "http://www.w3.org/2005/Atom",
         "arxiv": "http://arxiv.org/schemas/atom"}

_X_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/(@?[A-Za-z0-9_]{1,15})\b",
    re.IGNORECASE,
)
_GH_LINK_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))"
    r"(?:/([A-Za-z0-9._-]{1,100}))?",
    re.IGNORECASE,
)
_X_RESERVED = {"i", "intent", "home", "search", "hashtag", "share", "settings"}
_GH_RESERVED = {"features", "about", "pricing", "topics", "trending",
                "collections", "events", "sponsors", "readme", "orgs"}

# Institutions whose alumni are the thesis's whole point. Kept here rather
# than in seeds.yaml because it is signal MECHANICS (how we recognise a top
# lab), not targeting — the same split the rest of the codebase observes.
TOP_LABS = (
    "openai", "google deepmind", "deepmind", "google brain", "google research",
    "meta ai", "fair", "anthropic", "microsoft research", "nvidia",
    "apple machine learning", "allen institute", "ai2", "mistral",
    "stability ai", "cohere", "eleutherai", "berkeley ai research", "bair",
    "stanford ai lab", "sail", "mit csail", "cmu", "carnegie mellon",
)

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
)


# ------------------------------------------------------------ pure parsing


def extract_x_handle(text: str) -> str | None:
    """(pure, tested) First plausible X handle linked in a blob of text."""
    for match in _X_LINK_RE.finditer(text or ""):
        handle = match.group(1).lstrip("@")
        if handle.lower() not in _X_RESERVED:
            return handle
    return None


def extract_github_repo(text: str) -> str | None:
    """(pure, tested) First `owner/name` GitHub repo linked in the text.

    Papers routinely link their code, and a repo owner already resolves to
    an X handle through the GitHub source — so this is the highest-yield
    bridge available from a paper alone.
    """
    for match in _GH_LINK_RE.finditer(text or ""):
        owner, repo = match.group(1), match.group(2)
        if owner.lower() in _GH_RESERVED or not repo:
            continue
        return f"{owner}/{repo.rstrip('.,);')}"
    return None


def normalize_author(name: str) -> str:
    """(pure, tested) Author key for matching the same person across papers.

    Casefolded, punctuation-stripped, whitespace-collapsed. Deliberately
    NOT clever: initials-vs-full-name and transliteration collisions are
    real, so this is a conservative key that under-merges rather than
    fusing two researchers into one record.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", (name or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def is_top_lab(affiliation: str) -> bool:
    """(pure, tested) Does this affiliation name a frontier lab?"""
    text = (affiliation or "").lower()
    return any(lab in text for lab in TOP_LABS)


def parse_entries(xml_text: str) -> list[dict[str, Any]]:
    """(pure, tested) arXiv's Atom feed → paper dicts.

    `arxiv:affiliation` is OPTIONAL and in practice often absent — most
    submissions omit it. Papers are returned regardless, with affiliation
    empty, because a missing affiliation must read as "unknown" rather than
    "unaffiliated"; the caller decides what to do with the gap.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []

    papers: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", _ATOM):
        def _text(path: str) -> str:
            node = entry.find(path, _ATOM)
            return (node.text or "").strip() if node is not None else ""

        arxiv_url = _text("a:id")
        if not arxiv_url:
            continue
        authors: list[dict[str, str]] = []
        for author in entry.findall("a:author", _ATOM):
            name_node = author.find("a:name", _ATOM)
            name = (name_node.text or "").strip() if name_node is not None else ""
            if not name:
                continue
            affiliation_node = author.find("arxiv:affiliation", _ATOM)
            authors.append({
                "name": name,
                "affiliation": (
                    (affiliation_node.text or "").strip()
                    if affiliation_node is not None else ""
                ),
            })
        if not authors:
            continue

        comment_node = entry.find("arxiv:comment", _ATOM)
        papers.append({
            "arxiv_id": arxiv_url.rstrip("/").rsplit("/", 1)[-1],
            "url": arxiv_url,
            "title": re.sub(r"\s+", " ", _text("a:title")),
            "abstract": re.sub(r"\s+", " ", _text("a:summary")),
            "published": _text("a:published"),
            "updated": _text("a:updated"),
            "authors": authors,
            "categories": [
                c.get("term", "") for c in entry.findall("a:category", _ATOM)
                if c.get("term")
            ],
            "comment": (
                (comment_node.text or "").strip()
                if comment_node is not None else ""
            ),
        })
    return papers


def paper_links(paper: dict[str, Any]) -> dict[str, str]:
    """(pure, tested) Bridgeable identifiers found in a paper's own text.

    Abstract and comment only — the fields arXiv actually returns. The
    comment field is where "Code at github.com/..." usually lives.
    """
    blob = f"{paper.get('abstract', '')} {paper.get('comment', '')}"
    links: dict[str, str] = {}
    if handle := extract_x_handle(blob):
        links["x"] = handle
    if repo := extract_github_repo(blob):
        links["github_repo"] = repo
    return links


def build_query(categories: list[str], terms: list[str]) -> str:
    """(pure, tested) An arXiv `search_query` from categories + thesis terms.

    Categories are AND-ed against an OR of terms, which is what keeps the
    result set on-thesis instead of returning every cs.LG paper this month.
    """
    cat_clause = " OR ".join(f"cat:{c}" for c in categories if c)
    term_clause = " OR ".join(f'all:"{t}"' for t in terms if t)
    if cat_clause and term_clause:
        return f"({cat_clause}) AND ({term_clause})"
    return cat_clause or term_clause


def to_unlinked(paper: dict[str, Any], author: dict[str, str]) -> UnlinkedLead:
    """(pure, tested) An author we could not bridge → a manual-lookup lead."""
    affiliation = author.get("affiliation", "")
    bio = paper.get("title", "")
    if affiliation:
        bio = f"{affiliation} — {bio}"
    return UnlinkedLead(
        source="arxiv",
        ref=normalize_author(author["name"]),
        name=author["name"],
        bio=bio[:280],
        url=paper.get("url", ""),
        found_at=_parse_dt(paper.get("published", "")),
    )


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return None


# ----------------------------------------------------------------- source


class ArxivSource(DiscoverySource):
    """Recent on-thesis papers → candidate founders.

    Free and unauthenticated. Failures are logged and swallowed by
    `cli._run_discovery` like every other discovery source: a source being
    down must never abort a run.
    """

    name = "arxiv"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    @_retry
    async def _get(self, client: httpx.AsyncClient, **params: Any) -> str:
        response = await client.get(_API, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        categories = list(getattr(seeds, "arxiv_categories", []) or [])
        if not categories:
            return [], []
        terms = list(thesis.sectors or []) + list(thesis.keywords or [])
        query = build_query(categories, terms[:12])
        if not query:
            return [], []

        cutoff = datetime.now(timezone.utc) - timedelta(days=_WINDOW_DAYS)
        accounts: list[Account] = []
        unlinked: list[UnlinkedLead] = []
        seen_authors: set[str] = set()

        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                xml_text = await self._get(
                    client,
                    search_query=query,
                    start=0,
                    max_results=_RESULTS_PER_QUERY,
                    sortBy="submittedDate",
                    sortOrder="descending",
                )
            except Exception as exc:  # noqa: BLE001 — logged, never fatal
                _console.print(f"[yellow]arxiv query failed:[/] {exc}")
                return [], []

            for paper in parse_entries(xml_text):
                published = _parse_dt(paper.get("published", ""))
                if published and published < cutoff:
                    continue
                # Persist first: the affiliation history is the durable
                # asset here, and it accrues whether or not we can bridge
                # anyone to an X account today.
                try:
                    self.store.record_paper(paper)
                except Exception as exc:  # noqa: BLE001
                    _console.print(f"[yellow]arxiv store failed:[/] {exc}")

                links = paper_links(paper)
                handle = links.get("x")
                for index, author in enumerate(paper["authors"]):
                    key = normalize_author(author["name"])
                    if not key or key in seen_authors:
                        continue
                    seen_authors.add(key)
                    # A paper's linked handle belongs to whoever posted the
                    # work — in practice the first author. Attributing it to
                    # every co-author would manufacture identities.
                    if handle and index == 0:
                        accounts.append(Account(
                            id=handle.lower(), handle=handle,
                            name=author["name"],
                            bio=(author.get("affiliation") or "")[:200],
                            source="arxiv",
                            github_repo=links.get("github_repo"),
                        ))
                    else:
                        unlinked.append(to_unlinked(paper, author))
        return accounts, unlinked
