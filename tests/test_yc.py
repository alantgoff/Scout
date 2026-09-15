"""YC directory source — new-batch companies as scoreable leads.

The record already carries a website and a one-liner, so the tests are
about what is NOT allowed through: acquired/public companies, off-thesis
ones, and a record whose only link is not a company site. And about the
identity rule: a YC company is keyed by its domain with a real profile_url,
the same row `scout add <domain>` or RSS would make.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from scout.config import Seeds, Settings, Thesis
from scout.ingest.yc_src import YCSource, batch_slug, parse_batches, parse_companies
from scout.store import Store

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
THESIS = Thesis(keywords=["agentic", "robotics"], sectors=["ai infra"])

RECORDS = [
    {"id": 1, "name": "Acme Robotics", "slug": "acme-robotics", "batch": "Summer 2026",
     "status": "Active", "website": "https://www.acme-robotics.com/",
     "one_liner": "Agentic robots for warehouses", "long_description": "…",
     "tags": ["Robotics", "AI"], "industries": ["Industrials"],
     "url": "https://www.ycombinator.com/companies/acme-robotics", "launched_at": 1756000000},
    {"id": 2, "name": "Groomly", "slug": "groomly", "batch": "Summer 2026",
     "status": "Active", "website": "https://groomly.com", "one_liner": "Dog grooming on demand",
     "tags": ["Consumer"], "industries": ["Consumer"]},
    {"id": 3, "name": "OldCo", "slug": "oldco", "batch": "Summer 2026", "status": "Acquired",
     "website": "https://oldco.ai", "one_liner": "Agentic infra, acquired last week"},
    {"id": 4, "name": "Stealthy", "slug": "stealthy", "batch": "Summer 2026", "status": "Active",
     "website": "", "one_liner": "Agentic robotics, site coming",
     "url": "https://www.ycombinator.com/companies/stealthy"},
    {"id": 5, "name": "LinkOnly", "slug": "linkonly", "batch": "Summer 2026", "status": "Active",
     "website": "https://github.com/linkonly", "one_liner": "Open-source agentic tools"},
    {"id": 6, "name": "Acme Robotics (dupe)", "slug": "acme-robotics-2", "batch": "Winter 2026",
     "status": "Active", "website": "https://acme-robotics.com/about", "one_liner": "Robotics"},
]


def test_parse_batches_reads_any_shape_and_sorts_newest_first() -> None:
    assert parse_batches({"Summer 2025": 210, "Winter 2026": 240, "Fall 2025": 130}) == [
        "Winter 2026", "Fall 2025", "Summer 2025"]
    assert parse_batches({"batches": [{"name": "Summer 2026", "count": 200},
                                      {"name": "Spring 2026", "count": 150}]}) == [
        "Summer 2026", "Spring 2026"]
    assert parse_batches({"batches": {"summer-2026": {"name": "Summer 2026"}}}) == ["Summer 2026"]
    assert parse_batches(["Winter 2012", "Summer 2026", "junk", 42]) == ["Summer 2026", "Winter 2012"]
    assert parse_batches(None) == []
    assert parse_batches({"total": 5000}) == []
    assert batch_slug("Summer 2026") == "summer-2026"


def test_active_on_thesis_company_becomes_a_domain_keyed_account() -> None:
    accounts, unlinked = parse_companies(RECORDS, thesis=THESIS, now=NOW)
    handles = [a.handle for a in accounts]
    assert handles == ["acme-robotics"]  # groomly off-thesis, oldco acquired, dupe merged
    acme = accounts[0]
    assert acme.id == "yc:acme-robotics"
    assert acme.source == "yc"
    assert acme.name == "Acme Robotics"
    assert acme.bio == "Agentic robots for warehouses — YC Summer 2026"  # provenance travels
    # The identity rule: no invented x.com link.
    assert acme.profile_url == "https://www.acme-robotics.com/"
    assert acme.url == "https://www.acme-robotics.com/"
    # No usable site (empty, or a code host) → unlinked for the resolver.
    assert [(u.name, u.source) for u in unlinked] == [("Stealthy", "yc"), ("LinkOnly", "yc")]
    assert unlinked[0].ref == "https://www.ycombinator.com/companies/stealthy"


def test_no_thesis_terms_lets_everything_active_through_and_batches_filter() -> None:
    accounts, unlinked = parse_companies(RECORDS, thesis=Thesis(), now=NOW)
    assert {a.handle for a in accounts} == {"acme-robotics", "groomly"}
    accounts, _ = parse_companies(RECORDS, thesis=Thesis(), now=NOW, batches={"Winter 2026"})
    assert [a.id for a in accounts] == ["yc:acme-robotics-2"]
    assert parse_companies([{"junk": 1}, "nope", {"name": ""}], thesis=Thesis(), now=NOW) == ([], [])


class _Resp:
    def __init__(self, status: int, data=None) -> None:
        self.status_code, self._data = status, data

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class _Client:
    def __init__(self, *, batch_404: bool = False, meta_status: int = 200) -> None:
        self.urls: list[str] = []
        self.batch_404 = batch_404
        self.meta_status = meta_status

    async def get(self, url: str) -> _Resp:
        self.urls.append(url)
        if url.endswith("/meta.json"):
            return _Resp(self.meta_status, {"Summer 2026": 200, "Winter 2026": 240,
                                            "Summer 2025": 210})
        if url.endswith("/batches/summer-2026.json"):
            return _Resp(404) if self.batch_404 else _Resp(200, [RECORDS[0], RECORDS[1]])
        if url.endswith("/batches/winter-2026.json"):
            return _Resp(200, [RECORDS[5]])
        if url.endswith("/companies/all.json"):
            return _Resp(200, RECORDS)
        return _Resp(500)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


def _run(tmp_path: Path, monkeypatch, client: _Client, batches: int = 2):
    from scout.ingest import yc_src

    monkeypatch.setattr(yc_src.httpx, "AsyncClient", lambda **kw: client)
    store = Store(tmp_path / "t.db")
    source = YCSource(Settings(yc_batches=batches), store)
    return store, asyncio.run(source.discover(Seeds(), THESIS))


def test_discover_reads_the_latest_batches_only(tmp_path: Path, monkeypatch) -> None:
    client = _Client()
    store, (accounts, unlinked) = _run(tmp_path, monkeypatch, client)
    assert [u for u in client.urls if "/batches/" in u] == [
        "https://yc-oss.github.io/api/batches/summer-2026.json",
        "https://yc-oss.github.io/api/batches/winter-2026.json",
    ]
    assert not any(u.endswith("/companies/all.json") for u in client.urls)
    assert [a.handle for a in accounts] == ["acme-robotics"]
    assert store.get_account("acme-robotics") is not None  # persisted like every source
    assert unlinked == []


def test_a_missing_batch_file_falls_back_to_the_full_directory(tmp_path: Path, monkeypatch) -> None:
    client = _Client(batch_404=True)
    _, (accounts, unlinked) = _run(tmp_path, monkeypatch, client)
    assert any(u.endswith("/companies/all.json") for u in client.urls)
    assert [a.handle for a in accounts] == ["acme-robotics"]
    assert [u.name for u in unlinked] == ["Stealthy", "LinkOnly"]  # from the full file


def test_an_unreadable_meta_reads_nothing_instead_of_raising(tmp_path: Path, monkeypatch) -> None:
    client = _Client(meta_status=503)
    _, (accounts, unlinked) = _run(tmp_path, monkeypatch, client)
    assert (accounts, unlinked) == ([], [])
    assert len(client.urls) == 1  # nothing else was attempted
