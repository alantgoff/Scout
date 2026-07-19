"""Knowledge-graph pure logic — node ids, ingest from findings.

Persistence lives in store.py (kg_nodes / kg_edges tables + kg_* methods),
per the project convention. Everything here is pure and unit-tested.

Node ids are `"{type}:{normalized}"` reusing the company_key normalizer, so
"Eval-HQ.ai" and "EvalHQ" land on the same company node. Types: company,
person, org, sector. Rels: competes_with, founded_by, worked_at, in_sector,
invested_in.

Provenance: edges written by analysis carry provenance "agent"; edges the
user creates by hand carry "user" and are never overwritten or deleted by
agents (enforced in store.kg_upsert_edge).
"""

from __future__ import annotations

import re

from scout.companies import normalize_company
from scout.diligence.schema import Memo

NODE_TYPES = ("company", "person", "org", "sector")
RELS = ("competes_with", "founded_by", "worked_at", "in_sector", "invested_in")


def normalize_name(name: str) -> str | None:
    """Normalizer for non-company node names (people, orgs, sectors):
    lowercased alphanumerics, no TLD stripping. None when unusable."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key if len(key) > 1 else None


def node_id(node_type: str, name: str) -> str | None:
    """`"{type}:{normalized}"` — companies go through the company_key
    normalizer so KG nodes and the Startups track agree on identity."""
    if node_type == "company":
        key = normalize_company(name)
    else:
        key = normalize_name(name)
    return f"{node_type}:{key}" if key else None


def _org_from_background(entry: str) -> str:
    """verified_background entries are "Org — role/title" per the rubric;
    take the org part (tolerating plain hyphens and commas)."""
    for sep in ("—", " - ", ","):
        if sep in entry:
            return entry.split(sep, 1)[0].strip()
    return entry.strip()


def ingest(memo: Memo, sectors: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Nodes + edges to persist from one memo's findings: competitors (both
    directions), founders + their prior employers, and sector membership.
    Returns (nodes, edges) as kwargs-dicts for store.kg_upsert_node/edge."""
    company_id = f"company:{memo.company_key}"
    nodes: dict[str, dict] = {
        company_id: {
            "id": company_id,
            "type": "company",
            "name": memo.company_name or memo.company_key,
            "meta": {"url": memo.recon.website} if memo.recon.website else {},
        }
    }
    edges: list[dict] = []

    def add_node(nid: str, ntype: str, name: str, meta: dict | None = None) -> None:
        nodes.setdefault(nid, {"id": nid, "type": ntype, "name": name, "meta": meta or {}})

    def add_edge(src: str, rel: str, dst: str, source_url: str = "") -> None:
        edges.append(
            {
                "src": src, "rel": rel, "dst": dst,
                "provenance": "agent", "source_url": source_url,
                "memo_ref": memo.company_key,
            }
        )

    # Competitors — from the competition payload; both directions so
    # neighbor lookups are symmetric (mirrors kg_link_competitors).
    competition = memo.finding("competition")
    if competition is not None:
        for comp in competition.payload.get("competitors", []):
            name = (comp.get("name") or "").strip() if isinstance(comp, dict) else ""
            if not name:
                continue
            cid = node_id("company", name)
            if cid is None or cid == company_id:
                continue
            url = (comp.get("url") or "").strip()
            # Agent-researched URLs render as links in the UI — only keep
            # plain web schemes (never javascript:/data:/etc).
            if not url.startswith(("http://", "https://")):
                url = ""
            add_node(cid, "company", name, {"url": url} if url else {})
            add_edge(company_id, "competes_with", cid, source_url=url)
            add_edge(cid, "competes_with", company_id, source_url=url)

    # Founders + prior employers — from the founder_pedigree payload.
    pedigree = memo.finding("founder_pedigree")
    if pedigree is not None:
        for founder in pedigree.payload.get("founders", []):
            if not isinstance(founder, dict):
                continue
            fname = (founder.get("name") or "").strip()
            pid = node_id("person", fname) if fname else None
            if pid is None:
                continue
            add_node(pid, "person", fname)
            add_edge(company_id, "founded_by", pid)
            for background in founder.get("verified_background", []):
                org = _org_from_background(str(background))
                oid = node_id("org", org)
                if oid is not None:
                    add_node(oid, "org", org)
                    add_edge(pid, "worked_at", oid)

    # Sector membership — seeds kg_competitors_for_sectors so analysis #10
    # starts from the map built by #1-9.
    for sector in sectors or []:
        sid = node_id("sector", sector)
        if sid is not None:
            add_node(sid, "sector", sector.strip())
            add_edge(company_id, "in_sector", sid)

    return list(nodes.values()), edges
