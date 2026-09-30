"""Warm-intro paths — who in the firm's network connects to a company.

Sourcing finds a company; the next question is how to reach it, and a cold
email is the worst answer when a co-investor, a founder you backed or an
advisor you know sits one hop away. Every relationship this module walks
is already in the knowledge graph (scout.graph), derived from cited
evidence — the investors research found for a round, the founders it
named, the labs in their bios, the watchlist's follows. Nothing here is
inferred; a path exists only where two cited facts meet.

What the graph cannot know is who the FIRM knows. That is the Network:

- portfolio companies — pipeline status "won" (shown as Allocated)
- companies you are talking to — contacted / meeting / diligence
- co-investors — every investor the graph cites on a portfolio company
- a firm-kept list of funds and people you know (Settings → Your network,
  or `scout network`), for relationships Scout has no way to see

Paths are graded by who you would actually ask:

  3  direct: your firm already backs it; its founder is someone you backed
     or listed
  2  one strong hop: its investor co-invested with you, or is a fund you
     listed; you have met its founder on another deal
  1  weak: a founder shares a lab with someone you know; a person you
     listed follows it on X

The network and the paths are firm-private — they say who the firm knows.
They never reach the phone app (scout.publish has no access to them).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from scout.graph import company_node, node_key

PORTFOLIO_STATUSES = {"won"}
MET_STATUSES = {"contacted", "meeting", "diligence"}
_MAX_PATHS = 6
_KIND_ORDER = {"already_backed": 0, "founder_known": 1, "co_investor": 2,
               "shared_lab": 3, "watcher": 4}


class Network(BaseModel):
    """Who the firm knows, as graph node keys → labels."""

    firm_key: str = ""
    firm_label: str = ""
    portfolio: dict[str, str] = Field(default_factory=dict)  # company key → label
    met: dict[str, str] = Field(default_factory=dict)  # company key → label
    investors: dict[str, str] = Field(default_factory=dict)  # listed funds/angels
    people: dict[str, str] = Field(default_factory=dict)  # listed people (names)
    handles: dict[str, str] = Field(default_factory=dict)  # listed @handles

    @property
    def empty(self) -> bool:
        return not (self.portfolio or self.met or self.investors
                    or self.people or self.handles)


class WarmPath(BaseModel):
    kind: Literal["already_backed", "founder_known", "co_investor",
                  "shared_lab", "watcher"]
    strength: int  # 3 direct · 2 one strong hop · 1 weak
    via: str  # who to ask
    text: str  # one sentence a partner can act on
    evidence: str = ""


def parse_list(text: str | None) -> list[str]:
    """Newline- or comma-separated entries, blanks dropped, order kept."""
    items = re.split(r"[\n,]", text or "")
    return [" ".join(item.split()) for item in items if item.strip()]


def _split_person(entry: str) -> tuple[str, str]:
    """"Elad Gil (@eladgil)" → ("Elad Gil", "eladgil"); "@eladgil" →
    ("", "eladgil"); "Elad Gil" → ("Elad Gil", "")."""
    match = re.search(r"@([A-Za-z0-9_]{1,15})", entry)
    handle = match.group(1) if match else ""
    name = re.sub(r"\(?@[A-Za-z0-9_]{1,15}\)?", "", entry).strip(" -—()")
    return name, handle


def network_from(ledger: list, pipeline: dict[str, dict], *, firm_name: str = "",
                 investors_text: str = "", people_text: str = "") -> Network:
    """(pure) The firm's network from the ledger's statuses and its lists."""
    net = Network(firm_key=node_key(firm_name) if firm_name else "",
                  firm_label=firm_name)
    for entry in ledger:
        lead = entry.lead
        status = (pipeline.get(lead.account.handle.lower()) or {}).get("status") or ""
        if status in PORTFOLIO_STATUSES or status in MET_STATUSES:
            key, label = company_node(lead)
            (net.portfolio if status in PORTFOLIO_STATUSES else net.met)[key] = label
    for name in parse_list(investors_text):
        key = node_key(name)
        if len(key) > 1:
            net.investors[key] = name
    for entry in parse_list(people_text):
        name, handle = _split_person(entry)
        if name and len(node_key(name)) > 1:
            net.people[node_key(name)] = name
        if handle:
            net.handles[handle.lower()] = f"@{handle}"
    # A company you already backed is not "met" as well.
    for key in net.portfolio:
        net.met.pop(key, None)
    return net


def _index(edges: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    by_src: dict[str, list[dict]] = {}
    by_dst: dict[str, list[dict]] = {}
    for edge in edges:
        by_src.setdefault(edge["src_key"], []).append(edge)
        by_dst.setdefault(edge["dst_key"], []).append(edge)
    return by_src, by_dst


def warm_paths(edges: list[dict], net: Network) -> dict[str, list[WarmPath]]:
    """(pure, tested) Company key → its warm paths, strongest first.

    Portfolio companies get none — they are already yours. One pass builds
    what the network touches (co-investors, known founders, their labs);
    a second walks each company's own edges against it.
    """
    if net.empty and not net.firm_key:
        return {}
    by_src, by_dst = _index(edges)

    def label_of(company_key: str) -> str:
        for edge in by_dst.get(company_key, []):
            return edge["dst_label"]
        return company_key

    # Co-investors: every investor cited on a portfolio company.
    co_invested: dict[str, list[str]] = {}
    for company, label in net.portfolio.items():
        for edge in by_dst.get(company, []):
            if edge["rel"] == "invested_in":
                co_invested.setdefault(edge["src_key"], []).append(label)
    # Founders you know, of portfolio (backed) and met companies. A person
    # can found several; portfolio outranks met, so it is listed first.
    known_founders: dict[str, list[tuple[str, str, str, str]]] = {}
    for group, how in ((net.portfolio, "portfolio"), (net.met, "met")):
        for company, label in group.items():
            for edge in by_dst.get(company, []):
                if edge["rel"] == "founded":
                    known_founders.setdefault(edge["src_key"], []).append(
                        (edge["src_label"], company, label, how))
    known_people = dict(net.people)
    # Labs of everyone you know → (person, label, why, the company that makes
    # them known — "" for the list) so a company never paths to itself.
    known_by_lab: dict[str, list[tuple[str, str, str, str]]] = {}
    for person, ties in known_founders.items():
        for edge in by_src.get(person, []):
            if edge["rel"] != "alum_of":
                continue
            for plabel, ckey, clabel_known, how in ties:
                known_by_lab.setdefault(edge["dst_key"], []).append(
                    (person, plabel, f"founder of {clabel_known}"
                     + (", your portfolio" if how == "portfolio" else ", whom you've met"),
                     ckey))
    for person, plabel in known_people.items():
        for edge in by_src.get(person, []):
            if edge["rel"] == "alum_of":
                known_by_lab.setdefault(edge["dst_key"], []).append(
                    (person, plabel, "in your network", ""))

    companies = {e["dst_key"] for e in edges
                 if e["rel"] in ("invested_in", "founded", "follows")}
    out: dict[str, list[WarmPath]] = {}
    for company in companies:
        if company in net.portfolio:
            continue
        clabel = label_of(company)
        paths: list[WarmPath] = []
        incoming = by_dst.get(company, [])
        for edge in incoming:
            rel, src, slabel = edge["rel"], edge["src_key"], edge["src_label"]
            evidence = edge.get("evidence") or ""
            if rel == "invested_in":
                if net.firm_key and src == net.firm_key:
                    paths.append(WarmPath(
                        kind="already_backed", strength=3, via=net.firm_label,
                        text=f"{net.firm_label} is already on {clabel}'s cap table",
                        evidence=evidence))
                elif src in co_invested:
                    shared = ", ".join(sorted(set(co_invested[src]))[:3])
                    paths.append(WarmPath(
                        kind="co_investor", strength=2, via=slabel,
                        text=f"{slabel} backs {clabel} — and co-invested with you "
                             f"in {shared}", evidence=evidence))
                elif src in net.investors:
                    paths.append(WarmPath(
                        kind="co_investor", strength=2, via=slabel,
                        text=f"{slabel} backs {clabel} — a fund in your network",
                        evidence=evidence))
            elif rel == "founded":
                ties = [t for t in known_founders.get(src, []) if t[1] != company]
                if ties:
                    _plabel, _ckey, other, how = ties[0]
                    if how == "portfolio":
                        paths.append(WarmPath(
                            kind="founder_known", strength=3, via=slabel,
                            text=f"{slabel} founded {clabel} — and your portfolio "
                                 f"company {other}", evidence=evidence))
                    else:
                        paths.append(WarmPath(
                            kind="founder_known", strength=2, via=slabel,
                            text=f"{slabel} founded {clabel} — you've met them "
                                 f"on {other}", evidence=evidence))
                elif src in known_people:
                    paths.append(WarmPath(
                        kind="founder_known", strength=3, via=slabel,
                        text=f"{slabel} founded {clabel} — in your network",
                        evidence=evidence))
                for lab_edge in by_src.get(src, []):
                    if lab_edge["rel"] != "alum_of":
                        continue
                    for person, plabel, why, via_company in known_by_lab.get(
                            lab_edge["dst_key"], []):
                        if person == src or via_company == company:
                            continue
                        paths.append(WarmPath(
                            kind="shared_lab", strength=1, via=plabel,
                            text=f"{slabel} ({clabel}) and {plabel} ({why}) are "
                                 f"both {lab_edge['dst_label']} alumni",
                            evidence=lab_edge.get("evidence") or ""))
            elif rel == "follows" and src in net.handles:
                paths.append(WarmPath(
                    kind="watcher", strength=1, via=net.handles[src],
                    text=f"{net.handles[src]} follows {clabel} on X — in your network"))
        if not paths:
            continue
        seen: set[tuple[str, str]] = set()
        unique = []
        for path in sorted(paths, key=lambda p: (-p.strength, _KIND_ORDER[p.kind], p.via)):
            key = (path.kind, node_key(path.via))
            if key not in seen:
                seen.add(key)
                unique.append(path)
        out[company] = unique[:_MAX_PATHS]
    return out


def warmth(paths: list[WarmPath] | None) -> int:
    """A company's best path strength, 0 when it has none — the sort key."""
    return max((p.strength for p in paths or []), default=0)


def network_for(store, ledger: list | None = None, pipeline: dict | None = None,
                firm_name: str = "") -> Network:
    """The firm's network straight from a Store (the CLI and UI share it)."""
    ledger = store.load_lead_ledger() if ledger is None else ledger
    pipeline = store.all_pipeline() if pipeline is None else pipeline
    return network_from(
        ledger, pipeline, firm_name=firm_name,
        investors_text=store.get_setting("network_investors") or "",
        people_text=store.get_setting("network_people") or "",
    )
