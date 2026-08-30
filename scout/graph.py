"""The knowledge graph — Scout's cross-links, made first-class.

Everything the pipeline learns arrives attached to ONE startup: its backers
in `funding_investors`, its founders in `founders`, the lab a founder left
in `lab_move` or a bio, the acquirer inside `company_status_note`, the
smart money watching it in `followed_by`. Stored that way, the knowledge
only answers questions about one company at a time. The questions a fund
actually asks run SIDEWAYS: which investor keeps showing up across our
pipeline? which lab do our best founders come from? what else did this
acquirer buy? which two companies share a backer?

This module derives typed, evidence-carrying edges from those same fields —
the graph never learns anything the pipeline didn't; it only makes what was
already learned queryable across companies. Node identity is normalized
(so "a16z", "A16Z" and "Andreessen Horowitz" are one investor) and junk
names ("undisclosed investors") are dropped rather than becoming fake hubs.

Pure functions, no I/O — unit-tested. Persistence lives in Store
(rebuild_graph / graph_* queries); triggers live in the CLI.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from scout.companies import company_key, display_name, founder_like
from scout.ingest.arxiv_src import TOP_LABS
from scout.models import Lead

# Node types, fixed vocabulary: company | person | investor | lab | watcher.
REL_LABELS = {
    "invested_in": "backs",
    "founded": "founded",
    "alum_of": "alum of",
    "acquired_by": "acquired by",
    "follows": "watched by",
}

# One canonical key per famous investor the classifier spells many ways.
# Deliberately short: only names with a COMMON second spelling belong here —
# an alias map that tries to be complete becomes its own data-quality bug.
INVESTOR_ALIASES = {
    "a16z": "andreessen horowitz",
    "andreessen": "andreessen horowitz",
    "yc": "y combinator",
    "ycombinator": "y combinator",
    "gv": "google ventures",
    "lightspeed venture partners": "lightspeed",
    "sequoia capital": "sequoia",
    "khosla": "khosla ventures",
}

# Investor "names" that are really absences of a name. Matching is on the
# normalized key, so "Undisclosed Investors" and "undisclosed" both drop.
_JUNK_INVESTORS = {
    "undisclosed", "undisclosedinvestors", "angelinvestors", "angels",
    "various", "variousinvestors", "others", "otherinvestors", "investors",
    "none", "na", "unknown", "strategicinvestors", "existinginvestors",
}

# "ex-OpenAI", "prev @DeepMind", "formerly at Google Brain" → the affiliation
# text, matched against TOP_LABS below.
_EX_LAB_RE = re.compile(
    r"(?:ex[-\s@]+|prev(?:iously)?\s+(?:@\s*|at\s+)?|formerly\s+(?:@\s*|at\s+)?)"
    r"([a-z0-9][a-z0-9 .&-]{1,28})",
    re.IGNORECASE,
)


class Edge(BaseModel):
    """One typed, evidence-carrying link between two nodes."""

    src_type: str
    src_key: str
    src_label: str
    rel: str
    dst_type: str
    dst_key: str
    dst_label: str
    evidence: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src_key, self.rel, self.dst_key)


def node_key(name: str) -> str:
    """Canonical node identity: lowercased alphanumerics, aliases folded.

    The same normalization companies.company_key uses, so an investor that
    is ALSO a company in the database (an acquirer, a strategic) lands on
    one node, not two.
    """
    cleaned = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    aliased = INVESTOR_ALIASES.get(re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip())
    if aliased:
        cleaned = re.sub(r"[^a-z0-9]", "", aliased)
    return cleaned


def _investor_names(raw: list[str]) -> list[str]:
    """Clean investor names: lead-phrases stripped, junk dropped."""
    out = []
    for name in raw:
        name = re.sub(r"^\s*(?:led by|from|with)\s+", "", (name or "").strip(),
                      flags=re.IGNORECASE).strip(" .")
        key = node_key(name)
        if len(key) > 1 and key not in _JUNK_INVESTORS:
            out.append(name)
    return out


def _founder_name(entry: str) -> str:
    """"Matthieu Lapeyre — co-founder, ex-INRIA" → "Matthieu Lapeyre"."""
    return re.split(r"\s+—\s+|\s+-\s+|,", entry, maxsplit=1)[0].strip()


def _labs_in(text: str) -> list[tuple[str, str]]:
    """(lab, evidence-snippet) for every frontier-lab affiliation named as a
    PAST one in `text`. Matching is against arxiv_src.TOP_LABS — the one
    list of labs the codebase already maintains — so the graph and the
    lab_departure signal can never disagree about what counts as a lab."""
    found: list[tuple[str, str]] = []
    for match in _EX_LAB_RE.finditer(text or ""):
        candidate = match.group(1).lower()
        for lab in TOP_LABS:
            if candidate.startswith(lab) or lab in candidate:
                found.append((lab, match.group(0).strip()))
                break
    return found


def _acquirer_from_note(note: str) -> str | None:
    """The counterparty out of a status note — "Hugging Face, April 2025" or
    "by BigCo (all-stock)" → the name; None when the note carries no name
    (a bare date, an empty string). Conservative on purpose: a wrong
    acquirer node is worse than none."""
    head = re.split(r"[,(—]|\s+-\s+", (note or "").strip(), maxsplit=1)[0]
    head = re.sub(r"^\s*(?:by|to)\s+", "", head, flags=re.IGNORECASE).strip(" .")
    if len(node_key(head)) < 2 or not re.search(r"[a-zA-Z]", head):
        return None
    # A head that is only date-words is a date, not a company.
    if re.fullmatch(
        r"(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|q[1-4]|\d{4}|\s|early|late|mid)+",
        head, re.IGNORECASE,
    ):
        return None
    return head


def company_node(lead: Lead) -> tuple[str, str]:
    """(key, label) for the startup behind a lead. Handle-keyed when no
    company name exists yet, so stealth founders still get a node — and the
    key upgrades to the company's own once the classifier names it."""
    key = company_key(lead) or lead.account.handle.lower()
    return key, display_name(lead)


def edges_for_lead(lead: Lead) -> list[Edge]:
    """Every edge one lead's evidence supports. Pure; deduped by the caller
    across leads (the same investor appears on many rows)."""
    edges: list[Edge] = []
    ckey, clabel = company_node(lead)
    verdict = lead.llm
    account = lead.account

    def add(src_type, src, rel, dst_type, dst, evidence="",
            src_key_override=None, dst_key_override=None):
        skey = src_key_override or node_key(src)
        dkey = dst_key_override or node_key(dst)
        if len(skey) < 2 or len(dkey) < 2 or skey == dkey:
            return
        edges.append(Edge(
            src_type=src_type, src_key=skey, src_label=src, rel=rel,
            dst_type=dst_type, dst_key=dkey, dst_label=dst,
            evidence=(evidence or "")[:200],
        ))

    # Investors → company (research/classifier-sourced, evidence-cited).
    if verdict is not None:
        for investor in _investor_names(verdict.funding_investors):
            add("investor", investor, "invested_in", "company", clabel,
                verdict.funding_evidence or "", dst_key_override=ckey)

        # Named founders → company; their past lab when the descriptor says.
        for entry in verdict.founders:
            person = _founder_name(entry)
            if not person:
                continue
            add("person", person, "founded", "company", clabel,
                entry, dst_key_override=ckey)
            for lab, snippet in _labs_in(entry):
                add("person", person, "alum_of", "lab", lab, snippet)

        # Acquirer — the edge that answers "what else did they buy".
        if verdict.company_status in ("acquired", "merged"):
            acquirer = _acquirer_from_note(verdict.company_status_note)
            if acquirer:
                add("company", clabel, "acquired_by", "company", acquirer,
                    verdict.company_status_evidence or "",
                    src_key_override=ckey)

    # The account's own person, when it IS a person building this company.
    person_name = (account.name or f"@{account.handle}").strip()
    is_founder_account = (
        verdict is not None and verdict.account_type == "founder"
    ) or (verdict is None and founder_like(lead))
    if is_founder_account and node_key(person_name) != ckey:
        add("person", person_name, "founded", "company", clabel,
            f"@{account.handle}", dst_key_override=ckey)
    # Their past lab, from the published record first (arXiv affiliation
    # history), else from the bio's own words.
    if is_founder_account or verdict is None:
        if account.lab_move and "→" in account.lab_move:
            lab = account.lab_move.split("→")[0].strip()
            add("person", person_name, "alum_of", "lab", lab.lower(),
                "arXiv affiliation history")
        for lab, snippet in _labs_in(account.bio):
            add("person", person_name, "alum_of", "lab", lab, f"bio: {snippet}")

    # Smart money watching — follow edges from the investor watchlist.
    for watcher in account.followed_by:
        add("watcher", f"@{watcher.lstrip('@')}", "follows", "company", clabel,
            dst_key_override=ckey,
            src_key_override=watcher.lstrip("@").lower())

    return edges


def dedupe(edges: list[Edge]) -> list[Edge]:
    """First edge per (src, rel, dst) wins — callers order by lead recency,
    so the freshest evidence string is the one kept."""
    seen: dict[tuple, Edge] = {}
    for edge in edges:
        seen.setdefault(edge.key, edge)
    return list(seen.values())
