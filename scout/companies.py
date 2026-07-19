"""Startup identity + grouping — the STARTUP is the first-class object.

The unit scout discovers is an X account; the unit a VC sources is a STARTUP.
Every founder-like lead resolves to one via `startup_identity`:

  - the classifier named a company        → that company ("EvalHQ")
  - founder, no company name yet          → a synthesized identity tied to the
    person: "Ada Lin's stealth startup" (pre-launch) or "Ada Lin's unnamed
    startup" (launched, name unknown)
  - corporate account / commentator / no founder evidence → None (stays an
    account, never dressed up as a startup)

`group_by_company` then folds leads attributed to the same company into one
entry (primary = highest score, the rest secondary accounts); unnamed founders
pass through as singleton startups. Views and the report count startups, with
founders as members.

Pure functions, no I/O — unit-tested.
"""

from __future__ import annotations

import re

from scout.models import Lead, LedgerEntry

Pair = tuple[Lead, LedgerEntry | None]
Grouped = tuple[Lead, LedgerEntry | None, list[Lead]]  # primary, entry, secondaries

# Heuristic founder evidence — used only when no classifier verdict exists
# (heuristics-only mode): any of these firing marks the account founder-like.
FOUNDER_EVIDENCE_SIGNALS = {
    "bio_intent", "departure_signal", "bio_change", "launch_traction",
    "builder_evidence", "github_evidence",
}


def founder_like(lead: Lead) -> bool:
    """Is this lead someone building a company (or the company itself)?

    Classifier verdict when present (founder/startup account types); founder-
    evidence signals otherwise, so heuristics-only runs still resolve startups.
    """
    if lead.llm is not None:
        if lead.llm.account_type is not None:
            return lead.llm.account_type in ("founder", "startup")
        return lead.llm.is_founder
    return any(
        s.name in FOUNDER_EVIDENCE_SIGNALS and s.value > 0 for s in lead.signals
    )


def startup_identity(lead: Lead) -> tuple[str, bool] | None:
    """(display_name, synthesized) for the startup behind a lead; None when
    the lead isn't founder-like (keep it an account, not a fake startup).

    Named companies keep their real name. A founder without one gets a stable
    placeholder derived from their name: possessive + stage-aware descriptor —
    "stealth startup" pre-launch, "unnamed startup" once launched.
    """
    verdict = lead.llm
    if verdict is not None and verdict.company_name and company_key(lead):
        return verdict.company_name.strip(), False
    if not founder_like(lead):
        return None
    who = (lead.account.name or f"@{lead.account.handle}").strip()
    possessive = who + ("'" if who.lower().endswith("s") else "'s")
    stage = verdict.stage if verdict else None
    descriptor = "unnamed startup" if stage in ("launched", "scaling") else "stealth startup"
    return f"{possessive} {descriptor}", True


def company_key(lead: Lead) -> str | None:
    """Normalized grouping key for the startup behind a lead (None = unknown).

    Lowercased alphanumerics only, so "EvalHQ", "eval hq", and "Eval-HQ.ai"
    merge. Single-character keys are discarded as noise.
    """
    verdict = lead.llm
    if verdict is None or not verdict.company_name:
        return None
    key = re.sub(r"[^a-z0-9]", "", verdict.company_name.lower())
    # Strip a trailing TLD the classifier sometimes includes ("evalhq.ai").
    for tld in ("ai", "io", "com", "dev", "app", "xyz", "co"):
        raw = verdict.company_name.lower().strip()
        if raw.endswith("." + tld):
            key = re.sub(r"[^a-z0-9]", "", raw[: -len(tld) - 1])
            break
    return key if len(key) > 1 else None


def group_by_company(pairs: list[Pair]) -> list[Grouped]:
    """Fold (lead, entry) pairs sharing a company into single entries.

    Order-preserving: each group sits where its first member appeared; the
    primary is the group's highest-scoring lead; leads without a company key
    pass through as singletons.
    """
    members: dict[str, list[Pair]] = {}
    order: list[tuple[str, Pair | str]] = []
    for pair in pairs:
        key = company_key(pair[0])
        if key is None:
            order.append(("solo", pair))
        else:
            if key not in members:
                members[key] = []
                order.append(("group", key))
            members[key].append(pair)

    grouped: list[Grouped] = []
    for kind, payload in order:
        if kind == "solo":
            lead, entry = payload  # type: ignore[misc]
            grouped.append((lead, entry, []))
        else:
            group = sorted(members[payload], key=lambda p: -p[0].score)  # type: ignore[index]
            primary_lead, primary_entry = group[0]
            grouped.append((primary_lead, primary_entry, [p[0] for p in group[1:]]))
    return grouped
