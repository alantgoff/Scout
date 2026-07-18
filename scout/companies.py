"""Company grouping — fold X accounts that belong to the same startup.

The unit scout discovers is an X account; the unit a VC sources is a STARTUP.
A launched startup typically surfaces twice — the founder's account and the
company's own account. `group_by_company` folds leads that the classifier
attributed to the same company into one entry (primary = highest score,
the rest attached as secondary accounts), so the Startups view and the report
count companies, not accounts.

Pure functions, no I/O — unit-tested.
"""

from __future__ import annotations

import re

from scout.models import Lead, LedgerEntry

Pair = tuple[Lead, LedgerEntry | None]
Grouped = tuple[Lead, LedgerEntry | None, list[Lead]]  # primary, entry, secondaries


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
