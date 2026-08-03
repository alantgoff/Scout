"""Pure collaboration logic — stances, disagreement, mentions.

No Streamlit, no I/O (the dbfields.py convention): the UI and the store
call in; unit tests cover the math. The design principle is the same
provenance grammar the model side already has: every human judgment carries
an author, a lens, and a rationale — and DISAGREEMENT between partners is a
first-class, computable signal, not a merge conflict.
"""

from __future__ import annotations

import re

from scout.models import Vote, VoteSummary

# Stance values. The asymmetry is deliberate: a pass (-2) weighs like a
# strong yes (+2), so one conviction holder and one dissenter read as a
# genuine split, not a mild average.
STANCES: dict[str, int] = {"strong_yes": 2, "yes": 1, "unsure": 0, "pass": -2}
STANCE_LABELS = {
    "strong_yes": "Strong yes",
    "yes": "Yes",
    "unsure": "Unsure",
    "pass": "Pass",
}
# A split needs at least two voters spanning a real gap (yes vs pass at
# minimum — spread >= 3 under the values above; strong_yes vs unsure is 2,
# an interesting hedge but not a fight).
CONTESTED_SPREAD = 3


def vote_summary(votes: list[Vote]) -> VoteSummary | None:
    """Tally one startup's stances; None when there are no votes."""
    if not votes:
        return None
    values = [STANCES.get(v.stance, 0) for v in votes]
    spread = float(max(values) - min(values)) if len(values) > 1 else 0.0
    stamps = [s for s in (v.updated_at or v.created_at for v in votes) if s]
    return VoteSummary(
        handle=votes[0].handle,
        n_votes=len(votes),
        mean=round(sum(values) / len(values), 2),
        spread=spread,
        contested=len(votes) >= 2 and spread >= CONTESTED_SPREAD,
        by_actor={v.actor: v.stance for v in votes},
        latest_at=max(stamps) if stamps else None,
    )


def contested_sort_key(summary: VoteSummary | None) -> tuple:
    """Sort key for partner-meeting mode: widest split first, then the most
    recently voted-on. Startups with no votes sink to the end."""
    if summary is None:
        return (1, 0.0, "")
    latest = summary.latest_at.isoformat() if summary.latest_at else ""
    return (0, -summary.spread, -summary.n_votes, latest)


def disagreements(
    summaries: dict[str, VoteSummary], display_names: dict[str, str] | None = None
) -> list[tuple[str, str]]:
    """(handle, sentence) pairs for every contested startup — the partner
    meeting agenda. display_names maps actor id -> short name."""
    names = display_names or {}

    def short(actor: str) -> str:
        return names.get(actor) or actor.split("@")[0]

    out: list[tuple[str, str]] = []
    for handle, summary in summaries.items():
        if not summary.contested:
            continue
        sides: dict[str, list[str]] = {}
        for actor, stance in summary.by_actor.items():
            sides.setdefault(stance, []).append(short(actor))
        parts = [
            f"{' & '.join(sorted(who))}: {STANCE_LABELS.get(stance, stance)}"
            for stance, who in sorted(
                sides.items(), key=lambda kv: -STANCES.get(kv[0], 0)
            )
        ]
        out.append((handle, " · ".join(parts)))
    out.sort(key=lambda pair: contested_sort_key(summaries[pair[0]]))
    return out


_MENTION_RE = re.compile(r"@([A-Za-z0-9._+-]+(?:@[A-Za-z0-9.-]+)?)")


def parse_mentions(body: str, users: list[dict]) -> list[str]:
    """User ids @-mentioned in a comment body.

    Matches "@sara", "@sara.lin", or a full "@sara@firm.com" against the
    users table (id = lowercased email, name = display name). Name matches
    compare the token against the first name and the dotted full name.
    """
    if not body or not users:
        return []
    by_id = {u["id"]: u["id"] for u in users}
    by_alias: dict[str, str] = {}
    for u in users:
        by_alias[u["id"].split("@")[0].lower()] = u["id"]
        name = (u.get("name") or "").strip().lower()
        if name:
            by_alias[name.replace(" ", ".")] = u["id"]
            by_alias.setdefault(name.split(" ")[0], u["id"])
    found: list[str] = []
    for token in _MENTION_RE.findall(body):
        token = token.lower().rstrip(".")
        target = by_id.get(token) or by_alias.get(token)
        if target and target not in found:
            found.append(target)
    return found
