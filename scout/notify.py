"""Slack notifications — mention/assignment pings now, digests with the worker.

Design rules:
- Failure-tolerant: a Slack outage must never break a triage click; every
  send is tenacity-wrapped and errors are logged, not raised.
- Pure builders: functions that COMPOSE messages take plain data and return
  Block Kit dicts, so tests cover them without network. Only post_slack does
  I/O.
- Deep links: {app_base_url}?s=<handle>&p=<page> — the UI's query-param
  bridge routes them to the right page + selection.
"""

from __future__ import annotations

import httpx
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_random_exponential

from scout.models import Event
from scout.store import Store

_console = Console()

_SLACK_TIMEOUT_S = 10


def slack_configured(store: Store) -> bool:
    return bool(store.get_setting("slack_webhook_url"))


@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_random_exponential(multiplier=1, max=10))
def _post(webhook_url: str, payload: dict) -> None:
    resp = httpx.post(webhook_url, json=payload, timeout=_SLACK_TIMEOUT_S)
    resp.raise_for_status()


def post_slack(store: Store, text: str, blocks: list[dict] | None = None) -> bool:
    """Send one message to the firm channel. Returns False (and logs) on any
    failure — callers never branch on delivery."""
    webhook = store.get_setting("slack_webhook_url")
    if not webhook:
        return False
    payload: dict = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        _post(webhook, payload)
        return True
    except Exception as exc:  # noqa: BLE001 — notification failure is non-fatal
        _console.print(f"[yellow]Slack notification failed:[/] {exc}")
        return False


def deep_link(store: Store, handle: str, page: str = "Startups") -> str:
    base = (store.get_setting("app_base_url") or "").rstrip("/")
    if not base:
        return ""
    return f"{base}/?s={handle}&p={page}"


def _slack_name(store: Store, user_id: str) -> str:
    """`<@U…>` when the member mapped their Slack id, else their name."""
    user = store.get_user(user_id) or {}
    member_id = (user.get("slack_member_id") or "").strip()
    if member_id:
        return f"<@{member_id}>"
    return user.get("name") or user_id.split("@")[0]


def _actor_name(store: Store, actor: str) -> str:
    if actor.startswith(("agent:", "system:", "schedule:")):
        return "Scout"
    user = store.get_user(actor) or {}
    return user.get("name") or actor.split("@")[0]


# ------------------------------------------------------------ pure builders


def mention_message(store: Store, event: Event, startup_name: str) -> str:
    """A comment @-mention ping."""
    who = _actor_name(store, event.actor)
    mentions = " ".join(_slack_name(store, m) for m in
                        event.payload.get("mentions", []))
    preview = event.payload.get("preview", "")
    link = deep_link(store, event.handle or "", "Startups")
    line = f"{mentions} — {who} mentioned you on *{startup_name}*: “{preview}”"
    return f"{line}\n{link}" if link else line


def assignment_message(store: Store, event: Event, startup_name: str) -> str:
    who = _actor_name(store, event.actor)
    assignee = _slack_name(store, event.payload.get("assignee", ""))
    link = deep_link(store, event.handle or "", "Startups")
    line = f"{assignee} — {who} assigned *{startup_name}* to you"
    return f"{line}\n{link}" if link else line


# -------------------------------------------------------------- dispatchers


def ping_mentions(store: Store, event: Event, startup_name: str) -> None:
    """Called inline right after add_comment when the comment mentions
    someone. Per-user opt-out lives in users.settings_json."""
    if not event.payload.get("mentions"):
        return
    post_slack(store, mention_message(store, event, startup_name))


def ping_assignment(store: Store, event: Event, startup_name: str) -> None:
    if not event.payload.get("assignee"):
        return
    post_slack(store, assignment_message(store, event, startup_name))


# ------------------------------------------------------------------ digests


def digest_data(store: Store, since: datetime, window: str = "daily") -> dict:
    """Gather everything a digest reports on. Store reads only — the two
    builders below are pure, so message shape is testable without a DB.

    What earns a place is judged by one question: would a partner waking up
    in another timezone act differently for having read it? New high scorers
    yes; a re-scored startup nobody touched, no.
    """
    from scout.collab import vote_summary
    from scout.companies import display_name
    from scout.status import POSITIVE_STATUSES

    pipeline = store.all_pipeline()
    ledger = store.load_lead_ledger(include_demo=False)
    threshold = float(store.get_setting("digest_score_threshold") or 60.0)

    # New and untriaged: the actual inbox. A startup someone already moved
    # is not news to the firm.
    top_new: list[dict] = []
    for entry in ledger:
        first_seen = entry.first_seen_at
        if first_seen is None or first_seen < since:
            continue
        handle = entry.lead.account.handle.lower()
        status = (pipeline.get(handle, {}) or {}).get("status") or "new"
        if status != "new" or entry.lead.score < threshold:
            continue
        top_new.append({
            "handle": handle,
            "name": display_name(entry.lead),
            "score": entry.lead.score,
            "fit": entry.lead.llm.thesis_fit if entry.lead.llm else None,
            "why": ((entry.lead.llm.why_interesting
                     or entry.lead.llm.one_line_summary)
                    if entry.lead.llm else "") or "",
            "link": deep_link(store, handle, "Startups"),
        })
    top_new.sort(key=lambda item: -item["score"])
    top_new = top_new[:8]

    # Movers: already-triaged startups whose score jumped. A shortlisted
    # company gaining 15 points has usually just shipped or raised.
    movers = [
        {
            "handle": entry.lead.account.handle.lower(),
            "name": display_name(entry.lead),
            "score": entry.lead.score,
            "delta": entry.score_delta,
            "link": deep_link(store, entry.lead.account.handle.lower(), "Startups"),
        }
        for entry in ledger
        if entry.score_delta is not None and entry.score_delta >= 10
        and entry.last_seen_at is not None and entry.last_seen_at >= since
    ]
    movers.sort(key=lambda item: -(item["delta"] or 0))
    movers = movers[:5]

    events = store.events(since=since, limit=400)
    # Machine chatter is not activity — a partner cares what PEOPLE did.
    human_events = [e for e in events
                    if not e.actor.startswith(("agent:", "system:", "schedule:"))]
    by_actor: dict[str, int] = {}
    for event in human_events:
        by_actor[event.actor] = by_actor.get(event.actor, 0) + 1
    actors = [
        {"name": _actor_name(store, actor), "n": n}
        for actor, n in sorted(by_actor.items(), key=lambda kv: -kv[1])
    ]

    # Contested: the meeting agenda, delivered before the meeting.
    votes = store.all_votes()
    contested: list[dict] = []
    for handle, handle_votes in votes.items():
        summary = vote_summary(handle_votes)
        if summary is None or not summary.contested:
            continue
        row = pipeline.get(handle, {}) or {}
        if (row.get("status") or "new") == "passed":
            continue
        lead = next((e.lead for e in ledger
                     if e.lead.account.handle.lower() == handle), None)
        contested.append({
            "handle": handle,
            "name": display_name(lead) if lead else f"@{handle}",
            "detail": " · ".join(
                f"{_actor_name(store, actor)}: {stance.replace('_', ' ')}"
                for actor, stance in sorted(summary.by_actor.items())
            ),
            "link": deep_link(store, handle, "Shortlist"),
        })
    contested = contested[:5]

    # Memos waiting on a human — the one thing that blocks on a person.
    awaiting = [
        {
            "handle": handle,
            "name": (display_name(lead) if (lead := next(
                (e.lead for e in ledger
                 if e.lead.account.handle.lower() == handle), None))
                else f"@{handle}"),
            "link": deep_link(store, handle, "Memos"),
        }
        for handle, row in pipeline.items()
        if (row or {}).get("memo_review_status") == "requested"
    ][:5]

    n_shortlisted = sum(
        1 for row in pipeline.values()
        if (row.get("status") or "new") in POSITIVE_STATUSES
    )
    latest_run = store.latest_run() or {}

    data = {
        "window": window,
        "since": since,
        "top_new": top_new,
        "movers": movers,
        "contested": contested,
        "awaiting_review": awaiting,
        "actors": actors,
        "n_events": len(human_events),
        "n_shortlisted": n_shortlisted,
        "last_run_at": latest_run.get("created_at", ""),
        "base_url": (store.get_setting("app_base_url") or "").rstrip("/"),
    }
    data["has_content"] = bool(
        top_new or movers or contested or awaiting or human_events
    )
    return data


def digest_fallback_text(data: dict) -> str:
    """The notification-line summary Slack shows before blocks render."""
    label = "This week" if data["window"] == "weekly" else "Today"
    bits = []
    if data["top_new"]:
        bits.append(f"{len(data['top_new'])} new to review")
    if data["contested"]:
        bits.append(f"{len(data['contested'])} contested")
    if data["awaiting_review"]:
        bits.append(f"{len(data['awaiting_review'])} memo(s) awaiting review")
    if not bits:
        bits.append("no new leads")
    return f"Scout — {label.lower()}: " + ", ".join(bits)


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _link(name: str, url: str) -> str:
    return f"<{url}|{name}>" if url else f"*{name}*"


def digest_blocks(data: dict) -> list[dict]:
    """Block Kit for the digest. Pure — takes the dict digest_data built.

    Ordered by what should change someone's next hour: things needing a
    decision first, new candidates second, context last.
    """
    heading = "Scout — this week" if data["window"] == "weekly" else "Scout — today"
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": heading}}
    ]

    if data["awaiting_review"]:
        lines = "\n".join(f"• {_link(m['name'], m['link'])}"
                          for m in data["awaiting_review"])
        blocks.append(_section(f"*Memos awaiting review*\n{lines}"))

    if data["contested"]:
        lines = "\n".join(f"• {_link(c['name'], c['link'])} — {c['detail']}"
                          for c in data["contested"])
        blocks.append(_section(f"*Contested — worth a conversation*\n{lines}"))

    if data["top_new"]:
        lines = []
        for item in data["top_new"]:
            fit = f" · fit {item['fit']:.0%}" if item["fit"] is not None else ""
            why = f"\n   _{item['why'][:150]}_" if item["why"] else ""
            lines.append(
                f"• {_link(item['name'], item['link'])} — "
                f"*{item['score']:.0f}*{fit}{why}"
            )
        blocks.append(_section("*New, untriaged*\n" + "\n".join(lines)))
    else:
        blocks.append(_section("_No new startups above the digest threshold._"))

    if data["movers"]:
        lines = "\n".join(
            f"• {_link(m['name'], m['link'])} — {m['score']:.0f} "
            f"(+{m['delta']:.0f})" for m in data["movers"]
        )
        blocks.append(_section(f"*Moving up*\n{lines}"))

    if data["actors"]:
        who = ", ".join(f"{a['name']} ({a['n']})" for a in data["actors"])
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Activity: {who}"}],
        })

    footer = f"{data['n_shortlisted']} in the funnel"
    if data["base_url"]:
        footer += f" · <{data['base_url']}|open Scout>"
    blocks.append({"type": "context",
                   "elements": [{"type": "mrkdwn", "text": footer}]})
    return blocks


def send_digest(store: Store, since: datetime, window: str = "daily") -> bool:
    """Build and post one digest. Returns whether it went out."""
    data = digest_data(store, since, window=window)
    if not data["has_content"]:
        return False
    return post_slack(store, digest_fallback_text(data), digest_blocks(data))
