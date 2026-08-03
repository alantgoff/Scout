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
