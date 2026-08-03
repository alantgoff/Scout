"""Headless memo generation — the path the worker and CLI share with the UI.

The evidence assembly (site crawl, tweets, notes, the investor's own
categorization) used to live inside ui.py, which meant a memo could only be
produced by a human sitting in front of Streamlit. Moving it here lets a
scheduled job write memos overnight while leaving the UI's live narration
(`st.status` progress for deep research) exactly as it was — the UI passes
its own `on_event` and keeps its spinner; everything below the narration is
now the same code for both callers.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import urlparse

from scout.agents import investment_memo
from scout.config import Settings, Thesis
from scout.models import Lead
from scout.store import Store
from scout.web import bundle_text, fetch_site_bundle, normalize_site_url


def site_evidence(
    lead: Lead, settings: Settings, store: Store, depth: str
) -> tuple[str, str]:
    """(labeled site bundle, provenance note) for the memo dossier.

    Crawls the key pages of every candidate URL on file (company_url + the
    bio website), cache-first, and says WHOSE pages these appear to be — a
    founder's personal domain must never masquerade as product evidence
    (the Walden Robotics failure).
    """
    if depth == "quick":
        return "", ""
    verdict = lead.llm
    candidates = [u for u in [(verdict.company_url if verdict else None),
                              lead.account.website] if u]
    primary = next((u for u in candidates if normalize_site_url(u)), None)
    if primary is None:
        return "", "no website URL on file at all"
    try:
        pages = asyncio.run(fetch_site_bundle(
            primary, settings, store, extra_urls=tuple(candidates[1:])))
    except Exception:  # noqa: BLE001 — a dead site must not fail the memo
        pages = []
    site_text = bundle_text(pages, 9000 if depth == "standard" else 6000)
    host = urlparse(normalize_site_url(primary) or "").netloc
    if not site_text:
        return "", f"the URL on file ({host}) yielded no readable pages"
    company = (verdict.company_name or "").strip() if verdict else ""
    company_slug = re.sub(r"[^a-z0-9]", "", company.lower())
    host_slug = re.sub(r"[^a-z0-9]", "", host.lower())
    if company_slug and company_slug not in host_slug:
        note = (f"captured from {host}, which does NOT look like "
                f"{company}'s own domain — likely the founder's personal "
                "site; treat product claims accordingly")
    else:
        note = f"captured from {host}"
    return site_text, note


def memo_kwargs(
    lead: Lead,
    settings: Settings,
    store: Store,
    *,
    depth: str = "standard",
    focus: str = "",
    notes: str = "",
    attrs: dict | None = None,
) -> dict:
    """Everything investment_memo needs beyond (lead, thesis, settings).

    Without an Anthropic key the template renders regardless, so skip the
    crawl rather than pay for evidence nothing will read.
    """
    site_text, site_note = (
        site_evidence(lead, settings, store, depth)
        if settings.anthropic_api_key else ("", "")
    )
    return dict(
        site_text=site_text,
        site_note=site_note,
        tweets=store.get_tweets(lead.account.id, limit=12),
        notes=notes,
        depth=depth,
        focus=focus,
        attrs=attrs or None,
    )


def generate_memo(
    store: Store,
    settings: Settings,
    thesis: Thesis,
    lead: Lead,
    *,
    depth: str = "standard",
    focus: str = "",
    actor: str = "agent:memo",
    on_event=None,
) -> dict:
    """Write (or refuse to write) one memo. Returns a result summary.

    Two refusals are deliberate and both preserve stored work:
    - a fallback skeleton never replaces a real memo (no key, API error, or
      unusable output would otherwise trade a researched memo for a stub);
    - nothing is written when the agent returns nothing usable.

    The write goes through set_memo, so every generation appends a version
    and a human's edits stay restorable.
    """
    handle = lead.account.handle.lower()
    row = store.get_pipeline(handle) or {}
    existing = (row.get("brief") or "").strip()
    attrs = {
        key: value
        for key, value in (store.all_attrs().get(handle) or {}).items()
        if value not in (None, "", [], False)
    }
    memo, is_ai, meta = investment_memo(
        lead, thesis, settings,
        on_event=on_event,
        **memo_kwargs(lead, settings, store, depth=depth, focus=focus,
                      notes=row.get("notes") or "", attrs=attrs),
    )
    if not (memo or "").strip():
        return {"handle": handle, "written": False,
                "reason": "the agent returned nothing usable"}
    if not is_ai and existing:
        return {"handle": handle, "written": False,
                "reason": "generation fell back to the skeleton; kept the "
                          "existing memo"}
    store.set_memo(handle, memo, meta=meta, kind="generated", actor=actor)
    return {
        "handle": handle, "written": True, "is_ai": is_ai,
        "depth": meta.get("depth", depth),
        "sources": len(meta.get("sources", []) or []),
        "searches": meta.get("searches", 0),
    }
