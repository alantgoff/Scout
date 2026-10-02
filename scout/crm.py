"""CRM write-back: the startups a firm decides to pursue land in its CRM.

Scout is where companies are found and judged; the firm's CRM (Attio or
Affinity) is where relationships are run. Once a startup reaches the push
threshold (shortlisted, by default), Scout writes it there:

  1. the company record, matched by DOMAIN — an existing record is linked,
     never overwritten; a new one gets Scout's name and one-liner;
  2. an entry on the configured list (idempotent: Attio asserts by parent,
     Affinity is checked before adding);
  3. a "Sourced by Scout" note, once — what it is, why it surfaced, the
     score, the evidence-backed facts, a link back into Scout;
  4. the investment memo as a note, whenever its text has changed.

Rules this module keeps:

- **Domain or nothing.** A CRM record keyed by a guessed name is a
  duplicate waiting to happen, so a startup without a company domain (a
  stealth founder, a personal site) is skipped and says so. The domain
  comes from the classifier's company_url, a domain-keyed account, or a
  company account's own site — through the same publisher/social filter
  every source uses (web.company_domain).
- **Never clobber what people curated.** Existing records are only linked;
  the only fields Scout ever fills are empty ones.
- **Never delete.** A startup later passed in Scout stays in the CRM — the
  CRM's own pipeline is where that decision is recorded.
- **Resume, don't repeat.** Each step is recorded in `crm_links` as soon
  as it succeeds, so a failure halfway (or a retry by the worker) picks up
  where it stopped instead of writing a second note. Non-idempotent POSTs
  are retried only when the server cannot have processed them (connect
  failures, 429); idempotent PUT/PATCH/GET also retry 5xx.
- **Firm-private.** Votes, comments and member notes are not sent; what is
  sent is what Scout sourced and the memo the firm wrote.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Protocol

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from scout.companies import startup_identity
from scout.config import Settings
from scout.models import CrmPushResult, Lead
from scout.status import FUNNEL_STAGES, STATUS_LABELS
from scout.web import company_domain

ATTIO_BASE = "https://api.attio.com/v2"
AFFINITY_BASE = "https://api.affinity.co"
TIMEOUT_S = 20.0
DEFAULT_THRESHOLD = "shortlisted"
SYNC_LIMIT = 50  # companies per sync pass — a backlog drains over passes

SUMMARY_TITLE = "Sourced by Scout"
MEMO_TITLE = "Scout memo"


class CrmError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------- pure


def company_domain_for(lead: Lead) -> str | None:
    """The domain a CRM record for this startup is keyed by, or None.

    A founder's own website is NOT the company's — only a company account
    (account_type "startup") or a domain-keyed row may lend its site."""
    verdict = lead.llm
    candidates = []
    if verdict is not None and verdict.company_url:
        candidates.append(verdict.company_url)
    if lead.account.profile_url:
        candidates.append(lead.account.profile_url)
    if verdict is not None and verdict.account_type == "startup" and lead.account.website:
        candidates.append(lead.account.website)
    for url in candidates:
        domain = company_domain(url, own_host="")
        if domain:
            return domain
    return None


def company_name_for(lead: Lead, domain: str) -> str:
    """The real company name; a synthesized placeholder ("Ada Lin's stealth
    startup") must never become a CRM record's name, so the domain stands
    in for it."""
    identity = startup_identity(lead)
    if identity and not identity[1]:
        return identity[0]
    return domain


# Thresholds a firm may pick. "won" (Allocated) is never one: portfolio
# companies reach Scout through `scout add --status won` so warm paths know
# the cap tables — they are already in the CRM, and a "Sourced by Scout"
# note on them would be false.
THRESHOLDS = [s for s in FUNNEL_STAGES if s != "won"]


def push_statuses(threshold: str | None) -> set[str]:
    """The funnel from the threshold onward, short of Allocated. "new" and
    "passed" never push. A company that was pushed on its way to Allocated
    keeps its link; an explicit `scout crm push` still sends anything."""
    stage = threshold if threshold in THRESHOLDS else DEFAULT_THRESHOLD
    return set(THRESHOLDS[THRESHOLDS.index(stage):])


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode()).hexdigest()[:16]


def summary_note(lead: Lead, *, status: str = "", thesis_name: str = "",
                 link: str = "") -> str:
    """The "Sourced by Scout" note: what the company is and why it surfaced.

    Every line restates a field Scout already holds with its evidence —
    a round only with the source it was read in (the models.py validator
    already drops unevidenced rounds), signals with their own detail."""
    verdict = lead.llm
    lines: list[str] = []
    if verdict is not None and (verdict.product_summary or verdict.one_line_summary):
        lines.append((verdict.product_summary or verdict.one_line_summary).strip())
        lines.append("")
    facts = []
    if lead.score is not None:
        fit = (f", thesis fit {verdict.thesis_fit:.0%}"
               if verdict is not None and verdict.thesis_fit is not None else "")
        under = f' under "{thesis_name}"' if thesis_name else ""
        facts.append(f"Scout score {lead.score:.0f}/100{fit}{under}")
    if status:
        facts.append(f"Status in Scout: {STATUS_LABELS.get(status, status)}")
    if verdict is not None:
        shape = " · ".join(x for x in (
            verdict.stage and f"stage {verdict.stage}",
            " / ".join(x for x in (verdict.sector, verdict.subsector) if x),
            verdict.business_model,
        ) if x)
        if shape:
            facts.append(shape)
        if verdict.funding_stage and verdict.funding_stage != "unknown":
            round_ = verdict.funding_stage.replace("_", " ").title()
            amount = f" {verdict.funding_amount}" if verdict.funding_amount else ""
            investors = (f" — {', '.join(verdict.funding_investors)}"
                         if verdict.funding_investors else "")
            source = f" (source: {verdict.funding_evidence})" if verdict.funding_evidence else ""
            facts.append(f"Funding: {round_}{amount}{investors}{source}")
        if verdict.founders:
            facts.append("Founders: " + "; ".join(verdict.founders))
        if verdict.company_status and verdict.company_status != "active":
            note = f" — {verdict.company_status_note}" if verdict.company_status_note else ""
            facts.append(f"Company status: {verdict.company_status}{note}")
    lines += [f"- {fact}" for fact in facts]
    fired = sorted((s for s in lead.signals if s.value > 0),
                   key=lambda s: -(s.value * (s.weight or 0)))
    if fired:
        lines += ["", "Why it surfaced:"]
        lines += [f"- {s.name.replace('_', ' ')}"
                  + (f": {s.detail}" if s.detail else "") for s in fired[:6]]
    if verdict is not None and verdict.why_interesting:
        lines += ["", verdict.why_interesting.strip()]
    found = ", ".join(sorted(set(lead.account.sources or [lead.account.source]) - {""}))
    lines += ["", f"Found via {found or 'Scout'} — @{lead.account.handle}"]
    if link:
        lines.append(f"Open in Scout: {link}")
    return "\n".join(lines).strip() + "\n"


# --------------------------------------------------------------------- HTTP


def _retryable(idempotent: bool):
    def check(exc: BaseException) -> bool:
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True  # never reached the server
        if isinstance(exc, CrmError):
            # 429: rejected, not processed. 5xx: maybe processed — only a
            # request that is safe to repeat may go again.
            return exc.status == 429 or (idempotent and (exc.status or 0) >= 500)
        if isinstance(exc, httpx.TransportError):
            return idempotent
        return False
    return check


# Module-level so tests can make retries instant.
RETRY_WAIT = wait_random_exponential(multiplier=0.5, max=8)


class _Http:
    def __init__(self, base: str, headers: dict[str, str],
                 client: httpx.Client | None = None) -> None:
        self.base = base
        self.headers = headers
        self._client = client

    @property
    def client(self) -> httpx.Client:
        # Lazy: building a client (Settings page, doctor) must not open one.
        if self._client is None:
            self._client = httpx.Client(timeout=TIMEOUT_S)
        return self._client

    def request(self, method: str, path: str, *, json: Any = None,
                params: dict | None = None) -> Any:
        idempotent = method in ("GET", "PUT", "PATCH")

        def once() -> Any:
            resp = self.client.request(method, self.base + path, json=json,
                                       params=params, headers=self.headers)
            if resp.status_code >= 400:
                detail = resp.text[:300].replace("\n", " ")
                raise CrmError(f"{method} {path} → HTTP {resp.status_code}: {detail}",
                               resp.status_code)
            return resp.json() if resp.content else {}

        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=RETRY_WAIT,
            retry=retry_if_exception(_retryable(idempotent)),
            reraise=True,
        ):
            with attempt:
                return once()
        return {}  # unreachable


class CrmClient(Protocol):
    provider: str
    label: str
    list_ref: str | None

    def upsert_company(self, name: str, domain: str,
                       description: str) -> tuple[str, str | None, bool]: ...
    def ensure_listed(self, remote_id: str) -> str: ...
    def add_note(self, remote_id: str, title: str, content: str) -> str: ...


class AttioClient:
    """Attio REST v2. The company is ASSERTED by domain (PUT with
    matching_attribute=domains: found → that record, else created), then
    only its empty name/description are filled."""

    provider = "attio"
    label = "Attio"

    def __init__(self, api_key: str, list_ref: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.list_ref = (list_ref or "").strip() or None
        self.http = _Http(ATTIO_BASE, {"Authorization": f"Bearer {api_key}"}, client)

    def upsert_company(self, name, domain, description):
        body = self.http.request(
            "PUT", "/objects/companies/records",
            params={"matching_attribute": "domains"},
            json={"data": {"values": {"domains": [domain]}}},
        )
        record = body.get("data") or {}
        record_id = (record.get("id") or {}).get("record_id")
        if not record_id:
            raise CrmError("Attio returned no record id for the company")
        values = record.get("values") or {}
        fill = {}
        if name and not values.get("name"):
            fill["name"] = name
        if description and not values.get("description"):
            fill["description"] = description
        if fill:
            self.http.request("PATCH", f"/objects/companies/records/{record_id}",
                              json={"data": {"values": fill}})
        return record_id, record.get("web_url"), "name" in fill

    def ensure_listed(self, remote_id):
        body = self.http.request(
            "PUT", f"/lists/{self.list_ref}/entries",
            json={"data": {"parent_record_id": remote_id, "parent_object": "companies",
                           "entry_values": {}}},
        )
        entry_id = ((body.get("data") or {}).get("id") or {}).get("entry_id")
        return str(entry_id or "listed")

    def add_note(self, remote_id, title, content):
        body = self.http.request(
            "POST", "/notes",
            json={"data": {"parent_object": "companies", "parent_record_id": remote_id,
                           "title": title, "format": "markdown", "content": content}},
        )
        return str(((body.get("data") or {}).get("id") or {}).get("note_id") or "")


class AffinityClient:
    """Affinity API v1 (Basic auth, the key as password). Organizations are
    found by an EXACT domain match among the term search's results — a
    name match is never enough — and created only when none matches."""

    provider = "affinity"
    label = "Affinity"

    def __init__(self, api_key: str, list_id: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.list_ref = (str(list_id or "")).strip() or None
        token = base64.b64encode(f":{api_key}".encode()).decode()
        self.http = _Http(AFFINITY_BASE, {"Authorization": f"Basic {token}"}, client)

    def upsert_company(self, name, domain, description):
        body = self.http.request("GET", "/organizations", params={"term": domain})
        for org in (body.get("organizations") or []):
            domains = {d.lower() for d in (org.get("domains") or []) if d}
            if org.get("domain"):
                domains.add(org["domain"].lower())
            if domain in domains:
                return str(org["id"]), None, False
        org = self.http.request("POST", "/organizations",
                                json={"name": name or domain, "domain": domain})
        if not org.get("id"):
            raise CrmError("Affinity returned no organization id")
        return str(org["id"]), None, True

    def ensure_listed(self, remote_id):
        org = self.http.request("GET", f"/organizations/{remote_id}")
        for entry in org.get("list_entries") or []:
            if str(entry.get("list_id")) == self.list_ref:
                return str(entry.get("id"))
        entry = self.http.request("POST", f"/lists/{self.list_ref}/list-entries",
                                  json={"entity_id": int(remote_id)})
        return str(entry.get("id") or "listed")

    def add_note(self, remote_id, title, content):
        # v1 notes are plain text; the title leads the body.
        note = self.http.request("POST", "/notes", json={
            "organization_ids": [int(remote_id)], "content": f"{title}\n\n{content}"})
        return str(note.get("id") or "")


def clients_from(settings: Settings) -> list[CrmClient]:
    clients: list[CrmClient] = []
    if (settings.attio_api_key or "").strip():
        clients.append(AttioClient(settings.attio_api_key, settings.attio_list))
    if (settings.affinity_api_key or "").strip():
        clients.append(AffinityClient(settings.affinity_api_key, settings.affinity_list_id))
    return clients


def configured(settings: Settings) -> list[str]:
    """Labels of the connected CRMs — keys only, no client built (the UI
    asks on every rerun)."""
    return [label for label, key in (("Attio", settings.attio_api_key),
                                     ("Affinity", settings.affinity_api_key))
            if (key or "").strip()]


# ------------------------------------------------------------- orchestration


def push(store, client: CrmClient, lead: Lead, *, status: str = "",
         memo: str = "", thesis_name: str = "", link: str = "") -> CrmPushResult:
    """Write one startup to one CRM, resuming from whatever earlier pushes
    already did (store.crm_link). Never raises for a CRM failure — the
    error is recorded on the link and returned."""
    handle = lead.account.handle.lower()
    domain = company_domain_for(lead)
    result = CrmPushResult(provider=client.provider, handle=handle, domain=domain)
    if domain is None:
        result.skipped = ("no company domain — CRM records are keyed by domain, "
                          "and a guessed one makes duplicates")
        return result
    name = company_name_for(lead, domain)
    description = (lead.llm.one_line_summary if lead.llm else "") or ""
    for attempt in (1, 2):
        link_row = store.crm_link(client.provider, domain) or {}
        try:
            remote_id = link_row.get("remote_id")
            if not remote_id:
                remote_id, url, created = client.upsert_company(name, domain, description)
                result.created = created
                store.record_crm_step(client.provider, domain, handle,
                                      remote_id=remote_id, remote_url=url,
                                      created=created)
                link_row = store.crm_link(client.provider, domain) or {}
            result.remote_id = remote_id
            result.remote_url = link_row.get("remote_url")
            if client.list_ref and not link_row.get("list_entry_id"):
                entry = client.ensure_listed(remote_id)
                result.listed = True
                store.record_crm_step(client.provider, domain, handle, list_entry_id=entry)
            if not link_row.get("summary_note_id"):
                note_id = client.add_note(remote_id, f"{SUMMARY_TITLE} — {name}",
                                          summary_note(lead, status=status,
                                                       thesis_name=thesis_name, link=link))
                result.notes.append(SUMMARY_TITLE)
                store.record_crm_step(client.provider, domain, handle,
                                      summary_note_id=note_id or "added")
            if memo.strip() and text_hash(memo) != link_row.get("memo_hash"):
                note_id = client.add_note(remote_id, f"{MEMO_TITLE} — {name}", memo)
                result.notes.append(MEMO_TITLE)
                store.record_crm_step(client.provider, domain, handle,
                                      memo_hash=text_hash(memo), memo_note_id=note_id)
            store.record_crm_step(client.provider, domain, handle, error="",
                                  event=result.changed, result=result)
            return result
        except CrmError as exc:
            # A record someone deleted in the CRM: forget the stale link and
            # start over once, rather than failing on it forever.
            if exc.status == 404 and link_row.get("remote_id") and attempt == 1:
                store.forget_crm_link(client.provider, domain)
                continue
            result.error = str(exc)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        store.record_crm_step(client.provider, domain, handle, error=result.error)
        return result
    return result


def due(store, memo: str, provider: str, domain: str | None, has_list: bool) -> bool:
    """Whether a push would do anything — decided locally, so a sync over
    an unchanged pipeline makes no API calls at all."""
    if domain is None:
        return False
    link_row = store.crm_link(provider, domain)
    if not link_row or not link_row.get("remote_id") or not link_row.get("summary_note_id"):
        return True
    if has_list and not link_row.get("list_entry_id"):
        return True
    return bool(memo.strip()) and text_hash(memo) != link_row.get("memo_hash")


def pending(store, settings: Settings,
            clients: list[CrmClient] | None = None) -> list[str]:
    """Handles at or past the threshold that a sync would push now."""
    clients = clients if clients is not None else clients_from(settings)
    if not clients:
        return []
    statuses = push_statuses(store.get_setting("crm_push_threshold"))
    out = []
    for handle, row in sorted(store.all_pipeline().items()):
        if (row.get("status") or "new") not in statuses:
            continue
        lead = store.latest_lead(handle)
        if lead is None:
            continue
        domain = company_domain_for(lead)
        if any(due(store, row.get("brief") or "", c.provider, domain, bool(c.list_ref))
               for c in clients):
            out.append(handle)
    return out


def sync(store, settings: Settings, *, handles: list[str] | None = None,
         limit: int = SYNC_LIMIT, thesis_name: str = "",
         clients: list[CrmClient] | None = None) -> list[CrmPushResult]:
    """Push every startup at or past the threshold that has something new
    to say — or exactly `handles`, regardless of status (an explicit push).
    Returns what happened; failures are results, not exceptions."""
    from scout.notify import deep_link

    clients = clients if clients is not None else clients_from(settings)
    if not clients:
        return []
    pipeline = store.all_pipeline()
    if handles is not None:
        wanted = [h.lstrip("@").lower() for h in handles]
    else:
        statuses = push_statuses(store.get_setting("crm_push_threshold"))
        wanted = sorted(h for h, row in pipeline.items()
                        if (row.get("status") or "new") in statuses)
    results: list[CrmPushResult] = []
    pushed = 0
    for handle in wanted:
        if pushed >= limit:
            break
        lead = store.latest_lead(handle)
        if lead is None:
            continue
        row = pipeline.get(handle) or {}
        status, memo = row.get("status") or "", row.get("brief") or ""
        domain = company_domain_for(lead)
        attempted = False
        for client in clients:
            if handles is None and not due(store, memo, client.provider, domain,
                                           bool(client.list_ref)):
                continue
            attempted = True
            results.append(push(store, client, lead, status=status, memo=memo,
                                thesis_name=thesis_name,
                                link=deep_link(store, handle)))
        pushed += attempted
    return results


def summarize(results: list[CrmPushResult]) -> str:
    changed = sum(1 for r in results if r.changed)
    failed = sum(1 for r in results if r.error)
    skipped = sum(1 for r in results if r.skipped)
    parts = [f"{changed} updated"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped (no company domain)")
    return ", ".join(parts)
