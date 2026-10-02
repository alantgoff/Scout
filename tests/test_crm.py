"""CRM write-back (scout/crm.py) against in-memory fakes of the Attio and
Affinity APIs (httpx.MockTransport) — no network. What is pinned: a record
is keyed by the company's domain or not at all; existing CRM data is never
overwritten; a push resumes rather than repeats (no duplicate notes); only
requests the server cannot have processed are retried; and an unchanged
pipeline syncs with zero API calls."""

from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest
from tenacity import wait_none

from scout import crm, doctor
from scout.config import Settings
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.store import Store


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch):
    monkeypatch.setattr(crm, "RETRY_WAIT", wait_none())


# ------------------------------------------------------------------- fakes


class FakeAttio:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.entries: dict[tuple[str, str], str] = {}
        self.notes: list[dict] = []
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[tuple[str, str], list[int]] = {}

    def add_record(self, domain: str, name: str = "", description: str = "") -> str:
        rid = f"rec-{len(self.records) + 1}"
        self.records[rid] = {"domains": [domain], "name": name, "description": description}
        return rid

    def _out(self, rid: str) -> dict:
        rec = self.records[rid]
        values = {k: ([{"value": rec[k]}] if rec[k] else []) for k in ("name", "description")}
        values["domains"] = [{"domain": d} for d in rec["domains"]]
        return {"data": {"id": {"record_id": rid}, "values": values,
                         "web_url": f"https://app.attio.com/firm/company/{rid}"}}

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        assert request.headers["Authorization"] == "Bearer attio-key"
        for (m, prefix), codes in self.fail.items():
            if m == method and path.startswith(prefix) and codes:
                return httpx.Response(codes.pop(0), json={"message": "injected"})
        body = json.loads(request.content) if request.content else {}
        if method == "PUT" and path == "/v2/objects/companies/records":
            assert parse_qs(request.url.query.decode())["matching_attribute"] == ["domains"]
            domain = body["data"]["values"]["domains"][0]
            rid = next((r for r, rec in self.records.items() if domain in rec["domains"]),
                       None) or self.add_record(domain)
            return httpx.Response(200, json=self._out(rid))
        if method == "PATCH" and path.startswith("/v2/objects/companies/records/"):
            rid = path.rsplit("/", 1)[1]
            if rid not in self.records:
                return httpx.Response(404, json={"message": "not found"})
            self.records[rid].update(body["data"]["values"])
            return httpx.Response(200, json=self._out(rid))
        if method == "PUT" and path.startswith("/v2/lists/"):
            data = body["data"]
            assert data["parent_object"] == "companies"
            rid = data["parent_record_id"]
            if rid not in self.records:
                return httpx.Response(404, json={"message": "no such record"})
            key = (path.split("/")[3], rid)
            self.entries.setdefault(key, f"ent-{len(self.entries) + 1}")
            return httpx.Response(200, json={"data": {"id": {"entry_id": self.entries[key]}}})
        if method == "POST" and path == "/v2/notes":
            data = body["data"]
            if data["parent_record_id"] not in self.records:
                return httpx.Response(404, json={"message": "no such record"})
            assert data["format"] == "markdown" and data["parent_object"] == "companies"
            self.notes.append(data)
            return httpx.Response(200, json={"data": {"id": {"note_id": f"note-{len(self.notes)}"}}})
        return httpx.Response(400, json={"message": f"unexpected {method} {path}"})

    def client(self, list_ref: str | None = "dealflow") -> crm.AttioClient:
        return crm.AttioClient("attio-key", list_ref,
                               client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class FakeAffinity:
    def __init__(self) -> None:
        self.orgs: dict[int, dict] = {}
        self.entries: list[dict] = []
        self.notes: list[dict] = []
        self.calls: list[tuple[str, str]] = []

    def add_org(self, name: str, domain: str | None, domains: list[str] | None = None) -> int:
        oid = 1000 + len(self.orgs)
        self.orgs[oid] = {"id": oid, "name": name, "domain": domain,
                          "domains": domains if domains is not None else ([domain] if domain else [])}
        return oid

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))
        assert request.headers["Authorization"].startswith("Basic ")
        body = json.loads(request.content) if request.content else {}
        if method == "GET" and path == "/organizations":
            term = parse_qs(request.url.query.decode())["term"][0].lower()
            stem = term.split(".")[0]
            hits = [o for o in self.orgs.values()
                    if stem in o["name"].lower() or any(stem in d for d in o["domains"])]
            return httpx.Response(200, json={"organizations": hits, "next_page_token": None})
        if method == "POST" and path == "/organizations":
            return httpx.Response(200, json=self.orgs[self.add_org(body["name"], body["domain"])])
        if method == "GET" and path.startswith("/organizations/"):
            oid = int(path.rsplit("/", 1)[1])
            entries = [e for e in self.entries if e["entity_id"] == oid]
            return httpx.Response(200, json={**self.orgs[oid], "list_entries": entries})
        if method == "POST" and path.startswith("/lists/"):
            entry = {"id": 5000 + len(self.entries), "list_id": int(path.split("/")[2]),
                     "entity_id": body["entity_id"]}
            self.entries.append(entry)
            return httpx.Response(200, json=entry)
        if method == "POST" and path == "/notes":
            self.notes.append(body)
            return httpx.Response(200, json={"id": 9000 + len(self.notes)})
        return httpx.Response(400, json={"message": f"unexpected {method} {path}"})

    def client(self, list_id: str | None = "77") -> crm.AffinityClient:
        return crm.AffinityClient("aff-key", list_id,
                                  client=httpx.Client(transport=httpx.MockTransport(self.handler)))


# ------------------------------------------------------------------- leads


def _lead(handle="acme_ai", company_url="https://www.acme.ai/about", name="Acme AI",
          account_type="startup", website=None, **verdict) -> Lead:
    return Lead(
        account=Account(id=handle, handle=handle, name=name, website=website,
                        source="search", sources=["search", "hn"]),
        signals=[Signal(name="smart_money_follow", value=1.0, weight=15.0,
                        detail="followed by @sequoia"),
                 Signal(name="bio_intent", value=0.0, weight=10.0)],
        llm=LLMVerdict(handle=handle, account_type=account_type, is_founder=True,
                       stage="launched", sector="ai infra", company_name=name if
                       company_url else None, company_url=company_url,
                       one_line_summary="Evals for agents.", thesis_fit=0.8,
                       confidence=0.9, **verdict),
        score=71.0,
    )


def _store(tmp_path, *leads: Lead) -> Store:
    store = Store(tmp_path / "scout.db", actor="ada@firm.com")
    if leads:
        store.save_leads("run-1", list(leads))
    return store


# --------------------------------------------------------------- pure rules


def test_domain_comes_from_the_company_never_a_founders_own_site():
    assert crm.company_domain_for(_lead()) == "acme.ai"
    founder = _lead(company_url=None, account_type="founder",
                    website="https://adalin.dev")
    assert crm.company_domain_for(founder) is None
    company_acct = _lead(company_url=None, website="https://acme.ai")
    assert crm.company_domain_for(company_acct) == "acme.ai"
    article = _lead(company_url="https://techcrunch.com/2026/acme-raises")
    assert crm.company_domain_for(article) is None
    social = _lead(company_url="https://x.com/acme_ai", website=None)
    assert crm.company_domain_for(social) is None
    keyed = _lead(company_url=None, account_type="founder")
    keyed.account.profile_url = "https://acme.ai"
    assert crm.company_domain_for(keyed) == "acme.ai"


def test_a_synthesized_name_never_becomes_a_crm_record_name():
    stealth = _lead(company_url=None, name="Ada Lin", account_type="founder")
    assert crm.company_name_for(stealth, "acme.ai") == "acme.ai"
    assert crm.company_name_for(_lead(), "acme.ai") == "Acme AI"


def test_push_statuses_run_from_the_threshold_and_never_include_allocated():
    assert crm.push_statuses(None) == {"shortlisted", "contacted", "meeting", "diligence"}
    assert "longlisted" in crm.push_statuses("longlisted")
    assert crm.push_statuses("diligence") == {"diligence"}
    assert crm.push_statuses("won") == crm.push_statuses("shortlisted")  # not a threshold
    assert "won" not in crm.push_statuses("longlisted")


def test_summary_note_states_only_what_scout_holds():
    lead = _lead(funding_stage="seed", funding_amount="$4M",
                 funding_investors=["Sequoia"], funding_evidence="techcrunch.com/acme",
                 founders=["Ada Lin — ex-DeepMind"])
    note = crm.summary_note(lead, status="shortlisted", thesis_name="Agents",
                            link="https://scout.firm/?s=acme_ai&p=Startups")
    assert "Scout score 71/100, thesis fit 80% under \"Agents\"" in note
    assert "Status in Scout: Shortlisted" in note
    assert "Funding: Seed $4M — Sequoia (source: techcrunch.com/acme)" in note
    assert "Founders: Ada Lin — ex-DeepMind" in note
    assert "smart money follow: followed by @sequoia" in note
    assert "bio intent" not in note  # a signal that didn't fire isn't a reason
    assert "Found via hn, search — @acme_ai" in note
    assert "Open in Scout: https://scout.firm/" in note
    # A round without evidence never reaches a verdict (models.py validator),
    # so it can never reach the note either.
    bare = _lead(funding_stage="series_a")
    assert "Funding" not in crm.summary_note(bare)


# ------------------------------------------------------------------- Attio


def test_attio_first_push_creates_lists_and_notes_then_resumes_silently(tmp_path):
    fake, lead = FakeAttio(), _lead()
    store = _store(tmp_path, lead)
    result = crm.push(store, fake.client(), lead, status="shortlisted",
                      memo="# Memo\nPURSUE", thesis_name="Agents")
    assert not result.error and result.created and result.listed
    assert result.notes == [crm.SUMMARY_TITLE, crm.MEMO_TITLE]
    (rid, rec), = fake.records.items()
    assert rec == {"domains": ["acme.ai"], "name": "Acme AI",
                   "description": "Evals for agents."}
    assert fake.entries == {("dealflow", rid): "ent-1"}
    assert [n["title"] for n in fake.notes] == ["Sourced by Scout — Acme AI",
                                                "Scout memo — Acme AI"]
    assert result.remote_url.endswith(f"/company/{rid}")

    fake.calls.clear()
    again = crm.push(store, fake.client(), lead, status="shortlisted", memo="# Memo\nPURSUE")
    assert not again.changed and fake.calls == []  # everything already done

    edited = crm.push(store, fake.client(), lead, status="meeting", memo="# Memo v2\nPURSUE")
    assert edited.notes == [crm.MEMO_TITLE] and len(fake.notes) == 3
    assert fake.calls == [("POST", "/v2/notes")]


def test_attio_existing_record_is_linked_never_overwritten(tmp_path):
    fake, lead = FakeAttio(), _lead()
    rid = fake.add_record("acme.ai", name="ACME (curated)", description="")
    result = crm.push(crm_store := _store(tmp_path, lead), fake.client(), lead)
    assert not result.created
    assert fake.records[rid]["name"] == "ACME (curated)"   # the firm's name stands
    assert fake.records[rid]["description"] == "Evals for agents."  # empty → filled
    assert crm_store.crm_link("attio", "acme.ai")["remote_id"] == rid


def test_attio_without_a_list_writes_record_and_notes_only(tmp_path):
    fake, lead = FakeAttio(), _lead()
    result = crm.push(_store(tmp_path, lead), fake.client(list_ref=None), lead)
    assert not result.listed and fake.entries == {} and len(fake.notes) == 1


def test_a_failed_note_resumes_without_recreating_or_duplicating(tmp_path):
    fake, lead = FakeAttio(), _lead()
    store = _store(tmp_path, lead)
    fake.fail[("POST", "/v2/notes")] = [502]  # maybe processed → must not retry
    first = crm.push(store, fake.client(), lead)
    assert "HTTP 502" in first.error and fake.notes == []
    assert fake.calls.count(("POST", "/v2/notes")) == 1
    link = store.crm_link("attio", "acme.ai")
    assert link["remote_id"] and link["list_entry_id"] and "502" in link["last_error"]

    fake.calls.clear()
    second = crm.push(store, fake.client(), lead)
    assert not second.error and second.notes == [crm.SUMMARY_TITLE]
    assert fake.calls == [("POST", "/v2/notes")]  # no re-assert, no re-list
    assert store.crm_link("attio", "acme.ai")["last_error"] == ""


def test_retry_only_what_the_server_cannot_have_processed(tmp_path):
    fake, lead = FakeAttio(), _lead()
    fake.fail[("POST", "/v2/notes")] = [429]          # rejected, not processed
    fake.fail[("PUT", "/v2/objects/companies")] = [503]  # idempotent: safe again
    result = crm.push(_store(tmp_path, lead), fake.client(), lead)
    assert not result.error and len(fake.notes) == 1
    assert fake.calls.count(("PUT", "/v2/objects/companies/records")) == 2
    assert fake.calls.count(("POST", "/v2/notes")) == 2


def test_a_record_deleted_in_the_crm_is_relinked_once(tmp_path):
    fake, lead = FakeAttio(), _lead()
    store = _store(tmp_path, lead)
    crm.push(store, fake.client(), lead)
    fake.records.clear()  # someone deleted it in Attio
    result = crm.push(store, fake.client(), lead, memo="new memo")
    assert not result.error and result.created
    assert store.crm_link("attio", "acme.ai")["remote_id"] in fake.records


def test_no_domain_is_skipped_with_a_reason(tmp_path):
    fake = FakeAttio()
    lead = _lead(company_url=None, account_type="founder", name="Ada Lin")
    result = crm.push(_store(tmp_path, lead), fake.client(), lead)
    assert "no company domain" in result.skipped and fake.calls == []


# ---------------------------------------------------------------- Affinity


def test_affinity_matches_by_exact_domain_not_by_name(tmp_path):
    fake, lead = FakeAffinity(), _lead()
    decoy = fake.add_org("Acme AI Labs", "acmeailabs.com")  # term search returns it
    result = crm.push(_store(tmp_path, lead), fake.client(), lead)
    assert result.created and result.remote_id != str(decoy)
    assert fake.orgs[int(result.remote_id)]["domain"] == "acme.ai"


def test_affinity_existing_org_and_list_entry_are_reused(tmp_path):
    fake, lead = FakeAffinity(), _lead()
    oid = fake.add_org("Acme", None, domains=["acme.ai", "acme.com"])
    fake.entries.append({"id": 4999, "list_id": 77, "entity_id": oid})
    result = crm.push(_store(tmp_path, lead), fake.client(), lead, memo="memo text")
    assert not result.created and result.remote_id == str(oid)
    assert ("POST", "/lists/77/list-entries") not in fake.calls
    assert len(fake.orgs) == 1
    assert [n["organization_ids"] for n in fake.notes] == [[oid], [oid]]
    assert fake.notes[0]["content"].startswith("Sourced by Scout — Acme AI\n\n")


# -------------------------------------------------------------------- sync


def test_sync_pushes_the_threshold_set_and_an_unchanged_pipeline_makes_no_calls(tmp_path):
    fake = FakeAttio()
    leads = [_lead("acme_ai"), _lead("beta", "https://beta.io", "Beta"),
             _lead("portfolio", "https://port.co", "PortCo"),
             _lead("early", "https://early.dev", "Early"),
             _lead("ada", None, "Ada Lin", account_type="founder")]
    store = _store(tmp_path, *leads)
    for handle, status in (("acme_ai", "shortlisted"), ("beta", "meeting"),
                           ("portfolio", "won"), ("early", "longlisted"),
                           ("ada", "shortlisted")):
        store.set_pipeline(handle, status=status)
    clients = [fake.client()]
    results = crm.sync(store, Settings(), clients=clients)
    assert {r.handle for r in results if r.changed} == {"acme_ai", "beta"}
    assert [r.handle for r in results if r.skipped] == []  # no domain → not due at all
    assert {rec["name"] for rec in fake.records.values()} == {"Acme AI", "Beta"}
    assert crm.pending(store, Settings(), clients) == []

    fake.calls.clear()
    assert crm.sync(store, Settings(), clients=clients) == []
    assert fake.calls == []

    store.set_setting("crm_push_threshold", "longlisted")
    assert crm.pending(store, Settings(), clients) == ["early"]
    store.set_pipeline("beta", brief="# Beta memo")
    assert crm.pending(store, Settings(), clients) == ["beta", "early"]


def test_explicit_push_sends_any_status_and_reports_unknown_handles(tmp_path):
    fake = FakeAttio()
    store = _store(tmp_path, _lead("portfolio", "https://port.co", "PortCo"))
    store.set_pipeline("portfolio", status="won")
    results = crm.sync(store, Settings(), handles=["@portfolio", "ghost"],
                       clients=[fake.client()])
    assert [r.handle for r in results] == ["portfolio"] and results[0].changed


def test_a_push_that_changed_something_is_one_event_with_its_actor(tmp_path):
    fake, lead = FakeAttio(), _lead()
    store = _store(tmp_path, lead)
    crm.push(store, fake.client(), lead)
    crm.push(store, fake.client(), lead)  # nothing new → no second event
    events = [e for e in store.events(limit=50) if e.verb == "crm_pushed"]
    assert len(events) == 1
    assert events[0].actor == "ada@firm.com" and events[0].payload["provider"] == "attio"
    assert events[0].payload["notes"] == [crm.SUMMARY_TITLE]


def test_rename_handle_carries_the_crm_link(tmp_path):
    fake, lead = FakeAttio(), _lead("acme-ai")
    store = _store(tmp_path, lead)
    crm.push(store, fake.client(), lead)
    store.rename_handle("acme-ai", "acme_ai")
    assert store.crm_link("attio", "acme.ai")["handle"] == "acme_ai"


# ------------------------------------------------------------ worker & doctor


def test_worker_job_skips_when_unconfigured_and_raises_only_when_all_fail(tmp_path, monkeypatch):
    from scout import worker

    store = _store(tmp_path, _lead())
    store.set_pipeline("acme_ai", status="shortlisted")
    out = worker.handle_crm(store, Settings(attio_api_key=None, affinity_api_key=None),
                            {"payload": {}})
    assert "no CRM configured" in out["skipped"]

    fake = FakeAttio()
    fake.fail[("PUT", "/v2/objects/companies")] = [401]
    monkeypatch.setattr(crm, "clients_from", lambda settings: [fake.client()])
    with pytest.raises(RuntimeError, match="401"):
        worker.handle_crm(store, Settings(attio_api_key="k"), {"payload": {}})
    ok = worker.handle_crm(store, Settings(attio_api_key="k"), {"payload": {}})
    assert ok["summary"].startswith("1 updated")


def test_settings_read_crm_keys_and_a_blank_list_id_is_unset(monkeypatch):
    monkeypatch.setenv("ATTIO_API_KEY", "k")
    monkeypatch.setenv("AFFINITY_API_KEY", "")
    monkeypatch.setenv("AFFINITY_LIST_ID", "")
    settings = Settings(_env_file=None)
    assert crm.configured(settings) == ["Attio"]
    assert [c.provider for c in crm.clients_from(settings)] == ["attio"]


def _settings(**kw) -> Settings:
    base = {"attio_api_key": None, "attio_list": None,
            "affinity_api_key": None, "affinity_list_id": None}
    return Settings(_env_file=None, **{**base, **kw})


def test_doctor_judges_the_crm_list_as_write_back_will_use_it():
    attio = _settings(attio_api_key="k", attio_list="dealflow")
    people = json.dumps({"data": {"name": "Hiring", "parent_object": ["people"]}})
    companies = json.dumps({"data": {"name": "Dealflow", "parent_object": ["companies"]}})
    assert doctor._judge("Attio", 200, companies, attio).status == "ok"
    bad = doctor._judge("Attio", 200, people, attio)
    assert bad.status == "warn" and "not companies" in bad.detail
    assert doctor._judge("Attio", 404, "", attio).status == "warn"
    assert doctor._judge("Attio", 401, "", attio).status == "warn"  # never blocks the scan

    aff = _settings(affinity_api_key="k", affinity_list_id="77")
    lists = json.dumps([{"id": 77, "type": 0, "name": "People"},
                        {"id": 78, "type": 1, "name": "Deals"}])
    assert "not an organization list" in doctor._judge("Affinity", 200, lists, aff).detail
    ok = _settings(affinity_api_key="k", affinity_list_id="78")
    assert doctor._judge("Affinity", 200, lists, ok).status == "ok"
    assert "not among" in doctor._judge(
        "Affinity", 200, lists, _settings(affinity_api_key="k", affinity_list_id="9")).detail


def test_doctor_probes_crm_hosts_but_never_counts_them_as_discovery(tmp_path):
    from scout.config import Seeds, Thesis

    settings = _settings(attio_api_key="k", attio_list="dealflow", tw_cookies=None)
    thesis = Thesis(thesis="t", target_stages=["launched"])
    names = [t[0] for t in doctor._targets(settings, thesis, Seeds())]
    assert "Attio" in names
    # Every discovery host down, the CRM up: still "a run would read nothing".
    checks = doctor.network_checks(
        settings, thesis, Seeds(),
        probe=lambda url, h: ((200, json.dumps({"data": {"parent_object": ["companies"]}}))
                              if "attio" in url else (None, "down")))
    assert any(c.name == "Discovery" and c.status == "fail" for c in checks)
