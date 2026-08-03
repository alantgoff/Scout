"""The background layer: schedule arithmetic (pure), the queue's claim/retry/
lease semantics, and the digest builders.

No network anywhere — the digest tests assert on the Block Kit dicts, and
the worker tests substitute a fake handler for the subprocess ones.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scout import jobs as jobs_mod
from scout import notify, worker
from scout.jobs import (
    KIND_DIGEST,
    KIND_RUN,
    MAX_ATTEMPTS,
    ScheduleSpec,
    backoff_seconds,
    job_label,
    next_occurrence,
)
from scout.store import Store

UTC = timezone.utc


def make_store(tmp_path: Path, actor: str = "alan@firm.com") -> Store:
    return Store(tmp_path / "jobs.db", actor=actor)


# --- schedule arithmetic --------------------------------------------------------


def test_interval_schedule_is_just_an_offset() -> None:
    spec = ScheduleSpec(every_minutes=30)
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    assert next_occurrence(spec, now) == datetime(2026, 8, 3, 12, 30, tzinfo=UTC)
    assert spec.describe() == "every 30m"
    assert ScheduleSpec(every_minutes=60).describe() == "hourly"
    assert ScheduleSpec(every_minutes=240).describe() == "every 4h"


def test_daily_schedule_fires_at_local_wall_clock_time() -> None:
    """07:00 in London means 07:00 in London — the whole reason the spec
    carries a timezone rather than a UTC offset."""
    spec = ScheduleSpec(daily_at="07:00", tz="Europe/London")
    # August: BST (UTC+1), so 07:00 local is 06:00 UTC.
    summer = next_occurrence(spec, datetime(2026, 8, 3, 12, 0, tzinfo=UTC))
    assert summer == datetime(2026, 8, 4, 6, 0, tzinfo=UTC)
    # January: GMT (UTC+0), so 07:00 local is 07:00 UTC — same wall clock,
    # different UTC instant. A naive implementation drifts here.
    winter = next_occurrence(spec, datetime(2026, 1, 12, 12, 0, tzinfo=UTC))
    assert winter == datetime(2026, 1, 13, 7, 0, tzinfo=UTC)


def test_daily_schedule_survives_the_dst_transition() -> None:
    """Across the spring-forward weekend the local time must stay 07:00,
    even though the UTC instant shifts by an hour."""
    spec = ScheduleSpec(daily_at="07:00", tz="America/New_York")
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    # US DST begins 2026-03-08. Before and after, local time is unchanged.
    before = next_occurrence(spec, datetime(2026, 3, 6, 12, 0, tzinfo=UTC))
    after = next_occurrence(spec, datetime(2026, 3, 9, 12, 0, tzinfo=UTC))
    assert before.astimezone(zone).hour == 7
    assert after.astimezone(zone).hour == 7
    assert before.hour == 12 and after.hour == 11  # UTC moved, local did not


def test_weekday_mask_skips_the_weekend() -> None:
    spec = ScheduleSpec(daily_at="09:00", weekdays=[0, 1, 2, 3, 4], tz="UTC")
    # 2026-08-07 is a Friday; the next weekday firing is Monday the 10th.
    friday_evening = datetime(2026, 8, 7, 18, 0, tzinfo=UTC)
    assert next_occurrence(spec, friday_evening).date() == datetime(
        2026, 8, 10, tzinfo=UTC).date()
    assert spec.describe() == "weekdays at 09:00 UTC"


def test_next_occurrence_is_always_strictly_in_the_future() -> None:
    """Called at exactly the firing time, a schedule must advance a day —
    otherwise materializing it would loop forever on the same instant."""
    spec = ScheduleSpec(daily_at="09:00", tz="UTC")
    exactly_now = datetime(2026, 8, 3, 9, 0, tzinfo=UTC)
    assert next_occurrence(spec, exactly_now) == datetime(2026, 8, 4, 9, 0, tzinfo=UTC)


def test_invalid_specs_are_rejected_with_reasons() -> None:
    for spec in (
        ScheduleSpec(),                                    # neither field
        ScheduleSpec(every_minutes=30, daily_at="07:00"),  # both
        ScheduleSpec(every_minutes=1),                     # too frequent
        ScheduleSpec(daily_at="25:00"),                    # not a time
        ScheduleSpec(daily_at="07:00", weekdays=[9]),      # not a weekday
        ScheduleSpec(daily_at="07:00", tz="Mars/Olympus"), # not a timezone
    ):
        with pytest.raises(ValueError):
            spec.validate_spec()


def test_backoff_grows_then_caps() -> None:
    assert [backoff_seconds(n) for n in (1, 2, 3, 4)] == [60, 120, 240, 480]
    assert backoff_seconds(99) == 3600  # capped, never unbounded
    assert backoff_seconds(0) == 0


def test_job_label_names_its_subject() -> None:
    assert job_label(KIND_RUN, {"source": "twscrape"}) == "Sourcing run (twscrape)"
    assert job_label("generate_memo", {"handle": "acme"}) == "Memo — @acme"
    assert job_label(KIND_DIGEST, {}) == "Digest"


# --- the queue ------------------------------------------------------------------


def test_enqueue_and_claim_moves_the_job_to_running(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    job_id = store.enqueue_job(KIND_DIGEST, {"window": "daily"})
    assert store.job_queue_depth() == 1
    job = store.claim_job("worker-1")
    assert job is not None
    assert job["id"] == job_id
    assert job["status"] == "running"
    assert job["worker_id"] == "worker-1"
    assert job["attempts"] == 1
    assert job["lease_expires_at"]
    # The queue is now empty — a second worker gets nothing.
    assert store.claim_job("worker-2") is None
    store.finish_job(job_id, {"sent": True})
    done = store.get_job(job_id)
    assert done["status"] == "done" and done["result"] == {"sent": True}
    assert store.job_queue_depth() == 0


def test_two_workers_cannot_claim_the_same_job(tmp_path: Path) -> None:
    """The property the whole queue rests on."""
    store = make_store(tmp_path)
    store.enqueue_job(KIND_DIGEST)
    first = store.claim_job("worker-1")
    second = store.claim_job("worker-2")
    assert first is not None and second is None


def test_claim_respects_priority_then_age(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.enqueue_job(KIND_DIGEST, {"window": "daily"})
    store.enqueue_job(KIND_DIGEST, {"window": "weekly"})
    urgent = store.enqueue_job(KIND_RUN, {"source": "twscrape"}, priority=10)
    assert store.claim_job("w")["id"] == urgent          # priority first
    assert store.claim_job("w")["payload"]["window"] == "daily"  # then oldest


def test_run_after_defers_a_job(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    later = datetime.now(UTC) + timedelta(hours=1)
    store.enqueue_job(KIND_DIGEST, run_after=later)
    assert store.claim_job("w") is None  # not yet runnable
    assert store.job_queue_depth() == 1  # but still queued


def test_dedupe_stops_an_impatient_double_click(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.enqueue_job(KIND_RUN, {"source": "twscrape"}, dedupe=True)
    second = store.enqueue_job(KIND_RUN, {"source": "twscrape"}, dedupe=True)
    assert first is not None and second is None
    # A different payload is a different job.
    assert store.enqueue_job(KIND_RUN, {"source": "xapi"}, dedupe=True) is not None
    # Dedupe also covers a job that is already running, not just queued.
    store.claim_job("w")
    assert store.enqueue_job(KIND_RUN, {"source": "twscrape"}, dedupe=True) is None


def test_failure_retries_with_backoff_then_gives_up(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    job_id = store.enqueue_job(KIND_DIGEST, max_attempts=2)
    store.claim_job("w")
    assert store.fail_job(job_id, "slack timed out") is True
    row = store.get_job(job_id)
    assert row["status"] == "queued"       # back in the queue
    assert row["error"] == "slack timed out"
    assert datetime.fromisoformat(row["run_after"]) > datetime.now(UTC)  # deferred
    # Second failure exhausts the budget.
    store.db["jobs"].update(job_id, {"run_after": datetime.now(UTC).isoformat()})
    store.claim_job("w")
    assert store.fail_job(job_id, "slack timed out again") is False
    assert store.get_job(job_id)["status"] == "failed"


def test_a_dead_worker_gets_its_job_back(tmp_path: Path) -> None:
    """Without reaping, a container restart mid-run would strand the job in
    'running' forever."""
    store = make_store(tmp_path)
    job_id = store.enqueue_job(KIND_RUN)
    store.claim_job("doomed-worker")
    # Simulate the lease lapsing (the worker died without heartbeating).
    store.db["jobs"].update(job_id, {
        "lease_expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    })
    assert store.reap_stale_jobs() == 1
    row = store.get_job(job_id)
    assert row["status"] == "queued" and row["worker_id"] == ""
    # A live worker's job is untouched — heartbeating extends the lease.
    store.claim_job("healthy-worker")
    store.heartbeat_job(job_id)
    assert store.reap_stale_jobs() == 0
    assert store.get_job(job_id)["status"] == "running"


def test_reaping_respects_the_attempt_budget(tmp_path: Path) -> None:
    """A job that kills its worker every time must not requeue forever."""
    store = make_store(tmp_path)
    job_id = store.enqueue_job(KIND_RUN, max_attempts=1)
    store.claim_job("doomed")
    store.db["jobs"].update(job_id, {
        "lease_expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    })
    store.reap_stale_jobs()
    assert store.get_job(job_id)["status"] == "failed"


def test_only_queued_jobs_can_be_cancelled(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    job_id = store.enqueue_job(KIND_RUN)
    assert store.cancel_job(job_id) is True
    assert store.get_job(job_id)["status"] == "cancelled"
    other = store.enqueue_job(KIND_DIGEST)
    store.claim_job("w")
    assert store.cancel_job(other) is False  # its worker owns it


def test_unknown_job_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_store(tmp_path).enqueue_job("mine_bitcoin")


# --- schedules ------------------------------------------------------------------


def test_due_schedule_materializes_one_job_and_advances(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    spec = ScheduleSpec(every_minutes=60)
    schedule_id = store.upsert_schedule("Hourly run", KIND_RUN, spec,
                                        {"source": "twscrape"})
    row = next(s for s in store.schedules() if s["id"] == schedule_id)
    assert row["next_run_at"] and row["enabled"] is True
    # Not due yet.
    assert store.materialize_due_schedules() == []
    # Wind the clock forward past the firing time.
    due = datetime.fromisoformat(row["next_run_at"]) + timedelta(minutes=1)
    created = store.materialize_due_schedules(now=due)
    assert len(created) == 1
    job = store.get_job(created[0])
    assert job["kind"] == KIND_RUN
    assert job["payload"] == {"source": "twscrape"}
    assert job["requested_by"] == f"schedule:{schedule_id}"
    # And the schedule has moved on rather than firing again immediately.
    after = next(s for s in store.schedules() if s["id"] == schedule_id)
    assert datetime.fromisoformat(after["next_run_at"]) > due
    assert store.materialize_due_schedules(now=due) == []


def test_a_long_sleep_produces_one_job_not_forty(tmp_path: Path) -> None:
    """A machine asleep over a weekend should wake to one run, not a
    backlog of every missed occurrence."""
    store = make_store(tmp_path)
    store.upsert_schedule("Hourly", KIND_RUN, ScheduleSpec(every_minutes=60), {})
    much_later = datetime.now(UTC) + timedelta(days=3)
    assert len(store.materialize_due_schedules(now=much_later)) == 1


def test_disabled_schedule_never_fires(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    schedule_id = store.upsert_schedule(
        "Hourly", KIND_RUN, ScheduleSpec(every_minutes=60), {})
    store.set_schedule_enabled(schedule_id, False)
    row = next(s for s in store.schedules() if s["id"] == schedule_id)
    assert row["enabled"] is False and row["next_run_at"] is None
    assert store.materialize_due_schedules(
        now=datetime.now(UTC) + timedelta(days=2)) == []
    # Re-enabling recomputes the next firing.
    store.set_schedule_enabled(schedule_id, True)
    assert next(s for s in store.schedules()
                if s["id"] == schedule_id)["next_run_at"]


def test_schedule_does_not_stack_behind_a_slow_run(tmp_path: Path) -> None:
    """If the previous run is still going when the next is due, the schedule
    must not queue a second one."""
    store = make_store(tmp_path)
    store.upsert_schedule("Hourly", KIND_RUN, ScheduleSpec(every_minutes=60),
                          {"source": "twscrape"})
    first = store.materialize_due_schedules(now=datetime.now(UTC) + timedelta(hours=2))
    assert len(first) == 1
    store.claim_job("w")  # still running
    second = store.materialize_due_schedules(now=datetime.now(UTC) + timedelta(hours=4))
    assert second == []


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = worker.bootstrap_schedules(store)
    assert len(created) == 2  # a sourcing run and a digest
    assert worker.bootstrap_schedules(store) == []
    kinds = {s["kind"] for s in store.schedules()}
    assert kinds == {KIND_RUN, KIND_DIGEST}


# --- the worker loop ------------------------------------------------------------


def test_worker_runs_a_job_and_records_the_result(tmp_path: Path, monkeypatch) -> None:
    store = make_store(tmp_path)
    from scout.config import Settings

    seen: list[dict] = []

    def fake_handler(store_arg, settings_arg, job):
        seen.append(job)
        return {"ok": True, "kind": job["kind"]}

    monkeypatch.setitem(worker.HANDLERS, KIND_DIGEST, fake_handler)
    job_id = store.enqueue_job(KIND_DIGEST, {"window": "daily"})
    executed = worker.run_worker(store, Settings(), once=True)
    assert executed == 1
    assert seen and seen[0]["id"] == job_id
    assert store.get_job(job_id)["result"] == {"ok": True, "kind": KIND_DIGEST}


def test_worker_survives_a_handler_that_explodes(tmp_path: Path, monkeypatch) -> None:
    """The loop's contract: a bad handler costs one job, never the worker."""
    store = make_store(tmp_path)
    from scout.config import Settings

    def boom(store_arg, settings_arg, job):
        raise RuntimeError("the scraper fell over")

    monkeypatch.setitem(worker.HANDLERS, KIND_DIGEST, boom)
    job_id = store.enqueue_job(KIND_DIGEST, max_attempts=1)
    worker.run_worker(store, Settings(), once=True)
    row = store.get_job(job_id)
    assert row["status"] == "failed"
    assert "the scraper fell over" in row["error"]


def test_worker_drains_several_jobs_in_one_pass(tmp_path: Path, monkeypatch) -> None:
    store = make_store(tmp_path)
    from scout.config import Settings

    monkeypatch.setitem(worker.HANDLERS, KIND_DIGEST,
                        lambda s, cfg, job: {"ok": True})
    for window in ("daily", "weekly", "monthly"):
        store.enqueue_job(KIND_DIGEST, {"window": window})
    assert worker.run_worker(store, Settings(), once=True) == 3
    assert store.job_queue_depth() == 0


def test_worker_fails_a_job_with_no_handler(tmp_path: Path, monkeypatch) -> None:
    store = make_store(tmp_path)
    from scout.config import Settings

    monkeypatch.delitem(worker.HANDLERS, KIND_DIGEST)
    job_id = store.enqueue_job(KIND_DIGEST)
    worker.run_worker(store, Settings(), once=True)
    assert store.get_job(job_id)["status"] in ("queued", "failed")
    assert "no handler" in store.get_job(job_id)["error"]


# --- digests --------------------------------------------------------------------


def _seed_digest_store(tmp_path: Path) -> Store:
    """A store with a run, a fresh high scorer, and a partner disagreement."""
    from scout.models import Account, Lead, LLMVerdict, Signal

    store = make_store(tmp_path)
    store.ensure_user("alan@firm.com", name="Alan Goff")
    store.ensure_user("sara@firm.com", name="Sara Lin")
    store.set_setting("app_base_url", "https://scout.test")
    leads = []
    for i, (handle, score, fit) in enumerate([
        ("hotco", 88.0, 0.9), ("midco", 71.0, 0.6), ("lowco", 22.0, 0.2),
    ]):
        leads.append(Lead(
            account=Account(id=str(i), handle=handle, name=f"{handle} founder",
                            bio="building"),
            signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
            llm=LLMVerdict(handle=handle, is_founder=True, stage="launched",
                           sector="ai infra", company_name=handle.title(),
                           thesis_fit=fit, confidence=0.9,
                           why_interesting=f"{handle} is interesting"),
            score=score,
        ))
    store.record_run("run-1", source="twscrape", strategy_hash="abc",
                     thesis_statement="ai infra")
    store.save_leads("run-1", leads)
    return store


def test_digest_reports_only_what_changes_someone_s_next_hour(tmp_path: Path) -> None:
    store = _seed_digest_store(tmp_path)
    since = datetime.now(UTC) - timedelta(hours=24)
    data = notify.digest_data(store, since)
    handles = [item["handle"] for item in data["top_new"]]
    # Above the threshold and untriaged: newsworthy. Below it: not.
    assert handles == ["hotco", "midco"]
    assert data["top_new"][0]["score"] == 88.0
    assert data["top_new"][0]["link"] == "https://scout.test/?s=hotco&p=Startups"
    assert data["has_content"] is True

    # Once someone triages a startup it stops being news to the firm.
    store.set_pipeline("hotco", status="shortlisted")
    assert [i["handle"] for i in notify.digest_data(store, since)["top_new"]] == ["midco"]


def test_digest_carries_the_disagreement_and_the_review_queue(tmp_path: Path) -> None:
    store = _seed_digest_store(tmp_path)
    store.set_vote("hotco", "strong_yes", actor="alan@firm.com")
    store.set_vote("hotco", "pass", "crowded", actor="sara@firm.com")
    store.set_memo("midco", "# memo", kind="generated", actor="agent:memo")
    store.set_memo_review("midco", "requested", actor="alan@firm.com")

    data = notify.digest_data(store, datetime.now(UTC) - timedelta(hours=24))
    assert [c["handle"] for c in data["contested"]] == ["hotco"]
    assert "Alan Goff: strong yes" in data["contested"][0]["detail"]
    assert "Sara Lin: pass" in data["contested"][0]["detail"]
    assert [m["handle"] for m in data["awaiting_review"]] == ["midco"]

    blocks = notify.digest_blocks(data)
    rendered = str(blocks)
    # Things needing a decision come before things merely worth reading.
    assert rendered.index("awaiting review") < rendered.index("New, untriaged")
    assert "Contested" in rendered
    assert blocks[0]["type"] == "header"
    text = notify.digest_fallback_text(data)
    assert "1 contested" in text and "memo(s) awaiting review" in text


def test_digest_ignores_machine_activity(tmp_path: Path) -> None:
    """A digest reports what PEOPLE did — agent writes are not news."""
    store = _seed_digest_store(tmp_path)
    store.set_memo("hotco", "# memo", kind="generated", actor="agent:memo")
    data = notify.digest_data(store, datetime.now(UTC) - timedelta(hours=24))
    assert data["n_events"] == 0
    assert data["actors"] == []
    store.set_vote("hotco", "yes", actor="sara@firm.com")
    data = notify.digest_data(store, datetime.now(UTC) - timedelta(hours=24))
    assert data["n_events"] == 1
    assert data["actors"] == [{"name": "Sara Lin", "n": 1}]


def test_quiet_day_produces_no_digest(tmp_path: Path) -> None:
    """Nobody needs a daily message that says nothing happened."""
    store = make_store(tmp_path)
    data = notify.digest_data(store, datetime.now(UTC) - timedelta(hours=24))
    assert data["has_content"] is False
    assert notify.send_digest(store, datetime.now(UTC) - timedelta(hours=24)) is False


def test_digest_blocks_render_without_a_base_url(tmp_path: Path) -> None:
    """Slack links degrade to bold names rather than breaking the message."""
    store = _seed_digest_store(tmp_path)
    store.set_setting("app_base_url", "")
    data = notify.digest_data(store, datetime.now(UTC) - timedelta(hours=24))
    blocks = notify.digest_blocks(data)
    assert "*Hotco*" in str(blocks)  # no <url|name>, just the name
