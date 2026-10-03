"""The background worker: claims jobs, runs them, records what happened.

Shape of the thing:

    reap stale leases → materialize due schedules → claim one job → run it

One job at a time, on purpose. A firm of two runs a handful of jobs a day,
and serial execution means the X API budget guard, the SQLite writer, and
the scraper's rate limits each have exactly one contender. Concurrency here
would buy nothing and cost the invariants.

Long jobs (sourcing runs, memos) execute as SUBPROCESSES rather than inside
this loop. A scraper segfault or a wedged HTTP client then kills a child
that the worker reaps, instead of taking down the scheduler with it — and
the child's console output lands in a log file the UI can tail, which is the
same mechanism the UI's own manual runs already use.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console

from scout import jobs as jobs_mod
from scout import notify
from scout.config import Settings, load_thesis
from scout.store import Store

console = Console()

# How long to sleep when the queue is empty. Short enough that a UI button
# feels responsive, long enough to be invisible in CPU terms.
POLL_SECONDS = 5
# A sourcing run that has produced no output for this long is presumed hung.
CHILD_TIMEOUT_S = 3 * 3600


class _Heartbeat:
    """Keeps a running job's lease alive while a subprocess works.

    Without this the lease would lapse mid-run and the reaper would requeue
    a job that is in fact progressing fine — the classic double-execution
    bug in lease-based queues.
    """

    def __init__(self, store: Store, job_id: int) -> None:
        self._store, self._job_id = store, job_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Heartbeat":
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _beat(self) -> None:
        while not self._stop.wait(jobs_mod.HEARTBEAT_SECONDS):
            try:
                self._store.heartbeat_job(self._job_id)
            except Exception as exc:  # noqa: BLE001 — never kill the worker
                console.print(f"[yellow]heartbeat failed:[/] {exc}")


def _log_path(settings: Settings, kind: str) -> Path:
    log_dir = Path(settings.db_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{kind}-{stamp}.log"


class JobStopped(RuntimeError):
    """A member stopped the run this job was executing. Not a failure to
    retry: the job ends, and the queue does not bring it back."""


# How often a running child is checked for a stop request.
STOP_POLL_S = 5


def _run_cli(args: list[str], settings: Settings, kind: str,
             actor: str, store: Store | None = None) -> tuple[int, Path, str]:
    """Run `python -m scout.cli <args>` as a child, tee'd to a log file.

    Returns (exit_code, log_path, tail). The tail is the last few lines,
    which is what a failure message should carry — a job row storing a
    3MB scraper log helps nobody.
    """
    log_path = _log_path(settings, kind)
    env = {
        **os.environ,
        "TERM": "dumb", "NO_COLOR": "1", "COLUMNS": "120",
        "PYTHONUNBUFFERED": "1",
        "SCOUT_SCAN_LOG": str(log_path),
        # The run is attributed to whoever (or whatever) asked for it.
        "SCOUT_ACTOR": actor,
    }
    with open(log_path, "wb") as fh:
        fh.write(f"$ scout {' '.join(args)}\n".encode())
        fh.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "scout.cli", *args],
            cwd=Path(__file__).resolve().parent.parent,
            stdout=fh, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        stopped_by = None
        while True:
            try:
                code = proc.wait(timeout=STOP_POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if store is not None and (stopped_by := store.scan_stop_requested()):
                _kill_group(proc, signal.SIGTERM)
                code = -15
                break
            if time.monotonic() > deadline:
                # Kill the whole process group: scrapers spawn helpers, and a
                # bare terminate() would orphan them.
                _kill_group(proc, signal.SIGKILL)
                code = -9
                break
    if stopped_by:
        store.scan_finish("failed", f"stopped by {stopped_by}")
        raise JobStopped(f"stopped by {stopped_by}")
    return code, log_path, _tail(log_path)


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, escalating to SIGKILL if it
    doesn't exit."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass


def _tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


# --------------------------------------------------------------- handlers
# Each returns a result dict on success and raises on failure; the loop
# turns an exception into a retry-or-fail decision.


def handle_run(store: Store, settings: Settings, job: dict) -> dict:
    payload = job.get("payload") or {}
    args = ["run", "--source", payload.get("source", "twscrape")]
    if payload.get("max_accounts"):
        args += ["--max-accounts", str(payload["max_accounts"])]
    if payload.get("min_score"):
        args += ["--min-score", str(payload["min_score"])]
    if payload.get("ttl_days") is not None:
        args += ["--ttl-days", str(payload["ttl_days"])]
    code, log_path, tail = _run_cli(args, settings, "run",
                                    job.get("requested_by", "system:scout"), store)
    if code != 0:
        raise RuntimeError(
            f"sourcing run exited {code}\n{tail}" if code != -9
            else f"sourcing run timed out after {CHILD_TIMEOUT_S // 3600}h\n{tail}"
        )
    latest = store.latest_run() or {}
    return {"run_id": latest.get("id"), "leads": latest.get("n_leads"),
            "log_path": str(log_path)}


def handle_memo(store: Store, settings: Settings, job: dict) -> dict:
    payload = job.get("payload") or {}
    handle = (payload.get("handle") or "").lower()
    if not handle:
        raise ValueError("generate_memo needs a handle in its payload")
    args = ["memo", handle, "--depth", payload.get("depth", "standard")]
    if payload.get("focus"):
        args += ["--focus", payload["focus"]]
    code, log_path, tail = _run_cli(args, settings, f"memo-{handle}",
                                    job.get("requested_by", "system:scout"))
    if code != 0:
        raise RuntimeError(f"memo generation exited {code}\n{tail}")
    return {"handle": handle, "log_path": str(log_path)}


def handle_digest(store: Store, settings: Settings, job: dict) -> dict:
    """Digests run in-process: they are a few queries and one HTTP POST, so
    a subprocess would cost more than it protects."""
    payload = job.get("payload") or {}
    window = payload.get("window", "daily")
    hours = 168 if window == "weekly" else 24
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    data = notify.digest_data(store, since, window=window)
    if not data["has_content"] and not payload.get("always_send"):
        return {"sent": False, "reason": "nothing happened worth reporting"}
    sent = notify.post_slack(
        store, notify.digest_fallback_text(data), notify.digest_blocks(data)
    )
    return {"sent": sent, "window": window,
            "new_leads": len(data["top_new"]), "events": data["n_events"]}


def handle_refresh(store: Store, settings: Settings, job: dict) -> dict:
    """Re-research the tracked companies whose facts are oldest — the daily
    watch on longlisted+ companies for raises, acquisitions, shutdowns.
    Budget-gated inside the CLI against DAILY_SPEND_CAP_USD."""
    payload = job.get("payload") or {}
    args = ["refresh"]
    if payload.get("limit"):
        args += ["--limit", str(payload["limit"])]
    code, log_path, tail = _run_cli(args, settings, "refresh",
                                    job.get("requested_by", "system:scout"))
    if code != 0:
        raise RuntimeError(f"tracked refresh exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_resolve(store: Store, settings: Settings, job: dict) -> dict:
    """Work through the unlinked leads discovery could not key to a company
    — funding headlines, launches linking to a video — and turn them into
    scored leads. Budget-gated inside the CLI against DAILY_SPEND_CAP_USD."""
    payload = job.get("payload") or {}
    args = ["resolve"]
    if payload.get("limit"):
        args += ["--limit", str(payload["limit"])]
    code, log_path, tail = _run_cli(args, settings, "resolve",
                                    job.get("requested_by", "system:scout"))
    if code != 0:
        raise RuntimeError(f"unlinked-lead resolve exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_publish(store: Store, settings: Settings, job: dict) -> dict:
    """Render the phone app and deploy it wherever the box is configured
    to (GitHub Pages via DIGEST_REPO, Vercel via a linked project or
    VERCEL_TOKEN) — `scout publish --auto`. Nothing configured just
    renders docs/, so this never fails for want of a host."""
    code, log_path, tail = _run_cli(["publish", "--auto"], settings, "publish",
                                    job.get("requested_by", "system:scout"))
    if code != 0:
        raise RuntimeError(f"phone app publish exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_reclassify(store: Store, settings: Settings, job: dict) -> dict:
    """Re-score without discovery (`scout reclassify`): cache-first, so a
    rescore after a weights change costs little. `scope`: latest (default),
    stale (only startups scored under an older thesis version) or all."""
    payload = job.get("payload") or {}
    args = ["reclassify"]
    scope = payload.get("scope", "latest")
    if scope == "stale":
        args.append("--stale-only")
    elif scope == "all":
        args.append("--all")
    code, log_path, tail = _run_cli(args, settings, "reclassify",
                                    job.get("requested_by", "system:scout"), store)
    if code != 0:
        raise RuntimeError(f"rescore exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_preview(store: Store, settings: Settings, job: dict) -> dict:
    """Free discovery only (`scout source`) — what a run would find, unscored."""
    payload = job.get("payload") or {}
    args = ["source"]
    if payload.get("max_accounts"):
        args += ["--max-accounts", str(payload["max_accounts"])]
    code, log_path, tail = _run_cli(args, settings, "source",
                                    job.get("requested_by", "system:scout"), store)
    if code != 0:
        raise RuntimeError(f"discovery preview exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_verify(store: Store, settings: Settings, job: dict) -> dict:
    code, log_path, tail = _run_cli(["verify"], settings, "verify",
                                    job.get("requested_by", "system:scout"), store)
    if code != 0:
        raise RuntimeError(f"verification exited {code}\n{tail}")
    return {"log_path": str(log_path)}


def handle_crm(store: Store, settings: Settings, job: dict) -> dict:
    """Write startups at or past the push threshold to the firm's CRM —
    or exactly `handles` from the payload (a "Send to CRM" click). In-
    process: a few HTTP calls per company, and a sync over an unchanged
    pipeline makes none (crm.due decides locally). Raises only when every
    attempted push failed, so a single bad record retries alone next pass
    instead of failing the batch."""
    from scout import crm
    from scout.theses import resolve as resolve_thesis

    payload = job.get("payload") or {}
    if not crm.configured(settings):
        return {"skipped": "no CRM configured (ATTIO_API_KEY / AFFINITY_API_KEY)"}
    try:
        thesis_name = resolve_thesis(store).name or ""
    except Exception:  # noqa: BLE001 — a note without a thesis name is fine
        thesis_name = ""
    results = crm.sync(store, settings, handles=payload.get("handles"),
                       thesis_name=thesis_name)
    errors = [r for r in results if r.error]
    if errors and len(errors) == len(results):
        raise RuntimeError("; ".join(f"@{r.handle}: {r.error}" for r in errors[:3]))
    return {"summary": crm.summarize(results), "pushed": len(results),
            "errors": [f"@{r.handle} → {r.provider}: {r.error}" for r in errors]}


HANDLERS = {
    jobs_mod.KIND_RUN: handle_run,
    jobs_mod.KIND_MEMO: handle_memo,
    jobs_mod.KIND_DIGEST: handle_digest,
    jobs_mod.KIND_VERIFY: handle_verify,
    jobs_mod.KIND_REFRESH: handle_refresh,
    jobs_mod.KIND_RESOLVE: handle_resolve,
    jobs_mod.KIND_PUBLISH: handle_publish,
    jobs_mod.KIND_CRM: handle_crm,
    jobs_mod.KIND_RECLASSIFY: handle_reclassify,
    jobs_mod.KIND_PREVIEW: handle_preview,
}


# ------------------------------------------------------------------- loop


def execute_job(store: Store, settings: Settings, job: dict) -> bool:
    """Run one claimed job to completion. Returns True if it succeeded.

    Every failure path is caught: the worker's contract is that it keeps
    running no matter what a handler does.
    """
    handler = HANDLERS.get(job["kind"])
    label = jobs_mod.job_label(job["kind"], job.get("payload"))
    if handler is None:
        store.fail_job(job["id"], f"no handler for job kind {job['kind']!r}")
        return False
    console.print(f"[bold]▶ {label}[/bold] (job {job['id']})")
    started = time.monotonic()
    try:
        with _Heartbeat(store, job["id"]):
            result = handler(store, settings, job)
    except JobStopped as exc:
        store.fail_job(job["id"], str(exc), retry=False)
        console.print(f"[yellow]■ {label}[/yellow] — {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 — a handler must not stop the loop
        message = f"{type(exc).__name__}: {exc}"
        retrying = store.fail_job(job["id"], message)
        console.print(
            f"[red]✗ {label}[/red] — {message.splitlines()[0]}"
            + (" (will retry)" if retrying else " (giving up)")
        )
        return False
    elapsed = time.monotonic() - started
    store.finish_job(job["id"], result, log_path=result.get("log_path", ""))
    console.print(f"[green]✓ {label}[/green] in {elapsed:.0f}s")
    return True


def tick(store: Store, settings: Settings, worker_id: str) -> bool:
    """One pass: reap, schedule, run at most one job. True if work was done."""
    store.record_worker_heartbeat(worker_id)
    reaped = store.reap_stale_jobs()
    if reaped:
        console.print(f"[yellow]requeued {reaped} job(s) from a dead worker[/]")
    for job_id in store.materialize_due_schedules():
        console.print(f"[dim]schedule fired → job {job_id}[/dim]")
    job = store.claim_job(worker_id)
    if job is None:
        return False
    execute_job(store, settings, job)
    return True


def run_worker(
    store: Store,
    settings: Settings,
    *,
    once: bool = False,
    poll_seconds: int = POLL_SECONDS,
    max_jobs: int | None = None,
) -> int:
    """The worker loop. Returns the number of jobs executed.

    `once` drains the queue and returns — which is what a cron-driven
    deployment wants, and what the tests use. Without it this runs forever
    under systemd.
    """
    worker_id = f"{os.uname().nodename}:{os.getpid()}"
    console.print(f"[bold]scout worker[/bold] {worker_id} — "
                  f"db {store.db_path}")
    executed = 0
    stopping = threading.Event()

    def _stop(signum, _frame) -> None:
        console.print("\n[yellow]shutting down after the current job…[/]")
        stopping.set()

    # Only install handlers on the main thread (tests may call this
    # elsewhere); SIGTERM is what systemd and containers send.
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _stop)

    while not stopping.is_set():
        try:
            did_work = tick(store, settings, worker_id)
        except Exception as exc:  # noqa: BLE001 — the loop outlives everything
            console.print(f"[red]worker tick failed:[/] {type(exc).__name__}: {exc}")
            did_work = False
        if did_work:
            executed += 1
            if max_jobs is not None and executed >= max_jobs:
                break
            continue  # drain greedily before sleeping
        if once:
            break
        stopping.wait(poll_seconds)
    return executed


def bootstrap_schedules(store: Store, actor: str = "system:scout") -> list[int]:
    """Create the default schedules on a fresh install.

    Chosen for a firm that wants Scout to be a standing process rather than
    a tool someone remembers to open: source every morning, resolve what the
    run could not key, refresh the tracked list, digest, then publish the
    phone app — so the summary and the deployed app describe work that has
    finished.
    Idempotent — existing schedules of the same kind are left alone.
    """
    existing = {s["kind"] for s in store.schedules()}
    created: list[int] = []
    if jobs_mod.KIND_RUN not in existing:
        # Every day, not weekdays: momentum signals decay over a weekend
        # too, and the daily spend envelope (DAILY_SPEND_CAP_USD) is what
        # bounds cost — not skipping days.
        created.append(store.upsert_schedule(
            "Daily sourcing run", jobs_mod.KIND_RUN,
            jobs_mod.ScheduleSpec(daily_at="06:00", tz="UTC"),
            {"source": "twscrape"}, actor=actor,
        ))
    if jobs_mod.KIND_RESOLVE not in existing:
        # Right after the run: the headlines it just read with no company
        # behind them become scored leads before the refresh and the digest,
        # so "Acme raised" is a row in this morning's summary.
        created.append(store.upsert_schedule(
            "Unlinked-lead resolve", jobs_mod.KIND_RESOLVE,
            jobs_mod.ScheduleSpec(daily_at="06:30", tz="UTC"),
            {}, actor=actor,
        ))
    if jobs_mod.KIND_REFRESH not in existing:
        # After the run (cheap, cache-warm), before the digest (so a found
        # acquisition is in the morning summary, not tomorrow's).
        created.append(store.upsert_schedule(
            "Tracked-company refresh", jobs_mod.KIND_REFRESH,
            jobs_mod.ScheduleSpec(daily_at="06:45", tz="UTC"),
            {}, actor=actor,
        ))
    if jobs_mod.KIND_DIGEST not in existing:
        created.append(store.upsert_schedule(
            "Morning digest", jobs_mod.KIND_DIGEST,
            jobs_mod.ScheduleSpec(daily_at="07:30", tz="UTC"),
            {"window": "daily"}, actor=actor,
        ))
    if jobs_mod.KIND_PUBLISH not in existing:
        # Last: the deployed phone app reflects everything the morning did.
        created.append(store.upsert_schedule(
            "Phone app publish", jobs_mod.KIND_PUBLISH,
            jobs_mod.ScheduleSpec(daily_at="07:45", tz="UTC"),
            {}, actor=actor,
        ))
    return created
