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


def _run_cli(args: list[str], settings: Settings, kind: str,
             actor: str) -> tuple[int, Path, str]:
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
        try:
            code = proc.wait(timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # Kill the whole process group: scrapers spawn helpers, and a
            # bare terminate() would orphan them.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)
            code = -9
    return code, log_path, _tail(log_path)


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
    code, log_path, tail = _run_cli(args, settings, "run",
                                    job.get("requested_by", "system:scout"))
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


def handle_verify(store: Store, settings: Settings, job: dict) -> dict:
    code, log_path, tail = _run_cli(["verify"], settings, "verify",
                                    job.get("requested_by", "system:scout"))
    if code != 0:
        raise RuntimeError(f"verification exited {code}\n{tail}")
    return {"log_path": str(log_path)}


HANDLERS = {
    jobs_mod.KIND_RUN: handle_run,
    jobs_mod.KIND_MEMO: handle_memo,
    jobs_mod.KIND_DIGEST: handle_digest,
    jobs_mod.KIND_VERIFY: handle_verify,
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
    a tool someone remembers to open: source every weekday morning, and
    digest just after so the summary describes a run that has finished.
    Idempotent — existing schedules of the same kind are left alone.
    """
    existing = {s["kind"] for s in store.schedules()}
    created: list[int] = []
    if jobs_mod.KIND_RUN not in existing:
        created.append(store.upsert_schedule(
            "Weekday sourcing run", jobs_mod.KIND_RUN,
            jobs_mod.ScheduleSpec(daily_at="06:00", weekdays=[0, 1, 2, 3, 4],
                                  tz="UTC"),
            {"source": "twscrape"}, actor=actor,
        ))
    if jobs_mod.KIND_DIGEST not in existing:
        created.append(store.upsert_schedule(
            "Morning digest", jobs_mod.KIND_DIGEST,
            jobs_mod.ScheduleSpec(daily_at="07:30", weekdays=[0, 1, 2, 3, 4],
                                  tz="UTC"),
            {"window": "daily"}, actor=actor,
        ))
    return created
