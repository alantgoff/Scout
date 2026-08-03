"""Background work: job kinds, retry policy, and schedule arithmetic.

Everything here is pure — no database, no network, no clock reads except the
`now` you pass in — so the parts that are easy to get subtly wrong (DST
boundaries, backoff growth, weekday masks) are unit-testable. The queue
itself lives in Store (the single DB gateway) and the runner in
scout.worker.

Why a queue at all: sourcing runs take minutes and momentum signals decay in
days. A firm that has to remember to press "Run" misses hiring spikes and
launches — exactly the signals that are only valuable while they are fresh.
The queue also keeps long work off the Streamlit process, so one partner's
run never freezes the other partner's page.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

# Job kinds the worker knows how to dispatch. Kept as plain strings (not an
# Enum) because they round-trip through SQLite and JSON payloads.
KIND_RUN = "run_pipeline"
KIND_MEMO = "generate_memo"
KIND_DIGEST = "digest"
KIND_VERIFY = "verify"

JOB_KINDS = {KIND_RUN, KIND_MEMO, KIND_DIGEST, KIND_VERIFY}

JOB_LABELS = {
    KIND_RUN: "Sourcing run",
    KIND_MEMO: "Memo",
    KIND_DIGEST: "Digest",
    KIND_VERIFY: "Verification",
}

# Terminal states never re-enter the queue.
TERMINAL = {"done", "failed", "cancelled"}

# A claimed job holds its lease this long; a worker that dies without
# finishing has the job requeued once the lease lapses. Comfortably longer
# than the heartbeat interval so a slow-but-alive worker is never robbed.
LEASE_SECONDS = 900
HEARTBEAT_SECONDS = 60

MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 60
_BACKOFF_CAP_S = 3600


def backoff_seconds(attempt: int) -> int:
    """Delay before retrying a job that has failed `attempt` times.

    Deterministic (no jitter) on purpose: a single worker per firm means
    there is no thundering herd to spread out, and a predictable delay is
    far easier to reason about when a run is stuck.
    """
    if attempt < 1:
        return 0
    return min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_CAP_S)


class ScheduleSpec(BaseModel):
    """When a schedule fires.

    Two shapes, deliberately not full cron: cron is a foot-gun in a product
    where "every weekday at 7am" is the only thing anyone actually wants,
    and its DST behaviour is unintuitive.

    - every_minutes: a simple fixed interval.
    - daily_at + weekdays + tz: a wall-clock time in a named timezone, which
      is what "7am my time" means and survives DST correctly. weekdays uses
      Python's Monday=0 convention; empty means every day.
    """

    every_minutes: int | None = None
    daily_at: str | None = None  # "HH:MM"
    weekdays: list[int] = Field(default_factory=list)
    tz: str = "UTC"

    def validate_spec(self) -> None:
        """Raise ValueError on a spec that could never fire sensibly."""
        if self.every_minutes is None and not self.daily_at:
            raise ValueError("a schedule needs every_minutes or daily_at")
        if self.every_minutes is not None and self.daily_at:
            raise ValueError("set every_minutes or daily_at, not both")
        if self.every_minutes is not None and self.every_minutes < 5:
            raise ValueError("every_minutes must be at least 5")
        if self.daily_at is not None:
            _parse_hhmm(self.daily_at)  # raises on a bad time
        if any(d < 0 or d > 6 for d in self.weekdays):
            raise ValueError("weekdays must be 0 (Monday) through 6 (Sunday)")
        try:
            ZoneInfo(self.tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {self.tz}") from exc

    def describe(self) -> str:
        """One line a human can check at a glance — the whole point of
        keeping the spec small."""
        if self.every_minutes is not None:
            if self.every_minutes % 60 == 0:
                hours = self.every_minutes // 60
                return f"every {hours}h" if hours > 1 else "hourly"
            return f"every {self.every_minutes}m"
        days = "every day"
        if self.weekdays:
            if self.weekdays == [0, 1, 2, 3, 4]:
                days = "weekdays"
            elif self.weekdays == [5, 6]:
                days = "weekends"
            else:
                names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                days = ", ".join(names[d] for d in sorted(self.weekdays))
        return f"{days} at {self.daily_at} {self.tz}"


def _parse_hhmm(value: str) -> time:
    try:
        hour_s, _, minute_s = value.partition(":")
        hour, minute = int(hour_s), int(minute_s)
        return time(hour=hour, minute=minute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"daily_at must be HH:MM, got {value!r}") from exc


def next_occurrence(spec: ScheduleSpec, after: datetime) -> datetime:
    """The first firing strictly after `after`, as an aware UTC datetime.

    For daily schedules the arithmetic is done in the schedule's own
    timezone and converted back, which is what makes "07:00 Europe/London"
    stay at 07:00 through a DST switch instead of drifting an hour.
    """
    spec.validate_spec()
    if after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    after = after.astimezone(timezone.utc)

    if spec.every_minutes is not None:
        return after + timedelta(minutes=spec.every_minutes)

    zone = ZoneInfo(spec.tz)
    at = _parse_hhmm(spec.daily_at or "09:00")
    local = after.astimezone(zone)
    allowed = set(spec.weekdays) or set(range(7))
    # Walk forward day by day. Bounded at 8 to cover any weekday mask, and
    # cheap enough that clarity beats closed-form date maths.
    for offset in range(0, 9):
        candidate_date = (local + timedelta(days=offset)).date()
        candidate = datetime.combine(candidate_date, at, tzinfo=zone)
        # A nonexistent local time (spring-forward gap) still maps to a real
        # instant; comparing in UTC keeps the ordering honest either way.
        candidate_utc = candidate.astimezone(timezone.utc)
        if candidate_utc > after and candidate.weekday() in allowed:
            return candidate_utc
    # Unreachable for a valid spec — every mask hits within 8 days.
    raise ValueError(f"could not compute next run for {spec!r}")


def job_label(kind: str, payload: dict | None = None) -> str:
    """Human name for a job row, including its subject when it has one."""
    base = JOB_LABELS.get(kind, kind.replace("_", " "))
    payload = payload or {}
    if kind == KIND_MEMO and payload.get("handle"):
        return f"{base} — @{payload['handle']}"
    if kind == KIND_RUN and payload.get("source"):
        return f"{base} ({payload['source']})"
    if kind == KIND_DIGEST and payload.get("window"):
        return f"{base} ({payload['window']})"
    return base
