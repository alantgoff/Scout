"""SQLite persistence via sqlite-utils: cache, dedupe, seen-TTL, API budget ledger.

Every fetched account/tweet is upserted here so repeat runs are incremental
(and, in xapi mode, free).
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite_utils
from pydantic import ValidationError

from scout import jobs as jobs_mod
from scout.config import DEFAULT_DB_PATH
from scout.models import (
    Account,
    Comment,
    Event,
    Lead,
    LedgerEntry,
    LLMVerdict,
    MemoVersion,
    SitePage,
    Tweet,
    UnlinkedLead,
    Vote,
)

# Pre-~/.scout location: a scout.db relative to wherever scout was run from.
_LEGACY_DB_PATH = Path("scout.db")


class Store:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        *,
        cross_thread: bool = False,
        actor: str | None = None,
    ) -> None:
        db_path = Path(db_path).expanduser()
        # One-time migration: preserve spend history from the old cwd-relative
        # scout.db so moving the ledger home never resets the $25 guard.
        if (
            db_path == DEFAULT_DB_PATH
            and not db_path.exists()
            and _LEGACY_DB_PATH.exists()
        ):
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_LEGACY_DB_PATH, db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        # Who is making writes through this Store. The UI binds the logged-in
        # user's email after auth resolves; the CLI/worker bind namespaced
        # machine actors ("system:run", "agent:memo", "schedule:3"). Judgment
        # writes stamp it; None means unattributed (legacy callers).
        self.actor = actor
        if cross_thread:
            # Streamlit reruns the script (and polling fragments) on varying
            # threads; sqlite3's same-thread guard would raise
            # ProgrammingError mid-render. Cross-thread use is safe because
            # SQLite serializes statements internally and every read-merge-
            # write in this class runs inside write_tx() (BEGIN IMMEDIATE),
            # so concurrent sessions cannot lose each other's updates.
            import sqlite3

            self.db = sqlite_utils.Database(
                sqlite3.connect(str(db_path), check_same_thread=False)
            )
        else:
            self.db = sqlite_utils.Database(db_path)
        self._configure_connection()
        self._migrate_thesis_columns()
        self._ensure_collab_tables()
        self._ensure_job_tables()
        self._ensure_indexes()

    def _ensure_collab_tables(self) -> None:
        """Create the append-only collaboration tables up front.

        Unlike the rest of the store, these cannot be born from their first
        insert: sqlite-utils infers columns from the row it is given, and
        these tables need an auto-assigning INTEGER PRIMARY KEY that no row
        ever carries (events are ordered by it, comments and memo versions
        are referenced by it). Idempotent via if_not_exists.
        """
        self.db["events"].create(
            {
                "id": int, "at": str, "actor": str, "verb": str,
                "handle": str, "thesis_id": str, "payload_json": str,
                "notified": int,
            },
            pk="id",
            if_not_exists=True,
        )
        self.db["comments"].create(
            {
                "id": int, "handle": str, "actor": str, "body": str,
                "mentions_json": str, "memo_version_id": int,
                "created_at": str, "edited_at": str, "deleted_at": str,
            },
            pk="id",
            if_not_exists=True,
        )
        # The thesis registry is written by two different paths (metadata on
        # every run, full config on every save). Declaring the shape up front
        # means whichever runs first cannot leave the other's columns
        # missing — `is_active` in particular is read before either writes.
        self.db["theses"].create(
            {
                "id": str, "name": str, "statement": str,
                "current_version": str, "created_at": str, "archived_at": str,
                "is_active": int, "config_json": str,
                "config_updated_at": str, "config_updated_by": str,
            },
            pk="id",
            if_not_exists=True,
        )
        self.db["backtests"].create(
            {
                "id": int, "cutoff": str, "thesis_id": str, "threshold": float,
                "n_outcomes": int, "n_controls": int, "recall": float,
                "auc": float, "report_json": str, "created_at": str,
                "created_by": str,
            },
            pk="id",
            if_not_exists=True,
        )
        self.db["memo_versions"].create(
            {
                "id": int, "handle": str, "version_no": int, "body": str,
                "meta_json": str, "author": str, "kind": str,
                "created_at": str,
            },
            pk="id",
            if_not_exists=True,
        )

    def _configure_connection(self) -> None:
        """Concurrency posture for a multi-session deployment.

        WAL lets N readers proceed while one writer commits (and is what
        litestream replication requires); busy_timeout makes a second writer
        wait instead of raising 'database is locked'; synchronous=NORMAL is
        the standard WAL pairing (durability to the WAL, not fsync-per-txn).
        journal_mode persists in the file; the others are per-connection.
        """
        conn = self.db.conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")

    @contextmanager
    def write_tx(self):
        """One atomic write transaction, lost-update-safe.

        BEGIN IMMEDIATE takes the write lock BEFORE the first read, so a
        read-merge-write (set_pipeline, set_attrs, ...) cannot interleave
        with another session's — the deferred default would let two sessions
        read the same row and last-write-wins each other. Re-entrant: a
        write_tx inside an open write_tx joins it (single commit at the
        outermost exit). sqlite-utils' own atomic() blocks nest via
        SAVEPOINTs, so upsert/insert calls inside compose correctly.
        """
        conn = self.db.conn
        if conn.in_transaction:
            yield
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def _migrate_thesis_columns(self) -> None:
        """Add the thesis-provenance columns to tables that predate them.

        upsert(alter=True) grows a table on WRITE, but the staleness and
        registry queries READ these columns, and a database recorded before
        thesis identity existed has neither. Adding them up front means every
        query below can assume they are there.
        """
        for table, columns in (
            ("runs", {"thesis_id": str, "thesis_version": str}),
            ("llm_verdicts", {"thesis_id": str, "thesis_version": str}),
            ("pipeline", {"sourced_thesis_id": str}),
        ):
            if not self.db[table].exists():
                continue
            existing = set(self.db[table].columns_dict)
            missing = {k: v for k, v in columns.items() if k not in existing}
            if missing:
                self.db[table].add_column_alter_table = True
                for name, col_type in missing.items():
                    self.db[table].add_column(name, col_type)

    def _ensure_indexes(self) -> None:
        """Secondary indexes for the per-handle hot paths.

        Every composite pk here indexes the WRONG prefix for these lookups
        (leads is keyed run_id-first, follow_edges watcher-first, tweets by
        tweet id), so without these each per-account query in the pipeline
        was a full table scan — O(accounts × table) loops in aggregate.
        Idempotent, and skipped for tables that don't exist yet (sqlite-utils
        creates tables lazily on first write; the index lands on the next
        Store init, which every CLI command and UI rerun performs)."""
        for table, name, columns, expr in (
            # last_scored_at: handle = ? COLLATE NOCASE, newest first.
            ("leads", "idx_leads_handle",
             {"handle", "created_at"}, "handle collate nocase, created_at"),
            # get_tweets: account_id = ?, newest first.
            ("tweets", "idx_tweets_account",
             {"account_id", "created_at"}, "account_id, created_at"),
            # get_account: handle = ? COLLATE NOCASE.
            ("accounts", "idx_accounts_handle", {"handle"}, "handle collate nocase"),
            # record_bio: latest snapshot per handle (table grows every run).
            ("bio_snapshots", "idx_bio_snapshots_handle",
             {"handle", "seen_at"}, "handle, seen_at"),
            # recent_watchers_for/_map: followee = ? and first_seen >= ?.
            ("follow_edges", "idx_follow_edges_followee",
             {"followee", "first_seen"}, "followee, first_seen"),
            # verdict_history: handle = ?, newest archive first.
            ("llm_verdict_history", "idx_verdict_history_handle",
             {"handle", "archived_at"}, "handle, archived_at"),
            # Activity spine + collaboration state.
            ("events", "idx_events_handle", {"handle"}, "handle, id"),
            ("events", "idx_events_actor", {"actor"}, "actor, id"),
            ("votes", "idx_votes_handle", {"handle"}, "handle"),
            ("comments", "idx_comments_handle", {"handle"}, "handle, id"),
            ("comments", "idx_comments_memo",
             {"memo_version_id"}, "memo_version_id"),
            ("memo_versions", "idx_memo_versions_handle",
             {"handle", "version_no"}, "handle, version_no"),
        ):
            if not self.db[table].exists():
                continue
            if not columns <= set(self.db[table].columns_dict):
                continue  # legacy table shape — never let an index break init
            self.db.execute(f"create index if not exists {name} on {table}({expr})")
        # Partial index for the notifier's sweep — the unnotified set is tiny.
        if (
            self.db["events"].exists()
            and "notified" in self.db["events"].columns_dict
        ):
            self.db.execute(
                "create index if not exists idx_events_unnotified "
                "on events(notified) where notified = 0"
            )
        self.db.conn.commit()

    # ------------------------------------------------------------------ users

    def ensure_user(self, email: str, *, name: str = "") -> dict:
        """Provision-or-touch a user on login; returns the row.

        The id is the lowercased email (it is what OIDC asserts and what
        actor columns store). The FIRST user ever provisioned becomes admin,
        as does anyone listed in SCOUT_ADMIN_EMAILS — everyone else joins as
        member and an admin can promote them. last_seen_at is stamped here;
        the UI throttles calls to ~once a minute per session.
        """
        user_id = email.strip().lower()
        if not user_id:
            raise ValueError("ensure_user needs a non-empty email")
        now = datetime.now(timezone.utc).isoformat()
        admin_env = {
            e.strip().lower()
            for e in os.environ.get("SCOUT_ADMIN_EMAILS", "").split(",")
            if e.strip()
        }
        with self.write_tx():
            existing = self.get_user(user_id)
            if existing is None:
                first = not (
                    self.db["users"].exists() and self.db["users"].count > 0
                )
                role = "admin" if (first or user_id in admin_env) else "member"
                row = {
                    "id": user_id,
                    "name": name or user_id.split("@")[0],
                    "role": role,
                    "default_thesis_id": "",
                    "slack_member_id": "",
                    "settings_json": "{}",
                    "created_at": now,
                    "last_seen_at": now,
                }
                self.db["users"].insert(row, pk="id")
                return row
            updates: dict = {"last_seen_at": now}
            if name and name != existing.get("name"):
                updates["name"] = name
            if user_id in admin_env and existing.get("role") != "admin":
                updates["role"] = "admin"
            self.db["users"].update(user_id, updates)
            return {**existing, **updates}

    def get_user(self, user_id: str) -> dict | None:
        if not self.db["users"].exists():
            return None
        rows = list(
            self.db["users"].rows_where("id = ?", [user_id.strip().lower()], limit=1)
        )
        return dict(rows[0]) if rows else None

    def list_users(self) -> list[dict]:
        """All provisioned users, admins first then by name."""
        if not self.db["users"].exists():
            return []
        return [
            dict(r)
            for r in self.db["users"].rows_where(
                order_by="case role when 'admin' then 0 else 1 end, name"
            )
        ]

    def touch_user(self, user_id: str) -> None:
        """Presence heartbeat (throttled by the caller)."""
        if self.db["users"].exists():
            self.db.execute(
                "update users set last_seen_at = ? where id = ?",
                [datetime.now(timezone.utc).isoformat(), user_id.strip().lower()],
            )
            self.db.conn.commit()

    def update_user(self, user_id: str, **fields) -> None:
        """Update a member's own profile fields (name, Slack id, prefs).

        Role is deliberately NOT settable here — it is an authorization
        decision with its own guarded method, and folding it into a generic
        profile update is how privilege-escalation bugs happen.
        """
        allowed = {"name", "slack_member_id", "settings_json"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates or not self.db["users"].exists():
            return
        with self.write_tx():
            self.db["users"].update(user_id.strip().lower(), updates, alter=True)

    def set_user_role(self, user_id: str, role: str) -> None:
        if role not in ("admin", "member"):
            raise ValueError(f"unknown role: {role!r}")
        if self.db["users"].exists():
            self.db.execute(
                "update users set role = ? where id = ?",
                [role, user_id.strip().lower()],
            )
            self.db.conn.commit()

    # --------------------------------------------------------------- settings

    # Non-secret runtime knobs that may live in the settings table and
    # override the .env value at load time (key = Settings field name).
    # True secrets (API keys, cookies) are deliberately NOT overridable
    # here — they stay in the environment, never UI-writable.
    RUNTIME_SETTING_FIELDS: dict[str, type] = {
        "xapi_spend_cap_usd": float,
        "claude_model": str,
        "max_accounts": int,
        "ttl_days": int,
        "llm_max_candidates": int,
        "verdict_ttl_days": int,
    }

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        if not self.db["settings"].exists():
            return default
        rows = list(self.db["settings"].rows_where("key = ?", [key], limit=1))
        if not rows or rows[0].get("value") in (None, ""):
            return default
        return rows[0]["value"]

    def set_setting(self, key: str, value: str, *, actor: str | None = None) -> None:
        with self.write_tx():
            self.db["settings"].upsert(
                {
                    "key": key,
                    "value": str(value),
                    "updated_by": actor or self.actor or "",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                pk="key",
                alter=True,
            )

    def all_settings(self) -> dict[str, str]:
        if not self.db["settings"].exists():
            return {}
        return {r["key"]: r.get("value") or "" for r in self.db["settings"].rows}

    def apply_settings_overrides(self, settings) -> None:
        """Overlay DB-held runtime knobs onto a loaded Settings object.

        The firm's shared knobs (spend cap, model, run sizes) are edited in
        the UI and must apply to every session and worker run — a value in
        the settings table wins over the .env default. Unparseable values
        are ignored rather than crashing a run."""
        for field, cast in self.RUNTIME_SETTING_FIELDS.items():
            raw = self.get_setting(field)
            if raw in (None, ""):
                continue
            try:
                setattr(settings, field, cast(raw))
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------ cache

    def upsert_account(self, account: Account) -> None:
        self.upsert_accounts([account])

    def upsert_accounts(self, accounts: list[Account]) -> None:
        """Batched account upsert — one statement batch instead of one
        autocommitted transaction per row (sourcing used to re-upsert an
        account on every sighting of it)."""
        if not accounts:
            return
        rows = []
        for account in accounts:
            row = account.model_dump(mode="json")
            row["followed_by"] = json.dumps(row["followed_by"])
            row["sources"] = json.dumps(row.get("sources") or [])
            # Enrichment fields are recomputed from history every run — don't
            # persist stale values.
            row.pop("recent_followed_by", None)
            row.pop("bio_changed", None)
            rows.append(row)
        self.db["accounts"].upsert_all(rows, pk="id", alter=True)

    def upsert_tweets(self, tweets: list[Tweet]) -> None:
        if tweets:
            self.db["tweets"].upsert_all(
                [t.model_dump(mode="json") for t in tweets], pk="id"
            )

    def get_account(self, handle: str) -> Account | None:
        rows = list(
            self.db["accounts"].rows_where(
                "handle = ? COLLATE NOCASE", [handle], limit=1
            )
            if self.db["accounts"].exists()
            else []
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["followed_by"] = json.loads(row.get("followed_by") or "[]")
        row["sources"] = json.loads(row.get("sources") or "[]")
        # Drop stale enrichment columns from pre-v2 rows; recomputed per run.
        row.pop("recent_followed_by", None)
        row.pop("bio_changed", None)
        return Account.model_validate(row)

    def recent_discovered_accounts(self, days: int = 7) -> list[Account]:
        """Accounts discovered recently by any real strategy (demo/manual excluded).

        The `scout verify --discovered` shortlist source when there's no real
        scored run yet.
        """
        if not self.db["accounts"].exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.db["accounts"].rows_where(
            "source not in ('demo', 'manual', '') and fetched_at >= ?",
            [cutoff],
            order_by="fetched_at desc",
        )
        accounts = []
        for r in rows:
            row = dict(r)
            row["followed_by"] = json.loads(row.get("followed_by") or "[]")
            row["sources"] = json.loads(row.get("sources") or "[]")
            row.pop("recent_followed_by", None)
            row.pop("bio_changed", None)
            accounts.append(Account.model_validate(row))
        return accounts

    def get_tweets(self, account_id: str, limit: int = 20) -> list[Tweet]:
        if not self.db["tweets"].exists():
            return []
        rows = self.db["tweets"].rows_where(
            "account_id = ?", [account_id], order_by="created_at desc", limit=limit
        )
        return [Tweet.model_validate(dict(r)) for r in rows]

    # ------------------------------------------------------------- seen cache

    def last_scored_at(self, handle: str) -> datetime | None:
        if not self.db["leads"].exists():
            return None
        rows = list(
            self.db["leads"].rows_where(
                "handle = ? COLLATE NOCASE",
                [handle],
                order_by="created_at desc",
                limit=1,
            )
        )
        if not rows:
            return None
        return datetime.fromisoformat(rows[0]["created_at"])

    def recently_scored(self, handle: str, ttl_days: int) -> bool:
        last = self.last_scored_at(handle)
        if last is None:
            return False
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last < timedelta(days=ttl_days)

    def last_scored_map(self) -> dict[str, datetime]:
        """Newest scored-at per lowercased handle, across all runs — the
        batched form of `last_scored_at` for the pipeline's TTL skip (one
        grouped query instead of one query per candidate account)."""
        if not self.db["leads"].exists():
            return {}
        out: dict[str, datetime] = {}
        rows = self.db.execute(
            "select lower(handle), max(created_at) from leads group by lower(handle)"
        ).fetchall()
        for handle, created in rows:
            if not created:
                continue
            ts = datetime.fromisoformat(created)
            out[handle] = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return out

    # ------------------------------------------------------------------ leads

    def save_leads(self, run_id: str, leads: list[Lead]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db["leads"].upsert_all(
            [
                {
                    "run_id": run_id,
                    "handle": lead.account.handle,
                    "rank": lead.rank,
                    "score": lead.score,
                    "lead_json": lead.model_dump_json(),
                    "created_at": now,
                }
                for lead in leads
            ],
            pk=("run_id", "handle"),
        )

    def load_latest_leads(self) -> list[Lead]:
        if not self.db["leads"].exists():
            return []
        rows = list(self.db["leads"].rows_where(order_by="created_at desc", limit=1))
        if not rows:
            return []
        latest_run = rows[0]["run_id"]
        rows = self.db["leads"].rows_where(
            "run_id = ?", [latest_run], order_by="rank"
        )
        return [Lead.model_validate_json(r["lead_json"]) for r in rows]

    # ------------------------------------------------------------------- runs

    def record_run(
        self,
        run_id: str,
        *,
        source: str,
        strategy_hash: str,
        thesis_statement: str,
        config_json: str = "",
        thesis_id: str = "",
        thesis_version: str = "",
    ) -> None:
        """Record run provenance: which thesis, at which version, produced it.

        `thesis_id` is the durable identity and `thesis_version` the tuning
        fingerprint; `strategy_hash` is kept identical to the version so the
        existing strategy grouping keeps working unchanged. `requested_by`
        names who asked for the run (the store's bound actor by default).
        """
        self.db["runs"].upsert(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "strategy_hash": strategy_hash,
                "thesis_statement": thesis_statement,
                "config_json": config_json,
                "thesis_id": thesis_id,
                "thesis_version": thesis_version or strategy_hash,
                "requested_by": self.actor or "",
            },
            pk="run_id",
            alter=True,
        )

    # ---------------------------------------------------------------- theses

    def upsert_thesis(
        self,
        thesis_id: str,
        *,
        name: str,
        statement: str,
        version: str,
        make_active: bool = False,
    ) -> None:
        """Register a thesis and the version it is currently at.

        Called on every run, so the registry stays correct without a separate
        bookkeeping step. Recording the version here is what later makes
        staleness computable: a lead scored under an older version is stale by
        comparison with `theses.current_version`.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            existing = self.get_thesis(thesis_id)
            self.db["theses"].upsert(
                {
                    "id": thesis_id,
                    "name": name or thesis_id,
                    "statement": statement,
                    "current_version": version,
                    "created_at": (existing or {}).get("created_at") or now,
                    "archived_at": (existing or {}).get("archived_at"),
                    "is_active": 1 if make_active else (existing or {}).get("is_active", 0),
                },
                pk="id",
                alter=True,
            )
            if version:
                self.db["thesis_versions"].upsert(
                    {
                        "thesis_id": thesis_id,
                        "version": version,
                        "created_at": now,
                    },
                    pk=("thesis_id", "version"),
                    alter=True,
                )
            if make_active:
                self.set_active_thesis(thesis_id)

    def get_thesis(self, thesis_id: str) -> dict | None:
        if not self.db["theses"].exists():
            return None
        rows = list(self.db["theses"].rows_where("id = ?", [thesis_id], limit=1))
        return rows[0] if rows else None

    def save_thesis_config(self, thesis_id: str, config: dict) -> None:
        """Store a thesis's FULL configuration in the database.

        The registry above tracks identity and version; this holds the thing
        itself — weights, queries, disqualifiers, the lot. Keeping it in the
        database rather than only in thesis.yaml is what makes a thesis
        shared firm state: litestream backs it up, a replaced container
        keeps it, and two partners editing from different continents are
        reading and writing one copy instead of racing over a file.
        """
        with self.write_tx():
            self.db["theses"].upsert(
                {
                    "id": thesis_id,
                    "config_json": json.dumps(config, sort_keys=True, default=str),
                    "config_updated_at": datetime.now(timezone.utc).isoformat(),
                    "config_updated_by": self.actor or "",
                },
                pk="id",
                alter=True,
            )

    def get_thesis_config(self, thesis_id: str) -> dict | None:
        """The stored configuration, or None if only metadata is on file
        (theses backfilled from run history have no config)."""
        row = self.get_thesis(thesis_id)
        if not row or not row.get("config_json"):
            return None
        try:
            return json.loads(row["config_json"])
        except (TypeError, ValueError):
            return None

    def set_user_thesis(self, user_id: str, thesis_id: str) -> None:
        """Point one member at a thesis, without touching anyone else's.

        The multiplayer fix for the old single active thesis.yaml: one
        partner exploring a new space must not silently re-aim the other
        partner's workspace — or, worse, the scheduled run.
        """
        if not self.db["users"].exists():
            return
        with self.write_tx():
            self.db["users"].update(
                user_id.strip().lower(), {"active_thesis_id": thesis_id},
                alter=True,
            )

    def user_thesis_id(self, user_id: str) -> str | None:
        user = self.get_user(user_id) or {}
        return (user.get("active_thesis_id") or "").strip() or None

    def list_theses(self, include_archived: bool = False) -> list[dict]:
        """Registered theses with run/lead counts, most recently run first."""
        if not self.db["theses"].exists():
            return []
        where = "" if include_archived else "where t.archived_at is null"
        # A brand-new database has theses before it has runs or leads; the
        # counts degrade to zero rather than raising "no such table".
        has_runs = self.db["runs"].exists()
        has_leads = self.db["leads"].exists() and has_runs
        run_count = (
            "(select count(*) from runs r where r.thesis_id = t.id)" if has_runs else "0"
        )
        last_run = (
            "(select max(r.created_at) from runs r where r.thesis_id = t.id)"
            if has_runs else "null"
        )
        lead_count = (
            "(select count(distinct lower(l.handle)) from leads l where l.run_id in "
            "(select run_id from runs where thesis_id = t.id))" if has_leads else "0"
        )
        rows = self.db.execute(
            f"""
            select t.id, t.name, t.statement, t.current_version, t.is_active,
                   t.archived_at, {run_count} as run_count,
                   {last_run} as last_run_at, {lead_count} as lead_count
            from theses t {where}
            order by last_run_at desc nulls last, t.created_at desc
            """
        ).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "statement": r[2] or "",
                "current_version": r[3] or "", "is_active": bool(r[4]),
                "archived_at": r[5], "run_count": r[6] or 0,
                "last_run_at": r[7], "lead_count": r[8] or 0,
            }
            for r in rows
        ]

    def thesis_version_history(self, thesis_id: str) -> list[dict]:
        if not self.db["thesis_versions"].exists():
            return []
        rows = self.db.execute(
            """
            select v.version, v.created_at,
                   (select count(*) from runs r
                     where r.thesis_id = v.thesis_id and r.thesis_version = v.version)
            from thesis_versions v where v.thesis_id = ?
            order by v.created_at asc
            """,
            [thesis_id],
        ).fetchall()
        return [
            {"version": r[0], "created_at": r[1], "run_count": r[2] or 0, "n": i}
            for i, r in enumerate(rows, start=1)
        ]

    def set_active_thesis(self, thesis_id: str) -> None:
        if not self.db["theses"].exists():
            return
        with self.write_tx():
            self.db.execute("update theses set is_active = 0")
            self.db.execute(
                "update theses set is_active = 1 where id = ?", [thesis_id]
            )

    def active_thesis_id(self) -> str | None:
        if not self.db["theses"].exists():
            return None
        row = self.db.execute(
            "select id from theses where is_active = 1 limit 1"
        ).fetchone()
        return row[0] if row else None

    def archive_thesis(self, thesis_id: str, archived: bool = True) -> None:
        if not self.db["theses"].exists():
            return
        self.db.execute(
            "update theses set archived_at = ? where id = ?",
            [datetime.now(timezone.utc).isoformat() if archived else None, thesis_id],
        )
        self.db.conn.commit()

    def stale_handles(self, thesis_id: str, current_version: str = "") -> list[str]:
        """Handles whose newest score predates the thesis's current version.

        A lead is stale when the thesis has been retuned since it was scored —
        its number was produced by weights or a prompt that no longer apply.
        Kept read-only and cheap: it feeds a badge and a count, and rescoring
        only happens when the investor asks for it.
        """
        if not (self.db["leads"].exists() and self.db["runs"].exists()):
            return []
        version = current_version or (self.get_thesis(thesis_id) or {}).get(
            "current_version", ""
        )
        if not version:
            return []
        rows = self.db.execute(
            """
            with scored as (
              select lower(l.handle) as handle, r.thesis_version as version,
                     row_number() over (partition by lower(l.handle)
                                        order by l.created_at desc, l.run_id desc) as rn
              from leads l join runs r on r.run_id = l.run_id
              where r.thesis_id = ? and l.run_id not like 'demo-%'
            )
            select handle from scored where rn = 1 and version is not ?
            """,
            [thesis_id, version],
        ).fetchall()
        return [r[0] for r in rows]

    def backfill_verdict_provenance(self) -> int:
        """Attribute existing verdicts to the run that produced them.

        Verdicts recorded before provenance existed carry no thesis, so the
        "scored against" line would be blank for every startup already in the
        database — the whole point, invisible on exactly the data the investor
        has. The run a handle was last scored in names both the thesis and the
        version, so infer from there. Idempotent: only fills blanks.
        """
        if not (self.db["llm_verdicts"].exists() and self.db["leads"].exists()):
            return 0
        rows = self.db.execute(
            """
            with newest as (
              select lower(l.handle) as handle, r.thesis_id, r.thesis_version,
                     row_number() over (partition by lower(l.handle)
                                        order by l.created_at desc) as rn
              from leads l join runs r on r.run_id = l.run_id
              where r.thesis_id is not null and r.thesis_id != ''
            )
            select v.handle, n.thesis_id, n.thesis_version
            from llm_verdicts v join newest n on n.handle = v.handle and n.rn = 1
            where v.thesis_id is null or v.thesis_id = ''
            """
        ).fetchall()
        for handle, thesis_id, version in rows:
            self.db.execute(
                "update llm_verdicts set thesis_id = ?, thesis_version = ? "
                "where handle = ?",
                [thesis_id, version or "", handle],
            )
        self.db.conn.commit()
        return len(rows)

    def merge_thesis(self, from_id: str, to_id: str) -> int:
        """Repoint one thesis's runs onto another and retire the empty one.

        Naming a thesis that has already been running would otherwise strand
        its history: the runs were filed under a slug derived from the
        statement, and the moment it is given a proper id the two look like
        different theses. Same statement, same thesis — so adopt the history.
        """
        if from_id == to_id or not self.db["runs"].exists():
            return 0
        with self.write_tx():
            moved = self.db.execute(
                "select count(*) from runs where thesis_id = ?", [from_id]
            ).fetchone()[0]
            self.db.execute(
                "update runs set thesis_id = ? where thesis_id = ?", [to_id, from_id]
            )
            for table, column in (
                ("llm_verdicts", "thesis_id"),
                ("llm_verdict_history", "thesis_id"),
                ("pipeline", "sourced_thesis_id"),
            ):
                if self.db[table].exists() and column in self.db[table].columns_dict:
                    self.db.execute(
                        f"update {table} set {column} = ? where {column} = ?",
                        [to_id, from_id],
                    )
            if self.db["thesis_versions"].exists():
                self.db.execute(
                    "update or replace thesis_versions set thesis_id = ? "
                    "where thesis_id = ?",
                    [to_id, from_id],
                )
            if self.db["theses"].exists():
                self.db.execute("delete from theses where id = ?", [from_id])
        return moved

    def adopt_history(self, thesis_id: str, statement: str) -> int:
        """Claim any legacy runs of this same statement for `thesis_id`."""
        from scout.config import slugify

        legacy = slugify(statement)
        if not legacy or legacy == thesis_id:
            return 0
        row = self.db.execute(
            "select count(*) from runs where thesis_id = ?", [legacy]
        ).fetchone() if self.db["runs"].exists() else None
        return self.merge_thesis(legacy, thesis_id) if row and row[0] else 0

    def backfill_thesis_ids(self) -> int:
        """Give historical runs a thesis identity, derived from their statement.

        Runs recorded before identity existed carry only a strategy hash, which
        fragments one thesis across every tuning tweak it ever had. The
        statement is the field that stayed put, so grouping on it recovers the
        theses a human would name. Idempotent: only fills blanks.
        """
        if not self.db["runs"].exists():
            return 0
        from scout.config import slugify

        rows = self.db.execute(
            """
            select run_id, thesis_statement, strategy_hash from runs
            where (thesis_id is null or thesis_id = '') and run_id not like 'demo-%'
            """
        ).fetchall()
        seen: dict[str, tuple[str, str]] = {}
        for run_id, statement, strategy_hash in rows:
            slug = slugify(statement or "") or f"legacy-{(strategy_hash or '')[:8]}"
            seen[slug] = (statement or "", strategy_hash or "")
            self.db.execute(
                "update runs set thesis_id = ?, thesis_version = coalesce("
                "nullif(thesis_version, ''), strategy_hash) where run_id = ?",
                [slug, run_id],
            )
        self.db.conn.commit()
        for slug, (statement, _hash) in seen.items():
            latest = self.db.execute(
                "select thesis_version from runs where thesis_id = ? "
                "order by created_at desc limit 1",
                [slug],
            ).fetchone()
            self.upsert_thesis(
                slug,
                name=(statement or slug)[:60],
                statement=statement,
                version=(latest[0] if latest else "") or "",
            )
        return len(rows)

    def list_strategies(self) -> list[dict]:
        """Distinct sourcing strategies with run stats, newest-run first.

        A strategy = one thesis+seeds fingerprint; the statement is identical
        within a group by construction. Demo runs are excluded.
        """
        if not self.db["runs"].exists():
            return []
        rows = self.db.execute(
            """
            select strategy_hash,
                   max(thesis_statement) as thesis_statement,
                   count(*) as run_count,
                   min(created_at) as first_run_at,
                   max(created_at) as last_run_at
            from runs
            where run_id not like 'demo-%'
            group by strategy_hash
            order by last_run_at desc
            """
        ).fetchall()
        return [
            {
                "strategy_hash": r[0],
                "thesis_statement": r[1] or "",
                "run_count": r[2],
                "first_run_at": r[3],
                "last_run_at": r[4],
            }
            for r in rows
        ]

    def latest_run(self) -> dict | None:
        """The newest run row with its lead count — what a worker reports
        back and what a digest opens with."""
        if not self.db["runs"].exists():
            return None
        rows = list(self.db["runs"].rows_where(order_by="created_at desc", limit=1))
        if not rows:
            return None
        run = dict(rows[0])
        run["id"] = run.get("run_id")
        count = self.db.execute(
            "select count(*) from leads where run_id = ?", [run.get("run_id")]
        ).fetchone() if self.db["leads"].exists() else None
        run["n_leads"] = int(count[0]) if count else 0
        return run

    def last_real_run_at(self) -> datetime | None:
        """Timestamp of the newest non-demo run (None if only demo / nothing).

        Prefers the runs table; falls back to the leads table for legacy runs
        recorded before run provenance existed.
        """
        for table in ("runs", "leads"):
            if not self.db[table].exists():
                continue
            row = self.db.execute(
                f"select max(created_at) from {table} where run_id not like 'demo-%'"
            ).fetchone()
            if row and row[0]:
                ts = datetime.fromisoformat(row[0])
                return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return None

    # ----------------------------------------------------------------- ledger

    def load_lead_ledger(
        self,
        include_demo: bool = False,
        strategy_hash: str | None = None,
        thesis_id: str | None = None,
    ) -> list[LedgerEntry]:
        """Person-centric view: every handle's most recent Lead across ALL
        runs, with movement metadata (prev score, first/last seen, new-this-run).

        Partitioning is by lower(handle) — the leads pk is case-sensitive while
        the rest of the codebase treats handles NOCASE, so casings must merge.
        Ordering is (created_at, run_id) — never run_id alone (all rows in a
        run share one created_at; run_id prefixes sort lexicographically).
        verify- runs are always included: they are real re-scores.
        """
        if not self.db["leads"].exists():
            return []
        where = "1=1" if include_demo else "run_id not like 'demo-%'"
        params: list = []
        if strategy_hash:
            where += " and run_id in (select run_id from runs where strategy_hash = ?)"
            params.append(strategy_hash)
        if thesis_id:
            # Identity, not version: every run of this thesis, across all the
            # tuning it has been through.
            where += " and run_id in (select run_id from runs where thesis_id = ?)"
            params.append(thesis_id)
        sql = f"""
        with pool as (
          select run_id, handle, score, lead_json, created_at
          from leads where {where}
        ),
        ranked as (
          select run_id, handle, created_at,
                 row_number() over w as rn,
                 lead(score) over w as prev_score,
                 count(*) over p as times_seen,
                 min(created_at) over p as first_seen_at,
                 max(created_at) over p as last_seen_at,
                 first_value(run_id) over (
                     partition by lower(handle)
                     order by created_at asc, run_id asc
                     rows between unbounded preceding and unbounded following
                 ) as first_seen_run
          from pool
          window w as (partition by lower(handle) order by created_at desc, run_id desc),
                 p as (partition by lower(handle))
        )
        select l.lead_json, r.prev_score, r.first_seen_run, r.first_seen_at,
               r.last_seen_at, r.times_seen,
               r.first_seen_run = (
                   select run_id from pool
                   order by created_at desc, run_id desc limit 1
               ) as is_new
        from ranked r
        join pool l on l.run_id = r.run_id and l.handle = r.handle
        where r.rn = 1
        """
        entries = [
            LedgerEntry(
                lead=Lead.model_validate_json(row[0]),
                prev_score=row[1],
                first_seen_run=row[2] or "",
                first_seen_at=row[3],
                last_seen_at=row[4],
                times_seen=row[5] or 1,
                is_new=bool(row[6]),
            )
            for row in self.db.execute(sql, params).fetchall()
        ]
        entries.sort(key=lambda e: -e.lead.score)
        return entries

    # ----------------------------------------------------------- search cache

    def record_search(self, query: str, handles: list[str]) -> None:
        """Cache which account handles a paid search query surfaced."""
        self.db["searches"].upsert(
            {
                "query": query,
                "handles": json.dumps(handles),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            pk="query",
        )

    def cached_search(self, query: str, ttl_days: int) -> list[str] | None:
        """Handles from a previous run of `query`, or None if absent/expired."""
        if not self.db["searches"].exists():
            return None
        rows = list(self.db["searches"].rows_where("query = ?", [query], limit=1))
        if not rows:
            return None
        fetched = datetime.fromisoformat(rows[0]["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched >= timedelta(days=ttl_days):
            return None
        return json.loads(rows[0]["handles"])

    # ------------------------------------------------------------- site cache

    def record_site(self, page: SitePage) -> None:
        """Cache one fetched company site — failures too (negative caching)."""
        self.db["websites"].upsert(
            {
                "url": page.url,
                "final_url": page.final_url,
                "status": page.status,
                "text": page.text,
                "fetched_at": (page.fetched_at or datetime.now(timezone.utc)).isoformat(),
            },
            pk="url",
            alter=True,
        )

    def cached_site(self, url: str, ttl_days: int) -> SitePage | None:
        """Previously fetched site, or None when absent/expired. Failure rows
        expire after 1 day regardless of ttl_days, so a transient outage
        doesn't blind the classifier to a site for a whole TTL window."""
        if not self.db["websites"].exists():
            return None
        rows = list(self.db["websites"].rows_where("url = ?", [url], limit=1))
        if not rows:
            return None
        row = rows[0]
        fetched = datetime.fromisoformat(row["fetched_at"])
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        # ok/thin/non-html/too-large describe the site itself → full TTL;
        # error:* is likely transient → 1 day.
        settled = not (row.get("status") or "").startswith("error")
        effective_ttl = ttl_days if settled else min(1, ttl_days)
        if datetime.now(timezone.utc) - fetched >= timedelta(days=effective_ttl):
            return None
        return SitePage(
            url=row["url"], final_url=row.get("final_url") or "",
            status=row.get("status") or "", text=row.get("text") or "",
            fetched_at=fetched,
        )

    # ---------------------------------------------------- follow-graph edges

    def record_follow_snapshot(self, watcher: str, followee_handles: list[str]) -> None:
        """Upsert one watcher's following list as edges.

        X exposes no follow timestamps, so `first_seen` (our first observation
        of the edge) is the standard proxy every commercial tracker uses.
        A watcher's very first snapshot is the baseline: those edges are old
        follows of unknown age, distinguished via the follow_meta table.
        """
        watcher = watcher.lstrip("@").lower()
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            if not self._watcher_baseline(watcher):
                self.db["follow_meta"].insert(
                    {"watcher": watcher, "first_snapshot_at": now}, pk="watcher"
                )
            edges = self.db["follow_edges"]
            known: set[str] = set()
            if edges.exists():
                known = {
                    r["followee"]
                    for r in edges.rows_where("watcher = ?", [watcher])
                }
            # Two batched upserts (new edges / refreshes) instead of one
            # autocommitted upsert per followee — a watcher list is ~100 rows.
            # A refresh row carries last_seen only, so first_seen is preserved.
            seen: set[str] = set()
            new_rows: list[dict] = []
            refresh_rows: list[dict] = []
            for handle in followee_handles:
                followee = handle.lstrip("@").lower()
                if followee in seen:
                    continue
                seen.add(followee)
                if followee in known:
                    refresh_rows.append(
                        {"watcher": watcher, "followee": followee, "last_seen": now}
                    )
                else:
                    new_rows.append(
                        {
                            "watcher": watcher,
                            "followee": followee,
                            "first_seen": now,
                            "last_seen": now,
                        }
                    )
            if new_rows:
                edges.upsert_all(new_rows, pk=("watcher", "followee"))
            if refresh_rows:
                edges.upsert_all(refresh_rows, pk=("watcher", "followee"))

    def _watcher_baseline(self, watcher: str) -> str | None:
        if not self.db["follow_meta"].exists():
            return None
        rows = list(self.db["follow_meta"].rows_where("watcher = ?", [watcher], limit=1))
        return rows[0]["first_snapshot_at"] if rows else None

    def recent_watchers_for(self, handle: str, days: int) -> list[str]:
        """Watchers who NEWLY followed `handle` within the window.

        "Newly" = the edge's first_seen is inside the window AND strictly after
        that watcher's baseline snapshot (cold-start edges don't count — their
        real age is unknown).
        """
        if not self.db["follow_edges"].exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.db["follow_edges"].rows_where(
            "followee = ? and first_seen >= ?", [handle.lstrip("@").lower(), cutoff]
        )
        recent: list[str] = []
        for row in rows:
            baseline = self._watcher_baseline(row["watcher"])
            if baseline and row["first_seen"] > baseline:
                recent.append(row["watcher"])
        return recent

    def recent_watchers_map(self, days: int) -> dict[str, list[str]]:
        """Every followee's recent new watchers in ONE query — the batched
        form of `recent_watchers_for`, same rules (first_seen inside the
        window AND strictly after that watcher's baseline snapshot). The
        per-followee form scanned follow_edges once per candidate account,
        which made enrichment and the graph leg O(accounts × edges)."""
        if not (
            self.db["follow_edges"].exists() and self.db["follow_meta"].exists()
        ):
            return {}
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.db.execute(
            """
            select e.followee, e.watcher
            from follow_edges e
            join follow_meta m on m.watcher = e.watcher
            where e.first_seen >= ? and e.first_seen > m.first_snapshot_at
            """,
            [cutoff],
        ).fetchall()
        out: dict[str, list[str]] = {}
        for followee, watcher in rows:
            out.setdefault(followee, []).append(watcher)
        return out

    # ---------------------------------------------------------- verdict cache

    def record_verdict(
        self,
        handle: str,
        fingerprint: str,
        verdict: LLMVerdict,
        *,
        thesis_id: str = "",
        thesis_version: str = "",
    ) -> None:
        """Cache a Claude verdict keyed by handle + input fingerprint (a hash of
        bio, tweets, thesis, and model — see llm._fingerprint).

        The outgoing verdict is archived first. There is one row per handle, so
        rescoring under a new thesis would otherwise destroy the old judgement —
        and "this scored 0.20 under Edge AI, 0.75 under Novel Architectures" is
        exactly what makes a thesis change legible after the fact.
        """
        key = handle.lstrip("@").lower()
        # Callers in the classifier know the thesis but not the seeds, so they
        # cannot compute the version hash; the registry already knows what
        # version this thesis is at.
        if thesis_id and not thesis_version:
            thesis_version = (self.get_thesis(thesis_id) or {}).get(
                "current_version", ""
            ) or ""
        with self.write_tx():
            self._record_verdict_locked(
                key, fingerprint, verdict,
                thesis_id=thesis_id, thesis_version=thesis_version,
            )

    def _record_verdict_locked(
        self,
        key: str,
        fingerprint: str,
        verdict: LLMVerdict,
        *,
        thesis_id: str,
        thesis_version: str,
    ) -> None:
        """Archive-then-overwrite as one unit (caller holds write_tx)."""
        self._archive_verdict(key)
        self.db["llm_verdicts"].upsert(
            {
                "handle": key,
                "fingerprint": fingerprint,
                "verdict_json": verdict.model_dump_json(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "thesis_id": thesis_id,
                "thesis_version": thesis_version,
            },
            pk="handle",
            alter=True,
        )

    def _archive_verdict(self, key: str) -> None:
        """Copy the current verdict into history before it is overwritten."""
        if not self.db["llm_verdicts"].exists():
            return
        rows = list(self.db["llm_verdicts"].rows_where("handle = ?", [key], limit=1))
        if not rows:
            return
        prior = rows[0]
        self.db["llm_verdict_history"].insert(
            {
                "handle": key,
                "thesis_id": prior.get("thesis_id") or "",
                "thesis_version": prior.get("thesis_version") or "",
                "fingerprint": prior.get("fingerprint") or "",
                "verdict_json": prior.get("verdict_json") or "",
                "created_at": prior.get("created_at") or "",
                "archived_at": datetime.now(timezone.utc).isoformat(),
            },
            alter=True,
        )

    def verdict_provenance(self, handle: str) -> dict | None:
        """Which thesis and version produced this handle's CURRENT verdict."""
        if not self.db["llm_verdicts"].exists():
            return None
        rows = list(
            self.db["llm_verdicts"].rows_where(
                "handle = ?", [handle.lstrip("@").lower()], limit=1
            )
        )
        if not rows:
            return None
        return {
            "thesis_id": rows[0].get("thesis_id") or "",
            "thesis_version": rows[0].get("thesis_version") or "",
            "created_at": rows[0].get("created_at") or "",
        }

    def verdict_history(self, handle: str, limit: int = 10) -> list[dict]:
        """Prior verdicts for a handle, newest first — what it used to score."""
        if not self.db["llm_verdict_history"].exists():
            return []
        rows = list(
            self.db["llm_verdict_history"].rows_where(
                "handle = ?", [handle.lstrip("@").lower()],
                order_by="archived_at desc", limit=limit,
            )
        )
        out: list[dict] = []
        for row in rows:
            try:
                verdict = LLMVerdict.model_validate_json(row["verdict_json"])
            except (ValidationError, ValueError):
                continue
            out.append({
                "verdict": verdict,
                "thesis_id": row.get("thesis_id") or "",
                "thesis_version": row.get("thesis_version") or "",
                "created_at": row.get("created_at") or "",
            })
        return out

    def cached_verdict(
        self, handle: str, fingerprint: str, ttl_days: int
    ) -> LLMVerdict | None:
        """A previously cached verdict, or None when absent, stale, or the
        inputs changed (fingerprint mismatch)."""
        if ttl_days <= 0 or not self.db["llm_verdicts"].exists():
            return None
        rows = list(
            self.db["llm_verdicts"].rows_where(
                "handle = ?", [handle.lstrip("@").lower()], limit=1
            )
        )
        if not rows or rows[0]["fingerprint"] != fingerprint:
            return None
        created = datetime.fromisoformat(rows[0]["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - created >= timedelta(days=ttl_days):
            return None
        return LLMVerdict.model_validate_json(rows[0]["verdict_json"])

    # ------------------------------------------------------------ bio history

    def record_bio(self, handle: str, bio: str) -> str | None:
        """Snapshot a bio if it changed; return the PREVIOUS bio (None if first sighting)."""
        handle = handle.lstrip("@").lower()
        with self.write_tx():
            latest = None
            if self.db["bio_snapshots"].exists():
                rows = list(self.db["bio_snapshots"].rows_where(
                    "handle = ?", [handle], order_by="seen_at desc", limit=1
                ))
                latest = rows[0]["bio"] if rows else None
            if latest == bio:
                return latest  # unchanged — previous == current, no change seen
            self.db["bio_snapshots"].insert(
                {
                    "handle": handle,
                    "bio": bio,
                    "seen_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return latest

    # --------------------------------------------------------- unlinked leads

    def upsert_unlinked_leads(self, leads: list[UnlinkedLead]) -> None:
        if leads:
            self.db["unlinked_leads"].upsert_all(
                [lead.model_dump(mode="json") for lead in leads],
                pk=("source", "ref"),
            )

    def load_unlinked_leads(self, days: int = 30) -> list[UnlinkedLead]:
        if not self.db["unlinked_leads"].exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.db["unlinked_leads"].rows_where(
            "found_at >= ?", [cutoff], order_by="found_at desc"
        )
        return [UnlinkedLead.model_validate(dict(r)) for r in rows]

    # ------------------------------------------------------- deal-flow pipeline

    def set_pipeline(
        self,
        handle: str,
        *,
        status: str | None = None,
        notes: str | None = None,
        outreach: str | None = None,
        channel: str | None = None,
        brief: str | None = None,
        brief_edited: bool = False,
        brief_meta: dict | None = None,
        sourced_thesis_id: str | None = None,
        actor: str | None = None,
        expect_updated_at: str | None = None,
    ) -> bool:
        """Upsert deal-flow state for one lead (read-merge-write so partial
        updates never clobber the other fields).

        `brief` holds the investment memo markdown. A generation
        (brief_edited=False) stamps brief_at and clears brief_edited_at; a
        manual edit (brief_edited=True) stamps brief_edited_at and keeps the
        original generation time. `brief_meta` (depth, sources, searches —
        agents.investment_memo's meta) is stored as brief_meta_json alongside
        a generation and left untouched by edits.

        Runs inside write_tx so two sessions updating the same startup merge
        instead of clobbering; `updated_by` records the writer. Status and
        notes changes append events when an actor is bound. Returns False
        (writing nothing) when `expect_updated_at` is given and another
        session updated the row since — the optimistic-concurrency path for
        the free-text notes editor."""
        handle = handle.lstrip("@").lower()
        who = actor or self.actor
        with self.write_tx():
            row = {"handle": handle}
            if self.db["pipeline"].exists():
                existing = list(
                    self.db["pipeline"].rows_where("handle = ?", [handle], limit=1)
                )
                if existing:
                    row = dict(existing[0])
            if (
                expect_updated_at is not None
                and (row.get("updated_at") or "") not in ("", expect_updated_at)
            ):
                return False
            old_status = row.get("status") or "new"
            old_notes = row.get("notes") or ""
            now = datetime.now(timezone.utc).isoformat()
            if status is not None:
                row["status"] = status
            if notes is not None:
                row["notes"] = notes
            if outreach is not None:
                row["outreach"] = outreach
                row["outreach_at"] = now
            if channel is not None:
                row["channel"] = channel
            if brief is not None:
                row["brief"] = brief
                if brief_edited:
                    row["brief_edited_at"] = now
                else:
                    row["brief_at"] = now
                    row["brief_edited_at"] = None
            if brief_meta is not None:
                row["brief_meta_json"] = json.dumps(brief_meta)
            # Which thesis brought this company in. Triage itself stays global —
            # a company you shortlisted is shortlisted, whichever thesis found
            # it — but the attribution answers "why is this in my pipeline".
            # Only ever set once, so a later thesis cannot rewrite the origin.
            if sourced_thesis_id and not row.get("sourced_thesis_id"):
                row["sourced_thesis_id"] = sourced_thesis_id
            row["updated_at"] = now
            row["updated_by"] = who or ""
            self.db["pipeline"].upsert(row, pk="handle", alter=True)
            # Judgment events (actor-bound writers only — legacy callers
            # without an actor still work, they just leave no feed trail).
            if who:
                if status is not None and status != old_status:
                    self._append_event(
                        "status_changed", handle=handle, actor=who,
                        payload={"old": old_status, "new": status},
                    )
                if notes is not None and notes != old_notes:
                    self._append_event("notes_edited", handle=handle, actor=who)
        return True

    @staticmethod
    def _decode_pipeline(row: dict) -> dict:
        row["brief_meta"] = json.loads(row.get("brief_meta_json") or "{}")
        return row

    def get_pipeline(self, handle: str) -> dict:
        handle = handle.lstrip("@").lower()
        if not self.db["pipeline"].exists():
            return {}
        rows = list(self.db["pipeline"].rows_where("handle = ?", [handle], limit=1))
        return self._decode_pipeline(dict(rows[0])) if rows else {}

    def all_pipeline(self) -> dict[str, dict]:
        """All deal-flow rows keyed by lowercased handle."""
        if not self.db["pipeline"].exists():
            return {}
        return {r["handle"]: self._decode_pipeline(dict(r))
                for r in self.db["pipeline"].rows}

    def pipeline_counts(self) -> dict[str, int]:
        """Count of leads at each status (only statuses that appear)."""
        if not self.db["pipeline"].exists():
            return {}
        if "status" not in self.db["pipeline"].columns_dict:
            # Rows exist (notes/outreach) but nothing ever set a status.
            n = self.db.execute("select count(*) from pipeline").fetchone()[0]
            return {"new": n} if n else {}
        rows = self.db.execute(
            "select coalesce(nullif(status, ''), 'new'), count(*) "
            "from pipeline group by 1"
        ).fetchall()
        return {status: n for status, n in rows}

    # ------------------------------------------------------- activity events

    def _append_event(
        self,
        verb: str,
        *,
        handle: str | None = None,
        thesis_id: str | None = None,
        payload: dict | None = None,
        actor: str | None = None,
    ) -> None:
        """Append one row to the activity spine.

        Called INSIDE the same write_tx as the state write it describes, so
        an event exists iff its state change committed. Events power the
        activity feed, unread badges, Slack notifications, and audit — state
        is never derived from them. An event without an author is a bug;
        callers that may run actor-less (legacy CLI paths) skip emission
        rather than calling with None."""
        who = actor or self.actor
        if not who:
            raise ValueError(f"event {verb!r} needs an actor — bind store.actor")
        self.db["events"].insert(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "actor": who,
                "verb": verb,
                "handle": (handle or "").lstrip("@").lower() or None,
                "thesis_id": thesis_id or None,
                "payload_json": json.dumps(payload or {}),
                "notified": 0,
            },
            alter=True,
        )

    def events(
        self,
        *,
        limit: int = 200,
        handle: str | None = None,
        actor: str | None = None,
        verbs: list[str] | None = None,
        before_id: int | None = None,
        since: datetime | None = None,
    ) -> list[Event]:
        """Newest-first slice of the activity feed, with optional filters."""
        if not self.db["events"].exists():
            return []
        where = ["1=1"]
        params: list = []
        if since is not None:
            where.append("at >= ?")
            params.append(since.isoformat())
        if handle:
            where.append("handle = ?")
            params.append(handle.lstrip("@").lower())
        if actor:
            where.append("actor = ?")
            params.append(actor)
        if verbs:
            where.append(f"verb in ({','.join('?' * len(verbs))})")
            params.extend(verbs)
        if before_id is not None:
            where.append("id < ?")
            params.append(before_id)
        rows = self.db.execute(
            f"select id, at, actor, verb, handle, thesis_id, payload_json "
            f"from events where {' and '.join(where)} "
            f"order by id desc limit {int(limit)}",
            params,
        ).fetchall()
        return [
            Event(
                id=r[0], at=r[1], actor=r[2], verb=r[3], handle=r[4],
                thesis_id=r[5], payload=json.loads(r[6] or "{}"),
            )
            for r in rows
        ]

    def latest_event_id(self) -> int:
        if not self.db["events"].exists():
            return 0
        row = self.db.execute("select coalesce(max(id), 0) from events").fetchone()
        return int(row[0])

    def unread_count(self, actor: str) -> int:
        """Events since this member's read cursor, excluding their own."""
        if not self.db["events"].exists():
            return 0
        cursor = 0
        if self.db["read_cursors"].exists():
            row = self.db.execute(
                "select last_event_id from read_cursors where actor = ?", [actor]
            ).fetchone()
            cursor = int(row[0]) if row and row[0] else 0
        return int(self.db.execute(
            "select count(*) from events where id > ? and actor != ?",
            [cursor, actor],
        ).fetchone()[0])

    def mark_read(self, actor: str) -> None:
        with self.write_tx():
            self.db["read_cursors"].upsert(
                {
                    "actor": actor,
                    "last_event_id": self.latest_event_id(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                pk="actor",
                alter=True,
            )

    def unnotified_events(self, limit: int = 100) -> list[Event]:
        """Oldest-first events the notifier has not yet dispatched."""
        if not self.db["events"].exists():
            return []
        rows = self.db.execute(
            "select id, at, actor, verb, handle, thesis_id, payload_json "
            "from events where notified = 0 order by id asc limit ?",
            [limit],
        ).fetchall()
        return [
            Event(id=r[0], at=r[1], actor=r[2], verb=r[3], handle=r[4],
                  thesis_id=r[5], payload=json.loads(r[6] or "{}"))
            for r in rows
        ]

    def mark_notified(self, event_ids: list[int]) -> None:
        if not event_ids or not self.db["events"].exists():
            return
        with self.write_tx():
            self.db.execute(
                f"update events set notified = 1 "
                f"where id in ({','.join('?' * len(event_ids))})",
                event_ids,
            )

    # ------------------------------------------------------------------ votes

    def set_vote(
        self,
        handle: str,
        stance: str,
        rationale: str = "",
        *,
        thesis_id: str = "",
        actor: str | None = None,
    ) -> None:
        """One partner's stance on one startup. Re-voting updates the row
        (history is the vote_cast events); an empty stance clears it."""
        from scout.collab import STANCES

        who = actor or self.actor
        if not who:
            raise ValueError("a vote needs an author — bind store.actor")
        key = handle.lstrip("@").lower()
        if not stance:
            with self.write_tx():
                if self.db["votes"].exists():
                    self.db.execute(
                        "delete from votes where handle = ? and actor = ?",
                        [key, who],
                    )
                    self._append_event(
                        "vote_cleared", handle=key, actor=who, payload={}
                    )
            return
        if stance not in STANCES:
            raise ValueError(f"unknown stance: {stance!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            created = now
            if self.db["votes"].exists():
                prior = list(self.db["votes"].rows_where(
                    "handle = ? and actor = ?", [key, who], limit=1))
                if prior:
                    created = prior[0].get("created_at") or now
            self.db["votes"].upsert(
                {
                    "handle": key, "actor": who, "stance": stance,
                    "rationale": rationale.strip(),
                    "thesis_id": thesis_id or "",
                    "created_at": created, "updated_at": now,
                },
                pk=("handle", "actor"),
                alter=True,
            )
            self._append_event(
                "vote_cast", handle=key, thesis_id=thesis_id or None, actor=who,
                payload={"stance": stance, "rationale": rationale.strip()},
            )

    def votes_for(self, handle: str) -> list[Vote]:
        if not self.db["votes"].exists():
            return []
        return [
            Vote.model_validate(dict(r))
            for r in self.db["votes"].rows_where(
                "handle = ?", [handle.lstrip("@").lower()], order_by="created_at"
            )
        ]

    def all_votes(self) -> dict[str, list[Vote]]:
        """Every startup's votes, keyed by lowercased handle."""
        if not self.db["votes"].exists():
            return {}
        out: dict[str, list[Vote]] = {}
        for r in self.db["votes"].rows_where(order_by="handle, created_at"):
            vote = Vote.model_validate(dict(r))
            out.setdefault(vote.handle, []).append(vote)
        return out

    # --------------------------------------------------------------- comments

    def add_comment(
        self,
        handle: str,
        body: str,
        *,
        mentions: list[str] | None = None,
        memo_version_id: int | None = None,
        actor: str | None = None,
    ) -> int:
        """Append one comment; returns its id. Mentions are precomputed by
        the caller (collab.parse_mentions) so the event payload can drive
        notification pings."""
        who = actor or self.actor
        if not who:
            raise ValueError("a comment needs an author — bind store.actor")
        body = body.strip()
        if not body:
            raise ValueError("empty comment")
        key = handle.lstrip("@").lower()
        with self.write_tx():
            comment_id = (
                self.db["comments"]
                .insert(
                    {
                        "handle": key,
                        "actor": who,
                        "body": body,
                        "mentions_json": json.dumps(mentions or []),
                        "memo_version_id": memo_version_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "edited_at": None,
                        "deleted_at": None,
                    },
                    alter=True,
                )
                .last_pk
            )
            self._append_event(
                "comment_added", handle=key, actor=who,
                payload={
                    "comment_id": comment_id,
                    "mentions": mentions or [],
                    "memo_version_id": memo_version_id,
                    "preview": body[:140],
                },
            )
        return int(comment_id)

    def delete_comment(self, comment_id: int, *, actor: str | None = None) -> None:
        """Soft delete (threads keep their shape); the UI enforces who may."""
        if not self.db["comments"].exists():
            return
        with self.write_tx():
            self.db.execute(
                "update comments set deleted_at = ? where id = ?",
                [datetime.now(timezone.utc).isoformat(), comment_id],
            )

    def comments_for(self, handle: str, include_deleted: bool = False) -> list[Comment]:
        if not self.db["comments"].exists():
            return []
        where = "handle = ?" + ("" if include_deleted else " and deleted_at is null")
        out = []
        for r in self.db["comments"].rows_where(
            where, [handle.lstrip("@").lower()], order_by="id"
        ):
            row = dict(r)
            row["mentions"] = json.loads(row.pop("mentions_json", None) or "[]")
            out.append(Comment.model_validate(row))
        return out

    def all_comment_counts(self) -> dict[str, int]:
        if not self.db["comments"].exists():
            return {}
        rows = self.db.execute(
            "select handle, count(*) from comments "
            "where deleted_at is null group by handle"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ----------------------------------------------------------- memo history

    def set_memo(
        self,
        handle: str,
        body: str,
        *,
        meta: dict | None = None,
        kind: str = "generated",
        actor: str | None = None,
        restored_from: int | None = None,
    ) -> int:
        """Write the current memo AND an immutable version snapshot, atomically.

        pipeline.brief stays the current memo (every existing read path —
        Memos page, export, PDF — is untouched); memo_versions accumulates
        history, which is what makes regeneration safe: a human's edit is
        version n, a regeneration is version n+1, and any version can be
        re-promoted. Returns the new version_no."""
        if kind not in ("generated", "edited"):
            raise ValueError(f"unknown memo kind: {kind!r}")
        who = actor or self.actor or ("agent:memo" if kind == "generated" else "")
        if not who:
            raise ValueError("a memo write needs an author — bind store.actor")
        key = handle.lstrip("@").lower()
        with self.write_tx():
            version_no = 1
            if self.db["memo_versions"].exists():
                row = self.db.execute(
                    "select coalesce(max(version_no), 0) from memo_versions "
                    "where handle = ?",
                    [key],
                ).fetchone()
                version_no = int(row[0]) + 1
            self.db["memo_versions"].insert(
                {
                    "handle": key,
                    "version_no": version_no,
                    "body": body,
                    "meta_json": json.dumps(meta or {}),
                    "author": who,
                    "kind": kind,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                alter=True,
            )
            self.set_pipeline(
                key,
                brief=body,
                brief_edited=(kind == "edited"),
                brief_meta=meta if kind == "generated" else None,
                actor=who,
            )
            payload: dict = {"version_no": version_no}
            if restored_from is not None:
                payload["restored_from"] = restored_from
            self._append_event(
                "memo_generated" if kind == "generated" else "memo_edited",
                handle=key, actor=who, payload=payload,
            )
        return version_no

    def memo_versions(self, handle: str, limit: int = 25) -> list[MemoVersion]:
        """Version headers newest-first (bodies included — memos are text)."""
        if not self.db["memo_versions"].exists():
            return []
        out = []
        for r in self.db["memo_versions"].rows_where(
            "handle = ?", [handle.lstrip("@").lower()],
            order_by="version_no desc", limit=limit,
        ):
            row = dict(r)
            row["meta"] = json.loads(row.pop("meta_json", None) or "{}")
            out.append(MemoVersion.model_validate(row))
        return out

    def restore_memo_version(self, handle: str, version_no: int,
                             *, actor: str | None = None) -> int | None:
        """Re-promote an old version as the current memo (as a NEW version —
        history is append-only). Returns the new version_no."""
        versions = {v.version_no: v for v in self.memo_versions(handle, limit=1000)}
        old = versions.get(version_no)
        if old is None:
            return None
        return self.set_memo(
            handle, old.body, meta=old.meta or None, kind="edited",
            actor=actor, restored_from=version_no,
        )

    def set_memo_review(
        self,
        handle: str,
        status: str,
        *,
        version_no: int | None = None,
        actor: str | None = None,
    ) -> None:
        """Memo review state on the pipeline row: none / requested /
        approved / changes_requested. Approval pins a version_no, so the UI
        can flag a memo edited after its approval."""
        if status not in ("none", "requested", "approved", "changes_requested"):
            raise ValueError(f"unknown review status: {status!r}")
        who = actor or self.actor
        if not who:
            raise ValueError("a review action needs an author")
        key = handle.lstrip("@").lower()
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            row = {"handle": key}
            if self.db["pipeline"].exists():
                existing = list(self.db["pipeline"].rows_where(
                    "handle = ?", [key], limit=1))
                if existing:
                    row = dict(existing[0])
            row["memo_review_status"] = status
            if status == "requested":
                row["memo_review_requested_by"] = who
            elif status in ("approved", "changes_requested"):
                row["memo_reviewed_by"] = who
                row["memo_reviewed_at"] = now
                if status == "approved" and version_no is not None:
                    row["memo_approved_version"] = int(version_no)
            row["updated_at"] = now
            row["updated_by"] = who
            self.db["pipeline"].upsert(row, pk="handle", alter=True)
            verb = {
                "requested": "memo_review_requested",
                "approved": "memo_approved",
                "changes_requested": "memo_changes_requested",
                "none": "memo_review_cleared",
            }[status]
            self._append_event(verb, handle=key, actor=who,
                               payload={"version_no": version_no})

    # ------------------------------------------------------------- assignment

    def set_assignment(self, handle: str, assignee: str | None,
                       *, actor: str | None = None) -> None:
        """Give a startup an owner (empty/None clears). History = events."""
        who = actor or self.actor
        if not who:
            raise ValueError("an assignment needs an author")
        key = handle.lstrip("@").lower()
        target = (assignee or "").strip().lower()
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            row = {"handle": key}
            if self.db["pipeline"].exists():
                existing = list(self.db["pipeline"].rows_where(
                    "handle = ?", [key], limit=1))
                if existing:
                    row = dict(existing[0])
            row.update(assignee=target, assigned_by=who,
                       assigned_at=now if target else None,
                       updated_at=now, updated_by=who)
            self.db["pipeline"].upsert(row, pk="handle", alter=True)
            self._append_event(
                "assigned" if target else "unassigned",
                handle=key, actor=who, payload={"assignee": target},
            )

    # -------------------------------------------------------------- migration

    def migrate_multiplayer(self, owner: str) -> dict[str, int]:
        """Adopt a single-user database into the multiplayer schema.

        Idempotent — safe to re-run, and additive only (the pre-migration
        copy stays a valid rollback). Three backfills:

        1. Attribution: every judgment already in the database was made by
           the person who ran Scout alone, so stamp `owner` on the rows that
           have no author. Honest, not invented.
        2. Memo history: snapshot each existing memo as version 1, so the
           first regeneration after the upgrade cannot destroy work that
           predates versioning.
        3. Votes from triage: a shortlisted startup was a yes and a passed
           one was a pass — importing them (marked as imports, not as
           freshly-cast opinions) means disagreement and taste features have
           real data on day one instead of an empty slate.
        """
        from scout.status import POSITIVE_STATUSES

        owner = owner.strip().lower()
        if not owner:
            raise ValueError("migrate_multiplayer needs the owner's email")
        counts = {"attributed": 0, "memo_versions": 0, "votes": 0}
        self.ensure_user(owner)
        now = datetime.now(timezone.utc).isoformat()

        with self.write_tx():
            # 1. Attribution for pre-multiplayer judgment rows.
            for table, column in (
                ("pipeline", "updated_by"),
                ("score_overrides", "actor"),
                ("startup_attrs", "updated_by"),
            ):
                if not self.db[table].exists():
                    continue
                if column not in self.db[table].columns_dict:
                    self.db[table].add_column(column, str)
                cursor = self.db.execute(
                    f"update {table} set {column} = ? "
                    f"where {column} is null or {column} = ''",
                    [owner],
                )
                counts["attributed"] += cursor.rowcount if cursor.rowcount > 0 else 0

            if not self.db["pipeline"].exists():
                return counts

            # 2. Existing memos become version 1 (only where none exists).
            # `brief` is created lazily by the first memo write, so a
            # database where nobody has generated one yet has no column.
            versioned = {
                r[0] for r in self.db.execute(
                    "select distinct handle from memo_versions"
                ).fetchall()
            }
            memo_rows = (
                [
                    dict(r) for r in self.db["pipeline"].rows_where(
                        "brief is not null and brief != ''"
                    )
                ]
                if "brief" in self.db["pipeline"].columns_dict
                else []
            )
            for row in memo_rows:
                handle = row["handle"]
                if handle in versioned:
                    continue
                edited = bool(row.get("brief_edited_at"))
                self.db["memo_versions"].insert(
                    {
                        "handle": handle,
                        "version_no": 1,
                        "body": row.get("brief") or "",
                        "meta_json": row.get("brief_meta_json") or "{}",
                        # An edited memo is the human's; an untouched one is
                        # the agent's.
                        "author": owner if edited else "agent:memo",
                        "kind": "edited" if edited else "generated",
                        "created_at": (
                            row.get("brief_edited_at") or row.get("brief_at") or now
                        ),
                    },
                    alter=True,
                )
                counts["memo_versions"] += 1

            # 3. Triage becomes the owner's votes (never overwriting a real one).
            voted = set()
            if self.db["votes"].exists():
                voted = {
                    r[0] for r in self.db.execute(
                        "select handle from votes where actor = ?", [owner]
                    ).fetchall()
                }
            imports: list[dict] = []
            for row in self.db["pipeline"].rows:
                handle = row["handle"]
                status = row.get("status") or "new"
                if handle in voted:
                    continue
                if status in POSITIVE_STATUSES:
                    stance = "yes"
                elif status == "passed":
                    stance = "pass"
                else:
                    continue
                imports.append({
                    "handle": handle, "actor": owner, "stance": stance,
                    "rationale": f"imported from triage ({status})",
                    "thesis_id": row.get("sourced_thesis_id") or "",
                    "created_at": row.get("updated_at") or now,
                    "updated_at": row.get("updated_at") or now,
                })
            if imports:
                self.db["votes"].upsert_all(
                    imports, pk=("handle", "actor"), alter=True
                )
                counts["votes"] = len(imports)
                # One event for the import as a whole — not N fake vote_cast
                # events that would flood the activity feed on day one.
                self._append_event(
                    "votes_imported", actor=owner,
                    payload={"count": len(imports)},
                )
        return counts

    # ------------------------------------------- startup columns & attributes

    # The user-owned data layer behind the Database page: a column schema the
    # user controls (curated selects, custom fields) + per-startup values.
    # Types: select | multiselect | text | number | checkbox.

    def ensure_default_columns(self, defaults: list[dict]) -> None:
        """Seed the schema ONCE — only when the table has never existed.

        Deleting a builtin column must stick, so presence of the table (even
        emptied) suppresses re-seeding. `defaults`: [{key, label, type,
        options}] in display order."""
        if self.db["startup_columns"].exists():
            return
        for position, column in enumerate(defaults):
            self.save_column(
                column["key"], column["label"], column["type"],
                options=column.get("options") or [],
                builtin=True, position=position,
                ai_fill=column.get("ai_fill", True),
            )

    def save_column(
        self,
        key: str,
        label: str,
        col_type: str,
        *,
        options: list[str] | None = None,
        builtin: bool = False,
        position: int | None = None,
        ai_fill: bool = True,
    ) -> None:
        """Create or update one schema column (updating = editing options or
        label; the key is stable). Position defaults to end-of-list.
        `ai_fill=False` marks judgment columns (e.g. Priority) that
        auto-categorization must never fill."""
        key = key.strip()
        if not key:
            raise ValueError("column key must be non-empty")
        with self.write_tx():
            self._save_column_locked(
                key, label, col_type, options=options, builtin=builtin,
                position=position, ai_fill=ai_fill,
            )

    def _save_column_locked(
        self,
        key: str,
        label: str,
        col_type: str,
        *,
        options: list[str] | None,
        builtin: bool,
        position: int | None,
        ai_fill: bool,
    ) -> None:
        if position is None:
            existing = [c["position"] for c in self.list_columns()]
            position = (max(existing) + 1) if existing else 0
        self.db["startup_columns"].upsert(
            {
                "key": key,
                "label": label.strip() or key,
                "type": col_type,
                "options_json": json.dumps(options or []),
                "builtin": 1 if builtin else 0,
                "position": int(position),
                "ai_fill": 1 if ai_fill else 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            pk="key",
            alter=True,
            columns={"builtin": int, "position": int, "ai_fill": int},
        )

    def delete_column(self, key: str) -> None:
        """Drop a column from the schema. Values under the key are left in
        startup_attrs and simply never rendered again (harmless; a re-added
        column with the same key would resurface them)."""
        if self.db["startup_columns"].exists():
            self.db.execute("delete from startup_columns where key = ?", [key])
            self.db.conn.commit()

    def list_columns(self) -> list[dict]:
        """Schema columns in display order, options decoded."""
        if not self.db["startup_columns"].exists():
            return []
        rows = []
        for r in self.db["startup_columns"].rows_where(
                order_by="position, created_at"):
            row = dict(r)
            row["options"] = json.loads(row.get("options_json") or "[]")
            row["builtin"] = bool(row.get("builtin"))
            # Rows written before the flag existed default to AI-fillable.
            row["ai_fill"] = bool(row["ai_fill"]) if "ai_fill" in row else True
            rows.append(row)
        return rows

    def set_attrs(self, handle: str, changes: dict, *, actor: str | None = None) -> None:
        """Merge per-startup attribute values (column key → value). A None
        value deletes the key. Read-merge-write inside write_tx, like
        set_pipeline; `updated_by` records the writer."""
        handle = handle.lstrip("@").lower()
        with self.write_tx():
            values: dict = {}
            if self.db["startup_attrs"].exists():
                rows = list(self.db["startup_attrs"].rows_where(
                    "handle = ?", [handle], limit=1))
                if rows:
                    values = json.loads(rows[0].get("values_json") or "{}")
            for key, value in changes.items():
                if value is None:
                    values.pop(key, None)
                else:
                    values[key] = value
            self.db["startup_attrs"].upsert(
                {
                    "handle": handle,
                    "values_json": json.dumps(values),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "updated_by": actor or self.actor or "",
                },
                pk="handle",
                alter=True,
            )
            if actor or self.actor:
                self._append_event(
                    "attrs_changed", handle=handle, actor=actor,
                    payload={"keys": sorted(changes)},
                )

    def all_attrs(self) -> dict[str, dict]:
        """Every startup's attribute values, keyed by lowercased handle."""
        if not self.db["startup_attrs"].exists():
            return {}
        return {
            r["handle"]: json.loads(r.get("values_json") or "{}")
            for r in self.db["startup_attrs"].rows
        }

    # --------------------------------------------------------- score overrides

    def set_override(
        self,
        handle: str,
        *,
        quality: dict[str, float] | None = None,
        sections: dict[str, float] | None = None,
        fit: float | None = None,
        score: float | None = None,
        note: str = "",
        actor: str | None = None,
    ) -> None:
        """Persist the investor's manual scoring for one lead.

        `sections` maps scorecard section key → 0..100 (only the sections
        the investor actually adjusted); `quality` is the legacy dim → 0..1
        shape, applied only to pre-scorecard verdicts; `fit` overrides
        thesis_fit (0..1); `score` pins the FINAL score outright (0..100).
        Passing everything as None/empty clears the row — an override that
        overrides nothing shouldn't linger. Values replace the whole row
        (the UI form always submits its full current state); `actor` records
        whose numbers these are."""
        handle = handle.lstrip("@").lower()
        if not quality and not sections and fit is None and score is None:
            self.clear_override(handle)
            return
        with self.write_tx():
            self.db["score_overrides"].upsert(
                {
                    "handle": handle,
                    "quality_json": json.dumps(quality or {}),
                    "sections_json": json.dumps(sections or {}),
                    "fit": fit,
                    "score": score,
                    "note": note,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "actor": actor or self.actor or "",
                },
                pk="handle",
                alter=True,
                # Explicit REAL affinity: a None on the first write would
                # otherwise type these TEXT and hand back "88.0" strings.
                columns={"fit": float, "score": float},
            )
            if actor or self.actor:
                self._append_event(
                    "override_set", handle=handle, actor=actor,
                    payload={"fit": fit, "score": score, "note": note},
                )

    def clear_override(self, handle: str) -> None:
        handle = handle.lstrip("@").lower()
        if self.db["score_overrides"].exists():
            self.db.execute("delete from score_overrides where handle = ?", [handle])
            self.db.conn.commit()

    def all_overrides(self) -> dict[str, dict]:
        """All manual-score rows keyed by lowercased handle; quality_json /
        sections_json are decoded into `quality` / `sections` for callers."""
        if not self.db["score_overrides"].exists():
            return {}
        out: dict[str, dict] = {}
        for r in self.db["score_overrides"].rows:
            row = dict(r)
            row["quality"] = json.loads(row.get("quality_json") or "{}")
            row["sections"] = json.loads(row.get("sections_json") or "{}")
            # Defensive float coercion — rows written before the explicit
            # column typing may carry TEXT values.
            for key in ("fit", "score"):
                row[key] = (float(row[key])
                            if row.get(key) not in (None, "") else None)
            out[row["handle"]] = row
        return out

    # ------------------------------------------------------------- backtests

    def save_backtest(self, report: dict) -> int:
        """Keep every backtest run.

        Worth storing rather than just writing a file: a backtest's value
        compounds when you can show the numbers moving as the thesis is
        tuned, and an investor asking "has this got better?" deserves a
        series rather than one screenshot.
        """
        metrics = report.get("metrics") or {}
        return int(self.db["backtests"].insert(
            {
                "cutoff": report.get("cutoff"),
                "thesis_id": report.get("thesis_id") or "",
                "threshold": report.get("threshold"),
                "n_outcomes": len(report.get("verdicts") or []),
                "n_controls": len(report.get("controls") or []),
                "recall": metrics.get("recall"),
                "auc": metrics.get("auc"),
                "report_json": json.dumps(report, default=str),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_by": self.actor or "",
            },
            alter=True,
        ).last_pk)

    def backtests(self, limit: int = 20) -> list[dict]:
        if not self.db["backtests"].exists():
            return []
        rows = self.db["backtests"].rows_where(order_by="id desc", limit=limit)
        out = []
        for row in rows:
            row = dict(row)
            try:
                row["report"] = json.loads(row.get("report_json") or "{}")
            except (TypeError, ValueError):
                row["report"] = {}
            out.append(row)
        return out

    # ------------------------------------------------------------- job queue

    def _ensure_job_tables(self) -> None:
        """Explicit creation for the same reason as the collab tables: rows
        never carry the auto-assigned id the queue orders and claims by."""
        self.db["jobs"].create(
            {
                "id": int, "kind": str, "payload_json": str, "status": str,
                "priority": int, "attempts": int, "max_attempts": int,
                "requested_by": str, "schedule_id": int,
                "created_at": str, "run_after": str, "started_at": str,
                "finished_at": str, "heartbeat_at": str, "lease_expires_at": str,
                "worker_id": str, "result_json": str, "error": str,
                "log_path": str,
            },
            pk="id",
            if_not_exists=True,
        )
        self.db["schedules"].create(
            {
                "id": int, "name": str, "kind": str, "payload_json": str,
                "spec_json": str, "enabled": int, "created_by": str,
                "created_at": str, "updated_at": str,
                "last_run_at": str, "next_run_at": str, "last_job_id": int,
            },
            pk="id",
            if_not_exists=True,
        )

    def enqueue_job(
        self,
        kind: str,
        payload: dict | None = None,
        *,
        actor: str | None = None,
        priority: int = 0,
        run_after: datetime | str | None = None,
        max_attempts: int = jobs_mod.MAX_ATTEMPTS,
        schedule_id: int | None = None,
        dedupe: bool = False,
    ) -> int | None:
        """Add a job to the queue; returns its id.

        `dedupe` skips the insert when an equivalent job (same kind, same
        payload) is already queued or running — what a UI button wants, so
        an impatient double-click doesn't start two sourcing runs.
        """
        if kind not in jobs_mod.JOB_KINDS:
            raise ValueError(f"unknown job kind: {kind!r}")
        payload = payload or {}
        payload_json = json.dumps(payload, sort_keys=True)
        now = datetime.now(timezone.utc)
        when = run_after or now
        when_iso = when if isinstance(when, str) else when.isoformat()
        with self.write_tx():
            if dedupe:
                existing = self.db.execute(
                    "select id from jobs where kind = ? and payload_json = ? "
                    "and status in ('queued', 'running') limit 1",
                    [kind, payload_json],
                ).fetchone()
                if existing:
                    return None
            # Chained, not two lookups: db["jobs"] builds a fresh Table each
            # access, so a separate .last_pk read would see a different
            # object and return None.
            return int(
                self.db["jobs"].insert(
                    {
                        "kind": kind, "payload_json": payload_json,
                        "status": "queued", "priority": priority,
                        "attempts": 0, "max_attempts": max_attempts,
                        "requested_by": actor or self.actor or "system:scout",
                        "schedule_id": schedule_id or 0,
                        "created_at": now.isoformat(), "run_after": when_iso,
                        "started_at": None, "finished_at": None,
                        "heartbeat_at": None, "lease_expires_at": None,
                        "worker_id": "", "result_json": "{}", "error": "",
                        "log_path": "",
                    },
                    alter=True,
                ).last_pk
            )

    def claim_job(self, worker_id: str) -> dict | None:
        """Atomically take the next runnable job, or None.

        The claim is a single BEGIN IMMEDIATE transaction, so two workers
        racing for the same job cannot both win — one blocks, re-reads, and
        finds it already running. Highest priority first, then oldest.
        """
        now = datetime.now(timezone.utc)
        with self.write_tx():
            if not self.db["jobs"].exists():
                return None
            # rows_where (not db.execute) because this needs dicts — the raw
            # cursor yields tuples.
            rows = list(self.db["jobs"].rows_where(
                "status = 'queued' and run_after <= ?", [now.isoformat()],
                order_by="priority desc, id asc", limit=1,
            ))
            if not rows:
                return None
            job = dict(rows[0])
            lease = now + timedelta(seconds=jobs_mod.LEASE_SECONDS)
            self.db["jobs"].update(
                job["id"],
                {
                    "status": "running", "worker_id": worker_id,
                    "started_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                    "lease_expires_at": lease.isoformat(),
                    "attempts": (job.get("attempts") or 0) + 1,
                },
            )
            job = self._job_row(job["id"]) or job
        # Hydrated, so handlers read job["payload"] rather than re-parsing
        # the JSON themselves.
        return self._hydrate_job(job)

    def heartbeat_job(self, job_id: int) -> None:
        """Extend a running job's lease — proof the worker is still alive."""
        now = datetime.now(timezone.utc)
        with self.write_tx():
            row = self._job_row(job_id)
            if not row or row.get("status") != "running":
                return
            self.db["jobs"].update(
                job_id,
                {
                    "heartbeat_at": now.isoformat(),
                    "lease_expires_at": (
                        now + timedelta(seconds=jobs_mod.LEASE_SECONDS)
                    ).isoformat(),
                },
            )

    def finish_job(
        self, job_id: int, result: dict | None = None, log_path: str = ""
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.write_tx():
            if not self._job_row(job_id):
                return
            self.db["jobs"].update(
                job_id,
                {
                    "status": "done", "finished_at": now, "error": "",
                    "result_json": json.dumps(result or {}),
                    **({"log_path": log_path} if log_path else {}),
                },
            )

    def fail_job(self, job_id: int, error: str, log_path: str = "") -> bool:
        """Record a failure. Retries with backoff until max_attempts, then
        the job goes terminal. Returns True if it will be retried."""
        now = datetime.now(timezone.utc)
        with self.write_tx():
            row = self._job_row(job_id)
            if not row:
                return False
            attempts = int(row.get("attempts") or 0)
            max_attempts = int(row.get("max_attempts") or jobs_mod.MAX_ATTEMPTS)
            will_retry = attempts < max_attempts
            update = {
                "error": error[:2000],
                **({"log_path": log_path} if log_path else {}),
            }
            if will_retry:
                delay = jobs_mod.backoff_seconds(attempts)
                update.update(
                    status="queued", worker_id="", lease_expires_at=None,
                    run_after=(now + timedelta(seconds=delay)).isoformat(),
                )
            else:
                update.update(status="failed", finished_at=now.isoformat())
            self.db["jobs"].update(job_id, update)
            return will_retry

    def cancel_job(self, job_id: int, actor: str | None = None) -> bool:
        """Cancel a queued job. A running job is left alone — its worker owns
        it, and killing work mid-flight is the caller's business, not the
        queue's."""
        with self.write_tx():
            row = self._job_row(job_id)
            if not row or row.get("status") != "queued":
                return False
            self.db["jobs"].update(
                job_id,
                {
                    "status": "cancelled",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"cancelled by {actor or self.actor or 'someone'}",
                },
            )
            return True

    def reap_stale_jobs(self) -> int:
        """Requeue jobs whose worker died holding the lease.

        This is what makes the queue survive a container restart mid-run:
        without it a crashed worker's job would sit in 'running' forever.
        """
        now = datetime.now(timezone.utc)
        reaped = 0
        with self.write_tx():
            if not self.db["jobs"].exists():
                return 0
            stale = list(self.db["jobs"].rows_where(
                "status = 'running' and lease_expires_at is not null "
                "and lease_expires_at < ?", [now.isoformat()],
            ))
            for row in stale:
                job = dict(row)
                attempts = int(job.get("attempts") or 0)
                max_attempts = int(job.get("max_attempts") or jobs_mod.MAX_ATTEMPTS)
                if attempts >= max_attempts:
                    self.db["jobs"].update(job["id"], {
                        "status": "failed", "finished_at": now.isoformat(),
                        "error": "worker died and retries are exhausted",
                    })
                else:
                    self.db["jobs"].update(job["id"], {
                        "status": "queued", "worker_id": "",
                        "lease_expires_at": None,
                        "run_after": now.isoformat(),
                        "error": "worker died; requeued",
                    })
                reaped += 1
        return reaped

    def _job_row(self, job_id: int) -> dict | None:
        if not self.db["jobs"].exists():
            return None
        rows = list(self.db["jobs"].rows_where("id = ?", [job_id], limit=1))
        return dict(rows[0]) if rows else None

    def get_job(self, job_id: int) -> dict | None:
        row = self._job_row(job_id)
        return self._hydrate_job(row) if row else None

    @staticmethod
    def _hydrate_job(row: dict) -> dict:
        row = dict(row)
        row["payload"] = json.loads(row.get("payload_json") or "{}")
        row["result"] = json.loads(row.get("result_json") or "{}")
        return row

    def jobs(
        self, status: str | list[str] | None = None, limit: int = 50
    ) -> list[dict]:
        """Recent jobs, newest first."""
        if not self.db["jobs"].exists():
            return []
        where, params = "1=1", []
        if status:
            wanted = [status] if isinstance(status, str) else list(status)
            where = f"status in ({','.join('?' * len(wanted))})"
            params = wanted
        rows = self.db["jobs"].rows_where(
            where, params, order_by="id desc", limit=limit
        )
        return [self._hydrate_job(dict(r)) for r in rows]

    def job_queue_depth(self) -> int:
        if not self.db["jobs"].exists():
            return 0
        row = self.db.execute(
            "select count(*) from jobs where status in ('queued', 'running')"
        ).fetchone()
        return int(row[0]) if row else 0

    def record_worker_heartbeat(self, worker_id: str) -> None:
        """The worker's liveness beacon, written once per tick.

        The UI reads this to decide whether clicking Run should enqueue a
        job or launch a subprocess itself: with a worker deployed, queueing
        is right; on a laptop running only `scout ui`, queueing would mean
        nothing ever happens.
        """
        self.set_setting("worker_last_seen", datetime.now(timezone.utc).isoformat())
        self.set_setting("worker_id", worker_id)

    def worker_status(self, stale_after_s: int = 180) -> dict | None:
        """{"id", "last_seen", "alive"} — None if a worker has never run."""
        last_seen = self.get_setting("worker_last_seen")
        if not last_seen:
            return None
        try:
            seen_at = datetime.fromisoformat(last_seen)
        except ValueError:
            return None
        if seen_at.tzinfo is None:
            seen_at = seen_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - seen_at).total_seconds()
        return {
            "id": self.get_setting("worker_id") or "",
            "last_seen": seen_at,
            "alive": age <= stale_after_s,
            "age_s": age,
        }

    # -------------------------------------------------------------- schedules

    def upsert_schedule(
        self,
        name: str,
        kind: str,
        spec: "jobs_mod.ScheduleSpec",
        payload: dict | None = None,
        *,
        schedule_id: int | None = None,
        enabled: bool = True,
        actor: str | None = None,
    ) -> int:
        """Create or update a schedule, recomputing when it next fires."""
        if kind not in jobs_mod.JOB_KINDS:
            raise ValueError(f"unknown job kind: {kind!r}")
        spec.validate_spec()
        now = datetime.now(timezone.utc)
        next_at = jobs_mod.next_occurrence(spec, now) if enabled else None
        record = {
            "name": name.strip() or jobs_mod.JOB_LABELS.get(kind, kind),
            "kind": kind,
            "payload_json": json.dumps(payload or {}, sort_keys=True),
            "spec_json": spec.model_dump_json(),
            "enabled": 1 if enabled else 0,
            "updated_at": now.isoformat(),
            "next_run_at": next_at.isoformat() if next_at else None,
        }
        with self.write_tx():
            if schedule_id:
                self.db["schedules"].update(schedule_id, record)
                return schedule_id
            record.update(
                created_by=actor or self.actor or "system:scout",
                created_at=now.isoformat(),
                last_run_at=None, last_job_id=0,
            )
            return int(self.db["schedules"].insert(record, alter=True).last_pk)

    def set_schedule_enabled(self, schedule_id: int, enabled: bool) -> None:
        with self.write_tx():
            row = self._schedule_row(schedule_id)
            if not row:
                return
            spec = jobs_mod.ScheduleSpec.model_validate_json(
                row.get("spec_json") or "{}"
            )
            next_at = (
                jobs_mod.next_occurrence(spec, datetime.now(timezone.utc))
                if enabled else None
            )
            self.db["schedules"].update(schedule_id, {
                "enabled": 1 if enabled else 0,
                "next_run_at": next_at.isoformat() if next_at else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    def delete_schedule(self, schedule_id: int) -> None:
        with self.write_tx():
            if self._schedule_row(schedule_id):
                self.db["schedules"].delete(schedule_id)

    def _schedule_row(self, schedule_id: int) -> dict | None:
        if not self.db["schedules"].exists():
            return None
        rows = list(self.db["schedules"].rows_where("id = ?", [schedule_id], limit=1))
        return dict(rows[0]) if rows else None

    @staticmethod
    def _hydrate_schedule(row: dict) -> dict:
        row = dict(row)
        row["payload"] = json.loads(row.get("payload_json") or "{}")
        try:
            row["spec"] = jobs_mod.ScheduleSpec.model_validate_json(
                row.get("spec_json") or "{}"
            )
        except Exception:  # noqa: BLE001 — a corrupt spec must not hide the row
            row["spec"] = None
        row["enabled"] = bool(row.get("enabled"))
        return row

    def schedules(self) -> list[dict]:
        if not self.db["schedules"].exists():
            return []
        return [
            self._hydrate_schedule(dict(r))
            for r in self.db["schedules"].rows_where(order_by="id asc")
        ]

    def materialize_due_schedules(self, now: datetime | None = None) -> list[int]:
        """Enqueue a job for every schedule whose time has come, and advance
        it to its next firing. Returns the job ids created.

        Deliberately fires at most one job per schedule per pass, even if
        several occurrences were missed (a laptop asleep over a weekend
        should produce one run on wake, not forty).
        """
        now = now or datetime.now(timezone.utc)
        created: list[int] = []
        for row in self.schedules():
            if not row["enabled"] or row["spec"] is None:
                continue
            next_at = row.get("next_run_at")
            if not next_at:
                continue
            if datetime.fromisoformat(next_at) > now:
                continue
            job_id = self.enqueue_job(
                row["kind"], row["payload"],
                actor=f"schedule:{row['id']}",
                schedule_id=row["id"],
                # A schedule must never stack duplicates behind a slow run.
                dedupe=True,
            )
            following = jobs_mod.next_occurrence(row["spec"], now)
            with self.write_tx():
                self.db["schedules"].update(row["id"], {
                    "last_run_at": now.isoformat(),
                    "next_run_at": following.isoformat(),
                    **({"last_job_id": job_id} if job_id else {}),
                })
            if job_id:
                created.append(job_id)
        return created

    # ------------------------------------------------------------- scan status

    def scan_start(self, kind: str, pid: int, phases: list[str] | None = None) -> None:
        """Mark a scan (run / verify / source preview) as running. Single-row
        table: scans are serialized firm-wide by design — one sourcing run at
        a time keeps the paid-adapter budget guard race-free. (The jobs table
        supersedes this as the queue; this row remains the live-progress view.)

        `phases` declares the expected phase sequence up front so the UI can
        render a stepper (pending → running → done). `SCOUT_SCAN_LOG` (set by
        the launcher) is recorded so any session can tail the live console
        output."""
        now = datetime.now(timezone.utc).isoformat()
        self.db["scan"].upsert(
            {
                "id": 1, "kind": kind, "pid": pid, "phase": "starting",
                "detail": "", "status": "running",
                "started_at": now, "updated_at": now, "finished_at": None,
                # 0 (not None) so SQLite types the columns INTEGER on first
                # write; the UI treats 0-total as "no progress bar yet".
                "done": 0, "total": 0, "unit": "",
                "phases_json": json.dumps(phases or []),
                "phase_log_json": json.dumps([{"phase": "starting", "at": now}]),
                "log_path": os.environ.get("SCOUT_SCAN_LOG", ""),
            },
            pk="id",
            alter=True,
        )

    def scan_update(
        self,
        phase: str,
        detail: str = "",
        done: int | None = None,
        total: int | None = None,
        unit: str = "",
    ) -> None:
        """Advance the running scan's phase (no-op when nothing is running).

        `done`/`total` drive a progress bar when known (`unit` "items" or
        "s" — seconds means the phase is wall-clock-budgeted and the UI
        derives `done` from elapsed time). Counters reset on phase change
        unless explicitly passed."""
        with self.write_tx():
            row = self._scan_row()
            if not row or row.get("status") != "running":
                return
            now = datetime.now(timezone.utc).isoformat()
            if phase != row.get("phase"):
                log = json.loads(row.get("phase_log_json") or "[]")
                log.append({"phase": phase, "at": now})
                row["phase_log_json"] = json.dumps(log)
            row.update(phase=phase, detail=detail, updated_at=now,
                       done=done, total=total, unit=unit)
            self.db["scan"].upsert(row, pk="id", alter=True)

    def scan_finish(self, status: str = "done", detail: str = "") -> None:
        with self.write_tx():
            row = self._scan_row()
            if not row:
                return
            now = datetime.now(timezone.utc).isoformat()
            row.update(status=status, detail=detail or row.get("detail") or "",
                       updated_at=now, finished_at=now)
            self.db["scan"].upsert(row, pk="id", alter=True)
            # Completed-scan history — powers empirical time estimates in the UI.
            if row.get("started_at"):
                self.db["scan_history"].insert(
                    {
                        "kind": row.get("kind"), "status": status,
                        "started_at": row.get("started_at"), "finished_at": now,
                        "phase_log_json": row.get("phase_log_json") or "[]",
                        "detail": row.get("detail") or "",
                    },
                    alter=True,
                )

    def _scan_row(self) -> dict | None:
        if not self.db["scan"].exists():
            return None
        rows = list(self.db["scan"].rows_where("id = 1", limit=1))
        return dict(rows[0]) if rows else None

    def current_scan(self) -> dict | None:
        """The latest scan row, with liveness enforced: a 'running' scan whose
        process is gone is flipped to failed so the UI never shows a ghost."""
        row = self._scan_row()
        if not row:
            return None
        if row.get("status") == "running" and row.get("pid"):
            try:
                os.kill(int(row["pid"]), 0)
            except (ProcessLookupError, ValueError):
                self.scan_finish("failed", "process exited unexpectedly")
                row = self._scan_row()
            except PermissionError:
                pass  # process exists but isn't ours — treat as alive
        return row

    # ---------------------------------------------------------- budget ledger

    def record_xapi_usage(
        self, endpoint: str, posts_read: int, users_read: int, est_cost_usd: float
    ) -> None:
        self.db["xapi_usage"].insert(
            {
                "endpoint": endpoint,
                "posts_read": posts_read,
                "users_read": users_read,
                "est_cost_usd": est_cost_usd,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def xapi_spend_usd(self) -> float:
        """Cumulative estimated X API spend across all runs (the $25 guard)."""
        if not self.db["xapi_usage"].exists():
            return 0.0
        row = self.db.execute(
            "select coalesce(sum(est_cost_usd), 0) from xapi_usage"
        ).fetchone()
        return float(row[0])
