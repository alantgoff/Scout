"""SQLite persistence via sqlite-utils: cache, dedupe, seen-TTL, API budget ledger.

Every fetched account/tweet is upserted here so repeat runs are incremental
(and, in xapi mode, free).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite_utils
from pydantic import ValidationError

from scout.config import DEFAULT_DB_PATH
from scout.models import Account, Lead, LedgerEntry, LLMVerdict, SitePage, Tweet, UnlinkedLead

# Pre-~/.scout location: a scout.db relative to wherever scout was run from.
_LEGACY_DB_PATH = Path("scout.db")


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH, *, cross_thread: bool = False) -> None:
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
        if cross_thread:
            # Streamlit reruns the script (and 4s-polling fragments) on
            # varying threads; sqlite3's same-thread guard would raise
            # ProgrammingError mid-render. Safe here: the UI is single-user
            # and read-mostly, never writing from two threads at once.
            import sqlite3

            self.db = sqlite_utils.Database(
                sqlite3.connect(str(db_path), check_same_thread=False)
            )
        else:
            self.db = sqlite_utils.Database(db_path)
        self._migrate_thesis_columns()

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

    # ------------------------------------------------------------------ cache

    def upsert_account(self, account: Account) -> None:
        row = account.model_dump(mode="json")
        row["followed_by"] = json.dumps(row["followed_by"])
        row["sources"] = json.dumps(row.get("sources") or [])
        # Enrichment fields are recomputed from history every run — don't persist
        # stale values.
        row.pop("recent_followed_by", None)
        row.pop("bio_changed", None)
        self.db["accounts"].upsert(row, pk="id", alter=True)

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
        existing strategy grouping keeps working unchanged.
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
        self.db.execute("update theses set is_active = 0")
        self.db.execute("update theses set is_active = 1 where id = ?", [thesis_id])
        self.db.conn.commit()

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
                "update or replace thesis_versions set thesis_id = ? where thesis_id = ?",
                [to_id, from_id],
            )
        if self.db["theses"].exists():
            self.db.execute("delete from theses where id = ?", [from_id])
        self.db.conn.commit()
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
        for handle in followee_handles:
            followee = handle.lstrip("@").lower()
            if followee in known:
                # existing edge — refresh last_seen only, preserve first_seen
                edges.upsert(
                    {"watcher": watcher, "followee": followee, "last_seen": now},
                    pk=("watcher", "followee"),
                )
            else:
                edges.upsert(
                    {
                        "watcher": watcher,
                        "followee": followee,
                        "first_seen": now,
                        "last_seen": now,
                    },
                    pk=("watcher", "followee"),
                )

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
        latest = None
        if self.db["bio_snapshots"].exists():
            rows = list(self.db["bio_snapshots"].rows_where(
                "handle = ?", [handle], order_by="seen_at desc", limit=1
            ))
            latest = rows[0]["bio"] if rows else None
        if latest == bio:
            return latest  # unchanged — previous == current, caller sees no change
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
    ) -> None:
        """Upsert deal-flow state for one lead (read-merge-write so partial
        updates never clobber the other fields).

        `brief` holds the investment memo markdown. A generation
        (brief_edited=False) stamps brief_at and clears brief_edited_at; a
        manual edit (brief_edited=True) stamps brief_edited_at and keeps the
        original generation time. `brief_meta` (depth, sources, searches —
        agents.investment_memo's meta) is stored as brief_meta_json alongside
        a generation and left untouched by edits."""
        handle = handle.lstrip("@").lower()
        row = {"handle": handle}
        if self.db["pipeline"].exists():
            existing = list(self.db["pipeline"].rows_where("handle = ?", [handle], limit=1))
            if existing:
                row = dict(existing[0])
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
        # Which thesis brought this company in. Triage itself stays global — a
        # company you shortlisted is shortlisted, whichever thesis found it —
        # but the attribution answers "why is this in my pipeline". Only ever
        # set once, so a later thesis cannot rewrite the origin.
        if sourced_thesis_id and not row.get("sourced_thesis_id"):
            row["sourced_thesis_id"] = sourced_thesis_id
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.db["pipeline"].upsert(row, pk="handle", alter=True)

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
        counts: dict[str, int] = {}
        for row in self.all_pipeline().values():
            status = row.get("status") or "new"
            counts[status] = counts.get(status, 0) + 1
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

    def set_attrs(self, handle: str, changes: dict) -> None:
        """Merge per-startup attribute values (column key → value). A None
        value deletes the key. Read-merge-write, like set_pipeline."""
        handle = handle.lstrip("@").lower()
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
            },
            pk="handle",
            alter=True,
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
    ) -> None:
        """Persist the investor's manual scoring for one lead.

        `sections` maps scorecard section key → 0..100 (only the sections
        the investor actually adjusted); `quality` is the legacy dim → 0..1
        shape, applied only to pre-scorecard verdicts; `fit` overrides
        thesis_fit (0..1); `score` pins the FINAL score outright (0..100).
        Passing everything as None/empty clears the row — an override that
        overrides nothing shouldn't linger. Values replace the whole row
        (the UI form always submits its full current state)."""
        handle = handle.lstrip("@").lower()
        if not quality and not sections and fit is None and score is None:
            self.clear_override(handle)
            return
        self.db["score_overrides"].upsert(
            {
                "handle": handle,
                "quality_json": json.dumps(quality or {}),
                "sections_json": json.dumps(sections or {}),
                "fit": fit,
                "score": score,
                "note": note,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            pk="handle",
            alter=True,
            # Explicit REAL affinity: a None on the first write would
            # otherwise type these TEXT and hand back "88.0" strings.
            columns={"fit": float, "score": float},
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

    # ------------------------------------------------------------- scan status

    def scan_start(self, kind: str, pid: int, phases: list[str] | None = None) -> None:
        """Mark a scan (run / verify / source preview) as running. Single-row
        table: scout is single-user and runs are serialized by design.

        `phases` declares the expected phase sequence up front so the UI can
        render a stepper (pending → running → done). `SCOUT_SCAN_LOG` (set by
        the UI when it launches the process detached) is recorded so any
        session can tail the live console output."""
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
