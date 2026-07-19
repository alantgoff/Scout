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

from scout.config import DEFAULT_DB_PATH
from scout.diligence.schema import Memo
from scout.models import Account, Lead, LedgerEntry, LLMVerdict, Tweet, UnlinkedLead

# Pre-~/.scout location: a scout.db relative to wherever scout was run from.
_LEGACY_DB_PATH = Path("scout.db")


def _clean_tags(tags: list) -> list[str]:
    """Strip, drop empties, dedupe case-insensitively (first casing wins)."""
    seen: dict[str, str] = {}
    for tag in tags:
        cleaned = str(tag).strip()
        if cleaned and cleaned.lower() not in seen:
            seen[cleaned.lower()] = cleaned
    return list(seen.values())


def pipeline_tags(row: dict) -> list[str]:
    """Parse a pipeline row's tags column (JSON list; tolerant of legacy
    comma-separated strings and missing/garbage values)."""
    raw = (row or {}).get("tags")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return _clean_tags(data)
    except (json.JSONDecodeError, TypeError):
        pass
    return _clean_tags(str(raw).split(","))


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
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
        self.db = sqlite_utils.Database(db_path)

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
    ) -> None:
        """Record run provenance: which strategy (thesis+seeds fingerprint)
        produced it. Runs sharing a hash group together in the ledger UI."""
        self.db["runs"].upsert(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "strategy_hash": strategy_hash,
                "thesis_statement": thesis_statement,
                "config_json": config_json,
            },
            pk="run_id",
            alter=True,
        )

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
        self, include_demo: bool = False, strategy_hash: str | None = None
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

    def record_verdict(self, handle: str, fingerprint: str, verdict: LLMVerdict) -> None:
        """Cache a Claude verdict keyed by handle + input fingerprint (a hash of
        bio, tweets, thesis, and model — see llm._fingerprint)."""
        self.db["llm_verdicts"].upsert(
            {
                "handle": handle.lstrip("@").lower(),
                "fingerprint": fingerprint,
                "verdict_json": verdict.model_dump_json(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            pk="handle",
            alter=True,
        )

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
        tags: list[str] | None = None,
    ) -> None:
        """Upsert deal-flow state for one lead (read-merge-write so partial
        updates never clobber the other fields). `tags` REPLACES the tag list
        (use tag_lead/untag_lead for incremental edits)."""
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
            row["brief_at"] = now
        if tags is not None:
            row["tags"] = json.dumps(_clean_tags(tags))
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.db["pipeline"].upsert(row, pk="handle", alter=True)

    def tag_lead(self, handle: str, tag: str) -> None:
        """Add one tag to a lead (deduped case-insensitively, order kept)."""
        tag = tag.strip()
        if not tag:
            return
        current = pipeline_tags(self.get_pipeline(handle))
        if tag.lower() not in {t.lower() for t in current}:
            self.set_pipeline(handle, tags=[*current, tag])

    def untag_lead(self, handle: str, tag: str) -> None:
        current = pipeline_tags(self.get_pipeline(handle))
        kept = [t for t in current if t.lower() != tag.strip().lower()]
        if len(kept) != len(current):
            self.set_pipeline(handle, tags=kept)

    def get_pipeline(self, handle: str) -> dict:
        handle = handle.lstrip("@").lower()
        if not self.db["pipeline"].exists():
            return {}
        rows = list(self.db["pipeline"].rows_where("handle = ?", [handle], limit=1))
        return dict(rows[0]) if rows else {}

    def all_pipeline(self) -> dict[str, dict]:
        """All deal-flow rows keyed by lowercased handle."""
        if not self.db["pipeline"].exists():
            return {}
        return {r["handle"]: dict(r) for r in self.db["pipeline"].rows}

    def pipeline_counts(self) -> dict[str, int]:
        """Count of leads at each status (only statuses that appear)."""
        counts: dict[str, int] = {}
        for row in self.all_pipeline().values():
            status = row.get("status") or "new"
            counts[status] = counts.get(status, 0) + 1
        return counts

    # ------------------------------------------------------------- scan status

    def scan_start(self, kind: str, pid: int) -> None:
        """Mark a scan (run / verify / source preview) as running. Single-row
        table: scout is single-user and runs are serialized by design."""
        now = datetime.now(timezone.utc).isoformat()
        self.db["scan"].upsert(
            {
                "id": 1, "kind": kind, "pid": pid, "phase": "starting",
                "detail": "", "status": "running",
                "started_at": now, "updated_at": now, "finished_at": None,
            },
            pk="id",
            alter=True,
        )

    def scan_update(self, phase: str, detail: str = "") -> None:
        """Advance the running scan's phase (no-op when nothing is running)."""
        row = self._scan_row()
        if not row or row.get("status") != "running":
            return
        row.update(phase=phase, detail=detail,
                   updated_at=datetime.now(timezone.utc).isoformat())
        self.db["scan"].upsert(row, pk="id", alter=True)

    def scan_finish(self, status: str = "done", detail: str = "") -> None:
        row = self._scan_row()
        if not row:
            return
        now = datetime.now(timezone.utc).isoformat()
        row.update(status=status, detail=detail or row.get("detail") or "",
                   updated_at=now, finished_at=now)
        self.db["scan"].upsert(row, pk="id", alter=True)

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

    # ------------------------------------------------------------------- memos

    def save_memo(self, memo: Memo) -> None:
        """Persist a deep-analysis memo (one per company; re-analysis
        replaces). The fingerprint makes the cache: re-running with unchanged
        inputs is served from here (mirrors the llm_verdicts idiom)."""
        self.db["memos"].upsert(
            {
                "company_key": memo.company_key,
                "company_name": memo.company_name,
                "fingerprint": memo.fingerprint,
                "composite": memo.composite,
                "memo_json": memo.model_dump_json(),
                "memo_md": memo.memo_md,
                "cost_usd": memo.cost_usd,
                "created_at": memo.created_at
                or datetime.now(timezone.utc).isoformat(),
            },
            pk="company_key",
            alter=True,
        )

    def load_memo(self, company_key: str) -> Memo | None:
        if not self.db["memos"].exists():
            return None
        rows = list(
            self.db["memos"].rows_where("company_key = ?", [company_key], limit=1)
        )
        return Memo.model_validate_json(rows[0]["memo_json"]) if rows else None

    def list_memos(self) -> list[Memo]:
        """All stored memos, newest first."""
        if not self.db["memos"].exists():
            return []
        return [
            Memo.model_validate_json(r["memo_json"])
            for r in self.db["memos"].rows_where(order_by="created_at desc")
        ]

    # ------------------------------------------------- diligence budget ledger

    def record_diligence_usage(
        self,
        *,
        company_key: str,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        searches: int,
        est_cost_usd: float,
    ) -> None:
        """Append one diligence API call to the spend ledger (mirrors
        xapi_usage: estimates, recorded per call, never rewritten)."""
        self.db["diligence_usage"].insert(
            {
                "company_key": company_key,
                "stage": stage,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "searches": searches,
                "est_cost_usd": est_cost_usd,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def diligence_spend_usd(self, company_key: str | None = None) -> float:
        """Cumulative estimated diligence spend — all-time, or one company's."""
        if not self.db["diligence_usage"].exists():
            return 0.0
        if company_key is None:
            row = self.db.execute(
                "select coalesce(sum(est_cost_usd), 0) from diligence_usage"
            ).fetchone()
        else:
            row = self.db.execute(
                "select coalesce(sum(est_cost_usd), 0) from diligence_usage "
                "where company_key = ?",
                [company_key],
            ).fetchone()
        return float(row[0])

    # --------------------------------------------------------- knowledge graph

    def kg_upsert_node(
        self, id: str, type: str, name: str, meta: dict | None = None
    ) -> None:
        """Read-merge upsert: an ingest with an empty name/meta never clobbers
        richer data already on the node."""
        row = {"id": id, "type": type, "name": name, "meta": json.dumps(meta or {})}
        if self.db["kg_nodes"].exists():
            existing = list(self.db["kg_nodes"].rows_where("id = ?", [id], limit=1))
            if existing:
                old = dict(existing[0])
                # First non-empty display name wins — a later write keyed by
                # the normalized key must not clobber "Tuva AI" with "tuvaai".
                if old.get("name"):
                    row["name"] = old["name"]
                old_meta = json.loads(old.get("meta") or "{}")
                row["meta"] = json.dumps({**old_meta, **(meta or {})})
        self.db["kg_nodes"].upsert(row, pk="id", alter=True)

    def kg_upsert_edge(
        self,
        src: str,
        rel: str,
        dst: str,
        provenance: str = "agent",
        source_url: str = "",
        memo_ref: str = "",
    ) -> None:
        """Upsert one edge. INVARIANT: user-provenance edges are never
        overwritten (or deleted) by agents — a manual "X competes with Y"
        link survives every re-analysis."""
        existing = None
        if self.db["kg_edges"].exists():
            rows = list(
                self.db["kg_edges"].rows_where(
                    "src = ? and rel = ? and dst = ?", [src, rel, dst], limit=1
                )
            )
            existing = dict(rows[0]) if rows else None
        if existing is not None and existing.get("provenance") == "user" and provenance == "agent":
            return
        self.db["kg_edges"].upsert(
            {
                "src": src,
                "rel": rel,
                "dst": dst,
                "provenance": provenance,
                "source_url": source_url,
                "memo_ref": memo_ref,
                "created_at": (existing or {}).get("created_at")
                or datetime.now(timezone.utc).isoformat(),
            },
            pk=("src", "rel", "dst"),
            alter=True,
        )

    def _kg_node_names(self, ids: list[str]) -> dict[str, dict]:
        if not ids or not self.db["kg_nodes"].exists():
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self.db["kg_nodes"].rows_where(f"id in ({marks})", ids)
        return {r["id"]: dict(r) for r in rows}

    def kg_neighbors(self, company_key: str) -> list[dict]:
        """Every edge touching this company, with the far node resolved:
        [{"id", "type", "name", "meta", "rel", "direction", "provenance"}]."""
        node = f"company:{company_key}"
        if not self.db["kg_edges"].exists():
            return []
        edges: list[tuple[str, dict]] = []  # (far_node_id, edge_row)
        seen: set[tuple[str, str]] = set()
        for row in self.db["kg_edges"].rows_where("src = ? or dst = ?", [node, node]):
            far = row["dst"] if row["src"] == node else row["src"]
            if far == node or (far, row["rel"]) in seen:
                continue
            seen.add((far, row["rel"]))
            edges.append((far, dict(row)))
        names = self._kg_node_names([far for far, _ in edges])  # one batched query
        return [
            {
                "id": far,
                "type": (names.get(far) or {}).get("type") or far.split(":", 1)[0],
                "name": (names.get(far) or {}).get("name") or far.split(":", 1)[-1],
                "meta": json.loads((names.get(far) or {}).get("meta") or "{}"),
                "rel": row["rel"],
                "direction": "out" if row["src"] == node else "in",
                "provenance": row.get("provenance") or "agent",
            }
            for far, row in edges
        ]

    def kg_competitors_for_sectors(
        self, sector_keys: list[str], exclude_company_key: str | None = None
    ) -> list[str]:
        """Names of companies previously mapped into these sectors — the seed
        the competition agent starts from ("previously mapped in this space").
        sector_keys are normalized sector node keys (graph.normalize_name)."""
        if not sector_keys or not self.db["kg_edges"].exists():
            return []
        sector_ids = [f"sector:{k}" for k in sector_keys]
        marks = ",".join("?" for _ in sector_ids)
        company_ids = {
            r["src"]
            for r in self.db["kg_edges"].rows_where(
                f"rel = 'in_sector' and dst in ({marks})", sector_ids
            )
            if r["src"].startswith("company:")
        }
        if exclude_company_key:
            company_ids.discard(f"company:{exclude_company_key}")
        names = self._kg_node_names(sorted(company_ids))
        return sorted(
            (names.get(cid) or {}).get("name") or cid.split(":", 1)[-1]
            for cid in company_ids
        )

    def kg_link_competitors(self, a: str, b: str, provenance: str = "user") -> bool:
        """Manually link two companies as competitors (both directions).
        Accepts display names or keys; creates missing nodes. Returns False
        when either name normalizes to nothing."""
        from scout.diligence.graph import node_id

        aid, bid = node_id("company", a), node_id("company", b)
        if aid is None or bid is None or aid == bid:
            return False
        self.kg_upsert_node(aid, "company", a.strip())
        self.kg_upsert_node(bid, "company", b.strip())
        self.kg_upsert_edge(aid, "competes_with", bid, provenance=provenance)
        self.kg_upsert_edge(bid, "competes_with", aid, provenance=provenance)
        return True
