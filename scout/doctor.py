"""`scout doctor` — is this installation ready to source, and if not, why.

One question with a lot of moving parts: keys, X cookies, the thesis and
seeds, the database, the worker and its schedules, and whether each
discovery source's host is reachable from THIS machine with THIS config.
Each part is a Check with a status and — whenever it is not ok — the fix.

Severity is judged against one outcome: will the daily scan produce
classified leads?

- fail: it will not (no Anthropic key, an unwritable database, a thesis
  that does not load, no discovery source that can run, a key the API
  rejects).
- warn: it will, degraded (X not connected, SEC will refuse an anonymous
  User-Agent, the watchlist is still the example list, no worker).
- info: optional features and context.

Config checks read settings, thesis, seeds and the store only; network
checks take a `probe` callable so tests (and `--offline`) never touch the
network. Nothing here spends money: the Anthropic probe is the free models
endpoint, GitHub's is one search (of ~30/minute), and Slack is never
posted to.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from scout.config import Seeds, Settings, Thesis

Status = Literal["ok", "warn", "fail", "info"]
Probe = Callable[[str, dict[str, str]], tuple[int | None, str]]

# The watchlist seeds.yaml ships with. A run against it follows someone
# else's taste in investors — fine for a demo, wrong for a firm.
EXAMPLE_WATCHLIST = {
    "eladgil", "natfriedman", "danielgross", "martin_casado", "garrytan",
    "karpathy", "drjimfan", "dylan522p", "benedictevans", "chipstrat",
    "zebulgar", "alexandr_wang", "emollick", "hliriani", "gdb",
}
DEFAULT_SEC_UA = Settings.model_fields["sec_user_agent"].default
_PROBE_TIMEOUT_S = 8.0
_MAX_FEEDS_PROBED = 10


class Check(BaseModel):
    area: str
    name: str
    status: Status
    detail: str
    fix: str = ""


# --- config (no network) --------------------------------------------------------


def _cookies_check(settings: Settings) -> Check:
    path = settings.tw_cookies
    fix = ("Export your X session cookies to a file and set TW_COOKIES "
           "(README: 'Cookie setup'). Until then runs use the free sources only.")
    if path is None:
        return Check(area="Keys", name="X cookies (TW_COOKIES)", status="warn",
                     detail="not set — X discovery is off; runs fall back to the "
                            "free sources", fix=fix)
    if not path.is_file():
        return Check(area="Keys", name="X cookies (TW_COOKIES)", status="warn",
                     detail=f"{path} is not a file — X discovery is off", fix=fix)
    from scout.ingest.twscrape_src import _parse_cookies

    try:
        cookies = _parse_cookies(path.read_text(encoding="utf-8"))
    except (RuntimeError, OSError, ValueError) as exc:
        return Check(area="Keys", name="X cookies (TW_COOKIES)", status="warn",
                     detail=f"{path} could not be read as cookies ({exc})", fix=fix)
    missing = [k for k in ("ct0", "auth_token") if k not in cookies]
    if missing:
        return Check(area="Keys", name="X cookies (TW_COOKIES)", status="warn",
                     detail=f"{path} lacks {', '.join(missing)} — X will reject it",
                     fix="Re-export the cookies while logged in to x.com.")
    return Check(area="Keys", name="X cookies (TW_COOKIES)", status="ok",
                 detail=f"{path} (validity is only proven by a run — X gives no "
                        "free way to test a session)")


def _tracked(store) -> int:
    """Distinct scored handles, one COUNT — not a ledger parse: the UI
    reruns this on every click."""
    if not store.db["leads"].exists():
        return 0
    return store.db.execute(
        "select count(distinct lower(handle)) from leads "
        "where run_id not like 'demo-%'").fetchone()[0]


def config_checks(settings: Settings, thesis: Thesis | None, seeds: Seeds,
                  store, *, thesis_error: str = "") -> list[Check]:
    """Everything decidable without the network."""
    checks: list[Check] = []
    add = checks.append

    # -- keys
    if settings.anthropic_api_key:
        add(Check(area="Keys", name="Anthropic key", status="ok",
                  detail="set (validity checked under Network)"))
    else:
        add(Check(area="Keys", name="Anthropic key", status="fail",
                  detail="not set — scoring is keyword rules only; classification, "
                         "research, resolve and refresh do nothing",
                  fix="Set ANTHROPIC_API_KEY in .env (server: /etc/scout/scout.env)."))
    add(_cookies_check(settings))
    add(Check(area="Keys", name="X API token (paid)", status="info",
              detail="set" if settings.x_bearer_token else
              "not set — only needed for `run --source xapi`"))
    add(Check(area="Keys", name="GitHub token", status="ok" if settings.github_token else "info",
              detail="set" if settings.github_token else
              "not set — GitHub search works anonymously at lower rate limits",
              fix="" if settings.github_token else
              "Optional: a free fine-grained PAT in GITHUB_TOKEN."))

    # -- thesis & seeds
    if thesis is None:
        add(Check(area="Thesis & seeds", name="Thesis", status="fail",
                  detail=thesis_error or "did not load",
                  fix="Fix thesis.yaml, or pick one with `scout thesis use <id>`."))
        return checks
    sources = sorted(thesis.active_discovery_sources)
    add(Check(area="Thesis & seeds", name="Thesis", status="ok",
              detail=f"{thesis.name or thesis.id or 'unnamed'} · stages "
                     f"{', '.join(thesis.target_stages) or '—'} · free sources "
                     f"{', '.join(sources) or 'none'}"))
    if not sources and settings.tw_cookies is None:
        add(Check(area="Thesis & seeds", name="Discovery", status="fail",
                  detail="no free source is active for the target stages and X is "
                         "not connected — a run has nothing to read",
                  fix="Add a target stage (launched → GitHub, HN, RSS, SEC, YC) or "
                      "set TW_COOKIES."))
    watchers = {w.lower() for w in seeds.watchers}
    if not watchers:
        add(Check(area="Thesis & seeds", name="Investor watchlist", status="warn",
                  detail="empty — no smart-money follow signals",
                  fix="List the investors whose follows you trust in seeds.yaml "
                      "(Thesis page → Watchlist & discovery)."))
    elif watchers <= EXAMPLE_WATCHLIST:
        add(Check(area="Thesis & seeds", name="Investor watchlist", status="warn",
                  detail=f"still the example list ({len(watchers)} handles) — the "
                         "smart-money signals follow someone else's taste",
                  fix="Replace it with your own investors in seeds.yaml "
                      "(Thesis page → Watchlist & discovery)."))
    else:
        add(Check(area="Thesis & seeds", name="Investor watchlist", status="ok",
                  detail=f"{len(watchers)} handles"))
    if "github" in sources and not seeds.github_topics:
        add(Check(area="Thesis & seeds", name="GitHub topics", status="warn",
                  detail="GitHub is active but no topics are set — it finds nothing",
                  fix="Add github_topics to seeds.yaml."))
    if "rss" in sources:
        feeds = [f for f in seeds.rss_feeds if f.strip()]
        add(Check(area="Thesis & seeds", name="RSS feeds",
                  status="ok" if feeds else "info",
                  detail=f"{len(feeds)} feed(s)" if feeds else
                  "none configured — the RSS source reads nothing",
                  fix="" if feeds else "Add launch/funding feeds to rss_feeds in "
                                       "seeds.yaml (examples are in the file)."))
    if "sec" in sources:
        ua = settings.sec_user_agent or ""
        if "@" not in ua:
            add(Check(area="Thesis & seeds", name="SEC User-Agent", status="warn",
                      detail=f"{ua!r} has no contact email — SEC's fair-access "
                             "policy asks for one and may refuse the requests",
                      fix='Set SEC_USER_AGENT="<your firm> <you@firm.com>".'))
        else:
            add(Check(area="Thesis & seeds", name="SEC User-Agent", status="ok",
                      detail=ua))
        if not seeds.sec_industries:
            add(Check(area="Thesis & seeds", name="SEC industries", status="info",
                      detail="empty — every non-fund industry is kept"))

    # -- database, worker, budget
    db_path = Path(settings.db_path)
    parent = db_path.parent
    writable = os.access(db_path if db_path.exists() else parent, os.W_OK)
    add(Check(area="Database & worker", name="Database",
              status="ok" if writable else "fail",
              detail=f"{db_path} · {_tracked(store)} companies scored"
              if writable else f"{db_path} is not writable",
              fix="" if writable else "Fix permissions, or point DB_PATH somewhere writable."))
    schedules = store.schedules()
    enabled = [s for s in schedules if s.get("enabled")]
    add(Check(area="Database & worker", name="Schedules",
              status="ok" if enabled else "warn",
              detail=f"{len(enabled)} enabled: "
                     + ", ".join(s["name"] for s in enabled) if enabled else
              "none — nothing runs on its own",
              fix="" if enabled else "Run `scout worker --bootstrap --once` once."))
    worker = store.worker_status()
    if worker and worker["alive"]:
        add(Check(area="Database & worker", name="Worker", status="ok",
                  detail=f"alive (last seen {int(worker['age_s'])}s ago)"))
    else:
        add(Check(area="Database & worker", name="Worker", status="warn",
                  detail="never run" if worker is None else
                  f"offline (last seen {int(worker['age_s'] // 60)} min ago)",
                  fix="Start `scout worker` (systemd unit in deploy/)."))
    cap = settings.daily_spend_cap_usd
    spent = store.spend_today_usd()
    add(Check(area="Database & worker", name="Daily spend cap",
              status="info" if cap <= 0 else "ok",
              detail="uncapped (DAILY_SPEND_CAP_USD=0)" if cap <= 0 else
              f"${spent:.2f} of ${cap:.2f} spent today (UTC)"))

    # -- optional delivery
    slack = bool(store.get_setting("slack_webhook_url"))
    add(Check(area="Optional", name="Slack digest", status="ok" if slack else "info",
              detail="webhook set" if slack else "no webhook — digests are not posted",
              fix="" if slack else "Settings page → Slack webhook."))
    targets = []
    if settings.digest_repo:
        targets.append("GitHub Pages")
    if settings.vercel_token or (Path("docs") / ".vercel" / "project.json").exists():
        targets.append("Vercel")
    add(Check(area="Optional", name="Phone app", status="ok" if targets else "info",
              detail="deploys to " + " + ".join(targets) if targets else
              "renders to docs/ only — no deploy target",
              fix="" if targets else "Set DIGEST_REPO, or `vercel link` inside docs/ "
                                     "(README: 'The phone app on Vercel')."))
    if targets and "Vercel" in targets and shutil.which("vercel") is None:
        add(Check(area="Optional", name="Vercel CLI", status="warn",
                  detail="Vercel is configured but the CLI is not installed",
                  fix="npm i -g vercel"))
    return checks


# --- network ----------------------------------------------------------------------


def http_probe(url: str, headers: dict[str, str]) -> tuple[int | None, str]:
    """GET → (status, first 20k chars of the body) or (None, error)."""
    import httpx

    try:
        resp = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT_S,
                         follow_redirects=True)
        return resp.status_code, resp.text[:20_000]
    except Exception as exc:  # noqa: BLE001 — every failure is a finding
        return None, f"{type(exc).__name__}: {exc}"[:200]


def _targets(settings: Settings, thesis: Thesis, seeds: Seeds) -> list[tuple[str, str, dict]]:
    """(name, url, headers) for everything worth probing under this config."""
    ua = {"User-Agent": "scout/0.1 (+doctor)"}
    targets: list[tuple[str, str, dict]] = []
    if settings.anthropic_api_key:
        targets.append(("Anthropic API", "https://api.anthropic.com/v1/models?limit=1",
                        {"x-api-key": settings.anthropic_api_key,
                         "anthropic-version": "2023-06-01"}))
    sources = thesis.active_discovery_sources
    if "github" in sources:
        headers = {"Accept": "application/vnd.github+json", **ua}
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        # The search endpoint itself, not /rate_limit: some networks allow
        # one and block the other, and search is the call the source makes.
        # Costs one of the ~30 searches/minute.
        targets.append(("GitHub", "https://api.github.com/search/repositories"
                        "?q=stars:%3E1000&per_page=1", headers))
    if "hn" in sources:
        targets.append(("Hacker News", "https://hn.algolia.com/api/v1/search"
                        "?query=startup&hitsPerPage=1", ua))
    if "arxiv" in sources:
        targets.append(("arXiv", "https://export.arxiv.org/api/query"
                        "?search_query=all:ai&max_results=1", ua))
    if "sec" in sources:
        # With the REAL User-Agent: whether SEC accepts it is the question.
        targets.append(("SEC EDGAR", "https://www.sec.gov/Archives/edgar/daily-index/",
                        {"User-Agent": settings.sec_user_agent}))
    if "yc" in sources:
        targets.append(("YC directory", "https://yc-oss.github.io/api/meta.json", ua))
    if "rss" in sources:
        for feed in [f.strip() for f in seeds.rss_feeds if f.strip()][:_MAX_FEEDS_PROBED]:
            targets.append((f"RSS {feed}", feed, ua))
    return targets


def _judge(name: str, status: int | None, body: str) -> Check:
    area = "Network"
    if status is None:
        return Check(area=area, name=name, status="warn",
                     detail=f"unreachable — {body}",
                     fix="Check this machine's network / proxy for the host.")
    if name == "Anthropic API":
        if status == 200:
            return Check(area=area, name=name, status="ok", detail="key accepted")
        if status in (401, 403):
            return Check(area=area, name=name, status="fail",
                         detail=f"HTTP {status} — the key was rejected",
                         fix="Replace ANTHROPIC_API_KEY with a valid key.")
    if name == "GitHub" and status == 403 and "rate limit" in body.lower():
        return Check(area=area, name=name, status="warn",
                     detail="HTTP 403 — rate limited",
                     fix="Set GITHUB_TOKEN (a free PAT) for higher limits.")
    if name == "SEC EDGAR" and status == 403:
        return Check(area=area, name=name, status="warn",
                     detail="HTTP 403 — SEC refused this User-Agent",
                     fix='Set SEC_USER_AGENT="<your firm> <you@firm.com>".')
    if name.startswith("RSS ") and status == 200:
        import feedparser

        n = len(feedparser.parse(body).entries)
        return Check(area=area, name=name, status="ok" if n else "warn",
                     detail=f"{n} entries" if n else "reachable but not a feed (0 entries)",
                     fix="" if n else "Use the feed URL, not the page URL.")
    if 200 <= status < 300:
        return Check(area=area, name=name, status="ok", detail="reachable")
    return Check(area=area, name=name, status="warn", detail=f"HTTP {status}",
                 fix="Check the host from this machine; the source will skip it.")


def network_checks(settings: Settings, thesis: Thesis | None, seeds: Seeds,
                   probe: Probe = http_probe) -> list[Check]:
    """Probe every host the current config would use, concurrently. When
    every discovery host fails, that is a fail: a run would read nothing."""
    if thesis is None:
        return []
    targets = _targets(settings, thesis, seeds)
    if not targets:
        return []
    with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
        results = list(pool.map(lambda t: probe(t[1], t[2]), targets))
    checks = [_judge(name, status, body)
              for (name, _url, _h), (status, body) in zip(targets, results)]
    discovery = [c for c in checks if c.name != "Anthropic API"]
    if discovery and settings.tw_cookies is None and all(c.status != "ok" for c in discovery):
        checks.append(Check(
            area="Network", name="Discovery", status="fail",
            detail="no discovery source is reachable and X is not connected — "
                   "a run would read nothing",
            fix="Fix network access to at least one source, or set TW_COOKIES."))
    return checks


# --- verdict ------------------------------------------------------------------------


def verdict(checks: list[Check]) -> tuple[bool, str]:
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    if fails:
        return False, (f"Not ready — {len(fails)} blocking issue"
                       f"{'s' if len(fails) != 1 else ''}"
                       + (f", {len(warns)} warning{'s' if len(warns) != 1 else ''}"
                          if warns else "") + ".")
    if warns:
        return True, (f"Ready, degraded — {len(warns)} warning"
                      f"{'s' if len(warns) != 1 else ''}. The daily scan will run.")
    return True, "Ready — the daily scan will run with everything configured."
