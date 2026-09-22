"""Digest publisher tests — offline render against a seeded temp store."""

from __future__ import annotations

from pathlib import Path

import json

from typer.testing import CliRunner

from scout.cli import app
from scout.config import Thesis
from scout.models import Account, Lead, LLMVerdict, Signal
from scout.publish import build_digest, _icon_png
from scout.store import Store


def seeded_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "t.db")
    startup = Lead(
        account=Account(id="1", handle="ada_infra", name="Ada Lin",
                        bio="ex-OpenAI", website="https://evalhq.ai"),
        signals=[Signal(name="bio_intent", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="ada_infra", account_type="founder", is_founder=True,
                       stage="launched", sector="ai infra", subsector="agent evals",
                       business_model="b2b saas", company_name="EvalHQ",
                       company_url="https://evalhq.ai",
                       one_line_summary="Eval platform for agents.",
                       thesis_fit=0.8, confidence=0.9),
        score=46.0,
    )
    stealth = Lead(
        account=Account(id="2", handle="stealth_sam", name="Sam O"),
        signals=[Signal(name="departure_signal", value=1.0, weight=20.0)],
        llm=LLMVerdict(handle="stealth_sam", account_type="founder", is_founder=True,
                       stage="stealth", confidence=0.8),
        score=20.0,
    )
    store.save_leads("r1", [startup, stealth])
    store.record_run("r1", source="twscrape", strategy_hash="h", thesis_statement="t")
    store.set_pipeline("ada_infra", status="shortlisted",
                       brief="**What they're building** — evals")
    return store


def test_build_digest_renders_tracks_and_no_secrets(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    out = build_digest(store, Thesis(thesis="Vertical agents"), tmp_path / "docs")

    page = out.read_text(encoding="utf-8")
    assert "EvalHQ" in page  # startup titled by company
    assert "@ada_infra" in page
    assert "stealth_sam" in page  # pre-launch watch section
    assert "stealth startup" in page  # unnamed founder → synthesized identity
    assert "Shortlisted" in page  # pipeline status chip
    assert "What they&#x27;re building" in page or "Research brief" in page
    assert 'name="robots" content="noindex"' in page
    assert "apple-mobile-web-app-capable" in page
    # never anything key-shaped or env-flavored
    assert "ANTHROPIC" not in page and "BEARER" not in page and ".env" not in page

    icon = (tmp_path / "docs" / "icon.png").read_bytes()
    assert icon.startswith(b"\x89PNG\r\n\x1a\n")


def test_icon_png_is_wellformed() -> None:
    data = _icon_png(8)
    assert data.startswith(b"\x89PNG") and data.endswith(
        b"IEND" + (0xAE426082).to_bytes(4, "big")
    )


def test_digest_search_index_is_lowercased(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    page = build_digest(store, Thesis(), tmp_path / "docs").read_text(encoding="utf-8")
    assert 'data-search="' in page
    start = page.index('data-search="') + len('data-search="')
    blob = page[start:page.index('"', start)]
    assert blob == blob.lower()


# --- the four-view app ---------------------------------------------------------


def app_store(tmp_path: Path) -> Store:
    """A store exercising every view: a funded/acquired startup with
    connections, a mover, private notes and votes that must never publish."""
    store = Store(tmp_path / "app.db")
    pollen = Lead(
        account=Account(id="1", handle="pollenrobotics", name="Pollen Robotics",
                        website="https://pollen-robotics.com"),
        llm=LLMVerdict(
            handle="pollenrobotics", account_type="startup", stage="launched",
            company_name="Pollen Robotics", company_url="https://pollen-robotics.com",
            one_line_summary="Open-source robots.", thesis_fit=0.6, confidence=0.9,
            funding_stage="seed", funding_amount="€2.5M",
            funding_investors=["Bpifrance"], funding_evidence="techcrunch",
            hq="Bordeaux, France", founded_year=2016,
            founders=["Matthieu Lapeyre — co-founder"],
            company_status="acquired",
            company_status_note="Hugging Face, April 2025",
            company_status_evidence="techcrunch 2025-04-14",
        ),
        score=55.0,
    )
    sibling = Lead(
        account=Account(id="2", handle="acmebot", name="Acme Robotics"),
        llm=LLMVerdict(handle="acmebot", account_type="startup", stage="launched",
                       company_name="Acme Robotics", funding_investors=["Bpifrance"],
                       funding_evidence="press", one_line_summary="Robots too.",
                       confidence=0.8),
        score=48.0,
    )
    store.save_leads("r0", [sibling.model_copy(update={"score": 30.0})])
    store.save_leads("r1", [pollen, sibling])  # sibling +18 → a mover
    store.record_run("r1", source="twscrape", strategy_hash="h", thesis_statement="t")
    store.set_pipeline("pollenrobotics", status="shortlisted",
                       notes="PRIVATE-NOTE-do-not-publish")
    store.set_pipeline("acmebot", status="longlisted")
    store.rebuild_graph(store.load_lead_ledger())
    return store


def test_app_views_render_with_data_attributes_and_tabs(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    # Tabs + hash routing.
    for view in ("view-startups", "view-funnel", "view-graph", "view-alerts"):
        assert view in page
    assert 'data-view="funnel"' in page
    # Sort/filter attributes stamped on cards.
    assert 'data-score="55.0"' in page
    assert 'data-status="shortlisted"' in page
    assert 'data-roundrank="2"' in page  # seed
    # Toolbar controls exist.
    for control in ('id="sort"', 'id="fstatus"', 'id="fstage"', 'id="fround"'):
        assert control in page


def test_researched_facts_and_warning_reach_the_page(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "Bordeaux, France" in page and "founded 2016" in page
    assert "Matthieu Lapeyre" in page
    assert "Seed · €2.5M" in page          # the round chip
    assert "Backed by Bpifrance" in page
    assert "⚠ Acquired — Hugging Face, April 2025" in page
    # The connections cross-link: Bpifrance backs both companies.
    assert "also backs" in page


def test_funnel_and_alerts_views(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    # Funnel groups in FUNNEL_STAGES order: Longlisted before Shortlisted.
    assert page.index(">Longlisted") < page.index(">Shortlisted")
    assert 'data-target="card-pollenrobotics"' in page
    # Alerts: the acquisition with its source, and the mover.
    assert "No longer independent" in page
    assert "techcrunch 2025-04-14" in page
    assert "Moving up" in page and "▲ 18" in page


def test_graph_view_embeds_the_canvas(tmp_path: Path) -> None:
    store = app_store(tmp_path)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "Knowledge graph" in page
    assert "const NODES =" in page  # the canvas fragment landed
    assert "Bpifrance" in page


def test_adversarial_labels_cannot_break_out_of_the_graph_payload(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    evil = Lead(
        account=Account(id="1", handle="evilco", name="EvilCo"),
        llm=LLMVerdict(handle="evilco", account_type="startup", stage="launched",
                       company_name="EvilCo",
                       funding_investors=['</script><script>alert(1)',
                                          "Honest Capital"],
                       funding_evidence="press", confidence=0.8),
        score=10.0,
    )
    honest = Lead(
        account=Account(id="2", handle="other", name="OtherCo"),
        llm=LLMVerdict(handle="other", account_type="startup", stage="launched",
                       company_name="OtherCo",
                       funding_investors=["Honest Capital"],
                       funding_evidence="press", confidence=0.8),
        score=9.0,
    )
    store.save_leads("r1", [evil, honest])
    store.rebuild_graph(store.load_lead_ledger())
    page = build_digest(store, Thesis(thesis="t"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "</script><script>alert" not in page


def test_pwa_files_written_and_registered(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    out = build_digest(store, Thesis(thesis="t"), tmp_path / "docs")
    docs = out.parent
    assert (docs / "manifest.webmanifest").exists()
    sw = (docs / "sw.js").read_text(encoding="utf-8")
    assert "caches" in sw and "index.html" in sw
    page = out.read_text(encoding="utf-8")
    assert 'serviceWorker' in page and 'rel="manifest"' in page
    assert 'name="robots" content="noindex"' in page  # still private-ish


def test_private_judgments_never_publish(tmp_path: Path) -> None:
    """Notes are the firm's private judgment; spend is firm-internal.
    Neither may reach a public page."""
    store = app_store(tmp_path)
    store.record_llm_usage("classify", "m", cost_usd=7.77)
    page = build_digest(store, Thesis(thesis="Robots"),
                        tmp_path / "docs").read_text(encoding="utf-8")
    assert "PRIVATE-NOTE-do-not-publish" not in page
    assert "7.77" not in page and "spend" not in page.lower()


# --- Vercel: the same output directory, deployable with a password ------------

runner = CliRunner()
THESIS_PATH = Path(__file__).resolve().parent.parent / "thesis.yaml"


def test_vercel_files_are_written_and_gate_only_the_page(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    out = build_digest(store, Thesis(thesis="t"), tmp_path / "docs", stamp="20260915120000")
    docs = out.parent

    config = json.loads((docs / "vercel.json").read_text(encoding="utf-8"))
    headers = {h["key"]: h["value"] for rule in config["headers"] for h in rule["headers"]}
    assert headers["X-Robots-Tag"].startswith("noindex")
    revalidated = next(r for r in config["headers"] if "index.html" in r["source"])
    assert "must-revalidate" in revalidated["headers"][0]["value"]

    assert "cleanUrls" not in config  # it 308s index.html, which the sw pre-caches
    assert json.loads((docs / "package.json").read_text())["type"] == "module"

    middleware = (docs / "middleware.js").read_text(encoding="utf-8")
    assert 'runtime: "nodejs"' in middleware  # "edge" is deprecated for middleware
    assert "export default function middleware" in middleware
    assert "DIGEST_PASSWORD" in middleware and "WWW-Authenticate" in middleware
    # The manifest and icon are fetched without credentials by the browser;
    # gating them would break "Add to Home Screen".
    assert "manifest" in middleware and "icon" in middleware
    assert "if (!expected) return;" in middleware  # unset = open, never a lockout

    assert (docs / "robots.txt").read_text(encoding="utf-8").strip().endswith("Disallow: /")
    assert ".vercel/" in (docs / ".gitignore").read_text(encoding="utf-8")

    sw = (docs / "sw.js").read_text(encoding="utf-8")
    assert 'CACHE = "scout-digest-20260915120000"' in sw  # a new publish, a new cache
    assert "caches.delete" in sw                          # old caches evicted on activate
    assert "resp.ok" in sw                                # a 401 is never cached as the app

    # No file in the output carries a secret — the password lives in Vercel.
    for path in docs.iterdir():
        if path.is_file():
            assert "DIGEST_PASSWORD=" not in path.read_bytes().decode("utf-8", "ignore")


def _publish(tmp_path: Path, *args: str):
    return runner.invoke(
        app, ["publish", *args, "--thesis", str(THESIS_PATH)],
        env={"DB_PATH": str(tmp_path / "scout.db"), "DIGEST_REPO": "", "VERCEL_TOKEN": ""},
    )


def test_publish_auto_renders_only_when_no_host_is_configured(tmp_path: Path, monkeypatch) -> None:
    """The worker's daily form must never fail for want of a host."""
    monkeypatch.chdir(tmp_path)
    Store(tmp_path / "scout.db")
    result = _publish(tmp_path, "--auto")
    assert result.exit_code == 0, result.output
    assert "No deploy target configured" in result.output
    assert (tmp_path / "docs" / "middleware.js").exists()


def test_publish_vercel_without_the_cli_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    import shutil

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name, *a, **kw: None)
    Store(tmp_path / "scout.db")
    result = _publish(tmp_path, "--vercel")
    assert result.exit_code == 1
    assert "Vercel CLI not found" in result.output


def test_publish_vercel_deploys_a_linked_project_and_auto_finds_it(tmp_path: Path, monkeypatch) -> None:
    from scout import cli

    monkeypatch.chdir(tmp_path)
    Store(tmp_path / "scout.db")
    calls: list[Path] = []
    monkeypatch.setattr(cli, "_deploy_to_vercel",
                        lambda docs, settings: calls.append(docs) or "https://scout-digest.vercel.app")

    result = _publish(tmp_path, "--vercel")
    assert result.exit_code == 0, result.output
    assert "scout-digest.vercel.app" in result.output
    assert "DIGEST_PASSWORD" in result.output  # the reminder that the password lives in Vercel
    assert calls == [Path("docs")]

    # --auto deploys once docs/ is linked (`vercel link` writes .vercel/project.json).
    (tmp_path / "docs" / ".vercel").mkdir()
    (tmp_path / "docs" / ".vercel" / "project.json").write_text("{}")
    result = _publish(tmp_path, "--auto")
    assert result.exit_code == 0, result.output
    assert len(calls) == 2 and "No deploy target" not in result.output


_NODE_HARNESS = """
import middleware, { config } from "./mw.mjs";
const b64 = (s) => Buffer.from(s, "utf8").toString("base64");
const run = async (pw, auth) => {
  if (pw === null) delete process.env.DIGEST_PASSWORD; else process.env.DIGEST_PASSWORD = pw;
  const res = await middleware(new Request("https://x.test/",
    { headers: auth ? { authorization: auth } : {} }));
  return res === undefined ? "pass" : res.status;
};
const re = new RegExp("^" + config.matcher[0] + "$");
console.log(JSON.stringify({
  open: await run(null, null),
  noHeader: await run("hunter2", null),
  wrong: await run("hunter2", "Basic " + b64("u:nope")),
  right: await run("hunter2", "Basic " + b64("partner:hunter2")),
  colon: await run("a:b", "Basic " + b64("u:a:b")),
  unicode: await run("p\u00e4ssw\u00f6rd", "Basic " + b64("u:p\u00e4ssw\u00f6rd")),
  garbage: await run("hunter2", "Basic !!!"),
  bearer: await run("hunter2", "Bearer hunter2"),
  gated: ["/", "/index.html", "/sw.js", "/iconXpng"].map((p) => re.test(p)),
  exempt: ["/manifest.webmanifest", "/icon.png"].map((p) => re.test(p)),
}));
"""


def test_middleware_behaves_under_node(tmp_path: Path) -> None:
    """Run the generated middleware, not just read it. String checks passed
    while a non-ASCII password could never match (atob yields bytes, the
    browser sends UTF-8) and unescaped dots in the matcher exempted any
    path shaped like icon?png. Skipped where node is absent."""
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not installed")
    out = build_digest(seeded_store(tmp_path), Thesis(thesis="t"), tmp_path / "docs")
    shutil.copy(out.parent / "middleware.js", tmp_path / "mw.mjs")
    (tmp_path / "harness.mjs").write_text(_NODE_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", "harness.mjs"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    assert got["open"] == "pass"            # no password configured: never a lockout
    assert got["noHeader"] == 401 and got["wrong"] == 401
    assert got["right"] == "pass" and got["colon"] == "pass"
    assert got["unicode"] == "pass"
    assert got["garbage"] == 401 and got["bearer"] == 401
    assert got["gated"] == [True, True, True, True]
    assert got["exempt"] == [False, False]

