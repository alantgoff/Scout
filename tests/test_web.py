"""Website-evidence pure functions: URL normalization + text extraction.

House style — no network anywhere; the async fetcher is a thin wrapper and
stays untested like the ingest adapters.
"""

from __future__ import annotations

from scout.web import extract_site_text, normalize_site_url

REAL_PAGE = """
<!doctype html>
<html><head>
<title>Raindrop | AI Agent Monitoring &amp; Observability</title>
<meta name="description" content="Trace every run. Auto-fix silent failures. Ship confidently.">
<meta property="og:description" content="Agents fail silently. Fix them fast.">
<style>.hero { color: red; }</style>
<script>window.analytics = "junk";</script>
</head>
<body>
<nav>Product Docs Blog Careers</nav>
<main>
  <h1>Agents fail silently. Fix them fast.</h1>
  <p>Raindrop captures every production run — messages, tool calls, retries,
  and errors — in one place, and sends the real failures to Slack.</p>
  <script>console.log("more junk")</script>
  <svg><path d="M0 0"/></svg>
</main>
</body></html>
"""

SPA_PAGE = """
<!doctype html>
<html><head>
<title>Acme Robotics</title>
<meta property="og:description" content="Autonomous forklifts for mid-size warehouses.">
<script src="/bundle.js"></script>
</head>
<body><div id="root"></div><script>boot()</script></body></html>
"""


def test_extract_strips_junk_and_keeps_product_copy() -> None:
    text = extract_site_text(REAL_PAGE, max_chars=5000)
    assert "AI Agent Monitoring" in text
    assert "Trace every run" in text  # meta description
    assert "captures every production run" in text  # body copy
    assert "window.analytics" not in text  # script dropped
    assert "color: red" not in text  # style dropped
    assert "M0 0" not in text  # svg dropped


def test_extract_truncates() -> None:
    text = extract_site_text(REAL_PAGE, max_chars=60)
    assert len(text) <= 60


def test_extract_spa_falls_back_to_meta() -> None:
    text = extract_site_text(SPA_PAGE, max_chars=5000)
    assert "Autonomous forklifts" in text
    assert "Acme Robotics" in text


def test_normalize_strips_path_to_root() -> None:
    # The Raindrop case verbatim: a /careers link must become the homepage.
    assert normalize_site_url("http://raindrop.ai/careers") == "https://raindrop.ai/"
    assert normalize_site_url("raindrop.ai") == "https://raindrop.ai/"
    assert normalize_site_url("https://WWW.Example.COM/about?x=1#y") == "https://www.example.com/"


def test_normalize_rejects_junk() -> None:
    assert normalize_site_url(None) is None
    assert normalize_site_url("") is None
    assert normalize_site_url("mailto:a@b.co") is None
    assert normalize_site_url("ftp://files.example.com") is None
    assert normalize_site_url("https://localhost") is None
    assert normalize_site_url("http://127.0.0.1/admin") is None
    assert normalize_site_url("no-dots") is None


def test_normalize_skips_link_farms_and_socials() -> None:
    for url in ("https://linktr.ee/founder", "https://x.com/someone",
                "https://t.co/abc123", "https://www.instagram.com/someone",
                "https://github.com/org/repo", "https://calendly.com/founder"):
        assert normalize_site_url(url) is None, url
    # ...but real product domains survive
    assert normalize_site_url("https://linktree-competitor.com") is not None
