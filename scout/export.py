"""CSV + Markdown report writers, plus the rich top-10 terminal table (spec §5)."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scout.companies import startup_identity
from scout.config import Thesis
from scout.models import Lead

CSV_COLUMNS = [
    "rank",
    "startup",
    "handle",
    "name",
    "company_name",
    "company_url",
    "url",
    "score",
    "customer_type",
    "quality_score",
    "scorecard_rubric",
    "scorecard_band",
    "scorecard_sections",
    "scorecard",
    "scorecard_reasons",
    "quality",
    "quality_reasons",
    "product_summary",
    "grounding",
    "verification",
    "thesis_fit",
    "fit_reason",
    "value_add_fit",
    "value_add_reason",
    "value_add_levers",
    "stage",
    "sector",
    "subsector",
    "business_model",
    "tags",
    "one_line_summary",
    "why_interesting",
    "signals_hit",
    "followed_by",
    "bio",
    "evidence_links",
]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def _oneline(text: str, width: int = 140) -> str:
    """Collapse whitespace (bios have newlines) and truncate for compact output."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "…"


def write_csv(leads: list[Lead], out_dir: Path, thesis: Thesis | None = None) -> Path:
    """`thesis` supplies the rubric weights for the deterministic
    quality_score / scorecard columns; None (legacy callers) leaves them
    blank."""
    # Local import — score imports models only.
    from scout.score import company_quality, scorecard_score

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"leads_{_stamp()}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for lead in leads:
            llm = lead.llm
            q_score = company_quality(llm, thesis) if thesis is not None else None
            scorecard = scorecard_score(llm, thesis) if thesis is not None else None
            sc_result, sc_sections = scorecard if scorecard is not None else (None, [])
            identity = startup_identity(lead)
            writer.writerow(
                {
                    "rank": lead.rank if lead.rank is not None else "",
                    "startup": identity[0] if identity else "",
                    "handle": lead.account.handle,
                    "name": lead.account.name,
                    "company_name": (llm.company_name or "") if llm else "",
                    "company_url": (llm.company_url or "") if llm else "",
                    "url": lead.account.url,
                    "score": lead.score,
                    "customer_type": (llm.customer_type or "") if llm else "",
                    "quality_score": f"{q_score:.0f}" if q_score is not None else "",
                    "scorecard_rubric": sc_result.rubric_key if sc_result else "",
                    "scorecard_band": sc_result.band if sc_result else "",
                    "scorecard_sections": ";".join(
                        f"{s.key}={s.score:.0f}"
                        for s in sc_sections if s.score is not None
                    ),
                    "scorecard": (
                        ";".join(f"{k}={v:.0f}" for k, v in
                                 sorted(llm.scorecard.items(), key=lambda kv: -kv[1]))
                        if llm else ""
                    ),
                    "scorecard_reasons": (
                        " | ".join(f"{k}: {r}" for k, r in
                                   sorted(llm.scorecard_reasons.items()))
                        if llm else ""
                    ),
                    "quality": (
                        ";".join(f"{k}={v:.2f}" for k, v in
                                 sorted(llm.quality.items(), key=lambda kv: -kv[1]))
                        if llm else ""
                    ),
                    "quality_reasons": (
                        " | ".join(f"{k}: {r}" for k, r in sorted(llm.quality_reasons.items()))
                        if llm else ""
                    ),
                    "product_summary": (llm.product_summary or "") if llm else "",
                    "grounding": (llm.grounding or "") if llm else "",
                    "verification": (llm.verification or "") if llm else "",
                    "thesis_fit": (
                        f"{llm.thesis_fit:.2f}"
                        if llm and llm.thesis_fit is not None
                        else ""
                    ),
                    "fit_reason": llm.fit_reason if llm else "",
                    "value_add_fit": (
                        f"{llm.value_add_fit:.2f}"
                        if llm and llm.value_add_fit is not None
                        else ""
                    ),
                    "value_add_reason": llm.value_add_reason if llm else "",
                    "value_add_levers": (
                        ";".join(
                            f"{k}={v:.2f}"
                            for k, v in sorted(
                                llm.value_add_levers.items(), key=lambda kv: -kv[1]
                            )
                        )
                        if llm
                        else ""
                    ),
                    "stage": (llm.stage or "") if llm else "",
                    "sector": (llm.sector or "") if llm else "",
                    "subsector": (llm.subsector or "") if llm else "",
                    "business_model": (llm.business_model or "") if llm else "",
                    "tags": ";".join(llm.tags) if llm else "",
                    "one_line_summary": llm.one_line_summary if llm else "",
                    "why_interesting": llm.why_interesting if llm else "",
                    "signals_hit": ";".join(lead.signals_hit),
                    "followed_by": ";".join(lead.account.followed_by),
                    "bio": lead.account.bio,
                    "evidence_links": ";".join(lead.evidence_links),
                }
            )
    return path


PIPELINE_COLUMNS = [
    "startup", "handle", "name", "url", "status", "score", "thesis_fit",
    "value_add_fit", "notes", "channel", "outreach", "brief", "updated_at",
]


def pipeline_rows(store, thesis: Thesis | None = None) -> list[dict]:
    """Deal-flow rows (any status beyond 'new') joined with each lead's
    best-known ledger state. `store` is a scout.store.Store (duck-typed to
    avoid an import cycle in type-checking-averse callers). With a thesis,
    the investor's manual score overrides are applied so the export matches
    what the UI shows."""
    from scout.score import apply_override  # local import — avoids a cycle

    ledger = {e.lead.account.handle.lower(): e.lead for e in
              store.load_lead_ledger(include_demo=True)}
    if thesis is not None:
        overrides = store.all_overrides()
        for handle, lead in ledger.items():
            apply_override(lead, overrides.get(handle), thesis)
    # The user-owned database columns ride along as extra CSV columns.
    attr_columns = store.list_columns()
    attrs = store.all_attrs()
    rows: list[dict] = []
    for handle, entry in sorted(store.all_pipeline().items()):
        status = entry.get("status") or "new"
        if status == "new":
            continue
        lead = ledger.get(handle)
        verdict = lead.llm if lead else None
        identity = startup_identity(lead) if lead else None
        attr_cells = {}
        for column in attr_columns:
            value = (attrs.get(handle) or {}).get(column["key"])
            if isinstance(value, list):
                value = ";".join(str(v) for v in value)
            elif value is True:
                value = "yes"
            elif value in (None, False):
                value = ""
            attr_cells[column["label"]] = value
        rows.append({
            **attr_cells,
            "startup": identity[0] if identity else "",
            "handle": handle,
            "name": lead.account.name if lead else "",
            "url": lead.account.url if lead else f"https://x.com/{handle}",
            "status": status,
            "score": lead.score if lead else "",
            "thesis_fit": (
                f"{verdict.thesis_fit:.2f}"
                if verdict and verdict.thesis_fit is not None else ""
            ),
            "value_add_fit": (
                f"{verdict.value_add_fit:.2f}"
                if verdict and verdict.value_add_fit is not None else ""
            ),
            "notes": entry.get("notes") or "",
            "channel": entry.get("channel") or "",
            "outreach": entry.get("outreach") or "",
            "brief": entry.get("brief") or "",
            "updated_at": entry.get("updated_at") or "",
        })
    return rows


def write_pipeline_csv(rows: list[dict], out_dir: Path) -> Path:
    """Deal-flow export (status, notes, outreach, briefs) — CRM-import-ready.
    User-defined database columns present in the rows are appended after the
    fixed columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pipeline_{_stamp()}.csv"
    extra = [k for k in (rows[0] if rows else {}) if k not in PIPELINE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_COLUMNS + extra)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Print-friendly take on the app's editorial language: serif display headings,
# hairline section rules, warm-black ink on paper white.
_MEMO_PDF_CSS = """
body { font-family: "Helvetica", sans-serif; font-size: 10pt; color: #20180f;
       line-height: 1.55; }
h1 { font-family: "Georgia", serif; font-size: 21pt; font-weight: 600;
     margin: 0 0 2pt; line-height: 1.15; }
h2 { font-family: "Georgia", serif; font-size: 13.5pt; font-weight: 600;
     margin: 16pt 0 4pt; padding-bottom: 3pt;
     border-bottom: 0.6pt solid #c9beac; }
h3 { font-family: "Georgia", serif; font-size: 11.5pt; font-weight: 600;
     margin: 10pt 0 2pt; }
p { margin: 5pt 0; }
li { margin: 2pt 0; }
strong { color: #20180f; }
em { color: #4a4034; }
code { font-size: 9pt; }
blockquote { color: #4a4034; margin: 6pt 0 6pt 10pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th { font-size: 8pt; text-transform: uppercase; letter-spacing: 0.06em;
     color: #6b6052; text-align: left; padding: 4pt 8pt 4pt 0;
     border-bottom: 0.8pt solid #a89a83; }
td { font-size: 9pt; color: #4a4034; padding: 4.5pt 8pt 4.5pt 0;
     border-bottom: 0.5pt solid #d8cdbb; vertical-align: top; }
a { color: #7a5d1e; }
"""


def memo_pdf_bytes(memo_md: str, title: str, subtitle: str = "") -> bytes:
    """Render an investment memo's markdown to a styled PDF.

    Prepends a title block (startup name + a context line) to the memo body.
    Raises RuntimeError with an actionable message when the optional
    markdown-pdf dependency is missing."""
    try:
        from markdown_pdf import MarkdownPdf, Section
    except ImportError as exc:  # pragma: no cover — dep is in pyproject
        raise RuntimeError(
            "PDF export needs the markdown-pdf package — run `uv sync`."
        ) from exc
    import tempfile

    header = f"# {title}\n"
    if subtitle:
        header += f"*{subtitle}*\n\n"
    pdf = MarkdownPdf(toc_level=0, optimize=True)
    pdf.add_section(Section(header + memo_md.strip() + "\n"), user_css=_MEMO_PDF_CSS)
    pdf.meta["title"] = title
    pdf.meta["creationDate"] = ""  # fitz rejects Python datetimes here
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        pdf.save(str(path))
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _lead_card_lines(lead: Lead, index: int, firm: str = "") -> list[str]:
    account = lead.account
    llm = lead.llm
    # Startup-first: real company name, or the synthesized stealth identity
    # ("Ada Lin's stealth startup"); account title only for non-founders.
    identity = startup_identity(lead)
    title = identity[0] if identity else (account.name or account.handle)
    who = (llm.one_line_summary if llm else "") or _oneline(account.bio) or "(no bio)"
    facts = []
    if llm and llm.thesis_fit is not None:
        facts.append(f"fit {llm.thesis_fit:.0%}")
    if llm and llm.value_add_fit is not None:
        facts.append(f"{firm or 'firm'} lift {llm.value_add_fit:.0%}")
    if llm and llm.subsector:
        facts.append(llm.subsector)
    if llm and llm.business_model:
        facts.append(llm.business_model)
    if facts:
        who += f" ({' · '.join(facts)})"
    signals = ", ".join(lead.signals_hit) or "none"
    why_link = f"Signals: {signals}."
    if llm and llm.why_interesting:
        why_link += f" {llm.why_interesting}"
    why_link += f" {account.url}"
    if llm and llm.company_url:
        why_link += f" · {llm.company_url}"
    return [
        f"## {index}. {title} — @{account.handle} ({lead.score})",
        who,
        why_link,
        "",
    ]


def write_markdown(leads: list[Lead], thesis: Thesis, out_dir: Path) -> Path:
    """The one-pager: LAUNCHED STARTUPS first (the product), then the
    pre-launch watch list (people expected to launch soon — completeness)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{_stamp()}.md"
    date = datetime.now().strftime("%Y-%m-%d")

    def is_launched(lead: Lead) -> bool:
        return (lead.llm is not None
                and lead.llm.stage in ("launched", "scaling")
                and (lead.llm.account_type or "other") != "other")

    launched = [x for x in leads if is_launched(x)][:20]
    watch = [x for x in leads if not is_launched(x)][:10]

    lines = [f"# Scout — {thesis.thesis} — {date}", ""]
    lines += [f"## Launched startups ({len(launched)})", ""]
    if launched:
        for i, lead in enumerate(launched, start=1):
            lines += _lead_card_lines(lead, i, thesis.firm_name)
    else:
        lines += ["(none in this run)", ""]
    if watch:
        lines += [f"## Pre-launch watch ({len(watch)}) — expected to launch or found soon", ""]
        for i, lead in enumerate(watch, start=1):
            lines += _lead_card_lines(lead, i, thesis.firm_name)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_top_table(leads: list[Lead], n: int = 10) -> None:
    table = Table(title=f"Top {min(n, len(leads))} leads")
    table.add_column("Rank", justify="right")
    table.add_column("Handle", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Stage")
    table.add_column("Signals hit")
    table.add_column("One-liner", overflow="ellipsis", no_wrap=True)
    for lead in leads[:n]:
        llm = lead.llm
        one_liner = (llm.one_line_summary if llm else "") or lead.account.bio
        table.add_row(
            str(lead.rank) if lead.rank is not None else "",
            f"@{lead.account.handle}",
            f"{lead.score:.1f}",
            (llm.stage or "-") if llm else "-",
            ", ".join(lead.signals_hit),
            _oneline(one_liner, 60),
        )
    Console().print(table)
