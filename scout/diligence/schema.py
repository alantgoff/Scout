"""Diligence data models + pure parse helpers.

Structured outputs are NOT guaranteed on the research model, so every agent
keeps scout's proven JSON-in-text idiom: strip fences -> json.loads ->
validate -> sanitize/clamp, with a corrective retry driven by the caller
(research.py). All helpers here are pure and unit-tested.

Every factual claim an agent makes must carry an Evidence entry — a URL, or
an explicit source tag ("x-data" for claims derived from scout's own data,
"assumption" for stated assumptions). An honest gap goes in `unknowns`.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from scout.llmcall import strip_code_fences  # noqa: F401 — re-exported: the
# diligence parse helpers and their tests treat this module as the one-stop
# parsing surface.

# The 8 diligence dimensions, in display order.
DIMENSION_KEYS: tuple[str, ...] = (
    "founder_pedigree",
    "technical_differentiation",
    "wrapper_vs_proprietary",
    "deployment_model",
    "data_flywheel",
    "market_size",
    "defensibility_vs_labs",
    "competition",
)

DIMENSION_LABELS: dict[str, str] = {
    "founder_pedigree": "Founder pedigree",
    "technical_differentiation": "Technical differentiation",
    "wrapper_vs_proprietary": "Wrapper vs proprietary",
    "deployment_model": "Deployment model",
    "data_flywheel": "Data flywheel",
    "market_size": "Market size",
    "defensibility_vs_labs": "Defensibility vs labs",
    "competition": "Competition",
}

WRAPPER_CLASSES = (
    "thin_wrapper", "workflow_wrapper", "fine_tuned", "proprietary_model", "hybrid",
)
DEPLOYMENT_CLASSES = (
    "cloud_saas", "on_prem", "hybrid", "vpc_byoc", "api_only", "edge",
)
EVIDENCE_SOURCES = ("web", "x-data", "assumption")


# ------------------------------------------------------------------ core models


class Evidence(BaseModel):
    """One factual claim with its provenance. url may be "" only for claims
    derived from scout's own X data or explicitly tagged assumptions."""

    claim: str
    url: str = ""
    source: str = "web"  # "web" | "x-data" | "assumption"


class DimensionFinding(BaseModel):
    """One dimension agent's structured verdict."""

    key: str  # one of DIMENSION_KEYS
    score: float | None = None  # 0-10; None = analysis failed / insufficient data
    confidence: float = 0.0  # 0-1, downgradable by cross-examination
    summary: str = ""
    detail_md: str = ""  # the stepped analysis, markdown
    evidence: list[Evidence] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)  # explicit "couldn't determine"
    flags: list[str] = Field(default_factory=list)  # e.g. "thin_wrapper"
    payload: dict = Field(default_factory=dict)  # dimension-specific typed extras


class ReconFounder(BaseModel):
    name: str = ""
    role: str = ""
    background_hints: str = ""


class KeySource(BaseModel):
    url: str = ""
    title: str = ""
    what: str = ""  # what this source is good for


class ReconReport(BaseModel):
    """Shared ground-truth context handed to every dimension agent."""

    company_name: str = ""
    website: str = ""
    product_summary: str = ""
    founders: list[ReconFounder] = Field(default_factory=list)
    funding_events: list[str] = Field(default_factory=list)
    github_org: str = ""
    key_sources: list[KeySource] = Field(default_factory=list)


class CrossExam(BaseModel):
    """Cross-examination output: confidence DOWNGRADES only (a value is
    applied only when lower than the finding's own confidence) plus the
    contradictions that motivated them."""

    downgrades: dict[str, float] = Field(default_factory=dict)  # key -> confidence
    contradictions: list[str] = Field(default_factory=list)


class Memo(BaseModel):
    """A persisted deep-analysis result (store.memos, pk company_key)."""

    company_key: str
    company_name: str = ""
    fingerprint: str = ""
    composite: float | None = None  # 0-100 Diligence Score (NOT the screening score)
    flags: list[str] = Field(default_factory=list)  # e.g. "commodity_risk", "budget_capped"
    findings: list[DimensionFinding] = Field(default_factory=list)
    recon: ReconReport = Field(default_factory=ReconReport)
    memo_md: str = ""
    cost_usd: float = 0.0
    model: str = ""  # research model (recon + dimensions)
    synth_model: str = ""  # cross-exam + memo synthesis model
    created_at: str = ""  # ISO timestamp

    def finding(self, key: str) -> DimensionFinding | None:
        return next((f for f in self.findings if f.key == key), None)


# ----------------------------------------------------- dimension payload models


class PedigreeFounder(BaseModel):
    name: str = ""
    verified_background: list[str] = Field(default_factory=list)
    prior_exits: list[str] = Field(default_factory=list)


class FounderPedigreePayload(BaseModel):
    founders: list[PedigreeFounder] = Field(default_factory=list)


class WrapperPayload(BaseModel):
    classification: str = ""  # one of WRAPPER_CLASSES
    model_dependency: str = ""
    evidence_trail: list[str] = Field(default_factory=list)


class DeploymentPayload(BaseModel):
    classification: str = ""  # one of DEPLOYMENT_CLASSES
    compliance: list[str] = Field(default_factory=list)  # SOC2/HIPAA/FedRAMP mentions
    buyer_implications: str = ""


class TopDownEstimate(BaseModel):
    low: float | None = None  # USD millions
    base: float | None = None
    high: float | None = None
    year: int | None = None
    cagr: str = ""
    sources: list[str] = Field(default_factory=list)  # URLs


class BottomUpEstimate(BaseModel):
    buyer_count: float | None = None
    acv_low: float | None = None  # USD
    acv_high: float | None = None
    derived_range: str = ""
    assumptions: list[str] = Field(default_factory=list)


class MarketSizePayload(BaseModel):
    top_down: TopDownEstimate = Field(default_factory=TopDownEstimate)
    bottom_up: BottomUpEstimate = Field(default_factory=BottomUpEstimate)
    reconciliation: str = ""


class DefensibilityPayload(BaseModel):
    lab_roadmap_risk: str = ""  # is this on the labs' obvious roadmap?
    time_to_parity: str = ""
    surviving_moats: list[str] = Field(default_factory=list)


class Competitor(BaseModel):
    name: str = ""
    url: str = ""
    stage: str = ""
    note: str = ""


class CompetitionPayload(BaseModel):
    competitors: list[Competitor] = Field(default_factory=list)
    positioning_axis: str = ""
    whitespace: str = ""


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "founder_pedigree": FounderPedigreePayload,
    "wrapper_vs_proprietary": WrapperPayload,
    "deployment_model": DeploymentPayload,
    "market_size": MarketSizePayload,
    "defensibility_vs_labs": DefensibilityPayload,
    "competition": CompetitionPayload,
}


# -------------------------------------------------------------- parse helpers


def extract_json(text: str):
    """Parse the JSON object out of an agent reply. Research agents narrate
    around their JSON ("Let me search...") more than plain classifiers, so a
    failed direct parse falls back to the outermost {...} span — cheaper than
    a corrective retry that re-runs web research."""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


# Dimensions whose payload carries a `classification` drawn from a fixed
# vocabulary — normalized (and enforced) generically in sanitize_finding.
CLASSIFIED_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "wrapper_vs_proprietary": WRAPPER_CLASSES,
    "deployment_model": DEPLOYMENT_CLASSES,
}


def sanitize_finding(finding: DimensionFinding) -> DimensionFinding:
    """Clamp scores, normalize evidence sources, validate the payload against
    the dimension's typed model (in place; returned for chaining)."""
    if finding.score is not None:
        finding.score = _clamp(finding.score, 0.0, 10.0)
    finding.confidence = _clamp(finding.confidence, 0.0, 1.0)
    for ev in finding.evidence:
        if ev.source not in EVIDENCE_SOURCES:
            ev.source = "web" if ev.url else "assumption"
        # Law 1: a claim with no URL must be explicitly x-data or assumption.
        if not ev.url and ev.source == "web":
            ev.source = "assumption"
    model = PAYLOAD_MODELS.get(finding.key)
    if model is not None and finding.payload:
        try:
            validated = model.model_validate(finding.payload)
            allowed = CLASSIFIED_DIMENSIONS.get(finding.key)
            if allowed is not None:
                cleaned = (
                    validated.classification.strip().lower()  # type: ignore[attr-defined]
                    .replace(" ", "_").replace("-", "_")
                )
                validated.classification = cleaned  # type: ignore[attr-defined]
                # Off-vocabulary classes pass through (downstream exact-match
                # rules like the commodity cap just won't fire) but are
                # flagged so the gap is visible instead of silent.
                if cleaned not in allowed and "unrecognized_classification" not in finding.flags:
                    finding.flags.append("unrecognized_classification")
            finding.payload = validated.model_dump()
        except Exception:
            # Keep the raw payload (it may still be human-readable) but mark it.
            if "payload_invalid" not in finding.flags:
                finding.flags.append("payload_invalid")
    return finding


def parse_finding(text: str, key: str) -> DimensionFinding:
    """Parse one dimension agent's JSON reply. Pure — unit-testable.

    The dimension key is forced from the caller (an agent echoing the wrong
    key must not corrupt another dimension's slot). Unknown fields are
    tolerated; score/confidence are clamped; the payload is validated against
    the dimension's typed model.
    """
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    data["key"] = key
    return sanitize_finding(DimensionFinding.model_validate(data))


def parse_recon(text: str) -> ReconReport:
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return ReconReport.model_validate(data)


def parse_cross_exam(text: str) -> CrossExam:
    """Parse the cross-examination reply. Unknown dimension keys are dropped;
    confidences are clamped to 0-1."""
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    exam = CrossExam.model_validate(data)
    exam.downgrades = {
        key: _clamp(value, 0.0, 1.0)
        for key, value in exam.downgrades.items()
        if key in DIMENSION_KEYS
    }
    return exam


def apply_cross_exam(findings: list[DimensionFinding], exam: CrossExam) -> None:
    """Apply confidence DOWNGRADES in place — cross-examination may only
    lower confidence, never raise it."""
    for finding in findings:
        new = exam.downgrades.get(finding.key)
        if new is not None and new < finding.confidence:
            finding.confidence = new
            if "cross_exam_downgrade" not in finding.flags:
                finding.flags.append("cross_exam_downgrade")


def gap_finding(key: str, reason: str, flag: str = "analysis_failed") -> DimensionFinding:
    """The placeholder recorded when a dimension agent fails or is skipped —
    a failed dimension must never abort the run (or fake a score)."""
    return DimensionFinding(
        key=key, score=None, confidence=0.0,
        summary=f"Analysis unavailable: {reason}",
        unknowns=[reason], flags=[flag],
    )
