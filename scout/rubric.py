"""Startup scorecards: the B2B (enterprise readiness) and B2C (consumer
readiness) rubrics behind the company-quality component.

Derived from the V5 "Enterprise Readiness Evaluation Framework" scorecard,
refined for the evidence available at sourcing time (X + website dossiers,
not data rooms): the Strategic Fit section is dropped (the separate
thesis-fit component covers it, its 15% redistributed proportionally),
late-stage enterprise criteria are merged, and every criterion is scored
1-3 against explicit anchors. Higher is ALWAYS better — for risk criteria
a 3 means low risk.

customer_type routes the rubric: "b2c" scores against the consumer
scorecard; "b2b", "b2b2c", "mixed" (and None — product not established)
against the enterprise one. The founders & cap table keys are shared
across rubrics on purpose, so a customer_type correction keeps founder
scores meaningful.

Pure data + tiny helpers: no I/O, no scout imports. score.py does the
math over these definitions; signals/llm.py renders `prompt_block()` into
the classifier prompt — the prompt can never drift from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str  # stable id — the verdict's scorecard dict key
    label: str  # UI display name
    purpose: str  # one line: what it measures and why it matters
    weight: float  # sub-weight within its section (each section sums to 1.0)
    anchors: tuple[str, str, str]  # rating guide for scores 1 / 2 / 3


@dataclass(frozen=True)
class Section:
    key: str
    label: str
    weight: float  # default section weight (each rubric's sum to 100)
    criteria: tuple[Criterion, ...]


@dataclass(frozen=True)
class Rubric:
    key: str  # "b2b" | "b2c"
    label: str  # display name, e.g. "Enterprise readiness"
    sections: tuple[Section, ...]

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return tuple(c for s in self.sections for c in s.criteria)


def _c(key: str, label: str, purpose: str, weight: float, *anchors: str) -> Criterion:
    assert len(anchors) == 3
    return Criterion(key, label, purpose, weight, (anchors[0], anchors[1], anchors[2]))


# --- Shared founders & cap table criteria (identical keys across rubrics; the
# purpose/anchor text flavors differ where the buyer changes what "strong"
# looks like) -----------------------------------------------------------------

def _founders_section(dna: Criterion, exit_anchor: str) -> Section:
    return Section(
        "founders_captable", "Founders & cap table", 18, (
            _c("prev_founder_experience", "Prior founder experience",
               "Proven startup execution — prior founded companies or exits.", 0.20,
               "first-time founder with no operator signal",
               "senior operator or early employee at a notable startup",
               exit_anchor),
            dna,
            _c("investor_quality", "Investor quality",
               "Named backers signal access, guidance, and follow-on potential.", 0.20,
               "raised, but only unnamed or unknown backers",
               "credible seed funds or an accelerator badge (e.g. YC W25)",
               "top-tier funds or renowned operator-angels, by name"),
            _c("cap_table_risk", "Cap table cleanliness",
               "Ownership structure risk — studio/agency builds, corporate "
               "majorities, or crowded angel lists depress future rounds "
               "(3 = clean).", 0.10,
               "venture-studio/agency-built or corporate-controlled",
               "accelerator equity or a long visible angel list",
               "clean founder-led table, named institutional seed only"),
            _c("capital_and_stage", "Capital raised & stage",
               "Financial runway and maturity — how much has been raised to "
               "date.", 0.15,
               "under $1M raised or no funding evidence despite claims",
               "$1–10M raised",
               "$10M+ raised"),
            _c("funding_velocity", "Funding velocity",
               "Round-to-round pace — a signal of traction and investor "
               "conviction.", 0.20,
               "over ~30 months since the last round, or stalled",
               "raised again within ~24–30 months",
               "raised again within ~18–24 months, or oversubscribed"),
        ))


_B2B_FOUNDER_DNA = _c(
    "domain_expertise", "Domain expertise",
    "Depth in exactly this problem space — relevant but less predictive than "
    "prior execution.", 0.15,
    "no visible connection to the domain",
    "adjacent experience — nearby industry or function",
    "years building in exactly this domain, specifics cited")

_B2C_FOUNDER_DNA = _c(
    "consumer_product_dna", "Consumer product DNA",
    "Taste and consumer instinct — design craft, storytelling, "
    "audience-building track record.", 0.15,
    "no consumer product or audience signal",
    "some consumer product work or a modest audience",
    "shipped beloved consumer products or built a large audience")


# --- B2B: enterprise readiness ----------------------------------------------

B2B_RUBRIC = Rubric("b2b", "Enterprise readiness", (
    _founders_section(
        _B2B_FOUNDER_DNA,
        "prior founder with an exit or a company scaled to real revenue"),
    Section("market", "Market", 23, (
        _c("problem_urgency", "Problem urgency & severity",
           "Strength of the enterprise need — hair-on-fire vs nice-to-have.", 0.20,
           "nice-to-have, discretionary spend",
           "clear pain, but deferrable",
           "urgent budgeted pain buyers already pay to solve"),
        _c("tailwinds_timing", "Tailwinds & timing",
           "Macro and tech forces (AI waves, platform shifts) accelerating "
           "adoption right now.", 0.20,
           "headwinds, or clearly too early/too late",
           "moderate tailwind, plausible timing",
           "strong secular tailwind with the adoption window open now"),
        _c("competitive_landscape", "Competitive landscape",
           "Differentiation and GTM cost — open field vs saturated.", 0.15,
           "saturated, incumbent-dominated",
           "competitive with room to differentiate",
           "open or newly created category"),
        _c("market_size", "Market size",
           "Ceiling for a venture-scale outcome (TAM/SAM).", 0.15,
           "niche, under ~$1B",
           "solid, ~$1–10B",
           "large or fast-expanding, $10B+"),
        _c("analyst_media_visibility", "Analyst & media visibility",
           "Company and category visibility in credible media, analyst, and "
           "community discourse.", 0.10,
           "no coverage or community presence",
           "some blog, social, or community mentions",
           "G2/analyst listings, press, strong community buzz"),
        _c("procurement_alignment", "Procurement & sales-cycle fit",
           "Understands enterprise buying — security reviews, RFPs, long "
           "sales cycles.", 0.10,
           "no sign of an enterprise sales motion",
           "some enterprise-sales awareness visible",
           "RFP/SOC2-ready posture, named enterprise motion"),
        _c("regulatory_environment", "Regulatory environment",
           "Whether regulation accelerates or blocks adoption in the target "
           "sector (3 = tailwind).", 0.10,
           "regulation blocks or threatens the model",
           "regulation neutral",
           "regulation drives adoption — compliance mandates, incentives"),
    )),
    Section("technology", "Technology", 23, (
        _c("tech_moat_ip", "Technical moat & proprietary IP",
           "Unique defensible core — proprietary data, models, hardware, deep "
           "tech — hard to replicate.", 0.30,
           "thin wrapper on commodity APIs, no unique asset",
           "solid engineering, replicable with effort",
           "cited proprietary data/hardware/research advantage"),
        _c("scalable_architecture", "Scalable architecture",
           "Ability to scale reliably — cloud-native design, performance "
           "under load, resilience.", 0.20,
           "prototype-grade infrastructure signals",
           "modern stack, unproven at scale",
           "cloud-native and benchmarked or proven at scale"),
        _c("integration_extensibility", "Integrations & extensibility",
           "Fits enterprise stacks — APIs, SDKs, connectors to common "
           "platforms (Salesforce, Azure, SAP...).", 0.20,
           "closed product, no API",
           "basic API or a few integrations",
           "rich API/SDK plus an integration marketplace"),
        _c("security_compliance", "Security & compliance",
           "Observable security posture — SOC2/ISO pages, trust center, "
           "access controls, data residency.", 0.20,
           "nothing visible despite selling to enterprises",
           "security page or in-progress compliance claims",
           "SOC2/ISO badges, trust center, DPA on the site"),
        _c("deployment_flexibility", "Deployment flexibility",
           "Deployment models buyers need — SaaS, VPC, on-prem, hybrid.", 0.10,
           "single-tenant SaaS only where buyers need more",
           "some options documented",
           "SaaS plus VPC/on-prem/hybrid documented"),
    )),
    Section("product", "Product", 18, (
        _c("product_maturity", "Product maturity",
           "How deployable the product is today.", 0.25,
           "prototype or waitlist only",
           "beta or early GA",
           "live GA product in production use"),
        _c("feature_completeness", "Feature completeness",
           "Coverage of the target workflow.", 0.20,
           "single narrow use case",
           "core activities covered",
           "end-to-end workflow covered"),
        _c("usability_onboarding", "Docs & onboarding",
           "Documentation, support, and time-to-value for a new team.", 0.20,
           "no docs, high-touch onboarding only",
           "partial docs, some self-serve",
           "extensive docs, self-serve start, fast time-to-value"),
        _c("pricing_clarity", "Pricing clarity",
           "Transparent, scalable pricing aligned to value.", 0.15,
           "opaque — contact-us only",
           "some pricing clarity",
           "transparent tiers including an enterprise tier"),
        _c("enterprise_readiness", "Enterprise controls",
           "Sold-to-IT features — SSO, RBAC, audit logs, admin and "
           "governance tooling.", 0.20,
           "none visible",
           "partial — SSO or basic admin only",
           "enterprise-grade controls documented"),
    )),
    Section("traction_bm", "Traction & business model", 18, (
        _c("commercial_traction", "Commercial traction",
           "Evidence of enterprise commercial interest — revenue, contracts, "
           "usage.", 0.25,
           "waitlist or design-partner claims only",
           "a few named pilots or early logos",
           "multiple paying customers, case studies, or revenue claims"),
        _c("customer_quality", "Customer quality",
           "Caliber of customers — complexity fit and budget.", 0.20,
           "hobbyists or tiny SMBs only",
           "mid-market logos",
           "Fortune-500/enterprise logos or marquee design partners"),
        _c("strategic_partnerships", "Strategic partnerships",
           "Distribution and credibility through partners — clouds, SIs, "
           "platforms.", 0.15,
           "none visible",
           "partnerships in progress or minor ones",
           "named partnerships with majors (cloud marketplaces, SIs)"),
        _c("gtm_repeatability", "GTM repeatability",
           "Sales motion scalability — founder-led vs playbook.", 0.15,
           "purely founder-led, bespoke deals",
           "an emerging repeatable motion",
           "playbook-ready: self-serve funnel or sales team on a clear ICP"),
        _c("business_model_quality", "Business model",
           "Recurring, predictable revenue model.", 0.15,
           "one-off projects or services",
           "usage-based or mixed",
           "recurring SaaS/platform contracts"),
        _c("expansion_signals", "Expansion signals",
           "Existing customers growing usage — land-and-expand evidence.", 0.10,
           "churn or flat-usage signals",
           "some expansion mentions",
           "clear expansion — seats, regions, or products growing"),
    )),
))


# --- B2C: consumer readiness -------------------------------------------------

B2C_RUBRIC = Rubric("b2c", "Consumer readiness", (
    _founders_section(
        _B2C_FOUNDER_DNA,
        "prior founder with an exit or a product scaled to millions of users"),
    Section("market", "Market", 23, (
        _c("consumer_desire", "Consumer desire",
           "Strength and frequency of the want — daily-habit potential vs "
           "occasional nice-to-have.", 0.20,
           "occasional, low-intensity want",
           "real recurring desire",
           "intense frequent need or want — daily-use potential"),
        _c("cultural_tailwinds_timing", "Cultural tailwinds & timing",
           "Behavior shifts and cultural moments powering adoption now.", 0.20,
           "fading fad, or clearly too early",
           "moderate trend alignment",
           "riding a strong behavioral or cultural shift right now"),
        _c("competitive_landscape", "Competitive landscape",
           "Differentiation cost — open field vs crowded.", 0.15,
           "crowded, incumbent-owned",
           "competitive with a clear wedge",
           "open or newly created category"),
        _c("market_size", "Audience size",
           "Reachable audience and spend ceiling.", 0.15,
           "narrow niche audience",
           "large segment",
           "mass-market audience or huge spend category"),
        _c("virality_potential", "Virality potential",
           "Built-in shareability of the category — content loops, invites, "
           "status.", 0.15,
           "solo utility with nothing to share",
           "some social surface",
           "inherently viral loops — UGC, invites, status display"),
        _c("platform_regulatory_risk", "Platform & regulatory risk",
           "Dependence on app-store/social-platform policy and regulatory "
           "exposure (3 = low risk).", 0.15,
           "existential platform or regulatory dependence",
           "manageable exposure",
           "low dependence, multi-channel distribution"),
    )),
    Section("product_experience", "Product & experience", 23, (
        _c("product_craft", "Product craft",
           "Design quality and delight — the bar consumers judge in "
           "seconds.", 0.25,
           "clunky or generic",
           "solid but unremarkable",
           "exceptional craft users rave about"),
        _c("time_to_value", "Time to value",
           "How fast a new user hits the magic moment.", 0.20,
           "slow setup, delayed payoff",
           "value within the first session",
           "instant magic in the first minutes"),
        _c("retention_mechanics", "Retention mechanics",
           "Habit loops — streaks, personalization, accumulating value, "
           "social ties.", 0.25,
           "no reason to come back",
           "some habit hooks",
           "strong compounding loops — data, streaks, social graph"),
        _c("platform_presence", "Platform presence",
           "Quality presence where the audience lives — iOS, Android, "
           "web.", 0.15,
           "wrong platform or a single weak one",
           "solid on the one platform that matters",
           "polished on every key platform"),
        _c("differentiation", "Differentiation",
           "A hook competitors can't trivially copy — unique mechanic, "
           "content, or experience.", 0.15,
           "me-too clone",
           "a notable twist",
           "distinctive mechanic or experience of its own"),
    )),
    Section("growth_engagement", "Growth & engagement", 18, (
        _c("user_growth", "User growth",
           "Acquisition trajectory — downloads, signups, waitlist "
           "momentum.", 0.25,
           "flat or no growth evidence",
           "steady growth claims",
           "rapid growth — climbing rankings, big user milestones"),
        _c("engagement_retention", "Engagement & retention",
           "Depth of usage — DAU/MAU claims, session frequency, stickiness "
           "in review sentiment.", 0.25,
           "shallow one-shot usage signals",
           "decent engagement claims",
           "strong retention evidence — habitual use"),
        _c("organic_acquisition", "Organic acquisition",
           "Word-of-mouth and viral distribution vs paid-only growth.", 0.20,
           "paid-only or no acquisition story",
           "some organic traction",
           "strong organic/viral growth — UGC spread, referrals"),
        _c("community_strength", "Community strength",
           "Owned community — Discord/Reddit, UGC, superfans.", 0.15,
           "no community",
           "small active community",
           "large passionate community creating content"),
        _c("social_proof", "Social proof",
           "App-store ratings and rankings, press, creator "
           "endorsements.", 0.15,
           "poor or no ratings",
           "good ratings, modest visibility",
           "top rankings, glowing reviews, earned media"),
    )),
    Section("monetization_bm", "Monetization & business model", 18, (
        _c("monetization_model", "Monetization model",
           "Clarity and fit of the revenue model — subscriptions, "
           "transactions, ads.", 0.25,
           "no monetization path",
           "plausible model, unproven",
           "clear working model matched to how the product is used"),
        _c("unit_economics_signals", "Unit economics signals",
           "Margin structure and price point vs likely acquisition "
           "cost.", 0.20,
           "structurally thin margins or expensive acquisition",
           "plausible economics",
           "high-margin with evidence of cheap acquisition"),
        _c("willingness_to_pay", "Willingness to pay",
           "Evidence consumers actually pay — conversion claims, paid-tier "
           "rankings, revenue milestones.", 0.20,
           "free-only usage",
           "some paying users",
           "strong conversion or revenue evidence"),
        _c("ltv_expansion", "LTV expansion",
           "Growing spend per user — deepening subscriptions, upsells, "
           "cross-sell.", 0.15,
           "one-time purchase ceiling",
           "some expansion paths",
           "clearly expanding lifetime value mechanics"),
        _c("brand_strength", "Brand strength",
           "Brand as the consumer moat — recognition, loyalty, cultural "
           "cachet.", 0.20,
           "generic, no brand",
           "an emerging identity",
           "a loved brand with cultural pull"),
    )),
))


RUBRICS: dict[str, Rubric] = {"b2b": B2B_RUBRIC, "b2c": B2C_RUBRIC}


def rubric_for(customer_type: str | None) -> Rubric:
    """b2c → consumer rubric; b2b / b2b2c / mixed / None → enterprise."""
    return B2C_RUBRIC if customer_type == "b2c" else B2B_RUBRIC


def default_section_weights() -> dict[str, dict[str, float]]:
    """{'b2b': {section: weight}, 'b2c': {...}} — Thesis.scorecard_weights
    default factory. Fresh dicts each call (pydantic mutates defaults)."""
    return {
        rubric.key: {section.key: section.weight for section in rubric.sections}
        for rubric in RUBRICS.values()
    }


def criterion_labels() -> dict[str, str]:
    """Criterion key → display label, merged across rubrics (shared founder
    keys collide with identical labels, so order doesn't matter)."""
    return {c.key: c.label for rubric in RUBRICS.values() for c in rubric.criteria}


def section_labels() -> dict[str, str]:
    return {s.key: s.label for rubric in RUBRICS.values() for s in rubric.sections}


# Interpretation bands over the 0-100 scorecard total (V5 sheet: 80-100
# ready / 60-79 promising / <60 likely too early).
BAND_LABELS: dict[str, str] = {
    "strong": "Strong",
    "promising": "Promising",
    "weak": "Too early",
}
BAND_HELP = "Scorecard band — Strong ≥80 · Promising 60–79 · Too early <60"


def band_for(total: float) -> str:
    if total >= 80:
        return "strong"
    if total >= 60:
        return "promising"
    return "weak"


def _criterion_line(criterion: Criterion) -> str:
    a1, a2, a3 = criterion.anchors
    return (f"- {criterion.key}: {criterion.purpose} "
            f"[1 = {a1} | 2 = {a2} | 3 = {a3}]")


def prompt_block() -> str:
    """Both scorecards rendered as classifier-prompt text — the single
    source of truth the prompt substitutes via the {scorecard} placeholder,
    so the prompt can never drift from this registry."""
    lines = [
        "SCORECARD — judge the COMPANY itself, criterion by criterion, from "
        "evidence only. Fill in EXACTLY ONE scorecard per account, routed by "
        'customer_type: "b2c" -> the CONSUMER scorecard; "b2b", "b2b2c", or '
        '"mixed" -> the ENTERPRISE scorecard; product not established -> at '
        "most the founders & cap table criteria of the ENTERPRISE scorecard.",
        "Each criterion scores 1 (weak), 2 (moderate), or 3 (strong) against "
        "its [1 | 2 | 3] anchors. Higher is ALWAYS better — for risk criteria "
        "3 means LOW risk. Score ONLY criteria the evidence supports — OMIT "
        "the key entirely when there is no evidence (most early startups have "
        "5-12 scorable criteria; omitting is honest, guessing is wrong). Put "
        'integer scores in "scorecard" and a <=12-word evidence citation per '
        'scored criterion in "scorecard_reasons" (e.g. "website: SOC2 page, '
        'Okta + Vercel logos").',
    ]
    for rubric in (B2B_RUBRIC, B2C_RUBRIC):
        routing = ("customer_type b2b / b2b2c / mixed / unknown"
                   if rubric.key == "b2b" else "customer_type b2c")
        lines.append("")
        lines.append(f"{rubric.label.upper()} SCORECARD ({routing}):")
        for section in rubric.sections:
            lines.append(f"{section.label}:")
            lines.extend(_criterion_line(c) for c in section.criteria)
    return "\n".join(lines)
