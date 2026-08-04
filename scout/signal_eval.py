"""Which public signals actually predict that a startup will raise?

The backtest asks whether the composite score worked. This asks the more
useful question underneath it: which of the individual signals — bio
intent, a founder's departure, GitHub traction, HN attention, smart-money
follows — carry real information, how much, and whether that is decaying
as the world catches on.

That question is easy to answer badly. A fund with fifteen outcomes and
thirty controls can produce a number for any signal it likes, and most of
those numbers will be noise wearing a decimal point. So the design premise
here is that **the statistics must be harder on themselves than a skeptical
reader would be**:

- Every AUC ships with a bootstrap confidence interval. A point estimate
  alone, at these sample sizes, is close to meaningless.
- Significance comes from a permutation test against shuffled labels, not
  a parametric assumption nobody checked.
- Testing twelve signals at p<0.05 produces a false positive more often
  than not, so p-values are corrected for multiple comparisons
  (Benjamini-Hochberg) and it is the corrected q-value that decides.
- The minimum detectable effect for the given sample size is computed and
  stated up front, so "we found nothing" can be distinguished from "we
  could never have found anything with this much data".
- Signals are evaluated on their weight-free normalized VALUES. Scoring
  them through the current weights would measure our own assumptions
  reflected back at us.

Pure functions throughout: no I/O, no clock, seeded randomness. Everything
here is unit-tested against cases with known answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

# Bootstrap/permutation sizes. 2,000 is ample for two-decimal reporting and
# keeps a full evaluation well under a second at these sample sizes.
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 2000
DEFAULT_SEED = 20260101
ALPHA = 0.05

# Below this, per-signal claims are suppressed rather than reported with a
# caveat — a caveat under a number still leaves the number on the slide.
MIN_SAMPLES_PER_GROUP = 5


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties are handled the way AUC requires."""
    order = values.argsort(kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranked[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranked


def auc(positives, negatives) -> float:
    """Probability a random positive outranks a random negative.

    Rank-based (Mann-Whitney U), which is O(n log n) and gives ties exactly
    the half-credit the pairwise definition does — so a signal with no
    discrimination scores 0.5 rather than looking perfect.
    """
    pos = np.asarray(positives, dtype=float)
    neg = np.asarray(negatives, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    combined = np.concatenate([pos, neg])
    ranked = _ranks(combined)
    rank_sum = ranked[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def perfectly_separated(positives, negatives) -> bool:
    """True when every positive outranks every negative (or the reverse)."""
    point = auc(positives, negatives)
    return point in (0.0, 1.0)


def _separation_bound(n_pos: int, n_neg: int, alpha: float) -> float:
    """Confidence bound when separation is perfect.

    Treat the n_pos × n_neg pairwise comparisons as Bernoulli trials, all
    of which were won. The two-sided (1−α) bound on the underlying win rate
    is then (α/2)^(1/N). With 3 positives and 3 negatives that is 0.66 —
    which is the honest reading of "perfect, on nine comparisons".

    This is OPTIMISTIC, because the pairs share samples and so are not
    independent. It is used only to replace a degenerate interval with a
    conservative-in-spirit one; it is never presented as exact.
    """
    pairs = max(1, n_pos * n_neg)
    return float((alpha / 2) ** (1.0 / pairs))


def auc_ci(
    positives, negatives, n_boot: int = N_BOOTSTRAP, seed: int = DEFAULT_SEED,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """(auc, low, high) — percentile bootstrap over both groups.

    Assumption-free and, more importantly, honest about how wide the
    interval really is at n=15. Seeded so a report is reproducible: an
    investor re-running it must get the same numbers.

    One pathology is handled explicitly. When the groups separate perfectly,
    EVERY bootstrap resample reproduces the same ordering, so the percentile
    interval collapses to a point — and "AUC 1.00, 95% CI 1.00–1.00" from
    three companies a side is precisely the overconfidence this module
    exists to prevent. In that case the interval is replaced by a bound
    derived from the number of pairwise comparisons actually made, which
    shrinks as evidence accumulates instead of asserting certainty from the
    start.
    """
    pos = np.asarray(positives, dtype=float)
    neg = np.asarray(negatives, dtype=float)
    point = auc(pos, neg)
    if pos.size == 0 or neg.size == 0:
        return 0.0, 0.0, 0.0

    if point == 1.0:
        return 1.0, round(_separation_bound(pos.size, neg.size, alpha), 3), 1.0
    if point == 0.0:
        return 0.0, 0.0, round(1 - _separation_bound(pos.size, neg.size, alpha), 3)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samples[i] = auc(
            pos[rng.integers(0, pos.size, pos.size)],
            neg[rng.integers(0, neg.size, neg.size)],
        )
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(point, 3), round(float(low), 3), round(float(high), 3)


def permutation_p(
    positives, negatives, n_perm: int = N_PERMUTATIONS, seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided p-value for AUC against the null of 0.5.

    Two-sided on purpose: a signal that reliably points the WRONG way is a
    real finding (and a bug worth fixing), not a null result.

    The +1 in numerator and denominator is the standard correction that
    stops a p-value of exactly zero — with 2,000 permutations the strongest
    claim available is p ≈ 0.0005, and reporting 0 would overstate it.
    """
    pos = np.asarray(positives, dtype=float)
    neg = np.asarray(negatives, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 1.0
    observed = abs(auc(pos, neg) - 0.5)
    combined = np.concatenate([pos, neg])
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(combined)
        if abs(auc(shuffled[: pos.size], shuffled[pos.size:]) - 0.5) >= observed:
            at_least_as_extreme += 1
    return round((at_least_as_extreme + 1) / (n_perm + 1), 4)


def benjamini_hochberg(pvalues: list[float], alpha: float = ALPHA) -> list[float]:
    """Adjusted q-values controlling the false discovery rate.

    Testing a dozen signals at p<0.05 yields a false positive more often
    than not. BH is the right correction here rather than Bonferroni: it
    controls the expected PROPORTION of false discoveries instead of the
    chance of any, which is what a research question wants — some false
    leads are tolerable, a table full of them is not.

    Returns q-values in the caller's original order, enforced monotonic as
    the procedure requires.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvalues = [0.0] * m
    running_min = 1.0
    # Walk from the largest p downward, carrying the running minimum, which
    # is what makes the adjusted values monotonic.
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end  # 1-based rank of this p-value
        candidate = min(1.0, pvalues[idx] * m / rank)
        running_min = min(running_min, candidate)
        qvalues[idx] = round(running_min, 4)
    return qvalues


def min_detectable_auc(n_pos: int, n_neg: int, alpha: float = ALPHA) -> float:
    """The smallest AUC distinguishable from chance at this sample size.

    Uses the null standard error of the Mann-Whitney statistic,
    sqrt((n1 + n2 + 1) / (12 · n1 · n2)). The point is to let a report say
    "with 8 outcomes and 20 controls, nothing below 0.74 is detectable" —
    which turns a disappointing table into an actionable one: the answer is
    more data, not a worse thesis.
    """
    if n_pos < 1 or n_neg < 1:
        return 1.0
    se = ((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)) ** 0.5
    z = 1.959964 if abs(alpha - 0.05) < 1e-9 else 2.575829
    return round(min(1.0, 0.5 + z * se), 3)


class SignalFinding(BaseModel):
    """What the evidence says about one signal."""

    name: str
    coverage: float          # share of companies where the signal fired at all
    n_positive: int
    n_negative: int
    auc: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float = 1.0     # after multiple-comparison correction
    marginal_auc: float | None = None  # composite AUC lost by dropping it
    mean_when_raised: float = 0.0
    mean_when_not: float = 0.0
    redundant_with: list[str] = Field(default_factory=list)

    @property
    def significant(self) -> bool:
        """Corrected significance, and the interval must clear chance too."""
        return self.q_value <= ALPHA and (
            self.ci_low > 0.5 or self.ci_high < 0.5
        )

    @property
    def direction(self) -> str:
        if self.auc > 0.5:
            return "predictive"
        if self.auc < 0.5:
            return "inverted"
        return "flat"

    @property
    def verdict(self) -> str:
        """One phrase a partner can read without a statistics degree."""
        if not self.significant:
            return "no evidence either way"
        if self.direction == "inverted":
            return "points the WRONG way — worth investigating"
        if self.redundant_with:
            return f"predictive, but duplicates {', '.join(self.redundant_with)}"
        if self.marginal_auc is not None and self.marginal_auc < 0.005:
            return "predictive, but adds nothing beyond the other signals"
        return "predictive"


class SignalEvaluation(BaseModel):
    """The whole per-signal picture for one backtest."""

    findings: list[SignalFinding] = Field(default_factory=list)
    n_outcomes: int = 0
    n_controls: int = 0
    min_detectable: float = 1.0
    underpowered: bool = False
    composite_auc: float = 0.0
    notes: list[str] = Field(default_factory=list)
    redundant: list[tuple[str, str, float]] = Field(default_factory=list)

    @property
    def ranked(self) -> list[SignalFinding]:
        """Significant signals first, then by how far from chance."""
        return sorted(
            self.findings,
            key=lambda f: (not f.significant, -abs(f.auc - 0.5)),
        )


@dataclass
class Observation:
    """One company's signal values and what actually happened to it."""

    key: str
    raised: bool
    signal_values: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0


def evaluate_signals(
    observations: list[Observation],
    weights: dict[str, float] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATIONS,
) -> SignalEvaluation:
    """Per-signal predictive power across a backtest's companies.

    Signals are read from their normalized values, not their weighted
    contributions — the question is whether the SIGNAL carries information,
    which must not be entangled with how much weight we currently give it.
    """
    positives = [o for o in observations if o.raised]
    negatives = [o for o in observations if not o.raised]
    evaluation = SignalEvaluation(
        n_outcomes=len(positives),
        n_controls=len(negatives),
        min_detectable=min_detectable_auc(len(positives), len(negatives)),
        composite_auc=auc([o.composite for o in positives],
                          [o.composite for o in negatives]),
    )
    if len(positives) < MIN_SAMPLES_PER_GROUP or len(negatives) < MIN_SAMPLES_PER_GROUP:
        evaluation.underpowered = True
        evaluation.notes.append(
            f"Too few companies to evaluate signals: {len(positives)} that "
            f"raised and {len(negatives)} that did not, against a minimum of "
            f"{MIN_SAMPLES_PER_GROUP} each. Per-signal results are withheld "
            "rather than shown with a caveat — a caveat under a number still "
            "leaves the number on the slide."
        )
        return evaluation

    names = sorted({name for o in observations for name in o.signal_values})
    raw_p: list[float] = []
    for name in names:
        pos_values = [o.signal_values.get(name, 0.0) for o in positives]
        neg_values = [o.signal_values.get(name, 0.0) for o in negatives]
        fired = sum(1 for o in observations if o.signal_values.get(name, 0.0) > 0)
        point, low, high = auc_ci(pos_values, neg_values, n_boot=n_boot, seed=seed)
        p = permutation_p(pos_values, neg_values, n_perm=n_perm, seed=seed)
        raw_p.append(p)
        evaluation.findings.append(SignalFinding(
            name=name,
            coverage=round(fired / len(observations), 3),
            n_positive=len(positives), n_negative=len(negatives),
            auc=point, ci_low=low, ci_high=high, p_value=p,
            mean_when_raised=round(float(np.mean(pos_values)), 3),
            mean_when_not=round(float(np.mean(neg_values)), 3),
        ))

    for finding, q in zip(evaluation.findings, benjamini_hochberg(raw_p)):
        finding.q_value = q

    if weights:
        marginals = marginal_contributions(observations, weights, names)
        for finding in evaluation.findings:
            finding.marginal_auc = marginals.get(finding.name)

    # Double-counting check, independent of the leave-one-out numbers.
    evaluation.redundant = redundant_pairs(observations, names)
    by_name = {f.name: f for f in evaluation.findings}
    for first, second, _r in evaluation.redundant:
        # Attribute the duplication to the weaker of the pair, so the
        # suggestion is "drop this one", not "you have a problem somewhere".
        weaker, stronger = sorted(
            (first, second),
            key=lambda n: abs(by_name[n].auc - 0.5) if n in by_name else 0.0,
        )
        if weaker in by_name:
            by_name[weaker].redundant_with.append(stronger)
    if evaluation.redundant:
        evaluation.notes.append(
            f"{len(evaluation.redundant)} signal pair(s) correlate above 0.9 — "
            "two signals measuring the same thing, each carrying weight, is "
            "one signal carrying double."
        )

    evaluation.notes.append(
        f"With {len(positives)} outcomes and {len(negatives)} controls, the "
        f"smallest AUC distinguishable from chance is "
        f"{evaluation.min_detectable:.2f}. Signals below that are not weak — "
        "they are unmeasured."
    )
    evaluation.notes.append(
        f"{len(names)} signals were tested, so p-values are corrected for "
        "multiple comparisons (Benjamini-Hochberg). Significance below uses "
        "the corrected q-value."
    )
    return evaluation


def correlation(xs, ys) -> float:
    """Pearson correlation, 0.0 when either side is constant."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or y.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def redundant_pairs(
    observations: list[Observation], names: list[str], threshold: float = 0.9,
) -> list[tuple[str, str, float]]:
    """Signal pairs that are measuring the same thing.

    The direct diagnostic for double-counting, and the reason it exists
    separately from marginal contribution: leave-one-out cannot detect a
    duplicate when other signals are noisy, because dropping one duplicate
    lets the noise take more of the weight and the composite AUC falls for
    the wrong reason. Correlation answers the question outright.

    It matters because two signals at 0.99 correlation given 20 points of
    weight each are really one signal given 40 — a thesis can drift into
    that without anyone deciding to.
    """
    pairs: list[tuple[str, str, float]] = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            r = correlation(
                [o.signal_values.get(first, 0.0) for o in observations],
                [o.signal_values.get(second, 0.0) for o in observations],
            )
            if abs(r) >= threshold:
                pairs.append((first, second, r))
    return sorted(pairs, key=lambda p: -abs(p[2]))


def _composite(values: dict[str, float], weights: dict[str, float],
               exclude: str = "") -> float:
    """The weighted score, optionally with one signal removed."""
    total = sum(w for name, w in weights.items() if name != exclude) or 1.0
    return sum(
        values.get(name, 0.0) * weight
        for name, weight in weights.items()
        if name != exclude
    ) / total


def marginal_contributions(
    observations: list[Observation], weights: dict[str, float], names: list[str],
) -> dict[str, float]:
    """How much composite AUC each signal ADDS beyond the others.

    The number that matters for weighting decisions. A signal can be
    strongly predictive on its own and still contribute nothing here —
    which means it is redundant with something already in the score, and
    raising its weight would buy noise. Individually-impressive signals
    that add zero marginal value are the most common way a scoring model
    gets quietly worse.
    """
    positives = [o for o in observations if o.raised]
    negatives = [o for o in observations if not o.raised]
    if not positives or not negatives:
        return {}
    full = auc(
        [_composite(o.signal_values, weights) for o in positives],
        [_composite(o.signal_values, weights) for o in negatives],
    )
    out: dict[str, float] = {}
    for name in names:
        if name not in weights:
            continue
        without = auc(
            [_composite(o.signal_values, weights, exclude=name) for o in positives],
            [_composite(o.signal_values, weights, exclude=name) for o in negatives],
        )
        out[name] = round(full - without, 4)
    return out


class WeightSuggestion(BaseModel):
    """An evidence-derived weight, next to the one in use."""

    name: str
    current: float
    suggested: float
    reason: str

    @property
    def delta(self) -> float:
        return round(self.suggested - self.current, 1)


def suggest_weights(
    evaluation: SignalEvaluation,
    current_weights: dict[str, float],
    *,
    shrinkage: float = 0.5,
) -> list[WeightSuggestion]:
    """Weights informed by measured predictive power.

    Deliberately a shrunk heuristic rather than a fitted model. At fifteen
    outcomes a logistic regression would produce coefficients with
    confidence intervals spanning the plausible range, and dressing that up
    as "learned weights" would be the single most misleading thing this
    module could do. Instead:

    - only signals that survive multiple-comparison correction move at all;
    - the target weight is proportional to measured discrimination
      (AUC − 0.5), keeping the total weight budget unchanged so scores stay
      comparable across versions;
    - the move is shrunk halfway toward that target, because one backtest
      on one window is evidence, not proof;
    - a signal that points the wrong way is flagged for a human rather than
      silently inverted.
    """
    if evaluation.underpowered:
        return []
    budget = sum(current_weights.values())
    if budget <= 0:
        return []

    strengths: dict[str, float] = {}
    for finding in evaluation.findings:
        if finding.name not in current_weights:
            continue
        if not finding.significant or finding.direction == "inverted":
            continue
        strengths[finding.name] = max(0.0, finding.auc - 0.5)

    suggestions: list[WeightSuggestion] = []
    total_strength = sum(strengths.values())
    for finding in evaluation.findings:
        name = finding.name
        if name not in current_weights:
            continue
        current = current_weights[name]
        if finding.direction == "inverted" and finding.significant:
            suggestions.append(WeightSuggestion(
                name=name, current=current, suggested=current,
                reason=(f"points the wrong way (AUC {finding.auc:.2f}) — a human "
                        "should look at why before any weight changes"),
            ))
            continue
        if not finding.significant:
            suggestions.append(WeightSuggestion(
                name=name, current=current, suggested=current,
                reason=(f"no measurable effect at this sample size "
                        f"(AUC {finding.auc:.2f}, "
                        f"95% CI {finding.ci_low:.2f}–{finding.ci_high:.2f})"),
            ))
            continue
        target = budget * strengths[name] / total_strength if total_strength else current
        shrunk = current + shrinkage * (target - current)
        reason = (f"AUC {finding.auc:.2f} "
                  f"(CI {finding.ci_low:.2f}–{finding.ci_high:.2f}, "
                  f"q={finding.q_value:.3f})")
        if finding.redundant_with:
            reason += (f" but duplicates {', '.join(finding.redundant_with)} "
                       "— held back to avoid double-counting")
            shrunk = current
        elif finding.marginal_auc is not None and finding.marginal_auc < 0.005:
            reason += " but adds nothing beyond the other signals — held back"
            shrunk = current
        suggestions.append(WeightSuggestion(
            name=name, current=round(current, 1),
            suggested=round(shrunk, 1), reason=reason,
        ))
    return suggestions


# ------------------------------------------------------------- over time


class SignalTrend(BaseModel):
    """One signal's measured power across successive cutoffs."""

    name: str
    points: list[tuple[str, float, float, float]] = Field(default_factory=list)
    # (cutoff ISO date, auc, ci_low, ci_high)

    @property
    def first_auc(self) -> float | None:
        return self.points[0][1] if self.points else None

    @property
    def latest_auc(self) -> float | None:
        return self.points[-1][1] if self.points else None

    @property
    def decayed(self) -> bool:
        """True when a signal has measurably lost power.

        Requires the intervals to be disjoint, not merely the point
        estimates to differ — at these sample sizes two AUCs can look far
        apart and be perfectly consistent with no change at all. This is
        the check that stops the tool inventing a decay story every time
        noise moves a number.
        """
        if len(self.points) < 2:
            return False
        _, _first_auc, first_low, _first_high = self.points[0]
        _, _last_auc, _last_low, last_high = self.points[-1]
        # Disjoint intervals with the later one below: the whole condition.
        # (An earlier version also tested the point estimates, which is
        # implied by disjointness and only obscured the intent.)
        return last_high < first_low

    @property
    def summary(self) -> str:
        if self.first_auc is None or self.latest_auc is None:
            return "no data"
        if self.decayed:
            return (f"decayed — {self.first_auc:.2f} → {self.latest_auc:.2f}, "
                    "and the confidence intervals no longer overlap")
        change = self.latest_auc - self.first_auc
        if abs(change) < 0.05:
            return f"stable around {self.latest_auc:.2f}"
        direction = "stronger" if change > 0 else "weaker"
        return (f"{direction} ({self.first_auc:.2f} → {self.latest_auc:.2f}), "
                "though the intervals still overlap — not yet a real change")


def build_trends(
    evaluations: list[tuple[str, SignalEvaluation]],
) -> list[SignalTrend]:
    """Assemble per-signal series from evaluations at successive cutoffs.

    Input is (cutoff ISO date, evaluation), oldest first — which is what
    makes "is this signal wearing out?" answerable. A signal like "stealth
    in the bio" that worked in 2022 and became a meme by 2025 shows up here
    as a declining series long before anyone notices it by feel.
    """
    ordered = sorted(evaluations, key=lambda pair: pair[0])
    trends: dict[str, SignalTrend] = {}
    for cutoff, evaluation in ordered:
        if evaluation.underpowered:
            continue
        for finding in evaluation.findings:
            trend = trends.setdefault(finding.name, SignalTrend(name=finding.name))
            trend.points.append(
                (cutoff, finding.auc, finding.ci_low, finding.ci_high)
            )
    return sorted(trends.values(), key=lambda t: t.name)
