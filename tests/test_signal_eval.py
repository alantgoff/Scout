"""Per-signal predictive power: the statistics, tested against known answers.

A backtest whose arithmetic is wrong is worse than no backtest, because it
produces a confident number. So these tests check the properties that make
the results trustworthy rather than merely plausible:

- ties give exactly half credit, so a flat signal scores 0.5 not 1.0;
- the multiple-comparison correction actually suppresses the false
  positive you get for free when testing a dozen signals;
- an underpowered sample withholds results instead of decorating them;
- decay is only claimed when the confidence intervals separate.
"""

from __future__ import annotations

import numpy as np
import pytest

from scout import signal_eval as se
from scout.signal_eval import Observation


def obs(key: str, raised: bool, **signals) -> Observation:
    return Observation(key=key, raised=raised, signal_values=dict(signals))


# --- AUC ------------------------------------------------------------------------


def test_auc_matches_hand_computed_cases() -> None:
    assert se.auc([90, 80], [20, 10]) == 1.0     # perfect
    assert se.auc([10, 20], [80, 90]) == 0.0     # perfectly inverted
    assert se.auc([70, 70], [70, 70]) == 0.5     # all ties → no discrimination
    assert se.auc([90, 30], [40, 20]) == 0.75    # 3 of 4 pairs won
    assert se.auc([], [1, 2]) == 0.0             # undefined, not flattering


def test_auc_gives_ties_exactly_half_credit() -> None:
    """The property that stops a constant signal looking predictive."""
    # One positive tied with the single negative: half a win out of one pair.
    assert se.auc([50.0], [50.0]) == 0.5
    # Two positives, one ties and one wins: (1 + 0.5) / 2.
    assert se.auc([60.0, 50.0], [50.0]) == 0.75


def test_auc_agrees_with_the_pairwise_definition_on_random_data() -> None:
    """The fast rank implementation must equal the O(n²) definition."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        pos = rng.integers(0, 5, rng.integers(2, 12)).astype(float)
        neg = rng.integers(0, 5, rng.integers(2, 12)).astype(float)
        pairwise = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
        ) / (len(pos) * len(neg))
        assert se.auc(pos, neg) == pytest.approx(pairwise, abs=1e-9)


# --- confidence intervals -------------------------------------------------------


def test_ci_brackets_the_point_estimate_and_is_reproducible() -> None:
    pos = [0.9, 0.8, 0.85, 0.7, 0.95, 0.75]
    neg = [0.2, 0.1, 0.3, 0.15, 0.25, 0.05]
    point, low, high = se.auc_ci(pos, neg, n_boot=500)
    assert low <= point <= high
    # Seeded: an investor re-running the report must get identical numbers.
    assert se.auc_ci(pos, neg, n_boot=500) == (point, low, high)


def test_ci_is_embarrassingly_wide_at_small_n() -> None:
    """The point of showing intervals: six-a-side cannot prove much, and the
    interval should make that obvious rather than hiding it."""
    _point, low, high = se.auc_ci([0.9, 0.8, 0.7], [0.2, 0.3, 0.1], n_boot=500)
    assert high - low > 0.25


def test_noise_produces_an_interval_that_spans_chance() -> None:
    rng = np.random.default_rng(3)
    pos = rng.normal(0.5, 0.2, 30).tolist()
    neg = rng.normal(0.5, 0.2, 30).tolist()
    _point, low, high = se.auc_ci(pos, neg, n_boot=500)
    assert low < 0.5 < high  # cannot rule out "no effect", correctly


# --- significance ---------------------------------------------------------------


def test_permutation_p_is_large_for_noise_and_small_for_real_separation() -> None:
    rng = np.random.default_rng(11)
    noise_p = se.permutation_p(
        rng.normal(0, 1, 25).tolist(), rng.normal(0, 1, 25).tolist(), n_perm=500)
    assert noise_p > 0.1

    separated_p = se.permutation_p(
        [5.0] * 15 + [4.5] * 5, [0.0] * 15 + [0.5] * 5, n_perm=500)
    assert separated_p < 0.01


def test_permutation_p_never_claims_impossible_precision() -> None:
    """With 500 permutations the strongest claim is ~1/501; reporting 0
    would overstate the evidence."""
    p = se.permutation_p([9.0] * 20, [0.0] * 20, n_perm=500)
    assert p == pytest.approx(1 / 501, abs=1e-4)
    assert p > 0


def test_permutation_p_is_two_sided() -> None:
    """A signal reliably pointing the WRONG way is a finding, not a null."""
    inverted = se.permutation_p([0.0] * 15, [9.0] * 15, n_perm=500)
    assert inverted < 0.01


# --- multiple comparisons -------------------------------------------------------


def test_benjamini_hochberg_matches_the_textbook_computation() -> None:
    pvalues = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    q = se.benjamini_hochberg(pvalues)
    assert q[0] == pytest.approx(0.008, abs=0.001)   # 0.001 × 8/1
    assert q[1] == pytest.approx(0.032, abs=0.001)   # 0.008 × 8/2
    assert q[-1] == pytest.approx(0.205, abs=0.001)  # largest p unchanged
    # Monotonic, as the procedure requires.
    ordered = [q[i] for i in sorted(range(len(pvalues)), key=lambda i: pvalues[i])]
    assert ordered == sorted(ordered)


def test_correction_suppresses_the_free_false_positive() -> None:
    """Testing twelve signals, one lands at p=0.04 by chance. Uncorrected
    that reads as a discovery; corrected it correctly does not."""
    pvalues = [0.04] + [0.5] * 11
    q = se.benjamini_hochberg(pvalues)
    assert pvalues[0] < se.ALPHA          # would have been "significant"
    assert q[0] > se.ALPHA                # and is not, once corrected
    assert q == se.benjamini_hochberg(pvalues)  # deterministic


def test_bh_handles_empty_and_single_inputs() -> None:
    assert se.benjamini_hochberg([]) == []
    assert se.benjamini_hochberg([0.03]) == [0.03]


# --- power ----------------------------------------------------------------------


def test_min_detectable_auc_shrinks_as_samples_grow() -> None:
    """Turns 'we found nothing' into 'we could not have found anything'."""
    tiny = se.min_detectable_auc(5, 5)
    small = se.min_detectable_auc(15, 30)
    large = se.min_detectable_auc(200, 400)
    assert tiny > small > large
    assert tiny > 0.75   # five-a-side can only detect a near-perfect signal
    assert large < 0.56  # with hundreds, modest effects become visible


def test_min_detectable_auc_is_defined_for_degenerate_input() -> None:
    assert se.min_detectable_auc(0, 10) == 1.0
    assert se.min_detectable_auc(10, 0) == 1.0


# --- the evaluation -------------------------------------------------------------


def _dataset(n: int = 12) -> list[Observation]:
    """Companies where `strong` tracks the outcome, `noise` does not, and
    `flat` never fires at all."""
    rows = []
    for i in range(n):
        rows.append(obs(f"win{i}", True, strong=0.9 - i * 0.01,
                        noise=0.5 if i % 2 else 0.4, flat=0.0))
        rows.append(obs(f"lose{i}", False, strong=0.2 + i * 0.01,
                        noise=0.45 if i % 3 else 0.55, flat=0.0))
    return rows


def test_evaluation_separates_a_real_signal_from_noise() -> None:
    evaluation = se.evaluate_signals(_dataset(), n_boot=300, n_perm=300)
    findings = {f.name: f for f in evaluation.findings}

    assert findings["strong"].auc > 0.9
    assert findings["strong"].significant is True
    assert findings["strong"].verdict == "predictive"

    assert findings["noise"].significant is False
    assert findings["noise"].verdict == "no evidence either way"

    # A signal that never fires has no coverage and cannot be significant.
    assert findings["flat"].coverage == 0.0
    assert findings["flat"].auc == 0.5
    assert findings["flat"].significant is False

    # Ranking puts the real one first.
    assert evaluation.ranked[0].name == "strong"


def test_evaluation_reports_coverage_and_group_means() -> None:
    evaluation = se.evaluate_signals(_dataset(6), n_boot=200, n_perm=200)
    strong = next(f for f in evaluation.findings if f.name == "strong")
    assert strong.coverage == 1.0                 # fires for every company
    assert strong.mean_when_raised > strong.mean_when_not
    assert strong.n_positive == 6 and strong.n_negative == 6


def test_an_inverted_signal_is_flagged_not_buried() -> None:
    rows = []
    for i in range(12):
        rows.append(obs(f"w{i}", True, backwards=0.1))
        rows.append(obs(f"l{i}", False, backwards=0.9))
    evaluation = se.evaluate_signals(rows, n_boot=200, n_perm=200)
    finding = evaluation.findings[0]
    assert finding.direction == "inverted"
    assert finding.significant is True
    assert "WRONG way" in finding.verdict


def test_underpowered_samples_withhold_results_entirely() -> None:
    """A caveat under a number still leaves the number on the slide."""
    rows = [obs("w1", True, strong=0.9), obs("w2", True, strong=0.8),
            obs("l1", False, strong=0.1)]
    evaluation = se.evaluate_signals(rows, n_boot=100, n_perm=100)
    assert evaluation.underpowered is True
    assert evaluation.findings == []
    assert "Too few companies" in evaluation.notes[0]


def test_evaluation_states_what_it_could_not_have_detected() -> None:
    evaluation = se.evaluate_signals(_dataset(8), n_boot=200, n_perm=200)
    assert evaluation.min_detectable > 0.5
    assert any("not weak" in note and "unmeasured" in note
               for note in evaluation.notes)
    assert any("multiple comparisons" in note for note in evaluation.notes)


# --- marginal contribution ------------------------------------------------------


def test_a_duplicate_signal_shows_zero_marginal_value() -> None:
    """The most common way a scoring model quietly gets worse: adding a
    signal that is individually impressive and entirely redundant."""
    rows = []
    for i in range(12):
        value = 0.9 - i * 0.01
        rows.append(obs(f"w{i}", True, original=value, duplicate=value))
        rows.append(obs(f"l{i}", False, original=0.2 + i * 0.01,
                        duplicate=0.2 + i * 0.01))
    weights = {"original": 20.0, "duplicate": 20.0}
    marginals = se.marginal_contributions(rows, weights, ["original", "duplicate"])
    # Dropping either changes nothing — the other carries the same content.
    assert marginals["duplicate"] == pytest.approx(0.0, abs=0.01)
    assert marginals["original"] == pytest.approx(0.0, abs=0.01)


def test_a_uniquely_informative_signal_shows_positive_marginal_value() -> None:
    rows = []
    for i in range(12):
        rows.append(obs(f"w{i}", True, useful=0.9, useless=0.5))
        rows.append(obs(f"l{i}", False, useful=0.1, useless=0.5))
    weights = {"useful": 20.0, "useless": 20.0}
    marginals = se.marginal_contributions(rows, weights, ["useful", "useless"])
    assert marginals["useful"] > 0.2   # removing it destroys the ranking
    assert marginals["useless"] == pytest.approx(0.0, abs=0.01)


def test_evaluation_attaches_marginals_when_weights_are_supplied() -> None:
    rows = []
    for i in range(12):
        value = 0.9 - i * 0.01
        rows.append(obs(f"w{i}", True, original=value, duplicate=value))
        rows.append(obs(f"l{i}", False, original=0.2, duplicate=0.2))
    evaluation = se.evaluate_signals(
        rows, weights={"original": 20.0, "duplicate": 20.0},
        n_boot=200, n_perm=200)
    duplicate = next(f for f in evaluation.findings if f.name == "duplicate")
    assert duplicate.marginal_auc is not None
    # Named, not just labelled: the verdict says WHICH signal it duplicates.
    assert duplicate.verdict == "predictive, but duplicates original"


# --- weight suggestions ---------------------------------------------------------


def test_only_significant_signals_move_and_the_budget_is_preserved() -> None:
    evaluation = se.evaluate_signals(_dataset(), n_boot=300, n_perm=300)
    current = {"strong": 10.0, "noise": 30.0, "flat": 10.0}
    suggestions = {s.name: s for s in se.suggest_weights(evaluation, current)}

    # The real signal gains; the noisy one is left alone with a reason.
    assert suggestions["strong"].suggested > suggestions["strong"].current
    assert suggestions["noise"].suggested == suggestions["noise"].current
    assert "no measurable effect" in suggestions["noise"].reason
    assert "AUC" in suggestions["strong"].reason and "q=" in suggestions["strong"].reason


def test_suggestions_are_shrunk_rather_than_taken_at_face_value() -> None:
    """One backtest on one window is evidence, not proof — so the move is
    halfway to the target, never the whole way."""
    evaluation = se.evaluate_signals(_dataset(), n_boot=300, n_perm=300)
    current = {"strong": 10.0, "noise": 30.0}
    budget = sum(current.values())
    suggestions = {s.name: s for s in se.suggest_weights(evaluation, current)}
    # The unshrunk target would take essentially the whole budget, since
    # `strong` is the only significant signal; shrinkage keeps it short.
    assert suggestions["strong"].suggested < budget
    assert suggestions["strong"].suggested == pytest.approx(
        10.0 + 0.5 * (budget - 10.0), abs=0.6)


def test_an_underpowered_evaluation_suggests_nothing() -> None:
    rows = [obs("w", True, strong=0.9), obs("l", False, strong=0.1)]
    evaluation = se.evaluate_signals(rows, n_boot=50, n_perm=50)
    assert se.suggest_weights(evaluation, {"strong": 10.0}) == []


def test_an_inverted_signal_is_referred_to_a_human_not_auto_flipped() -> None:
    rows = []
    for i in range(12):
        rows.append(obs(f"w{i}", True, backwards=0.1))
        rows.append(obs(f"l{i}", False, backwards=0.9))
    evaluation = se.evaluate_signals(rows, n_boot=200, n_perm=200)
    suggestion = se.suggest_weights(evaluation, {"backwards": 20.0})[0]
    assert suggestion.suggested == 20.0        # unchanged
    assert "human should look" in suggestion.reason


# --- trends over time -----------------------------------------------------------


def _evaluation_with(name: str, auc_value: float, low: float, high: float
                     ) -> se.SignalEvaluation:
    return se.SignalEvaluation(findings=[se.SignalFinding(
        name=name, coverage=1.0, n_positive=10, n_negative=10,
        auc=auc_value, ci_low=low, ci_high=high, p_value=0.01, q_value=0.01,
    )], n_outcomes=10, n_controls=10)


def test_decay_is_claimed_only_when_the_intervals_separate() -> None:
    """Two AUCs can look far apart and be perfectly consistent with no
    change. Without this check the tool invents a decay story from noise."""
    overlapping = se.build_trends([
        ("2023-01-01", _evaluation_with("stealth", 0.80, 0.60, 0.95)),
        ("2025-01-01", _evaluation_with("stealth", 0.62, 0.45, 0.80)),
    ])[0]
    assert overlapping.decayed is False
    assert "not yet a real change" in overlapping.summary

    separated = se.build_trends([
        ("2023-01-01", _evaluation_with("stealth", 0.88, 0.80, 0.95)),
        ("2025-01-01", _evaluation_with("stealth", 0.55, 0.48, 0.62)),
    ])[0]
    assert separated.decayed is True
    assert "decayed" in separated.summary


def test_a_stable_signal_reads_as_stable() -> None:
    trend = se.build_trends([
        ("2023-01-01", _evaluation_with("github_stars", 0.74, 0.65, 0.83)),
        ("2025-01-01", _evaluation_with("github_stars", 0.76, 0.67, 0.85)),
    ])[0]
    assert trend.decayed is False
    assert "stable" in trend.summary


def test_trends_are_ordered_oldest_first_regardless_of_input_order() -> None:
    trend = se.build_trends([
        ("2025-01-01", _evaluation_with("x", 0.60, 0.5, 0.7)),
        ("2023-01-01", _evaluation_with("x", 0.80, 0.7, 0.9)),
    ])[0]
    assert [point[0] for point in trend.points] == ["2023-01-01", "2025-01-01"]
    assert trend.first_auc == 0.80 and trend.latest_auc == 0.60


def test_underpowered_evaluations_are_excluded_from_trends() -> None:
    weak = se.SignalEvaluation(underpowered=True, findings=[])
    trends = se.build_trends([
        ("2023-01-01", _evaluation_with("x", 0.80, 0.7, 0.9)),
        ("2025-01-01", weak),
    ])
    assert len(trends[0].points) == 1  # the underpowered window contributes nothing


def test_perfect_separation_does_not_claim_certainty() -> None:
    """The pathology that makes naive bootstrap backtests overconfident:
    with perfect separation every resample reproduces AUC 1.0, so the
    percentile interval collapses to a point. Three companies a side is not
    proof of a perfect signal."""
    point, low, high = se.auc_ci([0.9, 0.8, 0.7], [0.2, 0.3, 0.1], n_boot=500)
    assert point == 1.0 and high == 1.0
    assert low == pytest.approx(0.664, abs=0.01)   # bound from 9 pairings
    assert se.perfectly_separated([0.9, 0.8, 0.7], [0.2, 0.3, 0.1]) is True


def test_the_separation_bound_tightens_as_evidence_accumulates() -> None:
    """Perfect on 9 comparisons is weak; perfect on 400 is strong. The
    bound has to move, or it is not measuring anything."""
    _p, small_low, _h = se.auc_ci([0.9] * 3, [0.1] * 3, n_boot=200)
    _p, large_low, _h = se.auc_ci([0.9] * 20, [0.1] * 20, n_boot=200)
    assert small_low < large_low
    assert large_low > 0.99   # 400 clean pairings genuinely is strong evidence


def test_perfectly_inverted_separation_is_bounded_on_the_other_side() -> None:
    point, low, high = se.auc_ci([0.1, 0.2, 0.3], [0.7, 0.8, 0.9], n_boot=200)
    assert point == 0.0 and low == 0.0
    assert high == pytest.approx(0.336, abs=0.01)


# --- double-counting ------------------------------------------------------------


def test_correlation_finds_signals_measuring_the_same_thing() -> None:
    rows = []
    for i in range(12):
        value = 0.9 - i * 0.05
        rows.append(obs(f"w{i}", True, a=value, b=value, unrelated=(i % 3) / 3))
    pairs = se.redundant_pairs(rows, ["a", "b", "unrelated"])
    assert [(p[0], p[1]) for p in pairs] == [("a", "b")]
    assert pairs[0][2] == pytest.approx(1.0, abs=0.001)


def test_correlation_is_zero_for_a_constant_signal() -> None:
    """A signal that never varies cannot correlate with anything, and must
    not produce a divide-by-zero or a spurious 1.0."""
    rows = [obs(f"x{i}", i % 2 == 0, flat=0.0, varying=i / 10) for i in range(10)]
    assert se.correlation([o.signal_values["flat"] for o in rows],
                          [o.signal_values["varying"] for o in rows]) == 0.0
    assert se.redundant_pairs(rows, ["flat", "varying"]) == []


def test_duplicate_signals_are_caught_even_when_other_signals_are_noisy() -> None:
    """The case leave-one-out misses: dropping one duplicate lets the noise
    take more weight, so the composite AUC falls for the wrong reason and
    the duplicate looks uniquely valuable. Correlation answers it outright."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(14):
        value = 0.85 - i * 0.02
        rows.append(obs(f"w{i}", True, github=value, smart_money=value,
                        noise=float(rng.random())))
    for i in range(20):
        value = 0.25 + i * 0.01
        rows.append(obs(f"c{i}", False, github=value, smart_money=value,
                        noise=float(rng.random())))
    weights = {"github": 20.0, "smart_money": 20.0, "noise": 30.0}
    evaluation = se.evaluate_signals(rows, weights, n_boot=200, n_perm=200)

    # Leave-one-out alone would call both uniquely valuable…
    marginals = {f.name: f.marginal_auc for f in evaluation.findings}
    assert marginals["github"] > 0.05 and marginals["smart_money"] > 0.05
    # …but the correlation check names them as one signal counted twice.
    assert evaluation.redundant
    assert {evaluation.redundant[0][0], evaluation.redundant[0][1]} == {
        "github", "smart_money"}
    duplicated = [f for f in evaluation.findings if f.redundant_with]
    assert len(duplicated) == 1  # only the weaker of the pair is flagged
    assert "duplicates" in duplicated[0].verdict
    assert any("double" in note for note in evaluation.notes)


def test_a_duplicated_signal_is_held_back_from_a_weight_increase() -> None:
    """Raising both halves of a duplicated pair is how a scoring model
    silently doubles a signal's influence."""
    rows = []
    for i in range(14):
        value = 0.85 - i * 0.02
        rows.append(obs(f"w{i}", True, github=value, smart_money=value))
        rows.append(obs(f"c{i}", False, github=0.2 + i * 0.01,
                        smart_money=0.2 + i * 0.01))
    weights = {"github": 20.0, "smart_money": 20.0}
    evaluation = se.evaluate_signals(rows, weights, n_boot=200, n_perm=200)
    suggestions = {s.name: s for s in se.suggest_weights(evaluation, weights)}
    held = [s for s in suggestions.values() if "double-counting" in s.reason]
    assert held and held[0].suggested == held[0].current
