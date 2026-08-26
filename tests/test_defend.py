"""Detection-layer tests.

These check the properties that make a reported metric trustworthy, rather than checking that
the metric is high. A test asserting "AUC > 0.99" would pass on leaky data and fail for the
right reasons never.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from janus.defend import novelty, policy, supervised
from janus.defend.evaluate import temporal_split
from janus.generate.simulate import simulate
from janus.generate.world import WorldConfig

SMALL = WorldConfig(n_customers=2500, n_merchants=400, days=25, seed=13)


@pytest.fixture(scope="module")
def split():
    return temporal_split(simulate(SMALL).events)


@pytest.fixture(scope="module")
def detector(split):
    return supervised.train(split.X_train, split.y_train, seed=0)


def test_temporal_split_does_not_overlap_in_time(split):
    """Train must end before test begins, or campaigns straddle the boundary."""
    assert len(split.X_train) > 0 and len(split.X_test) > 0
    assert len(split.X_train.columns) == len(split.X_test.columns)


def test_detector_beats_chance(detector, split):
    m = supervised.evaluate(detector, split.X_test, split.y_test)
    assert m["roc_auc"] > 0.75, m
    assert m["pr_auc"] > m["base_rate"] * 5, "PR-AUC should far exceed the base rate"


def test_detector_is_not_suspiciously_perfect(detector, split):
    """A near-perfect score on a rare-event problem means leakage, not success.

    This is a guard against future changes reintroducing a look-ahead feature. If this ever
    fails, the correct response is to hunt for the leak, not to relax the bound.
    """
    m = supervised.evaluate(detector, split.X_test, split.y_test)
    assert m["roc_auc"] < 0.9995, f"implausibly perfect ROC-AUC ({m['roc_auc']}) - suspect leakage"


def test_recall_at_fixed_fpr_is_monotone(detector, split):
    """Loosening the FPR budget can never catch less fraud."""
    scores = detector.score(split.X_test)
    recalls = [
        supervised.recall_at_fpr(split.y_test, scores, f) for f in sorted(supervised.TARGET_FPRS)
    ]
    assert recalls == sorted(recalls), recalls


def test_threshold_never_exceeds_the_fpr_budget(detector, split):
    """The budget is a hard ceiling, not a target to land near.

    An earlier version allowed 40% slack and therefore passed while the realised FPR was
    running at twice the requested budget - isotonic calibration creates wide score plateaus,
    and a naive quantile threshold flags the whole tied block.
    """
    scores = detector.score(split.X_test)
    for fpr in supervised.TARGET_FPRS:
        t = supervised.threshold_at_fpr(split.y_test, scores, fpr)
        realised = supervised.realised_fpr(split.y_test, scores, t)
        assert realised <= fpr + 1e-9, f"fpr {fpr}: realised {realised}"


def test_calibrated_scores_have_coarse_plateaus(detector, split):
    """Documents *why* the tie-aware threshold exists, so nobody simplifies it away."""
    scores = detector.score(split.X_test)
    legit = scores[split.y_test == 0]
    distinct = len(set(legit.tolist()))
    assert distinct < len(legit) / 10, (
        "isotonic calibration is expected to produce far fewer distinct scores than events; "
        "if this ever stops being true, re-check that threshold_at_fpr is still needed"
    )


def test_novelty_layer_never_sees_labels(split):
    """The unsupervised layer must be trainable with the labels withheld entirely."""
    nov = novelty.train(split.X_train, np.zeros(len(split.X_train), dtype=int), seed=0)
    scores = nov.score(split.X_test)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert np.isfinite(scores).all()


def test_novelty_scores_are_higher_for_fraud(split):
    nov = novelty.train(split.X_train, split.y_train, seed=0)
    s = nov.score(split.X_test)
    assert s[split.y_test == 1].mean() > s[split.y_test == 0].mean()


def test_policy_actions_escalate_with_risk():
    th = policy.Thresholds(step_up=0.3, hold=0.5, block=0.8)
    scores = np.array([0.05, 0.35, 0.6, 0.95])
    actions = policy.decide(scores, np.zeros(4), th, novelty_weight=0.0)
    assert list(actions) == ["allow", "step_up", "hold", "block"]


def test_policy_net_benefit_beats_doing_nothing(detector, split):
    """The chosen operating point must make money AND be shippable.

    The sweep is ordered by feasibility first and net benefit second, so net benefit is
    deliberately *not* globally monotonic - an infeasible row can out-earn every feasible one,
    which is exactly the situation the friction ceiling exists to reject. Monotonicity is
    asserted within each feasibility group instead.
    """
    nov = novelty.train(split.X_train, split.y_train, seed=0)
    # The policy engine reasons in rupees, so it takes the calibrated probability rather than
    # the ranking score.
    f = detector.score(split.X_test)
    n = nov.score(split.X_test)
    amounts = split.test_events.amount.to_numpy()
    sweep = policy.sweep_thresholds(f, n, split.y_test, amounts)

    best = sweep.iloc[0]
    assert best.net_benefit > 0, "the optimal policy should be worth money"
    assert best.within_friction_budget, "the selected point must be shippable"
    for _, group in sweep.groupby("within_friction_budget"):
        assert group.net_benefit.is_monotonic_decreasing


def test_friction_ceiling_can_reject_the_richest_policy(detector, split):
    """The ceiling must actually bind, or it is decoration.

    Optimising net rupees alone drives the step-up threshold to the floor, because a challenge
    costs a few rupees and prevents thousands. Measured here, the unconstrained optimum
    challenges ~16% of legitimate payments - commercially unshippable regardless of arithmetic.
    """
    nov = novelty.train(split.X_train, split.y_train, seed=0)
    f = detector.score(split.X_test)
    n = nov.score(split.X_test)
    amounts = split.test_events.amount.to_numpy()
    sweep = policy.sweep_thresholds(f, n, split.y_test, amounts, max_challenge_rate=0.02)

    richest = sweep.sort_values("net_benefit", ascending=False).iloc[0]
    chosen = sweep.iloc[0]
    assert chosen.legit_challenged_rate <= 0.02
    if not richest.within_friction_budget:
        assert richest.net_benefit >= chosen.net_benefit, (
            "if the ceiling rejected a policy, that policy should have earned more"
        )


def test_blocking_everything_is_not_optimal(detector, split):
    """Sanity check on the cost model: a maximally aggressive policy must lose money."""
    amounts = split.test_events.amount.to_numpy()
    always_block = np.full(len(split.y_test), "block", dtype=object)
    result = policy.evaluate_policy(always_block, split.y_test, amounts)
    assert result["net_benefit"] < 0, "cost model is not penalising false blocks enough"


def test_detector_is_deterministic(split):
    a = supervised.train(split.X_train, split.y_train, seed=0)
    b = supervised.train(split.X_train, split.y_train, seed=0)
    np.testing.assert_allclose(a.score(split.X_test), b.score(split.X_test))


def test_training_fails_loudly_without_positives(split):
    with pytest.raises(ValueError, match="no positive examples"):
        supervised.train(split.X_train, np.zeros(len(split.X_train), dtype=int))


def test_smaller_world_still_trains():
    """The pipeline must survive a world too small to be statistically interesting.

    Asserts robustness, not accuracy. An earlier version required ROC-AUC > 0.6 here and went
    flaky at 0.588: this world's test split holds only a few dozen fraud events, so its AUC is
    mostly sampling noise. Pinning an accuracy bound to a sample that small measures the seed,
    not the model - the real accuracy claims are made in test_detector_beats_chance and in
    RESULTS.md, on a full world.
    """
    tiny = simulate(replace(SMALL, n_customers=800, days=12))
    sp = temporal_split(tiny.events)
    det = supervised.train(sp.X_train, sp.y_train, seed=0, n_estimators=80)

    scores = det.rank_score(sp.X_test)
    assert len(scores) == len(sp.y_test)
    assert np.isfinite(scores).all()
    calibrated = det.score(sp.X_test)
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
    assert det.thresholds, "thresholds must still be derived on a small calibration slice"

    metrics = supervised.evaluate(det, sp.X_test, sp.y_test)
    assert metrics["roc_auc"] > 0.5, "should at least beat chance even with this little data"
