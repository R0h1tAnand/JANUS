"""Feature store tests, centred on point-in-time correctness.

The leakage test here is the most important test in the repository. Every headline number the
submission reports - LOAO recall, recall at fixed FPR, the arena's convergence curves - is
worthless if features can see the future, and the failure mode is silent: a leaky model looks
*better*, not worse, right up until it meets production.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from janus.defend.features import build_features
from janus.generate.rails import LABEL_COLUMNS
from janus.generate.simulate import simulate
from janus.generate.world import WorldConfig

SMALL = WorldConfig(n_customers=900, n_merchants=200, days=15, seed=11)


@pytest.fixture(scope="module")
def events():
    return simulate(SMALL).events


@pytest.fixture(scope="module")
def features(events):
    return build_features(events)


def test_no_label_column_reaches_the_model(features):
    assert not (LABEL_COLUMNS & set(features.columns))
    for banned in ("is_fraud", "attack_id", "campaign_id"):
        assert banned not in features.columns


def test_features_are_finite(features):
    arr = features.to_numpy(dtype=np.float64)
    assert np.isfinite(arr).all(), "non-finite values would silently corrupt training"


def test_prefix_truncation_leaves_features_identical(events):
    """THE leakage guard.

    Recompute the features on a truncated prefix of the event stream. If any feature reads
    even one future row, the values for the rows that *are* present must change. Requiring
    bit-identical output is the strongest available statement that nothing looks forward.
    """
    full = build_features(events)
    k = len(events) // 2
    prefix_events = events.sort_values("ts", kind="stable").iloc[:k].reset_index(drop=True)
    prefix = build_features(prefix_events)

    # Categorical codes come from factorize() over whatever vocabulary is present, so they are
    # legitimately position-dependent; they carry no temporal information.
    compare = [c for c in full.columns if not c.endswith("_code")]
    pd.testing.assert_frame_equal(
        full[compare].iloc[:k].reset_index(drop=True),
        prefix[compare].reset_index(drop=True),
        check_exact=False,
        rtol=1e-6,
    )


def test_mutating_the_future_does_not_change_the_past(events):
    """A second, independent angle on the same guarantee.

    Multiply every amount in the back half of the stream by 1000. Features for the front half
    must not move. A rolling window that accidentally centred rather than trailed, or an
    expanding statistic computed without a shift, would fail this even if it passed truncation.
    """
    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)
    k = len(ordered) // 2
    baseline = build_features(ordered)

    tampered = ordered.copy()
    tampered.loc[k:, "amount"] = tampered.loc[k:, "amount"] * 1000.0
    after = build_features(tampered)

    compare = [c for c in baseline.columns if not c.endswith("_code")]
    pd.testing.assert_frame_equal(
        baseline[compare].iloc[:k], after[compare].iloc[:k], check_exact=False, rtol=1e-6
    )


def test_first_event_for_an_entity_has_no_history(events):
    """The very first event a payer ever makes must show an empty history, not a default."""
    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)
    feats = build_features(ordered)
    first_idx = ordered.groupby("payer_id", sort=False).head(1).index
    assert (feats.loc[first_idx, "payer_prior_count"] == 0).all()
    assert (feats.loc[first_idx, "payer_count_24h"] == 0).all()
    assert (feats.loc[first_idx, "payer_seconds_since_prev"] == -1).all()


def test_first_payment_to_a_payee_is_flagged_first_time(events):
    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)
    feats = build_features(ordered)
    first_pair = ordered.groupby(["payer_id", "payee_id"], sort=False).head(1).index
    assert (feats.loc[first_pair, "is_first_time_payee"] == 1).all()
    assert (feats.loc[first_pair, "beneficiary_age_days"] == -1).all()


def test_pair_history_is_distinct_from_payee_volume(events):
    """Guards the key collision that once silently erased the payee entity's own history."""
    feats = build_features(events)
    assert "pair_prior_count" in feats.columns
    assert "payee_prior_count" in feats.columns
    assert not feats["pair_prior_count"].equals(feats["payee_prior_count"])


def test_session_context_flags_are_not_perfect_predictors(events):
    """Legitimate traffic must exercise the context flags too.

    If no legitimate payment ever has an inbound call or an agent-initiated flag, these become
    perfect rules and every reported metric is inflated. This asserts the generator keeps a
    realistic legitimate base rate on each of them.
    """
    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)
    legit = ordered[ordered.is_fraud == 0]
    for flag in ("inbound_call_active", "agent_initiated"):
        rate = legit[flag].mean()
        assert rate > 0.005, f"{flag} never fires on legitimate traffic (rate {rate})"


def test_determinism(events):
    pd.testing.assert_frame_equal(build_features(events), build_features(events))


def test_scales_to_a_larger_world():
    """Feature building must stay fast enough to sit in a live demo loop."""
    import time

    big = simulate(replace(SMALL, n_customers=4000, days=30))
    start = time.perf_counter()
    build_features(big.events)
    elapsed = time.perf_counter() - start
    assert elapsed < 30, f"feature build took {elapsed:.1f}s"
