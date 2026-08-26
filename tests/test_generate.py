"""Generator tests.

These protect the properties the whole project rests on: that the data is reproducible, that
every attack card claiming a simulator has one that works, and that legitimate traffic has the
structure the detection features assume. A generator that silently loses its structure would
still produce impressive-looking metrics - on data that means nothing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from janus.generate.legit import generate_legit
from janus.generate.rails import EVENT_COLUMNS, LABEL_COLUMNS
from janus.generate.simulate import load_injectors, simulate
from janus.generate.world import WorldConfig, build_world

SMALL = WorldConfig(n_customers=1500, n_merchants=300, days=20, seed=7)


@pytest.fixture(scope="module")
def small_sim():
    return simulate(SMALL)


def test_simulation_is_deterministic():
    """Same seed, same data. A submission that cannot be reproduced cannot be checked."""
    a = simulate(SMALL).events
    b = simulate(SMALL).events
    assert len(a) == len(b)
    pd.testing.assert_frame_equal(
        a[["ts", "amount", "payer_id", "payee_id", "is_fraud", "attack_id"]],
        b[["ts", "amount", "payer_id", "payee_id", "is_fraud", "attack_id"]],
    )


def test_determinism_survives_a_fresh_interpreter():
    """Reproducibility must hold ACROSS processes, not just within one.

    `test_simulation_is_deterministic` compares two calls in a single interpreter, where
    Python's `hash()` is stable - so it passed happily while the simulator was in fact
    irreproducible between runs. Builtin `hash()` is randomised per process for strings, and
    it was being used to derive per-injector seeds: the same `--seed 13` produced 53,680 rows
    with 422 fraud events in one process and 53,767 with 506 in the next.

    This runs the simulation in subprocesses with deliberately different PYTHONHASHSEED values
    and requires byte-identical output. It is the only version of the determinism check that
    can fail on that bug.
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "import hashlib, numpy as np;"
        "from janus.generate.simulate import simulate;"
        "from janus.generate.world import WorldConfig;"
        "ev = simulate(WorldConfig(n_customers=900, n_merchants=150, days=12, seed=13)).events;"
        "print(__import__('json').dumps({"
        "'rows': len(ev), 'fraud': int(ev.is_fraud.sum()),"
        "'md5': hashlib.md5(np.ascontiguousarray(ev.amount.to_numpy(), dtype=np.float64)"
        ".tobytes()).hexdigest()}))"
    )

    results = []
    for hash_seed in ("0", "12345", "99999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600
        )
        assert out.returncode == 0, out.stderr[-2000:]
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    assert results[0] == results[1] == results[2], (
        f"simulation differs across processes: {results}. Something is deriving a random seed "
        "from builtin hash() - use janus.seeding.derive_seed instead."
    )


def test_different_seeds_give_different_data():
    a = simulate(SMALL).events
    b = simulate(replace(SMALL, seed=8)).events
    assert not a.amount.equals(b.amount)


def test_schema_is_exact(small_sim):
    assert list(small_sim.events.columns) == EVENT_COLUMNS


def test_no_nulls_in_decision_critical_fields(small_sim):
    critical = ["ts", "amount", "rail", "payer_id", "payee_id", "is_fraud"]
    assert not small_sim.events[critical].isna().any().any()


def test_amounts_are_positive_and_finite(small_sim):
    amt = small_sim.events.amount.to_numpy()
    assert np.isfinite(amt).all()
    assert (amt > 0).all()


def test_every_simulated_card_produces_events(small_sim):
    """Coverage cannot be claimed by a card whose injector silently yields nothing."""
    declared = set(load_injectors())
    produced = set(small_sim.events.loc[small_sim.events.is_fraud == 1, "attack_id"].unique())
    missing = declared - produced
    assert not missing, f"injectors that produced no fraud events: {sorted(missing)}"


def test_fraud_rate_calibration_hits_target():
    """On a world large enough to hold every family, the target should be met."""
    big = replace(SMALL, n_customers=12_000, days=45)
    res = simulate(big, target_fraud_rate=0.004)
    assert res.calibration_achieved, res.fraud_rate
    assert 0.002 < res.fraud_rate < 0.006, res.fraud_rate


def test_small_worlds_prefer_coverage_over_rate_target():
    """The one-campaign-per-family floor must win, and must be reported when it does."""
    res = simulate(SMALL, target_fraud_rate=0.0005)
    assert not res.calibration_achieved
    assert res.per_attack.shape[0] == len(load_injectors()), "no family may be dropped"


def test_uncalibrated_rate_is_higher_than_calibrated():
    assert simulate(SMALL, target_fraud_rate=None).fraud_rate > simulate(SMALL).fraud_rate


def test_leave_one_attack_out_removes_only_that_family():
    held = "VY-SOC-001"
    res = simulate(SMALL, exclude=[held])
    present = set(res.events.loc[res.events.is_fraud == 1, "attack_id"].unique())
    assert held not in present
    assert len(present) >= 15, "excluding one family should not disturb the others"


def test_labels_are_never_features():
    """Label columns must be identifiable so the feature store can refuse to read them."""
    assert LABEL_COLUMNS <= set(EVENT_COLUMNS)
    assert "is_fraud" in LABEL_COLUMNS


# --- structural realism of legitimate traffic ------------------------------------------

def test_legitimate_hours_are_bimodal():
    """Flat hours would make 'unusual hour' meaningless as a feature."""
    world = build_world(SMALL)
    hours = pd.to_datetime(generate_legit(world).ts).dt.hour
    counts = hours.value_counts().reindex(range(24), fill_value=0).to_numpy()
    night = counts[2:6].sum()
    peaks = counts[10:13].sum() + counts[19:22].sum()
    assert peaks > night * 4, "expected pronounced morning and evening peaks"


def test_round_number_spikes_exist():
    """Real UPI amount histograms are spiky; a smooth one is a synthetic-data tell."""
    world = build_world(SMALL)
    amounts = generate_legit(world).amount
    on_ladder = amounts.isin([100, 200, 500, 1000, 2000, 5000]).mean()
    assert on_ladder > 0.03, f"only {on_ladder:.2%} of amounts land on round values"


def test_most_p2p_goes_to_known_contacts():
    """First-time-beneficiary only discriminates if repeat payees are the norm."""
    world = build_world(SMALL)
    df = generate_legit(world)
    p2p = df[df.rail == "upi_p2p"]
    known = [
        int(row.payee_id) in set(world.contacts_of(int(row.payer_id)).tolist())
        for row in p2p.head(3000).itertuples()
    ]
    assert np.mean(known) > 0.6, "legitimate P2P should mostly reach existing contacts"


def test_fraud_beneficiaries_are_newer_than_legitimate_ones(small_sim):
    """The core signal the detector will lean on must actually be present in the data."""
    ev = small_sim.events
    p2p = ev[ev.payee_type == "person"]
    fraud_age = p2p.loc[p2p.is_fraud == 1, "payee_account_age_days"].median()
    legit_age = p2p.loc[p2p.is_fraud == 0, "payee_account_age_days"].median()
    assert fraud_age < legit_age


# --- guards against generator artefacts masquerading as fraud signal --------------------

def test_no_categorical_value_is_a_perfect_fraud_oracle(small_sim):
    """No single categorical value may be near-deterministic of fraud.

    This guards the most damaging failure mode in the whole project, and it has caught it
    twice. Legitimate traffic once used only six of the ten rails, never the `api` channel,
    never `cvv_only` authentication and never a non-empty payment reference - so each of those
    values appeared exclusively on fraud. The classifier duly reported ROC-AUC 0.9992, which
    measured nothing except which values the legitimate generator had declined to produce.

    Real fraud does not arrive on its own private enum. If this test fails, the fix is in the
    generator, never in the threshold.
    """
    ev = small_sim.events
    problems = []
    for col in ("rail", "channel", "auth_method", "payee_type", "merchant_category"):
        grouped = ev.groupby(col).is_fraud.agg(["mean", "size"])
        for value, row in grouped[grouped["size"] >= 30].iterrows():
            if row["mean"] > 0.90:
                problems.append(f"{col}={value!r} is {row['mean']:.1%} fraud over {int(row['size'])} events")
    assert not problems, "generator artefacts acting as perfect fraud oracles:\n  " + "\n  ".join(problems)


def test_numeric_ranges_overlap_between_classes(small_sim):
    """Fraud and legitimate traffic must share the same numeric universe.

    `device_age_days` once had a hard floor of 5 for legitimate events while attacker devices
    were 0-4 days old, making `device_age_days < 5` a 100% fraud oracle. Attackers buy handsets
    and connectivity from the same market as everyone else; what differs is the distribution,
    not the support.
    """
    ev = small_sim.events
    fraud = ev[ev.is_fraud == 1]
    legit = ev[ev.is_fraud == 0]
    for col in ("device_age_days", "ip_prefix", "payee_account_age_days", "amount"):
        lo = max(fraud[col].quantile(0.05), legit[col].quantile(0.05))
        hi = min(fraud[col].quantile(0.95), legit[col].quantile(0.95))
        assert lo < hi, f"{col}: fraud and legitimate ranges barely overlap"


def test_legitimate_traffic_uses_every_rail(small_sim):
    """Any rail that only fraud ever touches is a free answer for the classifier."""
    ev = small_sim.events
    legit_rails = set(ev.loc[ev.is_fraud == 0, "rail"].unique())
    fraud_rails = set(ev.loc[ev.is_fraud == 1, "rail"].unique())
    fraud_only = fraud_rails - legit_rails
    assert not fraud_only, f"rails used exclusively by fraud: {sorted(fraud_only)}"


def test_payment_references_are_not_fraud_exclusive(small_sim):
    """Legitimate transfers carry notes too, and they share vocabulary with scam ones."""
    ev = small_sim.events
    legit_with_note = (ev[ev.is_fraud == 0].payment_reference != "").mean()
    assert legit_with_note > 0.2, (
        f"only {legit_with_note:.1%} of legitimate transfers carry a note - any non-empty "
        "reference would then imply fraud"
    )
