"""Evaluation harnesses, including leave-one-attack-out.

Two splits, answering two different questions.

**Temporal holdout** answers "how well does the detector do against attacks it has seen?" It is
the number most submissions report, and on synthetic data with 21 known families it is close to
meaningless on its own - the model has seen every family in training, so it is being asked to
recognise something it was explicitly taught.

**Leave-one-attack-out** answers the question the brief actually poses. For each family, train a
detector on a world where that family *does not exist*, then test it against a world where it
does. This is the only honest proxy available for "emerging fraud": a genuinely novel attack is,
by definition, one absent from the training data.

LOAO is expected to be worse than the temporal holdout - sometimes much worse. Families whose
signature is shared with others (mule structures, first-time-payee sweeps) should generalise;
families with an idiosyncratic signature should not. That per-family spread is the useful
output, far more than any single averaged number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from janus.defend import supervised
from janus.defend.features import build_features
from janus.generate.simulate import load_injectors, simulate
from janus.generate.world import WorldConfig


@dataclass(slots=True)
class Split:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    test_events: pd.DataFrame


def temporal_split(events: pd.DataFrame, *, train_fraction: float = 0.7) -> Split:
    """Split on time, never at random.

    A random split lets the same campaign appear on both sides - the model sees three legs of
    an escalation sequence in training and is then asked about the fourth, which is not a test
    of anything. Splitting on time reproduces how the model is actually deployed.
    """
    ordered = events.sort_values("ts", kind="stable").reset_index(drop=True)
    X = build_features(ordered)
    y = ordered["is_fraud"].to_numpy()
    cut = int(len(ordered) * train_fraction)
    return Split(X.iloc[:cut], y[:cut], X.iloc[cut:], y[cut:], ordered.iloc[cut:])


def run_temporal(events: pd.DataFrame, *, seed: int = 0) -> dict:
    split = temporal_split(events)
    detector = supervised.train(split.X_train, split.y_train, seed=seed)
    metrics = supervised.evaluate(detector, split.X_test, split.y_test)
    scores = detector.rank_score(split.X_test)

    # Per-family recall at a deployable operating point, on families the model HAS seen.
    threshold = supervised.threshold_at_fpr(split.y_test, scores, 0.001)
    per_family = {}
    test = split.test_events
    for attack_id in sorted(test.loc[test.is_fraud == 1, "attack_id"].unique()):
        mask = (test.attack_id == attack_id).to_numpy() & (split.y_test == 1)
        if mask.sum():
            per_family[attack_id] = round(float((scores[mask] >= threshold).mean()), 4)

    return {"metrics": metrics, "per_family_recall_at_fpr001": per_family,
            "importance": supervised.feature_importance(detector, 15).to_dict("records")}


def run_loao(
    cfg: WorldConfig | None = None,
    *,
    seed: int = 0,
    families: list[str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Leave-one-attack-out across every simulated family.

    Training and test data come from **different world seeds** as well as different family
    sets, so the held-out family is unseen in the strongest sense available: neither the attack
    nor the population it runs against appeared during training.
    """
    cfg = cfg or WorldConfig()
    all_families = families or sorted(load_injectors())

    # One test world, containing everything, generated from a different seed than any training
    # world so no population overlap can flatter the result.
    test_sim = simulate(replace(cfg, seed=cfg.seed + 1000))
    test_events = test_sim.events.sort_values("ts", kind="stable").reset_index(drop=True)
    X_test = build_features(test_events)
    y_test = test_events["is_fraud"].to_numpy()

    rows = []
    for i, family in enumerate(all_families, 1):
        if progress:
            print(f"  [{i}/{len(all_families)}] holding out {family} ...", flush=True)
        train_sim = simulate(cfg, exclude=[family])
        train_events = train_sim.events.sort_values("ts", kind="stable").reset_index(drop=True)
        X_train = build_features(train_events)
        y_train = train_events["is_fraud"].to_numpy()

        detector = supervised.train(X_train, y_train, seed=seed)
        scores = detector.rank_score(X_test)

        # Threshold is set on legitimate traffic only, so it does not depend on the held-out
        # family being present - the same threshold the model would run at in production.
        threshold = supervised.threshold_at_fpr(y_test, scores, 0.001)
        held_mask = (test_events.attack_id == family).to_numpy() & (y_test == 1)
        seen_mask = (
            (test_events.attack_id != family).to_numpy() & (y_test == 1)
        )

        rows.append({
            "family": family,
            "n_held_out_events": int(held_mask.sum()),
            "recall_unseen": round(float((scores[held_mask] >= threshold).mean()), 4)
            if held_mask.sum() else None,
            "recall_seen_families": round(float((scores[seen_mask] >= threshold).mean()), 4)
            if seen_mask.sum() else None,
            "fpr": 0.001,
        })

    df = pd.DataFrame(rows)
    df["generalisation_gap"] = (df.recall_seen_families - df.recall_unseen).round(4)
    return df.sort_values("recall_unseen", ascending=False, na_position="last").reset_index(drop=True)


#: Folds with fewer test events than this are reported but excluded from the headline mean.
#: VY-SOC-002 (deepfake CFO fraud) is intrinsically low-volume - a handful of very large
#: payments - so a fold can land with two events and a recall of either 0.0 or 1.0. Neither
#: number means anything, and averaging them in would move the headline by several points.
MIN_RELIABLE_EVENTS = 15


def loao_summary(df: pd.DataFrame) -> dict:
    """Headline numbers from a LOAO table, including how many families fail outright."""
    reliable = df[df.n_held_out_events >= MIN_RELIABLE_EVENTS]
    unseen = reliable.recall_unseen.dropna()
    return {
        "n_low_volume_folds_excluded": int((df.n_held_out_events < MIN_RELIABLE_EVENTS).sum()),
        "min_reliable_events": MIN_RELIABLE_EVENTS,
        "families": int(len(df)),
        "mean_recall_unseen": round(float(unseen.mean()), 4),
        "median_recall_unseen": round(float(unseen.median()), 4),
        "mean_recall_seen": round(float(df.recall_seen_families.dropna().mean()), 4),
        "families_above_50pct": int((unseen >= 0.5).sum()),
        "families_below_20pct": int((unseen < 0.2).sum()),
        "worst_family": df.iloc[-1]["family"] if len(df) else None,
        # The gap between these two is the finding. A model that scores well on families it was
        # taught and poorly on families it was not has learned those specific attacks, not fraud.
        "generalisation_gap": round(
            float(reliable.recall_seen_families.dropna().mean() - unseen.mean()), 4
        ),
    }
