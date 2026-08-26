"""The two fidelity tests that are hard to game.

**Discriminator AUC.** Pool synthetic and real records, label them by origin, and train a
gradient-boosted classifier to tell them apart. If it cannot - AUC near 0.5 - the two are
distributionally interchangeable on the compared schema. This subsumes every marginal and
interaction test at once: any tell the KS table would find, the discriminator also finds, plus
the ones nobody thought to check.

It is also the most honest metric available, because it is adversarial by construction. A
generator author cannot tune toward it without genuinely fixing the distribution, and the
feature importances say exactly *which* column gave the game away - which turns a bad score
into a work item rather than an embarrassment.

**TSTR.** Train the fraud detector on synthetic data, test it on real data, and compare against
training on real. This asks the question that actually matters for the project's thesis: does
a model raised entirely on simulated attacks transfer to the real world? A high discriminator
AUC with good TSTR would still be a useful generator; the reverse would not be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#: The discriminator sees standardised amount, not raw. Handing it the currency level would
#: let it separate the datasets on the exchange rate alone and tell us nothing about the
#: generator - it would score ~1.0 on any pair of economies, including two real ones.
FEATURES = ["amount_z", "hour", "day_of_week"]


def _prepare(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    out = df[features or FEATURES].copy()
    # Category is compared separately as a distribution rather than one-hot encoded, so that a
    # vocabulary mismatch between datasets cannot masquerade as a distributional difference.
    return out.astype(float)


def discriminator_auc(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    features: list[str] | None = None,
    n_splits: int = 3,
    seed: int = 0,
) -> dict:
    """Cross-validated AUC for separating synthetic from real, plus what gave it away.

    ``features`` must be restricted to columns the reference dataset genuinely evidences;
    handing the discriminator a column the reference does not really have (PaySim's derived
    hour-of-day, for instance) guarantees a near-perfect AUC that says nothing.
    """
    features = [f for f in (features or FEATURES) if f in synthetic and f in reference]
    rng = np.random.default_rng(seed)
    n = min(len(synthetic), len(reference), 120_000)
    s = synthetic.iloc[rng.choice(len(synthetic), n, replace=False)]
    r = reference.iloc[rng.choice(len(reference), n, replace=False)]

    x = pd.concat([_prepare(s, features), _prepare(r, features)], ignore_index=True)
    y = np.r_[np.ones(n), np.zeros(n)]

    aucs, importances = [], []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(x, y):
        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.08, num_leaves=31,
            random_state=seed, verbose=-1,
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        aucs.append(roc_auc_score(y[test_idx], model.predict_proba(x.iloc[test_idx])[:, 1]))
        importances.append(model.feature_importances_)

    auc = float(np.mean(aucs))
    imp = np.mean(importances, axis=0)
    return {
        "auc": round(auc, 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "n_per_class": n,
        "verdict": _verdict(auc),
        "tells": [
            {"feature": f, "importance": round(float(i / imp.sum()), 3)}
            for f, i in sorted(zip(features, imp, strict=True), key=lambda t: -t[1])
        ],
    }


def _verdict(auc: float) -> str:
    """Interpretation bands for discriminator AUC.

    0.5 is indistinguishable. Anything above ~0.8 means a classifier can reliably spot the
    synthetic records, so any detector trained on them is at risk of learning the generator's
    fingerprint rather than the fraud.
    """
    if auc < 0.55:
        return "indistinguishable"
    if auc < 0.65:
        return "strong"
    if auc < 0.80:
        return "moderate"
    return "weak - the generator has a detectable fingerprint"


def tstr(
    synthetic: pd.DataFrame, reference: pd.DataFrame, *, seed: int = 0
) -> dict:
    """Train-on-synthetic / test-on-real, benchmarked against train-on-real.

    ``transfer_ratio`` is the headline: TSTR AUC divided by TRTR AUC. 1.0 means training on
    simulated data cost nothing at all; the number degrades gracefully and is far more
    informative than either AUC alone.
    """
    ref = reference[reference.is_fraud.notna()]
    if ref.is_fraud.nunique() < 2 or synthetic.is_fraud.nunique() < 2:
        return {"available": False, "reason": "one dataset has a single class"}

    rng = np.random.default_rng(seed)
    test_idx = rng.choice(len(ref), min(len(ref), 150_000), replace=False)
    real_test = ref.iloc[test_idx]
    real_rest = ref.drop(ref.index[test_idx])

    def _fit(train: pd.DataFrame) -> float:
        model = LGBMClassifier(
            n_estimators=250, learning_rate=0.06, num_leaves=31,
            random_state=seed, verbose=-1,
            # Reference fraud rates are extreme; without rebalancing the model predicts one class.
            class_weight="balanced",
        )
        model.fit(_prepare(train), train.is_fraud.to_numpy())
        return float(roc_auc_score(
            real_test.is_fraud.to_numpy(), model.predict_proba(_prepare(real_test))[:, 1]
        ))

    tstr_auc = _fit(synthetic)
    trtr_auc = _fit(real_rest) if len(real_rest) > 1000 and real_rest.is_fraud.nunique() > 1 else float("nan")
    ratio = tstr_auc / trtr_auc if trtr_auc == trtr_auc and trtr_auc > 0 else float("nan")
    return {
        "available": True,
        "tstr_auc": round(tstr_auc, 4),
        "trtr_auc": round(trtr_auc, 4) if trtr_auc == trtr_auc else None,
        "transfer_ratio": round(ratio, 4) if ratio == ratio else None,
        "n_real_test": len(real_test),
    }
