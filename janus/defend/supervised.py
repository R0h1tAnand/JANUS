"""The supervised detection layer.

Gradient-boosted trees rather than a neural network, for three reasons that all matter to the
brief's real-world-feasibility criterion: they score in tens of microseconds on CPU, they
handle the heavily mixed numeric/categorical feature space without embedding machinery, and
they are directly explainable via SHAP - which an analyst reviewing a held payment needs and a
regulator will ask for.

Metrics are reported at a **fixed false-positive rate** as well as in the usual aggregate
forms. A payments team does not buy AUC; it buys "how much fraud do we stop if we are willing
to challenge one legitimate payment in a thousand". Aggregate AUC at a 0.6% base rate can look
excellent while the model is useless at any operating point anyone would actually deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Operating points a payments organisation actually reasons about.
TARGET_FPRS = (0.0005, 0.001, 0.005, 0.01)

#: Fixed thread count for training. Must not be -1: see the note in :func:`train`.
N_TRAIN_THREADS = 4


@dataclass(slots=True)
class TrainedDetector:
    """A fitted detector plus the thresholds that make it operable."""

    model: LGBMClassifier
    calibrator: IsotonicRegression | None
    features: list[str]
    thresholds: dict[float, float] = field(default_factory=dict)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probability. Use for anything priced in rupees.

        Isotonic output is a step function, so this is well-calibrated but low-resolution -
        do not threshold on it. See :meth:`rank_score`.
        """
        raw = self.model.predict_proba(X[self.features])[:, 1]
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def score_fast(self, features: np.ndarray) -> np.ndarray:
        """Low-latency scoring path: raw booster on a float32 array.

        This is what a production authorisation path would actually call, and the difference is
        not marginal. Measured on this model: ``predict_proba`` on a one-row DataFrame takes
        ~7,700 microseconds, of which ~64 is the tree ensemble and the rest is pandas and
        sklearn validation overhead - schema checks, feature-name alignment, array copies -
        all of which a real scoring service does once at deploy time, not once per payment.

        Accepts a 2D array of shape (n, n_features) in :attr:`features` order. Returns the
        uncalibrated ranking score, matching :meth:`rank_score`.
        """
        arr = np.ascontiguousarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.model.booster_.predict(arr, num_threads=1)

    def rank_score(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated model score. Use for ranking, thresholding and AUC.

        Calibration and ranking are different jobs and conflating them cost real recall here.
        Isotonic regression is monotonic, so it preserves the *ordering* of events perfectly
        while collapsing 118k distinct scores onto ~68 plateaus. Thresholding on that means
        every operating point has to jump a whole plateau at a time: measured on one holdout,
        recall at a 0.1% budget fell from 0.89 to 0.40 purely from lost resolution, with no
        change to the model's actual discriminative power.

        So: threshold and rank on this, and reserve the calibrated probability for the policy
        engine, where a genuine probability is what the cost arithmetic needs.
        """
        return self.model.predict_proba(X[self.features])[:, 1]

    def threshold_for(self, fpr: float) -> float:
        return self.thresholds.get(fpr, 0.5)


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """The lowest score threshold whose realised false-positive rate is at most ``fpr``.

    Tie-aware, and it has to be. Isotonic calibration is a step function, so a calibrated
    detector emits very few distinct probabilities - in one measured run, 118,873 legitimate
    events collapsed onto 68 distinct scores with 96,078 of them sharing a single value. A
    plain ``np.quantile`` lands *on* one of those plateaus, and ``scores >= t`` then sweeps up
    the entire tied block: the realised FPR came out at 2x the requested budget, silently
    inflating every recall-at-fixed-FPR number that depended on it.

    This walks up to the next distinct score above the plateau instead, which guarantees the
    budget is respected. The cost is that the realised FPR can land *below* target when the
    plateaus are coarse - that is the correct direction to err, and
    :func:`realised_fpr` reports what actually happened so the gap is visible.
    """
    legit = np.sort(scores[y_true == 0])
    n = len(legit)
    if n == 0:
        return 1.0

    allowed = int(np.floor(fpr * n))
    if allowed <= 0:
        return float(np.nextafter(legit[-1], np.inf))

    candidate = legit[n - allowed]
    # searchsorted(..., "left") gives the first index of the tied block, so n - that index is
    # how many legitimate events this threshold would actually flag.
    flagged = n - int(np.searchsorted(legit, candidate, side="left"))
    if flagged <= allowed:
        return float(candidate)

    # The plateau is too wide. Step to the next distinct score above it.
    above = int(np.searchsorted(legit, candidate, side="right"))
    if above >= n:
        return float(np.nextafter(legit[-1], np.inf))
    return float(legit[above])


def realised_fpr(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """The false-positive rate a threshold actually produces, which may be below target."""
    legit = scores[y_true == 0]
    return float((legit >= threshold).mean()) if len(legit) else 0.0


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """Fraction of fraud caught while challenging at most ``fpr`` of legitimate payments."""
    t = threshold_at_fpr(y_true, scores, fpr)
    fraud = scores[y_true == 1]
    return float((fraud >= t).mean()) if len(fraud) else 0.0


def train(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 0,
    calibrate: bool = True,
    n_estimators: int = 400,
) -> TrainedDetector:
    """Fit the detector, then calibrate and derive operating thresholds on a held-out slice.

    The split is **temporal**, not random: the calibration slice is the most recent portion of
    the training window. Random splits let a model calibrate against the same campaigns it
    trained on, which flatters the thresholds it will actually run at.
    """
    features = list(X.columns)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        raise ValueError("cannot train a detector with no positive examples")

    cut = int(len(X) * 0.8)
    X_fit, y_fit = X.iloc[:cut], y[:cut]
    X_cal, y_cal = X.iloc[cut:], y[cut:]

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        # At a 0.6% base rate an unweighted model puts almost no gradient on the positives.
        scale_pos_weight=max(1.0, n_neg / max(n_pos, 1)),
        random_state=seed,
        # Reproducibility is a stated property of this project, and `n_jobs=-1` quietly
        # breaks it. LightGBM parallelises histogram construction across threads, and with
        # -1 the thread count follows whatever the machine happens to have free - so the same
        # seed on the same data produced ROC-AUC 0.8629 on an idle box and 0.7555 while a test
        # suite was running. `deterministic` plus `force_row_wise` and a fixed thread count
        # pin the result regardless of load. It costs some training speed and buys a number
        # a reviewer can actually reproduce.
        n_jobs=N_TRAIN_THREADS,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(X_fit[features], y_fit)

    calibrator = None
    raw_cal = model.predict_proba(X_cal[features])[:, 1] if len(X_cal) else np.array([])
    if calibrate and len(X_cal) > 500 and len(np.unique(y_cal)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal, y_cal)

    detector = TrainedDetector(model=model, calibrator=calibrator, features=features)
    if len(X_cal) and len(np.unique(y_cal)) > 1:
        rank_scores = detector.rank_score(X_cal)
        detector.thresholds = {f: threshold_at_fpr(y_cal, rank_scores, f) for f in TARGET_FPRS}
    return detector


def evaluate(
    detector: TrainedDetector, X: pd.DataFrame, y: np.ndarray, *, scores: np.ndarray | None = None
) -> dict:
    """Aggregate and operating-point metrics on a test split.

    Computed on the ranking score, not the calibrated probability - see
    :meth:`TrainedDetector.rank_score` for why that distinction matters here.
    """
    s = detector.rank_score(X) if scores is None else scores
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "n_fraud": int(y.sum()), "degenerate": True}

    out = {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "base_rate": round(float(y.mean()), 5),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
    }
    for fpr in TARGET_FPRS:
        t = threshold_at_fpr(y, s, fpr)
        pred = (s >= t).astype(int)
        out[f"recall@fpr{fpr}"] = round(recall_at_fpr(y, s, fpr), 4)
        # Reported alongside the target so a coarse calibration plateau is visible rather
        # than being quietly absorbed into a flattering recall figure.
        out[f"realised_fpr@{fpr}"] = round(realised_fpr(y, s, t), 5)
        out[f"precision@fpr{fpr}"] = round(float(precision_score(y, pred, zero_division=0)), 4)
        out[f"f1@fpr{fpr}"] = round(float(f1_score(y, pred, zero_division=0)), 4)

    # The default 0.5 operating point, reported for completeness rather than for use.
    pred_half = (s >= 0.5).astype(int)
    out["recall@0.5"] = round(float(recall_score(y, pred_half, zero_division=0)), 4)
    out["precision@0.5"] = round(float(precision_score(y, pred_half, zero_division=0)), 4)
    return out


def feature_importance(detector: TrainedDetector, top: int = 20) -> pd.DataFrame:
    imp = detector.model.feature_importances_.astype(float)
    return (
        pd.DataFrame({"feature": detector.features, "importance": imp / imp.sum()})
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
