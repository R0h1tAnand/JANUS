"""Unsupervised novelty layer.

The supervised model can only recognise what it was taught. This layer is trained on
*legitimate traffic only* and never sees a fraud label, so its notion of suspicious is
"unlike normal" rather than "like known fraud". That makes it the component that has any
chance against a family the atlas has not imagined yet.

It is deliberately kept as a separate score rather than folded into the classifier. Blending
them into a single number would let the supervised model's confidence drown out the novelty
signal exactly when the novelty signal is the only one that is right - which is the case that
matters. The policy engine combines them explicitly, where the trade-off is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import QuantileTransformer


@dataclass(slots=True)
class NoveltyDetector:
    forest: IsolationForest
    scaler: QuantileTransformer
    features: list[str]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Novelty in [0, 1]; higher is more unlike the legitimate baseline."""
        z = self.scaler.transform(X[self.features])
        # score_samples is higher for inliers, so negate and squash to [0, 1].
        raw = -self.forest.score_samples(z)
        return (raw - raw.min()) / max(float(np.ptp(raw)), 1e-9)


def train(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 150,
    max_samples: int = 50_000,
) -> NoveltyDetector:
    """Fit on legitimate rows only.

    Training on the full mixture would defeat the purpose: the model would learn that fraud is
    part of normal, which is precisely the assumption this layer exists to avoid making.
    """
    features = list(X.columns)
    legit = X[y == 0]
    if len(legit) < 1000:
        legit = X

    # Quantile transform first: isolation forests split on raw values, and the feature space
    # spans counts in single digits alongside rupee sums in the millions.
    scaler = QuantileTransformer(
        n_quantiles=min(1000, len(legit)), output_distribution="uniform", random_state=seed
    )
    z = scaler.fit_transform(legit[features])

    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, len(z)),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(z)
    return NoveltyDetector(forest=forest, scaler=scaler, features=features)
