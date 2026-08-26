"""Turn a risk score into a reason an analyst can act on.

A held payment needs a justification a human can read, act on, and defend to the customer -
and increasingly to a regulator. SHAP gives the per-event contribution of each feature; this
module maps those contributions back onto the signal families from ``janus.identify.signals``
so the explanation reads as "money moving to a brand-new counterparty during an active inbound
call" rather than as a list of column names and floats.

That mapping is the payoff from making observables declare a signal family back in Pillar 1:
the same vocabulary now runs end to end, from the attack card that predicted the signal to the
alert an analyst reads at three in the morning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janus.defend.supervised import TrainedDetector
from janus.identify.signals import SignalFamily

#: Which signal family each feature serves. Features are named after what they measure, so
#: this is a prefix/keyword mapping rather than a per-feature table.
FEATURE_FAMILY: list[tuple[str, SignalFamily]] = [
    ("is_first_time_payee", SignalFamily.BENEFICIARY_NOVELTY),
    ("beneficiary_age_days", SignalFamily.BENEFICIARY_NOVELTY),
    ("pair_prior_count", SignalFamily.BENEFICIARY_NOVELTY),
    ("personal_payee_fan_in", SignalFamily.GRAPH_FANIN_FANOUT),
    ("payee_distinct_payers", SignalFamily.GRAPH_FANIN_FANOUT),
    ("payee_fan_in_ratio", SignalFamily.GRAPH_FANIN_FANOUT),
    ("payee_balance_retention", SignalFamily.GRAPH_PASSTHROUGH),
    ("payer_gap_regularity", SignalFamily.TEMPORAL_REGULARITY),
    ("seconds_since_prev", SignalFamily.TEMPORAL_REGULARITY),
    ("hour_deviation_from_personal", SignalFamily.TEMPORAL_REGULARITY),
    ("amount_z_vs_personal", SignalFamily.AMOUNT_ANOMALY),
    ("amount_ratio_to_personal", SignalFamily.AMOUNT_ANOMALY),
    ("log_amount", SignalFamily.AMOUNT_ANOMALY),
    ("amount_is_round", SignalFamily.THRESHOLD_HUGGING),
    ("device_age_days", SignalFamily.DEVICE_NOVELTY),
    ("device_", SignalFamily.DEVICE_NOVELTY),
    ("ip_", SignalFamily.DEVICE_NOVELTY),
    ("screen_share_active", SignalFamily.SESSION_CONTEXT),
    ("inbound_call_active", SignalFamily.SESSION_CONTEXT),
    ("agent_initiated", SignalFamily.AGENT_ACTION),
    ("step_up_required", SignalFamily.AUTH_ANOMALY),
    ("auth_method_code", SignalFamily.AUTH_ANOMALY),
    ("is_declined", SignalFamily.DECLINE_PROBING),
    ("geo_mismatch", SignalFamily.GEO_ANOMALY),
    ("payee_account_age_days", SignalFamily.TENURE_MISMATCH),
    ("payee_is_merchant", SignalFamily.MERCHANT_LIFECYCLE),
    ("merchant_category_code", SignalFamily.MERCHANT_LIFECYCLE),
    ("payer_count", SignalFamily.ESCALATION_SEQUENCE),
    ("payer_sum", SignalFamily.ESCALATION_SEQUENCE),
    ("payee_count", SignalFamily.GRAPH_FANIN_FANOUT),
    ("payee_sum", SignalFamily.GRAPH_PASSTHROUGH),
]

#: Human-readable phrasing per family, used to compose alert text.
FAMILY_PHRASE: dict[SignalFamily, str] = {
    SignalFamily.BENEFICIARY_NOVELTY: "payment to a counterparty with no prior relationship",
    SignalFamily.GRAPH_FANIN_FANOUT: "recipient is collecting from many unrelated payers",
    SignalFamily.GRAPH_PASSTHROUGH: "recipient passes funds straight through without retaining them",
    SignalFamily.TEMPORAL_REGULARITY: "timing is machine-like or wrong for this customer",
    SignalFamily.AMOUNT_ANOMALY: "amount is out of pattern for this customer",
    SignalFamily.THRESHOLD_HUGGING: "value sits suspiciously close to a control limit",
    SignalFamily.DEVICE_NOVELTY: "device or network is unfamiliar for this account",
    SignalFamily.SESSION_CONTEXT: "an inbound call or screen share overlaps this payment",
    SignalFamily.AGENT_ACTION: "initiated by an automated agent, not a person",
    SignalFamily.AUTH_ANOMALY: "authentication context contradicts the claimed identity",
    SignalFamily.DECLINE_PROBING: "pattern resembles a search for a working instrument",
    SignalFamily.GEO_ANOMALY: "geography is inconsistent with this account's history",
    SignalFamily.TENURE_MISMATCH: "counterparty is far newer than this customer's norm",
    SignalFamily.MERCHANT_LIFECYCLE: "acceptance point is unusually new or fast-growing",
    SignalFamily.ESCALATION_SEQUENCE: "spend is escalating within a short window",
}


def family_of_feature(name: str) -> SignalFamily | None:
    for prefix, family in FEATURE_FAMILY:
        if name.startswith(prefix) or prefix in name:
            return family
    return None


@dataclass(slots=True)
class Explanation:
    """Why one event scored the way it did."""

    score: float
    top_features: list[tuple[str, float]]
    families: list[tuple[str, float]]
    reasons: list[str]

    def headline(self) -> str:
        return self.reasons[0] if self.reasons else "no dominant risk driver"


class Explainer:
    """SHAP-backed explanations, with a fast fallback when SHAP is unavailable."""

    def __init__(self, detector: TrainedDetector):
        self.detector = detector
        try:
            import shap

            self._explainer = shap.TreeExplainer(detector.model)
        except Exception:  # noqa: BLE001 - an explanation failure must never break scoring
            self._explainer = None

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self._explainer is None:
            # Fallback: global importance scaled by how far each feature sits from its median.
            imp = self.detector.model.feature_importances_.astype(float)
            z = (X[self.detector.features].to_numpy(dtype=float) - X[self.detector.features].median().to_numpy())
            return z * imp
        values = self._explainer.shap_values(X[self.detector.features])
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values)

    def explain(self, X: pd.DataFrame, *, top: int = 5) -> list[Explanation]:
        """Per-row explanations for a (small) batch of events."""
        scores = self.detector.score(X)
        contributions = self.shap_values(X)
        names = self.detector.features
        out: list[Explanation] = []

        for i in range(len(X)):
            row = contributions[i]
            order = np.argsort(-row)[:top]
            top_feats = [(names[j], float(row[j])) for j in order if row[j] > 0]

            family_totals: dict[str, float] = {}
            for name, value in top_feats:
                fam = family_of_feature(name)
                if fam is not None:
                    family_totals[str(fam)] = family_totals.get(str(fam), 0.0) + value

            ranked = sorted(family_totals.items(), key=lambda kv: -kv[1])
            reasons = [
                FAMILY_PHRASE.get(SignalFamily(fam), fam.replace("_", " "))
                for fam, _ in ranked
            ]
            out.append(
                Explanation(
                    score=float(scores[i]),
                    top_features=top_feats,
                    families=ranked,
                    reasons=reasons,
                )
            )
        return out
