"""The unified payment event schema.

Every rail - UPI, IMPS/NEFT/RTGS, card CNP/CP, AePS, wallet - emits into one wide table with
rail-specific columns left null. A single schema is what lets one detector reason across rails,
which matters because the attacks in the atlas deliberately cross them: a smishing campaign
starts on SMS and lands on card-CNP, while its proceeds launder over UPI P2P.

Field names follow ISO 8583 / ISO 20022 semantics where an equivalent exists, so that the
mapping into a live authorisation stream is a rename rather than a redesign. The mapping is
recorded in :data:`ISO_MAPPING` and rendered in the feasibility section of the walkthrough.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from scipy import stats


class Channel(StrEnum):
    """How the payment instruction reached the rail."""

    MOBILE_APP = "mobile_app"
    WEB = "web"
    POS = "pos"
    ATM = "atm"
    IVR = "ivr"
    API = "api"
    MICRO_ATM = "micro_atm"
    AGENT_ASSISTED = "agent_assisted"


class AuthMethod(StrEnum):
    UPI_PIN = "upi_pin"
    OTP_3DS = "otp_3ds"
    CVV_ONLY = "cvv_only"
    BIOMETRIC = "biometric"
    TOKEN_DEVICE = "token_device"
    NETBANKING = "netbanking"
    NONE = "none"


class Decision(StrEnum):
    """What the rail did with the instruction, before any Janus model sees it."""

    APPROVED = "approved"
    DECLINED = "declined"
    STEP_UP = "step_up"


#: Merchant categories with INR ticket-size parameters, in **log10 rupees**.
#:
#: These were originally guessed and were badly wrong in a way the fidelity harness caught.
#: Measured against Sparkov, real card categories span only ~0.95 decades of median ticket size
#: with a mean within-category spread of ~0.49 decades. The first version of this table spanned
#: 2.13 decades between categories while being far too tight within them, which produced an
#: aggregate amount distribution with too much spread and a right skew where real data is left-
#: skewed - the largest single tell the discriminator was picking up on.
#:
#: Recalibrated to ~1.45 decades between categories: wider than the US reference, because Indian
#: card portfolios genuinely disperse more between a metro ride and a jewellery purchase, but
#: nothing like the original range. ``round_pref`` is the probability an amount snaps to a
#: psychologically round figure - very high for recharges and P2P, near zero for fuel or
#: groceries where the amount is simply whatever the bill was.
#:
#: KNOWN AND DELIBERATE GAP: after recalibration the synthetic aggregate matches the reference on
#: spread (sd 0.61 vs 0.60) but not on skew (~0.0 vs -0.44). Sparkov's left skew is produced by a
#: US-specific structure in which grocery and fuel are simultaneously the highest-frequency,
#: highest-ticket and lowest-variance categories, which packs mass at the top of the range. That
#: is not true of Indian payments, where those same categories carry a long tail of small
#: transactions. Matching the skew would require mis-stating Indian ticket sizes to reproduce an
#: artefact of the US reference, so the gap is reported in the fidelity output instead of tuned
#: away. It is the largest known residual and it is stated rather than hidden.
MCC_PROFILES: dict[str, dict] = {
    "transport":        {"mcc": 4121, "mu": 2.30, "sigma": 0.34, "round_pref": 0.15},
    "telecom_recharge": {"mcc": 4814, "mu": 2.45, "sigma": 0.26, "round_pref": 0.80},
    "entertainment":    {"mcc": 7832, "mu": 2.60, "sigma": 0.44, "round_pref": 0.20},
    "pharmacy":         {"mcc": 5912, "mu": 2.65, "sigma": 0.48, "round_pref": 0.05},
    "digital_goods":    {"mcc": 5815, "mu": 2.70, "sigma": 0.42, "round_pref": 0.35},
    "restaurant":       {"mcc": 5812, "mu": 2.78, "sigma": 0.52, "round_pref": 0.10},
    "grocery":          {"mcc": 5411, "mu": 2.90, "sigma": 0.40, "round_pref": 0.05},
    "fuel":             {"mcc": 5541, "mu": 3.05, "sigma": 0.28, "round_pref": 0.25},
    "ecommerce":        {"mcc": 5399, "mu": 3.06, "sigma": 0.66, "round_pref": 0.05},
    "utilities":        {"mcc": 4900, "mu": 3.16, "sigma": 0.45, "round_pref": 0.02},
    "healthcare":       {"mcc": 8062, "mu": 3.30, "sigma": 0.58, "round_pref": 0.10},
    "travel":           {"mcc": 4722, "mu": 3.52, "sigma": 0.72, "round_pref": 0.05},
    "education":        {"mcc": 8220, "mu": 3.62, "sigma": 0.55, "round_pref": 0.30},
    "electronics":      {"mcc": 5732, "mu": 3.66, "sigma": 0.62, "round_pref": 0.10},
    "jewellery":        {"mcc": 5944, "mu": 3.78, "sigma": 0.68, "round_pref": 0.15},
}

#: Amounts people actually type when sending money to another person, in rupees.
#: Real UPI P2P data is dominated by these; a smooth log-normal is a tell-tale sign of
#: synthetic data, so the legitimate generator snaps to this ladder most of the time.
ROUND_AMOUNTS = np.array(
    [10, 20, 50, 100, 150, 200, 250, 300, 500, 700, 1000, 1500, 2000, 2500,
     3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000],
    dtype=np.float64,
)

#: Column order of the event table. Kept explicit so the parquet schema is stable across runs
#: and so a reviewer can see the whole record at a glance.
EVENT_COLUMNS: list[str] = [
    # identity of the event
    "event_id", "ts", "rail", "channel",
    # parties
    "payer_id", "payer_account", "payee_id", "payee_handle", "payee_type",
    # value
    "amount", "currency", "mcc", "merchant_category",
    # instrument and context
    "device_id", "ip_prefix", "payer_city", "payee_city",
    "auth_method", "step_up_required", "decision",
    # free text carried on the rail
    "payment_reference",
    # session-level context available to a real issuer at decision time.
    # Note what is absent: beneficiary_age_days is NOT a raw field, because it is derivable
    # from the payer's own history. Handing it over from the generator would bake the
    # answer into the data; janus.defend.features computes it point-in-time instead.
    "inbound_call_active", "screen_share_active", "agent_initiated",
    "payee_account_age_days", "device_age_days",
    # labels - never features
    "is_fraud", "attack_id", "campaign_id",
]

#: Columns that must never reach the model. Enforced in janus.defend.features.
LABEL_COLUMNS: set[str] = {"is_fraud", "attack_id", "campaign_id"}

#: How each Janus field maps onto a live authorisation message. Used by the feasibility
#: section to show the schema is a rename away from production, not an invention.
ISO_MAPPING: dict[str, str] = {
    "ts": "ISO8583 DE-07 transmission date/time | ISO20022 pacs.008 CreDtTm",
    "amount": "ISO8583 DE-04 amount, transaction | ISO20022 IntrBkSttlmAmt",
    "currency": "ISO8583 DE-49 currency code | ISO20022 Ccy",
    "mcc": "ISO8583 DE-18 merchant category code",
    "payer_account": "ISO8583 DE-02 primary account number | ISO20022 DbtrAcct",
    "payee_handle": "ISO8583 DE-42 card acceptor id | UPI VPA | ISO20022 CdtrAcct",
    "channel": "ISO8583 DE-22 POS entry mode",
    "auth_method": "ISO8583 DE-22/DE-52 | EMV 3DS AReq deviceChannel",
    "decision": "ISO8583 DE-39 response code",
    "payment_reference": "ISO20022 RmtInf/Ustrd | UPI transaction note",
    "device_id": "EMV 3DS AReq browser/SDK fingerprint",
}


#: Skewness parameter for the log-amount distribution.
#: Measured against both reference datasets, real transaction amounts are *left*-skewed in log
#: space - Sparkov -0.44, PaySim -0.35 - because a hard floor on transaction size truncates the
#: bottom while the top is compressed by what people can actually afford. A plain log-normal is
#: symmetric in log space and, once income scaling and round-snapping are applied, comes out
#: right-skewed at about +0.40. That mismatch was the single largest contributor to the
#: discriminator being able to spot synthetic records.
#: Set to -4 after a parameter sweep. Note honestly that this closes only part of the gap: the
#: aggregate log-amount skew lands near 0 rather than the reference's -0.44, because most of the
#: residual is contributed by the *category mixture* rather than the per-draw deviate. See the
#: note on MCC_PROFILES for why that remaining gap is left in place rather than tuned away.
LOG_SKEW = -4.0


def sample_amounts(
    rng: np.random.Generator, n: int, mu: np.ndarray | float, sigma: np.ndarray | float
) -> np.ndarray:
    """Draw amounts whose log10 is skew-normal rather than normal.

    ``mu`` and ``sigma`` are in log10 rupees, matching :data:`MCC_PROFILES` and the units the
    fidelity report quotes, so a discrepancy can be read straight off the comparison table.
    """
    z = stats.skewnorm.rvs(a=LOG_SKEW, size=n, random_state=rng)
    delta = LOG_SKEW / np.sqrt(1 + LOG_SKEW**2)
    mean = delta * np.sqrt(2 / np.pi)
    sd = np.sqrt(1 - 2 * delta**2 / np.pi)
    return np.power(10.0, mu + sigma * (z - mean) / sd)


def snap_to_round(amounts: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Snap the masked subset of ``amounts`` to the nearest human-typed round amount.

    Vectorised: finds each amount's nearest neighbour on the :data:`ROUND_AMOUNTS` ladder
    rather than rounding to a fixed decimal, so the resulting spikes sit exactly where they
    sit in real UPI data.
    """
    out = amounts.copy()
    if not mask.any():
        return out
    idx = np.abs(amounts[mask, None] - ROUND_AMOUNTS[None, :]).argmin(axis=1)
    out[mask] = ROUND_AMOUNTS[idx]
    return out
