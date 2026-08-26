"""Generate the legitimate transaction stream.

This is the part of the simulator that decides whether anything downstream is worth trusting.
Fraud detection is a rare-event problem: at a 0.3% base rate, the model spends essentially all
of its capacity learning what *normal* looks like. If normal is wrong - smooth amounts, uniform
hours, random payees - then every fraud signal is trivially separable and the reported metrics
are meaningless.

Four properties are modelled explicitly because the atlas's observables depend on them:

* **Amounts** are log-normal per merchant category, then snapped to the human round-number
  ladder at a category-specific rate. Real UPI amount histograms are spiky, not smooth.
* **Timing** is a bimodal circadian mixture with a per-customer phase offset, so "unusual hour"
  is defined relative to a person rather than to a global clock.
* **Payees** come mostly from the customer's contact graph and merchant affinity set, which is
  what gives "first-time beneficiary" and "never seen merchant" their discriminative power.
* **Weekday effects** shift volume toward weekends for retail categories.

Everything is vectorised; a 60-day, 10k-customer world generates in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.rails import (
    MCC_PROFILES,
    AuthMethod,
    Channel,
    Decision,
    sample_amounts,
    snap_to_round,
)
from janus.generate.world import CATEGORY_NAMES, CITY_NAMES, World

#: Bimodal hour-of-day mixture: a late-morning peak and a larger post-work evening peak.
HOUR_MODES = np.array([11.2, 20.1])
HOUR_SIGMA = np.array([2.3, 2.6])
HOUR_WEIGHTS = np.array([0.42, 0.58])

#: Relative transaction volume by weekday (Mon=0). Retail spend concentrates Fri-Sun.
WEEKDAY_WEIGHTS = np.array([0.92, 0.90, 0.95, 1.00, 1.18, 1.32, 1.15])

#: Every rail the world carries. Legitimate traffic MUST exercise all of them.
#:
#: This is not a cosmetic point. An earlier revision generated legitimate traffic on only six
#: rails while attacks used ten, which made `rail == card_token` a 100% fraud oracle - along
#: with `channel == api`, `auth_method == cvv_only`, and every non-empty payment reference.
#: Four perfect separators, all of them artefacts of what the legitimate generator declined to
#: produce rather than anything about fraud. Tokenised wallet payments, API-initiated mandate
#: debits, CVV-only low-value authorisations and transfer notes are all overwhelmingly
#: legitimate in the real world, and the detector has to learn them as such.
CARD_RAILS = {"card_cnp": 0.42, "card_cp": 0.30, "card_token": 0.28}
NON_CARD_RAILS = {
    "upi_p2m": 0.575, "upi_p2p": 0.300, "upi_mandate": 0.045,
    "upi_collect": 0.020, "imps": 0.035, "neft": 0.017, "rtgs": 0.008,
}

#: Transfer notes people genuinely write. Deliberately overlapping with scam vocabulary -
#: real customers do send "urgent", "medical" and "fees" - so the text layer cannot succeed by
#: keyword-matching alone.
LEGIT_REFERENCES = [
    "", "", "", "", "", "", "", "",  # most transfers carry no note at all
    "rent", "split bill", "groceries", "thanks", "dinner", "fees", "emi",
    "medical", "urgent", "school fees", "petrol", "cab", "gift", "loan repayment",
    "monthly", "advance", "settlement", "insurance premium", "maintenance",
]

#: Probability a P2P payment goes to someone already in the payer's contact graph.
#: The residual is genuine new-payee activity, and it is what stops "first-time beneficiary"
#: from being a perfect fraud oracle - real people do pay new people.
P_KNOWN_CONTACT = 0.88
#: Probability a merchant payment goes to one of the customer's regular merchants.
P_KNOWN_MERCHANT = 0.82


def _sample_hours(rng: np.random.Generator, cust: np.ndarray, world: World) -> np.ndarray:
    """Bimodal circadian hours with a per-customer phase offset."""
    mode = rng.choice(2, size=len(cust), p=HOUR_WEIGHTS)
    hours = rng.normal(HOUR_MODES[mode], HOUR_SIGMA[mode])
    hours += world.cust_circadian[cust]
    return np.mod(hours, 24.0)


def _sample_amounts(
    rng: np.random.Generator, cust: np.ndarray, cat_idx: np.ndarray, world: World
) -> np.ndarray:
    """Log-normal per category, scaled by the payer's income band, then round-snapped."""
    mu = np.array([MCC_PROFILES[c]["mu"] for c in CATEGORY_NAMES])[cat_idx]
    sigma = np.array([MCC_PROFILES[c]["sigma"] for c in CATEGORY_NAMES])[cat_idx]
    round_pref = np.array([MCC_PROFILES[c]["round_pref"] for c in CATEGORY_NAMES])[cat_idx]

    # sigma * 0.72 and an income exponent of 0.30 come from a sweep against Sparkov's measured
    # log-amount spread (sd 0.60); they bring the synthetic spread to 0.61.
    amounts = sample_amounts(rng, len(cust), mu, sigma * 0.72)
    amounts *= world.cust_income_mult[cust] ** 0.30
    snap_mask = rng.random(len(amounts)) < round_pref
    amounts = snap_to_round(amounts, snap_mask, rng)
    return np.round(amounts, 2).clip(1.0, 2_000_000.0)


def _ragged_pick(
    rng: np.random.Generator, cust: np.ndarray, flat: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Pick one entry per event from each customer's CSR neighbour list, vectorised."""
    sizes = (offset[cust + 1] - offset[cust]).astype(np.int64)
    sizes = np.maximum(sizes, 1)
    picks = offset[cust] + (rng.random(len(cust)) * sizes).astype(np.int64)
    return flat[np.minimum(picks, len(flat) - 1)]


def generate_legit(world: World, *, seed: int | None = None) -> pd.DataFrame:
    """Generate the full legitimate event stream for the world's configured window."""
    cfg = world.cfg
    rng = np.random.default_rng(cfg.seed + 1 if seed is None else seed)

    counts = rng.poisson(world.cust_rate * cfg.days)
    cust = np.repeat(np.arange(world.n_customers), counts)
    n = len(cust)
    if n == 0:
        return pd.DataFrame(columns=["event_id"])

    # --- when ---------------------------------------------------------------------
    day_w = np.resize(WEEKDAY_WEIGHTS, cfg.days)
    day_w = day_w / day_w.sum()
    day = rng.choice(cfg.days, size=n, p=day_w)
    hour = _sample_hours(rng, cust, world)
    seconds = (day * 86400 + hour * 3600 + rng.random(n) * 60).astype(np.int64)
    ts = np.datetime64("2026-01-01T00:00:00") + seconds.astype("timedelta64[s]")

    # --- which rail ---------------------------------------------------------------
    u = rng.random(n)
    is_card = u < world.cust_card_share[cust]

    card_names = np.array(list(CARD_RAILS))
    card_p = np.array(list(CARD_RAILS.values()))
    other_names = np.array(list(NON_CARD_RAILS))
    other_p = np.array(list(NON_CARD_RAILS.values()))

    rail = np.where(
        is_card,
        card_names[rng.choice(len(card_names), size=n, p=card_p / card_p.sum())],
        other_names[rng.choice(len(other_names), size=n, p=other_p / other_p.sum())],
    )

    is_p2m = rail == "upi_p2m"
    is_bank = np.isin(rail, ["imps", "neft", "rtgs"])
    is_mandate = rail == "upi_mandate"
    is_collect = rail == "upi_collect"

    # Mandate debits and collect requests are merchant-directed too.
    merchant_side = is_p2m | is_card | is_mandate | is_collect

    # --- who is paid --------------------------------------------------------------
    merch = _ragged_pick(rng, cust, world.affinity_flat, world.affinity_offset)
    stray_merchant = rng.random(n) > P_KNOWN_MERCHANT
    random_merch = rng.choice(
        world.n_merchants, size=n, p=world.merch_popularity / world.merch_popularity.sum()
    )
    merch = np.where(stray_merchant, random_merch, merch)

    payee_person = _ragged_pick(rng, cust, world.contacts_flat, world.contacts_offset)
    stray_person = rng.random(n) > P_KNOWN_CONTACT
    payee_person = np.where(stray_person, rng.integers(0, world.n_customers, n), payee_person)

    # --- how much -----------------------------------------------------------------
    cat_idx = np.where(merchant_side, world.merch_category[merch], -1)
    # Person-to-person transfers use their own ticket profile, not a merchant category.
    # P2P carries a genuinely wider spread than card spend - the same rail moves a 50-rupee
    # split bill and a 2-lakh rent payment - which the PaySim reference confirms (sd 0.83 vs
    # Sparkov's 0.60 in log10 rupees).
    p2p_amt = sample_amounts(rng, n, 3.08, 0.70) * world.cust_income_mult[cust] ** 0.30
    p2p_amt = snap_to_round(p2p_amt, rng.random(n) < 0.62, rng)
    # Bank rails carry larger tickets (rent, fees, settlements).
    p2p_amt = np.where(is_bank, p2p_amt * rng.uniform(2.5, 6.0, n), p2p_amt)
    merch_amt = _sample_amounts(rng, cust, np.maximum(cat_idx, 0), world)
    amount = np.where(merchant_side, merch_amt, np.round(p2p_amt, 2)).clip(1.0, 5_000_000.0)

    # --- context ------------------------------------------------------------------
    # Channel follows from the rail. Mandate debits are system-initiated over an API, which is
    # why `api` must be a common legitimate channel rather than an attack fingerprint.
    channel = np.select(
        [
            np.isin(rail, ["card_cp", "card_token"]),
            rail == "card_cnp",
            is_mandate,
            is_bank,
        ],
        [
            str(Channel.POS),
            np.where(rng.random(n) < 0.06, str(Channel.API), str(Channel.WEB)),
            str(Channel.API),
            np.where(rng.random(n) < 0.25, str(Channel.API), str(Channel.WEB)),
        ],
        default=str(Channel.MOBILE_APP),
    )

    # Low-value card-not-present authorisations legitimately skip the 3DS challenge, so
    # `cvv_only` is a normal condition rather than a fraud signature.
    low_value_cnp = (rail == "card_cnp") & (amount < 2000)
    auth = np.select(
        [
            low_value_cnp,
            rail == "card_cnp",
            np.isin(rail, ["card_cp", "card_token"]),
            is_mandate,
            is_bank,
        ],
        [
            str(AuthMethod.CVV_ONLY),
            str(AuthMethod.OTP_3DS),
            str(AuthMethod.TOKEN_DEVICE),
            str(AuthMethod.NONE),
            str(AuthMethod.NETBANKING),
        ],
        default=str(AuthMethod.UPI_PIN),
    )

    # A small, realistic share of legitimate traffic is declined or challenged. Without this
    # the decline-based features would be perfectly fraud-predictive, which they are not.
    dec_u = rng.random(n)
    decision = np.where(
        dec_u < 0.028, str(Decision.DECLINED),
        np.where(dec_u < 0.045, str(Decision.STEP_UP), str(Decision.APPROVED)),
    )

    payee_city = np.where(
        merchant_side, CITY_NAMES[world.merch_city[merch]], CITY_NAMES[world.cust_city[payee_person]]
    )
    payee_handle = np.where(
        merchant_side,
        np.char.add("merchant", merch.astype(str)),
        np.array([f"user{p:06d}@januspsp" for p in payee_person], dtype=object),
    )
    payee_acct_age = np.where(merchant_side, world.merch_age[merch], world.cust_account_age[payee_person])

    df = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "ts": ts,
        "rail": rail.astype(str),
        "channel": channel,
        "payer_id": cust,
        "payer_account": cust,
        "payee_id": np.where(merchant_side, merch + 1_000_000, payee_person),
        "payee_handle": payee_handle.astype(str),
        "payee_type": np.where(merchant_side, "merchant", "person"),
        "amount": amount,
        "currency": "INR",
        "mcc": np.where(merchant_side, [MCC_PROFILES[c]["mcc"] for c in CATEGORY_NAMES[np.maximum(cat_idx, 0)]], 0),
        "merchant_category": np.where(merchant_side, CATEGORY_NAMES[np.maximum(cat_idx, 0)], "p2p"),
        "device_id": world.cust_device[cust],
        "ip_prefix": world.cust_ip_prefix[cust],
        "payer_city": CITY_NAMES[world.cust_city[cust]],
        "payee_city": payee_city,
        "auth_method": auth,
        "step_up_required": (decision == str(Decision.STEP_UP)).astype(np.int8),
        "decision": decision,
        "payment_reference": rng.choice(LEGIT_REFERENCES, size=n),
        # Legitimate base rates for session context. These MUST be non-zero. An earlier
        # revision left them at zero for all legitimate traffic, which made
        # `inbound_call_active` and `agent_initiated` perfect fraud predictors and would have
        # inflated every headline metric in the submission. People genuinely do take calls
        # while paying, genuinely do share screens with support, and increasingly do delegate
        # purchases to agents - so the model has to learn these as weak contextual priors
        # rather than as rules.
        "inbound_call_active": (rng.random(n) < 0.017).astype(np.int8),
        "screen_share_active": (rng.random(n) < 0.004).astype(np.int8),
        "agent_initiated": (rng.random(n) < 0.021).astype(np.int8),
        "payee_account_age_days": payee_acct_age.astype(np.float32),
        "device_age_days": world.cust_device_age[cust].astype(np.float32),
        "is_fraud": np.int8(0),
        "attack_id": "",
        "campaign_id": "",
    })
    return df.sort_values("ts", ignore_index=True)
