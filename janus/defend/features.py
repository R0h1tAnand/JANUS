"""Point-in-time-correct feature store.

Every feature here answers the question "what could the issuer have known at the instant this
authorisation arrived?" - and nothing else. That constraint is not a nicety. Rolling aggregates
computed carelessly are the single most common way a fraud model reports a spectacular offline
AUC and then fails in production: if a card's 24-hour transaction count includes the *whole*
fraud burst, the model learns to recognise fraud from evidence that will not exist at decision
time.

So the windowing primitive is built to exclude the present row structurally rather than by
convention. For an event at position ``i`` within its entity's time-sorted history, a window
covers ``[t_i - W, t_i)`` - the slice strictly before ``i``. There is no way to accidentally
include the current event, and ``tests/test_features.py`` proves it by recomputing features on
truncated prefixes of the data and asserting the values are bit-identical.

Features are grouped by the signal family from ``janus.identify.signals`` that motivates them,
so each one can be traced back to the attack cards that asked for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janus.generate.rails import LABEL_COLUMNS
from janus.identify.signals import SignalFamily

HOUR_NS = 3_600_000_000_000
WINDOWS: dict[str, int] = {"1h": HOUR_NS, "24h": 24 * HOUR_NS, "7d": 168 * HOUR_NS}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """What a feature measures and which attack cards motivated it."""

    name: str
    family: SignalFamily
    description: str


def _window_stats(
    entity: np.ndarray, ts_ns: np.ndarray, amount: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-entity rolling counts, sums and recency over strictly prior events.

    Returns arrays aligned to the *input* order. The caller passes data already sorted by
    timestamp; this function sorts by (entity, ts) internally and scatters results back.
    """
    n = len(entity)
    order = np.lexsort((ts_ns, entity))
    e, t, a = entity[order], ts_ns[order], amount[order]

    # Group boundaries in the (entity, ts)-sorted array.
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    ends = np.r_[starts[1:], n]

    out = {f"count_{w}": np.zeros(n, dtype=np.float32) for w in WINDOWS}
    out |= {f"sum_{w}": np.zeros(n, dtype=np.float32) for w in WINDOWS}
    out["prior_count"] = np.zeros(n, dtype=np.float32)
    out["seconds_since_prev"] = np.full(n, -1.0, dtype=np.float32)

    for s, en in zip(starts, ends, strict=True):
        gt, ga = t[s:en], a[s:en]
        m = en - s
        pos = np.arange(m)
        prefix = np.r_[0.0, np.cumsum(ga)]

        out["prior_count"][s:en] = pos
        # Recency: gap to the immediately preceding event for this entity.
        gaps = np.full(m, -1.0)
        if m > 1:
            gaps[1:] = (gt[1:] - gt[:-1]) / 1e9
        out["seconds_since_prev"][s:en] = gaps

        for w, span in WINDOWS.items():
            # 'left' so an event exactly W ago is included; `pos` as the upper bound is what
            # excludes the current event, structurally.
            lo = np.searchsorted(gt, gt - span, side="left")
            out[f"count_{w}"][s:en] = pos - lo
            out[f"sum_{w}"][s:en] = prefix[pos] - prefix[lo]

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    return {k: v[inverse] for k, v in out.items()}


def _pair_history(
    payer: np.ndarray, payee: np.ndarray, ts_ns: np.ndarray
) -> dict[str, np.ndarray]:
    """How much prior relationship exists between this exact payer and payee.

    ``beneficiary_age_days`` is computed here rather than carried on the event, because the
    generator handing it over would bake the answer into the data. First-time payees get -1,
    which trees split on cleanly and which is not confusable with a real age.
    """
    n = len(payer)
    key = np.stack([payer, payee])
    order = np.lexsort((ts_ns, key[1], key[0]))
    p, q, t = payer[order], payee[order], ts_ns[order]

    new_pair = np.r_[True, (p[1:] != p[:-1]) | (q[1:] != q[:-1])]
    starts = np.flatnonzero(new_pair)
    ends = np.r_[starts[1:], n]

    prior = np.zeros(n, dtype=np.float32)
    age_days = np.full(n, -1.0, dtype=np.float32)
    for s, en in zip(starts, ends, strict=True):
        m = en - s
        prior[s:en] = np.arange(m)
        first = t[s]
        ages = (t[s:en] - first) / 8.64e13
        ages[0] = -1.0  # no prior relationship at the first-ever payment
        age_days[s:en] = ages

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    return {
        # Named `pair_` rather than `payee_` deliberately: `_window_stats` already emits
        # `payee_prior_count` for the payee entity, and an earlier revision of this module
        # silently overwrote it here. They measure different things - how busy this payee has
        # been, versus how much history this specific payer has with them - and conflating
        # them destroys the first-time-beneficiary signal.
        "pair_prior_count": prior[inverse],
        "beneficiary_age_days": age_days[inverse],
        "is_first_time_payee": (prior[inverse] == 0).astype(np.float32),
    }


def _expanding_baseline(
    entity: np.ndarray, ts_ns: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding mean and standard deviation over an entity's strictly prior values.

    Used for personal baselines: "is this amount unusual *for this person*", which is a far
    stronger signal than "is this amount unusual for the population".
    """
    n = len(entity)
    order = np.lexsort((ts_ns, entity))
    e, v = entity[order], value[order]
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    ends = np.r_[starts[1:], n]

    mean = np.zeros(n, dtype=np.float64)
    std = np.zeros(n, dtype=np.float64)
    for s, en in zip(starts, ends, strict=True):
        gv = v[s:en]
        m = en - s
        pos = np.arange(m)
        csum = np.r_[0.0, np.cumsum(gv)]
        csq = np.r_[0.0, np.cumsum(gv**2)]
        cnt = np.maximum(pos, 1)
        mu = csum[pos] / cnt
        var = np.maximum(csq[pos] / cnt - mu**2, 0.0)
        mean[s:en] = mu
        std[s:en] = np.sqrt(var)

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    return mean[inverse], std[inverse]


def _payee_fan_in(
    payee: np.ndarray, payer: np.ndarray, ts_ns: np.ndarray, span: int
) -> np.ndarray:
    """Distinct prior payers to this payee within a window.

    The core mule signal: a personal account collecting from many unrelated people. Computed
    exactly rather than approximated, because the approximations all confuse "many payments"
    with "many payers", and the difference between those two is the entire signal.
    """
    n = len(payee)
    order = np.lexsort((ts_ns, payee))
    q, p, t = payee[order], payer[order], ts_ns[order]
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], n]

    out = np.zeros(n, dtype=np.float32)
    for s, en in zip(starts, ends, strict=True):
        gt, gp = t[s:en], p[s:en]
        m = en - s
        lo = np.searchsorted(gt, gt - span, side="left")
        # `lo` is non-decreasing because `gt` is sorted, so the window only ever slides
        # forward. That lets a running multiset replace a per-event unique() call, turning
        # this from O(events x window) into O(events).
        counts: dict[int, int] = {}
        distinct = 0
        left = 0
        for i in range(m):
            if i > 0:
                payer_in = int(gp[i - 1])
                c = counts.get(payer_in, 0)
                if c == 0:
                    distinct += 1
                counts[payer_in] = c + 1
            while left < lo[i]:
                payer_out = int(gp[left])
                counts[payer_out] -= 1
                if counts[payer_out] == 0:
                    distinct -= 1
                left += 1
            out[s + i] = distinct

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    return out[inverse]


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature matrix for a time-sorted event frame."""
    df = events.sort_values("ts", kind="stable").reset_index(drop=True)
    ts_ns = df["ts"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    amount = df["amount"].to_numpy(dtype=np.float64)
    payer = df["payer_id"].to_numpy(dtype=np.int64)
    payee = df["payee_id"].to_numpy(dtype=np.int64)
    device = df["device_id"].to_numpy(dtype=np.int64)
    ip = df["ip_prefix"].to_numpy(dtype=np.int64)

    feats: dict[str, np.ndarray] = {}

    # --- velocity and recency per entity -------------------------------------------------
    for label, ent in (("payer", payer), ("payee", payee), ("device", device), ("ip", ip)):
        for k, v in _window_stats(ent, ts_ns, amount).items():
            feats[f"{label}_{k}"] = v

    # --- relationship history ------------------------------------------------------------
    feats |= _pair_history(payer, payee, ts_ns)

    # --- personal baselines --------------------------------------------------------------
    log_amount = np.log10(np.clip(amount, 1.0, None))
    mean_amt, std_amt = _expanding_baseline(payer, ts_ns, log_amount)
    feats["amount_z_vs_personal"] = np.where(
        std_amt > 1e-6, (log_amount - mean_amt) / np.maximum(std_amt, 1e-6), 0.0
    ).astype(np.float32)
    feats["amount_ratio_to_personal_mean"] = (
        log_amount / np.maximum(mean_amt, 1e-6)
    ).astype(np.float32)

    hour = df["ts"].dt.hour.to_numpy(dtype=np.float64)
    mean_hour, std_hour = _expanding_baseline(payer, ts_ns, hour)
    # Circular distance: 23:00 and 01:00 are two hours apart, not twenty-two.
    raw = np.abs(hour - mean_hour)
    feats["hour_deviation_from_personal"] = np.minimum(raw, 24 - raw).astype(np.float32)

    # --- graph structure -----------------------------------------------------------------
    feats["payee_distinct_payers_24h"] = _payee_fan_in(payee, payer, ts_ns, WINDOWS["24h"])
    feats["payee_distinct_payers_7d"] = _payee_fan_in(payee, payer, ts_ns, WINDOWS["7d"])
    # Balance retention: an account that pays out everything it takes in is a pipe, not a
    # destination. Both legs are strictly prior, so this stays point-in-time correct.
    #
    # Getting the outflow leg right is subtle. The first version of this used
    # `groupby(payer).max()` over `payer_sum_7d`, which silently took the maximum across an
    # entity's ENTIRE history - including events after the current one. The prefix-truncation
    # test in tests/test_features.py caught it. Correct version: an as-of join that looks up
    # what the payee's own outbound 7-day sum was at the moment this payment arrived, with
    # `allow_exact_matches=False` so simultaneous events cannot leak into each other.
    inflow = feats["payee_sum_7d"]
    outbound_history = pd.DataFrame(
        {"entity": payer, "ts": df["ts"], "outflow_7d": feats["payer_sum_7d"]}
    ).sort_values("ts", kind="stable")
    lookup = (
        pd.DataFrame({"entity": payee, "ts": df["ts"]})
        .reset_index()
        .sort_values("ts", kind="stable")
    )
    joined = pd.merge_asof(
        lookup, outbound_history, on="ts", by="entity",
        direction="backward", allow_exact_matches=False,
    )
    payee_outflow = (
        joined.sort_values("index")["outflow_7d"].fillna(0.0).to_numpy(dtype=np.float32)
    )
    feats["payee_balance_retention"] = np.where(
        inflow > 0, 1.0 - np.minimum(payee_outflow / np.maximum(inflow, 1.0), 2.0), 1.0
    ).astype(np.float32)
    # Fan-in only counts as a mule signal on a *personal* account. Merchants legitimately
    # collect from hundreds of unrelated payers - the raw distinct-payer count is actually
    # higher for legitimate traffic than fraudulent, because merchants dominate it. What is
    # anomalous is a personal handle with merchant-shaped traffic, which is exactly the
    # QR-substitution and mule-collection signature.
    is_merchant_payee = (df["payee_type"] == "merchant").to_numpy()
    feats["personal_payee_fan_in_24h"] = np.where(
        is_merchant_payee, 0.0, feats["payee_distinct_payers_24h"]
    ).astype(np.float32)
    feats["personal_payee_fan_in_7d"] = np.where(
        is_merchant_payee, 0.0, feats["payee_distinct_payers_7d"]
    ).astype(np.float32)
    feats["payee_fan_in_ratio"] = (
        feats["payee_distinct_payers_24h"] / np.maximum(feats["payee_count_24h"], 1.0)
    ).astype(np.float32)

    # --- burst regularity ----------------------------------------------------------------
    # Machine-paced attacks show near-constant inter-arrival times; humans do not.
    gaps = feats["payer_seconds_since_prev"]
    gap_mean, gap_std = _expanding_baseline(payer, ts_ns, np.where(gaps > 0, gaps, np.nan_to_num(gaps)))
    feats["payer_gap_regularity"] = np.where(
        gap_mean > 1, gap_std / np.maximum(gap_mean, 1.0), 1.0
    ).astype(np.float32)

    # --- static context available at decision time ---------------------------------------
    feats["log_amount"] = log_amount.astype(np.float32)
    feats["hour"] = hour.astype(np.float32)
    feats["day_of_week"] = df["ts"].dt.dayofweek.to_numpy(dtype=np.float32)
    feats["is_weekend"] = (feats["day_of_week"] >= 5).astype(np.float32)
    feats["payee_account_age_days"] = df["payee_account_age_days"].to_numpy(dtype=np.float32)
    feats["device_age_days"] = df["device_age_days"].to_numpy(dtype=np.float32)
    feats["inbound_call_active"] = df["inbound_call_active"].to_numpy(dtype=np.float32)
    feats["screen_share_active"] = df["screen_share_active"].to_numpy(dtype=np.float32)
    feats["agent_initiated"] = df["agent_initiated"].to_numpy(dtype=np.float32)
    feats["step_up_required"] = df["step_up_required"].to_numpy(dtype=np.float32)
    feats["is_declined"] = (df["decision"] == "declined").to_numpy(dtype=np.float32)
    feats["geo_mismatch"] = (df["payer_city"] != df["payee_city"]).to_numpy(dtype=np.float32)
    feats["payee_is_merchant"] = (df["payee_type"] == "merchant").to_numpy(dtype=np.float32)
    feats["amount_is_round"] = (
        np.isin(amount, [100, 200, 500, 1000, 2000, 5000, 10000])
    ).astype(np.float32)

    # Categorical context, integer-coded for LightGBM.
    for col in ("rail", "channel", "auth_method", "merchant_category"):
        feats[f"{col}_code"] = pd.factorize(df[col])[0].astype(np.float32)

    out = pd.DataFrame(feats)
    leaked = LABEL_COLUMNS & set(out.columns)
    if leaked:
        raise AssertionError(f"label columns leaked into features: {leaked}")
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def feature_names(events: pd.DataFrame | None = None) -> list[str]:
    """Stable feature ordering, for model persistence and for the dashboard."""
    if events is not None:
        return list(build_features(events).columns)
    return []
