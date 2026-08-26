"""Project Janus events and public reference data onto a common comparable schema.

Fidelity can only be measured on features both datasets actually have. Sparkov and PaySim
carry different columns from each other and from Janus, so each is mapped onto a small shared
frame - amount, timing, category, counterparty type, fraud label - and every comparison is made
there.

The projection is deliberately narrow and stated up front. Claiming fidelity on a schema the
reference data does not contain would be unfalsifiable, so the honest move is to compare on the
intersection and say plainly what is outside it. What is *not* compared: graph structure,
device signals and session context, none of which exist in either public dataset. Those are
defended on construction rather than on measurement, and the report says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

#: The shared comparison schema. ``amount_z`` is the standardised log-amount.
COMMON_COLUMNS = ["amount", "hour", "day_of_week", "log_amount", "amount_z", "category", "is_fraud"]

#: Which Janus rails are comparable to which reference dataset. Comparing everything to
#: everything is what produced the first, misleading fidelity read: an India-weighted event mix
#: that is ~72% UPI was being scored against a US card-only dataset, so most of the measured
#: divergence was the rail mix and the currency, not the generator. Like is now compared to like.
COMPARABLE_COLUMNS: dict[str, list[str]] = {
    # Sparkov carries genuine wall-clock timestamps, so temporal marginals are evidence.
    "sparkov": ["amount_z", "log_amount", "hour", "day_of_week"],
    # PaySim's `step` is a simulation tick index, not a clock: its 743 steps have no diurnal
    # structure at all, so hour and weekday derived from it are artefacts. Measuring against
    # them produced a spurious 0.98 discriminator AUC in an earlier run. PaySim contributes
    # amount-distribution evidence only, and the report says so rather than quietly averaging
    # a meaningless number into the headline.
    "paysim": ["amount_z", "log_amount"],
}

RAIL_SEGMENTS: dict[str, list[str]] = {
    "sparkov": ["card_cnp", "card_cp", "card_token"],
    "paysim": ["upi_p2p", "upi_p2m", "imps", "neft", "rtgs", "upi_collect"],
}

#: Sparkov merchant categories mapped onto Janus's category vocabulary.
SPARKOV_CATEGORY_MAP = {
    "grocery_pos": "grocery", "grocery_net": "grocery",
    "gas_transport": "fuel",
    "food_dining": "restaurant",
    "shopping_pos": "ecommerce", "shopping_net": "ecommerce",
    "home": "ecommerce", "misc_pos": "ecommerce", "misc_net": "ecommerce",
    "health_fitness": "healthcare", "personal_care": "healthcare",
    "kids_pets": "ecommerce",
    "entertainment": "entertainment",
    "travel": "travel",
}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived amount columns.

    ``amount_z`` standardises log-amount *within* each dataset. That is the column the
    cross-economy comparison relies on: a US card portfolio and an Indian one have genuinely
    different absolute ticket sizes, and no FX or PPP constant makes that difference go away.
    What is legitimately comparable is the *shape* of the spend distribution - its spread and
    skew - which standardising isolates. Raw ``log_amount`` is still reported alongside, so the
    location difference is visible rather than quietly normalised away.
    """
    df = df[df.amount > 0].copy()
    df["log_amount"] = np.log10(df.amount)
    std = df["log_amount"].std()
    df["amount_z"] = (df["log_amount"] - df["log_amount"].mean()) / (std if std > 0 else 1.0)
    return df[COMMON_COLUMNS].reset_index(drop=True)


def project_janus(events: pd.DataFrame, *, segment: str | None = None) -> pd.DataFrame:
    """Project a Janus event frame onto the comparison schema.

    ``segment`` names a reference dataset and restricts events to the rails that dataset can
    fairly be compared against - see :data:`RAIL_SEGMENTS`.
    """
    if segment:
        events = events[events["rail"].isin(RAIL_SEGMENTS[segment])]
    ts = pd.to_datetime(events["ts"])
    return _finish(pd.DataFrame({
        "amount": events["amount"].astype(float),
        "hour": ts.dt.hour.astype(int),
        "day_of_week": ts.dt.dayofweek.astype(int),
        "category": events["merchant_category"].astype(str),
        "is_fraud": events["is_fraud"].astype(int),
    }))


def load_sparkov(path: Path | None = None, nrows: int | None = 300_000) -> pd.DataFrame:
    """Load Sparkov card transactions onto the comparison schema."""
    path = path or DATA_DIR / "sparkov.csv"
    df = pd.read_csv(
        path, nrows=nrows,
        usecols=["trans_date_trans_time", "category", "amt", "is_fraud"],
    )
    ts = pd.to_datetime(df["trans_date_trans_time"], format="%m/%d/%y %H:%M")
    return _finish(pd.DataFrame({
        # Converted to INR for readability only. The comparison that carries weight is on
        # ``amount_z``, which is invariant to this constant.
        "amount": df["amt"].astype(float) * 83.0,
        "hour": ts.dt.hour.astype(int),
        "day_of_week": ts.dt.dayofweek.astype(int),
        "category": df["category"].map(SPARKOV_CATEGORY_MAP).fillna("ecommerce"),
        "is_fraud": df["is_fraud"].astype(int),
    }))


def load_paysim(path: Path | None = None, nrows: int | None = 400_000) -> pd.DataFrame:
    """Load PaySim mobile-money transfers onto the comparison schema.

    PaySim has no wall-clock timestamp - ``step`` is an hour index from an arbitrary origin -
    so hour-of-day and weekday are derived from it. That makes its temporal columns weaker
    evidence than Sparkov's, and the report weights them accordingly.
    """
    path = path or DATA_DIR / "paysim.csv"
    df = pd.read_csv(path, nrows=nrows, usecols=["step", "type", "amount", "isFraud"])
    # Only push-type transfers are comparable to UPI P2P; cash-in and debit are different rails.
    df = df[df["type"].isin(["TRANSFER", "PAYMENT", "CASH_OUT"])]
    return _finish(pd.DataFrame({
        "amount": df["amount"].astype(float),
        "hour": (df["step"] % 24).astype(int),
        "day_of_week": ((df["step"] // 24) % 7).astype(int),
        "category": np.where(df["type"] == "PAYMENT", "ecommerce", "p2p"),
        "is_fraud": df["isFraud"].astype(int),
    }))


def available_references() -> dict[str, Path]:
    """Which reference datasets have actually been fetched."""
    return {
        name: DATA_DIR / f"{name}.csv"
        for name in ("sparkov", "paysim")
        if (DATA_DIR / f"{name}.csv").exists()
    }


def load_reference(name: str) -> pd.DataFrame:
    return {"sparkov": load_sparkov, "paysim": load_paysim}[name]()
