"""Shared machinery for attack injectors.

An injector takes the world and its legitimate event stream and returns additional events that
realise one attack card. Injectors *manipulate the existing world* - they reuse real customers,
real devices, real merchants - rather than inventing a parallel population, because fraud that
arrives on unfamiliar entities is trivially separable and teaches the detector nothing.

Every injector exposes the same :class:`EvasionParams` knobs. Those knobs are the attacker's
genuine degrees of freedom - how much to take, how fast, through how aged a mule, how much to
prefer an existing relationship over a fresh one - and they are exactly what the Red agent in
``janus.loop`` searches over when it looks for variants the deployed detector misses. Keeping
them in one place is what makes the adversarial loop possible without rewriting 21 injectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from janus.generate.rails import EVENT_COLUMNS, MCC_PROFILES, AuthMethod, Channel, Decision
from janus.generate.world import CATEGORY_NAMES, CITY_NAMES, World

SECOND = np.timedelta64(1, "s")
MINUTE = np.timedelta64(60, "s")
HOUR = np.timedelta64(3600, "s")
DAY = np.timedelta64(86400, "s")


@dataclass(frozen=True, slots=True)
class EvasionParams:
    """The attacker's controllable attributes.

    Bounds are deliberately conservative: an attacker who scales amounts to 0.05x evades every
    amount-based rule but also destroys their own economics. The Red agent optimises expected
    *value*, not evasion alone, so these ranges describe what a rational operator would try.
    """

    amount_scale: float = 1.0      # multiplier on attack ticket sizes
    delay_scale: float = 1.0       # multiplier on inter-event delays (slower = stealthier)
    burst_scale: float = 1.0       # multiplier on events per campaign
    mule_age_pref: float = 0.0     # 0 = any mule, 1 = strongly prefer aged accounts
    novelty_avoid: float = 0.0     # probability of routing via an existing contact
    hour_blend: float = 0.0        # 0 = attack's natural hour, 1 = victim's normal hour

    BOUNDS = {
        "amount_scale": (0.15, 2.5),
        "delay_scale": (0.25, 12.0),
        "burst_scale": (0.2, 2.0),
        "mule_age_pref": (0.0, 1.0),
        "novelty_avoid": (0.0, 0.9),
        "hour_blend": (0.0, 1.0),
    }

    def clipped(self) -> EvasionParams:
        return replace(self, **{k: float(np.clip(getattr(self, k), *b)) for k, b in self.BOUNDS.items()})

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class InjectionContext:
    """Everything an injector needs to place an attack inside the world."""

    world: World
    legit: pd.DataFrame
    rng: np.random.Generator
    intensity: float = 1.0
    evasion: EvasionParams = EvasionParams()

    def campaigns(self, base: int) -> int:
        """How many independent campaigns to run, after intensity and burst scaling."""
        n = base * self.intensity * self.evasion.burst_scale
        return max(1, int(round(n)))

    def pick_mules(self, n: int) -> np.ndarray:
        """Sample mule accounts, optionally biased toward aged ones.

        Mule ageing is the single most effective evasion in the whole system: tenure features
        are heavily weighted by any realistic model, so an operator who can source older
        accounts buys a large drop in risk score for a purely logistical cost.
        """
        mules = self.world.mules()
        if len(mules) == 0:
            mules = np.arange(min(50, self.world.n_customers))
        ages = self.world.cust_account_age[mules]
        w = np.ones(len(mules)) if self.evasion.mule_age_pref <= 0 else (
            (ages / ages.max()) ** (3.0 * self.evasion.mule_age_pref)
        )
        w = w / w.sum()
        return self.rng.choice(mules, size=n, replace=len(mules) < n, p=w)

    def attack_hour(self, victims: np.ndarray, natural_hour: float) -> np.ndarray:
        """Blend the attack's natural hour toward each victim's own routine.

        ``hour_blend`` lets the Red agent trade attack convenience for temporal camouflage.
        """
        personal = 11.2 + self.world.cust_circadian[victims] + self.rng.normal(0, 2.4, len(victims))
        blended = (1 - self.evasion.hour_blend) * natural_hour + self.evasion.hour_blend * personal
        return np.mod(blended + self.rng.normal(0, 0.8, len(victims)), 24.0)


def blank_events(n: int) -> dict[str, np.ndarray | list]:
    """A dict of event columns pre-filled with neutral defaults.

    Injectors override only the fields their attack actually changes, so each one reads as a
    description of the attack rather than a wall of column assignments.
    """
    return {
        "event_id": np.zeros(n, dtype=np.int64),
        "ts": np.full(n, np.datetime64("2026-01-01T00:00:00")),
        "rail": ["upi_p2p"] * n,
        "channel": [str(Channel.MOBILE_APP)] * n,
        "payer_id": np.zeros(n, dtype=np.int64),
        "payer_account": np.zeros(n, dtype=np.int64),
        "payee_id": np.zeros(n, dtype=np.int64),
        "payee_handle": [""] * n,
        "payee_type": ["person"] * n,
        "amount": np.zeros(n, dtype=np.float64),
        "currency": ["INR"] * n,
        "mcc": np.zeros(n, dtype=np.int64),
        "merchant_category": ["p2p"] * n,
        "device_id": np.zeros(n, dtype=np.int64),
        "ip_prefix": np.zeros(n, dtype=np.int64),
        "payer_city": [""] * n,
        "payee_city": [""] * n,
        "auth_method": [str(AuthMethod.UPI_PIN)] * n,
        "step_up_required": np.zeros(n, dtype=np.int8),
        "decision": [str(Decision.APPROVED)] * n,
        "payment_reference": [""] * n,
        "inbound_call_active": np.zeros(n, dtype=np.int8),
        "screen_share_active": np.zeros(n, dtype=np.int8),
        "agent_initiated": np.zeros(n, dtype=np.int8),
        "payee_account_age_days": np.zeros(n, dtype=np.float32),
        "device_age_days": np.zeros(n, dtype=np.float32),
        "is_fraud": np.ones(n, dtype=np.int8),
        "attack_id": [""] * n,
        "campaign_id": [""] * n,
    }


def finalise(cols: dict, card_id: str) -> pd.DataFrame:
    """Assemble an injector's column dict into a validated event frame."""
    df = pd.DataFrame(cols)
    df["attack_id"] = card_id
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{card_id} injector produced events missing columns: {sorted(missing)}")
    return df[EVENT_COLUMNS]


def victim_context(ctx: InjectionContext, victims: np.ndarray) -> dict[str, np.ndarray]:
    """Device, city and IP for a set of victims - attacks ride the victim's own footprint."""
    w = ctx.world
    return {
        "device_id": w.cust_device[victims],
        "ip_prefix": w.cust_ip_prefix[victims],
        "payer_city": CITY_NAMES[w.cust_city[victims]],
        "device_age_days": w.cust_device_age[victims].astype(np.float32),
    }


def place_in_window(
    ctx: InjectionContext, victims: np.ndarray, hour: np.ndarray
) -> np.ndarray:
    """Convert per-victim hours into timestamps somewhere inside the simulation window."""
    day = ctx.rng.integers(0, max(1, ctx.world.cfg.days), size=len(victims))
    secs = (day * 86400 + hour * 3600 + ctx.rng.random(len(victims)) * 60).astype(np.int64)
    return np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")


def merchant_context(ctx: InjectionContext, merch: np.ndarray) -> dict[str, np.ndarray | list]:
    """Payee-side columns for merchant-directed attacks."""
    w = ctx.world
    categories = CATEGORY_NAMES[w.merch_category[merch]]
    return {
        "payee_id": merch + 1_000_000,
        "payee_handle": [f"merchant{int(x)}" for x in merch],
        "payee_type": ["merchant"] * len(merch),
        "merchant_category": categories.tolist(),
        "mcc": np.array([MCC_PROFILES[c]["mcc"] for c in categories], dtype=np.int64),
        "payee_account_age_days": w.merch_age[merch].astype(np.float32),
        "payee_city": CITY_NAMES[w.merch_city[merch]].tolist(),
    }


def merchants_in(world: World, categories: set[str]) -> np.ndarray:
    """Merchant ids whose category is in ``categories``, or all merchants if none match."""
    wanted = [i for i, c in enumerate(CATEGORY_NAMES) if c in categories]
    pool = np.flatnonzero(np.isin(world.merch_category, wanted))
    return pool if len(pool) else np.arange(world.n_merchants)


def attacker_device(ctx: InjectionContext, n: int) -> dict[str, np.ndarray]:
    """Device/IP columns for an attack executed from the attacker's own hardware.

    Critically, the IP space and the device-age distribution OVERLAP legitimate traffic. An
    earlier version drew attacker IPs from a disjoint range and attacker device ages from
    0-4 days against a legitimate floor of 5, which handed the classifier two artefacts that
    separated fraud perfectly and had nothing to do with fraud. Attackers buy connectivity and
    handsets from the same market everyone else does; what differs is that their devices skew
    *newer*, not that they occupy a different universe.
    """
    ages = ctx.rng.gamma(shape=1.4, scale=16.0, size=n)
    return {
        "device_id": (ctx.rng.integers(0, 20_000, n) + 900_000).astype(np.int64),
        "device_age_days": np.clip(ages, 0, 400).astype(np.float32),
        "ip_prefix": ctx.rng.integers(0, 4096, n).astype(np.int64),
    }


class Injector(ABC):
    """One attack card, realised as events inside the world."""

    card_id: str
    #: Rough number of independent campaigns at intensity 1.0.
    base_campaigns: int = 30

    @abstractmethod
    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        """Produce fraudulent events for this attack."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.card_id}>"
