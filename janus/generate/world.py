"""The synthetic payment economy that attacks are injected into.

Fidelity comes from generating fraud *inside* a coherent world rather than sampling it from a
separate distribution. If legitimate behaviour has no structure - no regular merchants, no
social graph, no personal rhythm - then "first-time beneficiary" and "out of pattern hour"
are not signals at all, and a detector trained on that data learns nothing transferable.

So the world carries the structure the atlas's observables depend on:

* a **contact graph**, built by preferential attachment, so P2P has a notion of "known payee"
* **merchant affinity** per customer, so card spend has a notion of "never seen merchant"
* per-customer **circadian phase**, so "out of pattern hour" is personal rather than global
* **account, device and beneficiary ages**, so tenure and novelty features have real content

Everything is derived from a single seed and stored as flat numpy arrays in CSR-style layout,
which keeps generation vectorised and the whole world reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from janus.generate.rails import MCC_PROFILES

#: Indian cities with rough population weights, used for geo assignment and geo-anomaly features.
CITIES: list[tuple[str, float]] = [
    ("Mumbai", 12.5), ("Delhi", 11.0), ("Bengaluru", 8.4), ("Hyderabad", 6.8),
    ("Ahmedabad", 5.6), ("Chennai", 5.0), ("Kolkata", 4.5), ("Surat", 4.5),
    ("Pune", 3.1), ("Jaipur", 3.0), ("Lucknow", 2.8), ("Kanpur", 2.7),
    ("Nagpur", 2.4), ("Indore", 2.1), ("Bhopal", 1.8), ("Patna", 1.7),
    ("Vadodara", 1.7), ("Ludhiana", 1.6), ("Agra", 1.5), ("Nashik", 1.5),
    ("Kochi", 1.4), ("Coimbatore", 1.4), ("Guwahati", 1.1), ("Chandigarh", 1.0),
]

AGE_BANDS = ["18-25", "25-35", "35-45", "45-55", "55-70", "70+"]
AGE_WEIGHTS = np.array([0.18, 0.28, 0.22, 0.15, 0.12, 0.05])

TENURE_LEVELS = ["low", "medium", "high"]
#: Digital tenure conditioned on age. Older cohorts skew toward low tenure, which is what makes
#: them disproportionately targeted by the social-engineering families in the atlas.
TENURE_BY_AGE = np.array([
    [0.05, 0.25, 0.70],   # 18-25
    [0.08, 0.32, 0.60],   # 25-35
    [0.15, 0.45, 0.40],   # 35-45
    [0.30, 0.45, 0.25],   # 45-55
    [0.50, 0.35, 0.15],   # 55-70
    [0.70, 0.25, 0.05],   # 70+
])

INCOME_MULTIPLIER = np.array([0.45, 0.7, 1.0, 1.6, 2.8, 5.0])
INCOME_WEIGHTS = np.array([0.20, 0.26, 0.24, 0.17, 0.09, 0.04])


@dataclass(slots=True)
class WorldConfig:
    n_customers: int = 10_000
    n_merchants: int = 2_000
    days: int = 60
    seed: int = 42
    #: Fraction of customers whose accounts are available to laundering operations. Mule supply
    #: is a real constraint on attacker throughput, so it is modelled explicitly.
    mule_fraction: float = 0.012
    #: Mean legitimate transactions per customer per day.
    base_rate: float = 1.1


@dataclass(slots=True)
class World:
    """Entity tables for one synthetic economy. All arrays are indexed by entity id."""

    cfg: WorldConfig

    # --- customers -----------------------------------------------------------------
    cust_age_band: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_tenure: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_income_mult: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_city: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_circadian: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_account_age: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_device: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_device_age: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_ip_prefix: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_is_mule: np.ndarray = field(default_factory=lambda: np.array([]))
    cust_card_share: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- contact graph (CSR) -------------------------------------------------------
    contacts_flat: np.ndarray = field(default_factory=lambda: np.array([]))
    contacts_offset: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- merchant affinity (CSR) ---------------------------------------------------
    affinity_flat: np.ndarray = field(default_factory=lambda: np.array([]))
    affinity_offset: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- merchants -----------------------------------------------------------------
    merch_category: np.ndarray = field(default_factory=lambda: np.array([]))
    merch_city: np.ndarray = field(default_factory=lambda: np.array([]))
    merch_age: np.ndarray = field(default_factory=lambda: np.array([]))
    merch_popularity: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_customers(self) -> int:
        return len(self.cust_rate)

    @property
    def n_merchants(self) -> int:
        return len(self.merch_age)

    def contacts_of(self, cid: int) -> np.ndarray:
        return self.contacts_flat[self.contacts_offset[cid] : self.contacts_offset[cid + 1]]

    def merchants_of(self, cid: int) -> np.ndarray:
        return self.affinity_flat[self.affinity_offset[cid] : self.affinity_offset[cid + 1]]

    def vpa(self, cid: int) -> str:
        """Stable VPA handle for a customer. Deterministic so events are reproducible."""
        return f"user{cid:06d}@januspsp"

    def mules(self) -> np.ndarray:
        return np.flatnonzero(self.cust_is_mule)

    def victims_matching(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        age_bands: list[str] | None = None,
        tenure: str | None = None,
    ) -> np.ndarray:
        """Sample victims fitting an attack card's victim profile.

        Targeting is part of attack fidelity: a distress-call scam that lands uniformly across
        the customer base produces a detector that learns nothing about who is actually at risk.
        Falls back to the full population when a profile is too narrow to fill the sample.
        """
        mask = ~self.cust_is_mule.astype(bool)
        if age_bands:
            wanted = {AGE_BANDS.index(a) for a in age_bands if a in AGE_BANDS}
            mask &= np.isin(self.cust_age_band, list(wanted))
        if tenure:
            mask &= self.cust_tenure == TENURE_LEVELS.index(tenure)
        pool = np.flatnonzero(mask)
        if len(pool) < n:
            pool = np.flatnonzero(~self.cust_is_mule.astype(bool))
        return rng.choice(pool, size=min(n, len(pool)), replace=False)


def _preferential_attachment(
    rng: np.random.Generator, n_nodes: int, sizes: np.ndarray, n_targets: int, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a CSR adjacency where target choice is weighted by ``weights``.

    Preferential attachment matters here: it produces the heavy-tailed degree distribution real
    payment graphs have, which is what makes graph features like fan-in discriminative. Uniform
    attachment would make every account look equally central and the mule signal would vanish.
    """
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    p = weights / weights.sum()
    flat = rng.choice(n_targets, size=int(offsets[-1]), p=p)
    return flat, offsets


def build_world(cfg: WorldConfig | None = None) -> World:
    """Construct a fully specified synthetic economy from a seed."""
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    n, m = cfg.n_customers, cfg.n_merchants

    city_p = np.array([w for _, w in CITIES])
    city_p = city_p / city_p.sum()

    age_band = rng.choice(len(AGE_BANDS), size=n, p=AGE_WEIGHTS)
    tenure = np.array([rng.choice(3, p=TENURE_BY_AGE[a]) for a in age_band])
    income = rng.choice(len(INCOME_MULTIPLIER), size=n, p=INCOME_WEIGHTS)

    # Activity rate: gamma gives the long right tail real portfolios have - a small number of
    # very active customers alongside a large dormant-ish majority.
    rate = rng.gamma(shape=2.0, scale=cfg.base_rate / 2.0, size=n)
    rate *= 1.0 + 0.35 * (tenure - 1)
    rate = np.clip(rate, 0.05, 20.0)

    # Personal circadian phase in hours, so "unusual hour" is defined per customer.
    circadian = rng.normal(0.0, 1.6, size=n)

    account_age = rng.integers(30, 4200, size=n).astype(np.float64)
    # Tenure correlates with account age; a high-tenure user with a 40-day-old account is rare.
    account_age = np.clip(account_age * (0.55 + 0.35 * tenure), 20, 5000)

    device = np.arange(n)
    device_age = np.clip(account_age * rng.uniform(0.15, 0.95, size=n), 0, 2500)
    # Device churn. Without this, legitimate device ages had a hard floor at 5 days while
    # attacker devices were 0-4 days old, which made `device_age_days < 5` a *perfect* fraud
    # oracle - 1,570 fraud events and zero legitimate ones - and drove the reported ROC-AUC to
    # 0.9992 on an artefact. Roughly 6% of a real customer base is on a handset less than six
    # weeks old at any moment, so the overlap has to be there.
    churned = rng.random(n) < 0.06
    device_age[churned] = rng.uniform(0, 45, size=int(churned.sum()))
    ip_prefix = rng.integers(0, 4096, size=n)

    is_mule = np.zeros(n, dtype=np.int8)
    n_mules = max(1, int(cfg.mule_fraction * n))
    # Mule accounts skew young and low-tenure: that is who gets recruited.
    recruit_w = 1.0 / (1.0 + account_age / 365.0)
    recruit_w /= recruit_w.sum()
    is_mule[rng.choice(n, size=n_mules, replace=False, p=recruit_w)] = 1

    # Card usage share rises with income and tenure; UPI dominates the low end.
    card_share = np.clip(0.05 + 0.06 * income + 0.05 * tenure + rng.normal(0, 0.04, n), 0.01, 0.75)

    # --- merchants ---------------------------------------------------------------
    cats = list(MCC_PROFILES)
    merch_category = rng.integers(0, len(cats), size=m)
    merch_city = rng.choice(len(CITIES), size=m, p=city_p)
    merch_age = rng.gamma(shape=1.6, scale=520, size=m).clip(1, 5000)
    # Zipf popularity: a handful of merchants take most of the volume.
    ranks = np.arange(1, m + 1)
    merch_popularity = 1.0 / ranks**0.85
    rng.shuffle(merch_popularity)

    # --- contact graph -----------------------------------------------------------
    n_contacts = rng.integers(3, 18, size=n)
    # Attachment weight by activity: busy people are in more people's contact lists.
    flat_c, off_c = _preferential_attachment(rng, n, n_contacts, n, rate)

    # --- merchant affinity -------------------------------------------------------
    n_affinity = rng.integers(4, 26, size=n)
    flat_a, off_a = _preferential_attachment(rng, n, n_affinity, m, merch_popularity)

    return World(
        cfg=cfg,
        cust_age_band=age_band, cust_tenure=tenure,
        cust_income_mult=INCOME_MULTIPLIER[income],
        cust_city=rng.choice(len(CITIES), size=n, p=city_p),
        cust_rate=rate, cust_circadian=circadian,
        cust_account_age=account_age, cust_device=device, cust_device_age=device_age,
        cust_ip_prefix=ip_prefix, cust_is_mule=is_mule, cust_card_share=card_share,
        contacts_flat=flat_c, contacts_offset=off_c,
        affinity_flat=flat_a, affinity_offset=off_a,
        merch_category=merch_category, merch_city=merch_city,
        merch_age=merch_age, merch_popularity=merch_popularity,
    )


CITY_NAMES = np.array([c for c, _ in CITIES])
CATEGORY_NAMES = np.array(list(MCC_PROFILES))
