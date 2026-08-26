"""Orchestrate a full simulation: legitimate economy plus every simulated attack.

The registry is driven by the atlas rather than by an import list, so coverage cannot silently
drift: adding a card with ``simulated: true`` and no injector fails atlas validation, and
removing an injector fails it too.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from janus.generate.injectors.base import EvasionParams, InjectionContext, Injector
from janus.generate.legit import generate_legit
from janus.generate.rails import EVENT_COLUMNS
from janus.generate.world import World, WorldConfig, build_world
from janus.identify.loader import simulated_cards
from janus.seeding import derive_seed


def load_injectors(only: list[str] | None = None) -> dict[str, Injector]:
    """Import every simulated card's injector, keyed by attack card id."""
    out: dict[str, Injector] = {}
    for card in simulated_cards():
        if only and card.id not in only:
            continue
        module = importlib.import_module(f"janus.generate.injectors.{card.injector}")
        injector = module.INJECTOR
        if injector.card_id != card.id:
            raise ValueError(
                f"injector {card.injector} declares card_id {injector.card_id!r} "
                f"but is referenced by {card.id}"
            )
        out[card.id] = injector
    return out


def _calibrate_fraud_rate(events: pd.DataFrame, target: float, seed: int) -> pd.DataFrame:
    """Drop whole attack campaigns until the fraud rate meets ``target``.

    Campaign-level rather than event-level sampling: removing individual events from a
    campaign would destroy the escalation sequences and fan-in structure that make the data
    worth training on. Proportional allocation across families preserves the relative mix.
    """
    fraud_mask = events.is_fraud == 1
    n_legit = int((~fraud_mask).sum())
    n_fraud = int(fraud_mask.sum())
    if n_fraud == 0:
        return events
    # Solve target = f / (legit + f) for the fraud-event budget.
    budget = target * n_legit / (1 - target)
    if n_fraud <= budget:
        return events

    rng = np.random.default_rng(seed + 99)
    keep_fraction = budget / n_fraud
    keep_campaigns: set[str] = set()
    for _, group in events[fraud_mask].groupby("attack_id"):
        campaigns = group.campaign_id.unique()
        # Floor of one campaign per family, always. Dropping a family entirely to hit a rate
        # target would silently destroy the coverage the atlas claims and break the
        # leave-one-attack-out folds, so on small worlds the floor wins and the achieved rate
        # comes in above target. `SimulationResult.calibration_achieved` reports when that
        # happens rather than leaving the discrepancy to be discovered later.
        n_keep = max(1, int(round(len(campaigns) * keep_fraction)))
        keep_campaigns.update(rng.choice(campaigns, size=n_keep, replace=False).tolist())

    # Rows belonging to a dropped campaign go entirely, including its non-fraud legs
    # (grooming credits, bust-out warm-up) so partial campaigns never leak into the data.
    in_kept = events.campaign_id.isin(keep_campaigns)
    return events[(events.campaign_id == "") | in_kept].reset_index(drop=True)


@dataclass(slots=True)
class SimulationResult:
    events: pd.DataFrame
    world: World
    fraud_rate: float
    per_attack: pd.DataFrame
    target_fraud_rate: float | None = None

    @property
    def calibration_achieved(self) -> bool:
        """Whether the requested fraud rate was actually reached.

        False means the one-campaign-per-family floor bound before the target did - typical on
        small worlds, where 21 families cannot fit inside a 0.4% budget.
        """
        if self.target_fraud_rate is None:
            return True
        return self.fraud_rate <= self.target_fraud_rate * 1.25

    def summary(self) -> str:
        n = len(self.events)
        f = int(self.events.is_fraud.sum())
        note = "" if self.calibration_achieved else " [rate floored by family coverage]"
        return (
            f"{n:,} events over {self.world.cfg.days} days | "
            f"{f:,} fraudulent ({self.fraud_rate:.3%}){note} | "
            f"{self.per_attack.shape[0]} attack families"
        )


def simulate(
    cfg: WorldConfig | None = None,
    *,
    intensity: float = 1.0,
    evasion: EvasionParams | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    target_fraud_rate: float | None = 0.006,
) -> SimulationResult:
    """Build a world, generate legitimate traffic, and inject every simulated attack.

    ``exclude`` powers leave-one-attack-out evaluation: the held-out family is generated
    separately for the test split, so the model has genuinely never seen it during training.

    ``target_fraud_rate`` calibrates the class balance. It matters more than it looks: attack
    volume is *absolute* - an operator runs a given number of campaigns regardless of how large
    the target institution is - while legitimate volume scales with the customer base. So the
    raw fraud rate out of the simulator is an artefact of the world's size, not a property of
    the attacks, and left uncalibrated a 10k-customer world yields ~1.7% fraud, which no real
    portfolio survives. Calibration drops whole *campaigns* rather than individual events, so
    the ones that remain keep their internal structure intact.

    The default of 0.6% is deliberately toward the high end of published payment-fraud rates:
    India's real-time push rails carry more fraud than card portfolios, and an inflated base
    rate is the conservative choice for a defender - it makes the precision numbers we report
    harder to achieve, not easier.
    """
    cfg = cfg or WorldConfig()
    world = build_world(cfg)
    legit = generate_legit(world)

    injectors = load_injectors(only)
    if exclude:
        injectors = {k: v for k, v in injectors.items() if k not in exclude}

    frames = [legit]
    for card_id, injector in sorted(injectors.items()):
        # Each injector gets its own derived seed so adding or removing one family does not
        # perturb the others - essential for LOAO folds to be comparable.
        # derive_seed, not hash(): builtin hash is per-process randomised for strings, which
        # made the whole simulation irreproducible across runs.
        rng = np.random.default_rng(derive_seed(cfg.seed, card_id))
        ctx = InjectionContext(
            world=world, legit=legit, rng=rng,
            intensity=intensity, evasion=(evasion or EvasionParams()).clipped(),
        )
        df = injector.generate(ctx)
        if len(df):
            frames.append(df[EVENT_COLUMNS])

    events = pd.concat(frames, ignore_index=True)
    if target_fraud_rate is not None:
        events = _calibrate_fraud_rate(events, target_fraud_rate, cfg.seed)
    events = events.sort_values("ts", ignore_index=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)

    fraud = events[events.is_fraud == 1]
    per_attack = (
        fraud.groupby("attack_id")
        .agg(events=("event_id", "size"),
             campaigns=("campaign_id", "nunique"),
             median_amount=("amount", "median"),
             total_value=("amount", "sum"))
        .sort_values("total_value", ascending=False)
    )
    return SimulationResult(
        events=events,
        world=world,
        fraud_rate=float(events.is_fraud.mean()),
        per_attack=per_attack,
        target_fraud_rate=target_fraud_rate,
    )
