"""The Red agent: an evolutionary search for attack variants the live detector misses.

This is what makes the system a loop rather than a pipeline. The Red agent treats the deployed
detector as a black box - it can submit transactions and observe whether they would be flagged,
which is exactly the feedback a real operator gets from probe transactions - and searches the
space of *controllable attack attributes* for a region the detector scores as legitimate.

Fitness is deliberately **rupees that got through**, not evasion rate. Optimising evasion alone
produces a degenerate attacker who sends one-rupee transfers at midnight and evades everything
while earning nothing. A real operator trades stealth against yield, and the interesting
variants - the ones a defender should actually worry about - are the ones that keep the money
worth taking. That single choice of objective is what makes the search produce plausible
attacks rather than noise.

The search is gradient-free (the detector is a tree ensemble and the parameters feed a
stochastic simulator), so a small evolutionary algorithm is the right tool: tournament
selection, Gaussian mutation in the bounded parameter space, and uniform crossover.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from janus.defend.features import build_features
from janus.defend.supervised import TrainedDetector
from janus.generate.injectors.base import EvasionParams, InjectionContext, Injector
from janus.generate.rails import EVENT_COLUMNS
from janus.generate.world import World
from janus.seeding import derive_seed

PARAM_NAMES = [f.name for f in fields(EvasionParams) if f.name != "BOUNDS"]


@dataclass(slots=True)
class Candidate:
    params: EvasionParams
    value_through: float = 0.0
    evasion_rate: float = 0.0
    total_value: float = 0.0

    @property
    def fitness(self) -> float:
        return self.value_through


def random_params(rng: np.random.Generator) -> EvasionParams:
    kwargs = {
        name: float(rng.uniform(*EvasionParams.BOUNDS[name]))
        for name in PARAM_NAMES
        if name in EvasionParams.BOUNDS
    }
    return EvasionParams(**kwargs).clipped()


def mutate(params: EvasionParams, rng: np.random.Generator, scale: float = 0.18) -> EvasionParams:
    """Gaussian perturbation proportional to each parameter's own range."""
    kwargs = {}
    for name in PARAM_NAMES:
        lo, hi = EvasionParams.BOUNDS[name]
        current = getattr(params, name)
        kwargs[name] = float(current + rng.normal(0, (hi - lo) * scale))
    return EvasionParams(**kwargs).clipped()


def crossover(a: EvasionParams, b: EvasionParams, rng: np.random.Generator) -> EvasionParams:
    kwargs = {n: (getattr(a, n) if rng.random() < 0.5 else getattr(b, n)) for n in PARAM_NAMES}
    return EvasionParams(**kwargs).clipped()


class RedAgent:
    """Black-box evolutionary attacker.

    Holds a fixed legitimate base so that only the attack events are regenerated per candidate.
    Re-simulating the whole world for every fitness evaluation would be both slower and wrong -
    the attacker does not get to change the bank's customers, only their own behaviour.
    """

    def __init__(
        self,
        world: World,
        legit: pd.DataFrame,
        injectors: dict[str, Injector],
        *,
        seed: int = 0,
        population: int = 12,
    ):
        self.world = world
        self.legit = legit
        self.injectors = injectors
        self.rng = np.random.default_rng(seed)
        self.population_size = population
        self.population: list[Candidate] = [
            Candidate(params=random_params(self.rng)) for _ in range(population)
        ]
        # Seed one candidate with the baseline behaviour, so the search can never do worse
        # than the attacker's starting point.
        self.population[0] = Candidate(params=EvasionParams())

    def _simulate(self, params: EvasionParams, *, sample: int = 0) -> pd.DataFrame:
        """Generate attacks under one configuration.

        The per-injector seed is derived from the *parameters*, not from the agent's own RNG.
        That makes fitness a deterministic function of the candidate: re-evaluating the same
        configuration returns the same score, so elitism actually preserves the best individual
        rather than re-rolling it. An earlier version drew seeds from the shared RNG, which made
        fitness noisy enough that the search could discard a good candidate simply by measuring
        it twice.

        ``sample`` lets a caller draw a genuinely fresh instance of the same configuration -
        used when handing evasive events to Blue, so Blue learns the *pattern* rather than
        memorising the exact transactions Red was scored on.
        """
        frames = [self.legit]
        param_key = tuple(round(getattr(params, n), 6) for n in PARAM_NAMES)
        for card_id, injector in sorted(self.injectors.items()):
            rng = np.random.default_rng(derive_seed(card_id, param_key, sample))
            ctx = InjectionContext(
                world=self.world, legit=self.legit, rng=rng, intensity=1.0, evasion=params
            )
            df = injector.generate(ctx)
            if len(df):
                frames.append(df[EVENT_COLUMNS])
        events = pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable")
        return events.reset_index(drop=True)

    def score_candidate(
        self, candidate: Candidate, detector: TrainedDetector, threshold: float
    ) -> Candidate:
        """Run one attack configuration past the deployed detector and measure what got through."""
        events = self._simulate(candidate.params)
        X = build_features(events)
        scores = detector.rank_score(X)

        fraud = events["is_fraud"].to_numpy() == 1
        amounts = events["amount"].to_numpy()
        undetected = fraud & (scores < threshold)

        candidate.total_value = float(amounts[fraud].sum())
        candidate.value_through = float(amounts[undetected].sum())
        candidate.evasion_rate = float(undetected.sum() / max(fraud.sum(), 1))
        return candidate

    def evolve(
        self, detector: TrainedDetector, threshold: float, *, generations: int = 3
    ) -> Candidate:
        """Run the search and return the most profitable evasive configuration found."""
        for cand in self.population:
            self.score_candidate(cand, detector, threshold)

        for _ in range(generations):
            self.population.sort(key=lambda c: -c.fitness)
            # Elitism: the best two survive untouched, so the search cannot regress.
            survivors = self.population[: max(2, self.population_size // 3)]
            children: list[Candidate] = []
            while len(children) + len(survivors) < self.population_size:
                a, b = self.rng.choice(len(survivors), size=2, replace=len(survivors) < 2)
                child = mutate(
                    crossover(survivors[a].params, survivors[b].params, self.rng), self.rng
                )
                children.append(Candidate(params=child))
            for child in children:
                self.score_candidate(child, detector, threshold)
            self.population = survivors + children

        self.population.sort(key=lambda c: -c.fitness)
        return self.population[0]

    def best_attack_events(self, params: EvasionParams, *, sample: int = 1) -> pd.DataFrame:
        """Draw a fresh instance of a configuration, for handing to Blue."""
        return self._simulate(params, sample=sample)


def describe(params: EvasionParams, baseline: EvasionParams | None = None) -> str:
    """Human-readable summary of what the attacker learned to change."""
    baseline = baseline or EvasionParams()
    parts = []
    for name in PARAM_NAMES:
        now, was = getattr(params, name), getattr(baseline, name)
        if abs(now - was) < 0.05:
            continue
        direction = "up" if now > was else "down"
        parts.append(f"{name} {direction} ({was:.2f} -> {now:.2f})")
    return "; ".join(parts) if parts else "no material change from baseline"


def default_replace(params: EvasionParams, **kwargs: float) -> EvasionParams:
    return replace(params, **kwargs).clipped()
