"""The Ideator: proposes attack vectors the atlas does not yet contain.

The brief asks for identification to be *thorough and exhaustive*. A hand-written list can
never claim that, so the Ideator treats the attack space as a combinatorial product

    rail x genai_enabler x surface x monetization

and searches the ~10^4 cells that product spans for combinations that are (a) plausible given
how the existing atlas behaves, and (b) not already covered.

Plausibility is learned from the atlas itself rather than asserted. Every card is evidence
that certain rails, enablers, surfaces and monetisation paths co-occur in workable attacks;
a candidate whose components never co-occur is probably incoherent, while one whose components
each co-occur strongly but which has never been combined as a whole is exactly what we want.

This runs fully offline and deterministically. When an LLM adapter is configured it is used
only to write prose for the top-ranked candidates - never to decide the ranking, so the output
is reproducible with or without a key.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from janus.identify.loader import atlas
from janus.identify.schema import AttackCard, GenAIEnabler, Monetization, Rail, Surface

AXES = ("rails", "genai_enablers", "surfaces", "monetization")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed (rail, enabler, surface, monetization) attack vector."""

    rail: Rail
    enabler: GenAIEnabler
    surface: Surface
    monetization: Monetization

    plausibility: float
    extrapolation: float
    impact: float
    nearest_card: str
    nearest_similarity: float

    @property
    def priority(self) -> float:
        """Ranking score: coherence x expected loss.

        Deliberately *not* multiplied by ``extrapolation``. An earlier version was, and it
        ranked the atlas's least coherent corners highest - because extrapolation and
        plausibility are anti-correlated by construction, so their product peaks on
        incoherent middle ground. The vectors a defender actually needs are the ones that
        are highly plausible and simply have not been assembled yet.
        """
        return round(self.plausibility * self.impact, 4)

    @property
    def tier(self) -> str:
        """Recombinations reuse only attested pairs; extrapolations reach past the atlas.

        Both are useful and they are useful for different reasons - a recombination is a
        near-term gap in coverage, an extrapolation is a research question - so they are
        labelled rather than blended into one score.
        """
        return "recombination" if self.extrapolation == 0 else "extrapolation"

    def title(self) -> str:
        return (
            f"{self.enabler.value.replace('_', ' ').title()} on "
            f"{self.rail.value.replace('_', ' ').upper()} via "
            f"{self.surface.value.replace('_', ' ')} -> "
            f"{self.monetization.value.replace('_', ' ')}"
        )


@dataclass
class _AtlasStats:
    """Empirical co-occurrence statistics derived from the atlas."""

    axis_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    pair_counts: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    severity_by_value: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    n_cards: int = 0

    def compatibility(self, axis_a: str, val_a: str, axis_b: str, val_b: str) -> float:
        """How strongly two axis values co-occur in real attacks, in [0, 1].

        A smoothed Ochiai (cosine) coefficient over card counts, deliberately *not* PMI:
        PMI rewards rarity, which made the first version of this ranker surface nonsense
        like "localisation on RTGS via QR sticker" - four values that are individually rare
        and have never co-occurred, which PMI reads as a surprising and therefore interesting
        association. Co-occurrence strength is the property we actually want.

        The support weight then discounts values attested in only one or two cards, so a
        single card cannot manufacture a confident-looking association on its own.
        """
        c_ab = self.pair_counts[(axis_a, axis_b)][(val_a, val_b)]
        c_a = self.axis_counts[axis_a][val_a]
        c_b = self.axis_counts[axis_b][val_b]
        ochiai = math.sqrt(((c_ab + 0.5) / (c_a + 1)) * ((c_ab + 0.5) / (c_b + 1)))
        support = min(1.0, math.sqrt(min(c_a, c_b) / 3)) if min(c_a, c_b) else 0.0
        return ochiai * support

    def co_attested(self, axis_a: str, val_a: str, axis_b: str, val_b: str) -> bool:
        """Whether any single existing card already pairs these two values."""
        return self.pair_counts[(axis_a, axis_b)][(val_a, val_b)] > 0


def _build_stats(cards: tuple[AttackCard, ...]) -> _AtlasStats:
    stats = _AtlasStats(n_cards=len(cards))
    for card in cards:
        values = {axis: [str(v) for v in getattr(card, axis)] for axis in AXES}
        for axis, vals in values.items():
            for v in set(vals):
                stats.axis_counts[axis][v] += 1
                stats.severity_by_value[v].append(card.severity)
        for axis_a, axis_b in itertools.combinations(AXES, 2):
            for va, vb in itertools.product(set(values[axis_a]), set(values[axis_b])):
                stats.pair_counts[(axis_a, axis_b)][(va, vb)] += 1
    return stats


def _coverage_tuples(cards: tuple[AttackCard, ...]) -> dict[tuple[str, ...], str]:
    """Every concrete axis-tuple the atlas already realises, mapped to its card id."""
    covered: dict[tuple[str, ...], str] = {}
    for card in cards:
        for combo in itertools.product(
            card.rails, card.genai_enablers, card.surfaces or [Surface.WEB], card.monetization
        ):
            covered.setdefault(tuple(str(c) for c in combo), card.id)
    return covered


def _impact(stats: _AtlasStats, rail: Rail, monetization: Monetization) -> float:
    """Expected loss magnitude, from observed severities of cards sharing these values."""
    observed = stats.severity_by_value[str(rail)] + stats.severity_by_value[str(monetization)]
    return (sum(observed) / len(observed)) / 5 if observed else 0.5


def propose(
    top_n: int = 20,
    *,
    min_plausibility: float = 0.0,
    min_extrapolation: float = 0.0,
) -> list[Candidate]:
    """Search the attack product space and return the highest-priority uncovered vectors.

    Raise ``min_extrapolation`` to ask specifically for vectors that reach beyond the
    combinations the atlas already attests - the research-question end of the output.
    """
    cards = atlas()
    stats = _build_stats(cards)
    covered = _coverage_tuples(cards)

    # Similarity of a candidate to the closest covered tuple: how many of the four axis
    # values it shares. A candidate differing in one axis is a variant; three, a new vector.
    covered_sets = [(set(t), cid) for t, cid in covered.items()]

    candidates: list[Candidate] = []
    for rail, enabler, surface, monet in itertools.product(
        Rail, GenAIEnabler, Surface, Monetization
    ):
        key = (str(rail), str(enabler), str(surface), str(monet))
        if key in covered:
            continue

        vals = {"rails": str(rail), "genai_enablers": str(enabler),
                "surfaces": str(surface), "monetization": str(monet)}
        pairs = list(itertools.combinations(AXES, 2))
        scores = [stats.compatibility(a, vals[a], b, vals[b]) for a, b in pairs]
        plausibility = math.prod(scores) ** (1 / len(scores))
        if plausibility < min_plausibility:
            continue

        # Measured at the pair level rather than the tuple level. Every candidate here is an
        # unrealised 4-tuple, so tuple-level distance is near-constant and cannot rank; how
        # many of its six constituent pairs are already attested does discriminate.
        attested = sum(stats.co_attested(a, vals[a], b, vals[b]) for a, b in pairs)
        extrapolation = 1.0 - attested / len(pairs)

        key_set = set(key)
        best_sim, best_card = 0.0, ""
        for cov_set, cid in covered_sets:
            sim = len(key_set & cov_set) / 4
            if sim > best_sim:
                best_sim, best_card = sim, cid

        candidates.append(
            Candidate(
                rail=rail, enabler=enabler, surface=surface, monetization=monet,
                plausibility=round(plausibility, 4),
                extrapolation=round(extrapolation, 4),
                impact=round(_impact(stats, rail, monet), 4),
                nearest_card=best_card,
                nearest_similarity=round(best_sim, 4),
            )
        )

    candidates = [c for c in candidates if c.extrapolation >= min_extrapolation]
    candidates.sort(key=lambda c: (-c.priority, c.rail.value, c.enabler.value))
    return candidates[:top_n]


def search_space_size() -> int:
    return len(Rail) * len(GenAIEnabler) * len(Surface) * len(Monetization)


def coverage_ratio() -> float:
    """Fraction of the product space the atlas explicitly realises.

    Reported honestly: it is small, because most of the product space is incoherent
    (you cannot run a QR-sticker attack over RTGS). The number is useful as a denominator
    for the Ideator's search, not as a scorecard.
    """
    return len(_coverage_tuples(atlas())) / search_space_size()
