"""Atlas integrity tests.

These guard the claims the README and the walkthrough make about coverage. If a card is
malformed, duplicated, or claims a simulation it cannot deliver, this fails loudly rather
than letting the submission overstate itself.
"""

from __future__ import annotations

from collections import Counter

import pytest

from janus.identify.ideator import propose
from janus.identify.loader import atlas, coverage, load_cards
from janus.identify.matrix import genai_intensity, matrix
from janus.identify.schema import GenAIEnabler, Rail
from janus.identify.signals import OBSERVABLE_SIGNALS, SignalFamily


def test_every_card_validates():
    cards = load_cards(check_injectors=False)
    assert len(cards) >= 40, "atlas should stay broad; breadth is an explicit judging criterion"


def test_card_ids_unique():
    ids = [c.id for c in atlas()]
    assert len(ids) == len(set(ids))


def test_simulated_cards_declare_an_injector():
    for card in atlas():
        assert card.simulated == (card.injector is not None), card.id


def test_every_rail_and_enabler_is_covered():
    """A gap on either axis means the atlas has a blind spot worth knowing about."""
    cov = coverage()
    uncovered_rails = [r for r, n in cov["rails"].items() if n == 0]
    uncovered_enablers = [e for e, n in cov["genai_enablers"].items() if n == 0]
    assert not uncovered_rails, f"rails with no attack card: {uncovered_rails}"
    assert not uncovered_enablers, f"enablers with no attack card: {uncovered_enablers}"
    assert len(cov["rails"]) == len(Rail)
    assert len(cov["genai_enablers"]) == len(GenAIEnabler)


def test_kill_chains_are_substantive():
    for card in atlas():
        assert len({s.phase for s in card.kill_chain}) >= 2, card.id
        assert any(s.genai_used for s in card.kill_chain), (
            f"{card.id} claims a GenAI enabler but no kill-chain step uses it"
        )


def test_every_observable_is_registered_to_a_signal_family():
    """No card may describe a signal that no feature family owns."""
    for card in atlas():
        for obs in card.observables:
            assert obs in OBSERVABLE_SIGNALS, f"{card.id}: unregistered observable {obs!r}"


def test_signal_families_are_shared_vocabulary():
    """Signal families must recur across cards, or the defence cannot generalise.

    Observable *names* are deliberately specific and mostly card-unique - that is what makes
    them informative. The families beneath them are what features implement, and those must
    be heavily reused, otherwise every attack would need its own bespoke detector and the
    leave-one-attack-out evaluation could never generalise to an unseen family.
    """
    counts = Counter(f for card in atlas() for f in card.signal_families)
    shared = sum(1 for n in counts.values() if n > 1)
    assert shared / len(counts) > 0.8, f"only {shared}/{len(counts)} families are reused"
    assert counts.most_common(1)[0][1] >= 10


def test_no_orphan_signal_families():
    """Every family in the registry must be reachable from at least one attack card."""
    used = {f for card in atlas() for f in card.signal_families}
    orphans = set(SignalFamily) - used
    assert not orphans, f"families no card motivates: {orphans}"


def test_matrix_spans_the_kill_chain():
    grid = matrix()
    populated = [p for p, cards in grid.items() if cards]
    assert len(populated) >= 8, f"kill-chain coverage is thin: {populated}"


def test_genai_concentrates_in_the_front_half():
    """The atlas's central empirical claim, asserted as a test.

    GenAI should materially enable preparation and persuasion far more than it enables the
    mechanics of moving money. If this ever flips, the walkthrough's thesis is wrong and
    the failure should be visible.
    """
    intensity = genai_intensity()
    front = max(intensity["resource_dev"], intensity["pretext"], intensity["trust_build"])
    back = max(intensity["monetize"], intensity["launder"])
    assert front > 0.7 > back


@pytest.mark.parametrize("extrapolate", [False, True])
def test_ideator_returns_ranked_uncovered_vectors(extrapolate):
    cands = propose(10, min_extrapolation=0.5 if extrapolate else 0.0)
    assert len(cands) == 10
    assert cands == sorted(cands, key=lambda c: (-c.priority, c.rail.value, c.enabler.value))
    assert all(0 <= c.plausibility <= 1 for c in cands)
    if extrapolate:
        assert all(c.tier == "extrapolation" for c in cands)


def test_ideator_is_deterministic():
    """Reproducibility is a submission requirement; the ranker must not drift between runs."""
    assert [c.title() for c in propose(20)] == [c.title() for c in propose(20)]
