"""Render the atlas as an ATT&CK-style matrix.

Columns are kill-chain phases, cells are the techniques that operate in that phase. The same
structure serves the terminal (`janus atlas matrix`) and the web Atlas view, so the CLI and
the dashboard can never drift apart.
"""

from __future__ import annotations

from janus.identify.loader import atlas
from janus.identify.schema import PHASE_ORDER, AttackCard, Phase


def matrix() -> dict[str, list[dict]]:
    """Map each kill-chain phase to the techniques that operate in it.

    A card appears in every phase its kill chain touches, so column counts sum to more than
    the card total - the same property the real ATT&CK matrix has.
    """
    grid: dict[str, list[dict]] = {str(p): [] for p in PHASE_ORDER}
    for card in sorted(atlas(), key=lambda c: c.id):
        for phase in {s.phase for s in card.kill_chain}:
            grid[str(phase)].append(
                {
                    "id": card.id,
                    "name": card.name,
                    "family": card.family,
                    "simulated": card.simulated,
                    "risk_score": card.risk_score,
                    "status": str(card.status),
                    "genai_used_here": any(
                        s.genai_used for s in card.kill_chain if s.phase == phase
                    ),
                }
            )
    return grid


def genai_intensity() -> dict[str, float]:
    """Fraction of steps in each phase that GenAI materially enables.

    This is the atlas's central empirical claim, stated as a number: generative AI is not
    uniformly distributed across the kill chain. It concentrates where the bottleneck used
    to be human effort - pretext, trust-building, resource development - and barely touches
    the phases governed by the payment rail's own mechanics.
    """
    counts: dict[str, list[int]] = {str(p): [] for p in PHASE_ORDER}
    for card in atlas():
        for step in card.kill_chain:
            counts[str(step.phase)].append(int(step.genai_used))
    return {p: (sum(v) / len(v) if v else 0.0) for p, v in counts.items()}


def cards_for_phase(phase: Phase) -> list[AttackCard]:
    return [c for c in atlas() if any(s.phase == phase for s in c.kill_chain)]
