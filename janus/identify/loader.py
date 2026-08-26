"""Load and validate the attack atlas.

The atlas is the single source of truth shared by all three pillars, so loading is strict:
a malformed card, a duplicate id, or a ``simulated`` card whose injector does not resolve is
a hard failure. That guarantee is what lets the README claim a simulated-coverage number
without anyone having to take it on trust.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

from janus.identify.schema import AttackCard, GenAIEnabler, Monetization, Phase, Rail, Surface

ATLAS_DIR = Path(__file__).parent / "atlas"
INJECTOR_PACKAGE = "janus.generate.injectors"


class AtlasError(RuntimeError):
    """Raised when the atlas is internally inconsistent."""


def _iter_documents(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [doc for doc in yaml.safe_load_all(fh) if doc is not None]


def load_cards(atlas_dir: Path | None = None, *, check_injectors: bool = True) -> list[AttackCard]:
    """Parse every YAML document under ``atlas_dir`` into validated :class:`AttackCard` objects."""
    directory = atlas_dir or ATLAS_DIR
    cards: list[AttackCard] = []
    errors: list[str] = []

    for path in sorted(directory.glob("*.yaml")):
        for i, doc in enumerate(_iter_documents(path)):
            try:
                cards.append(AttackCard.model_validate(doc))
            except Exception as exc:  # noqa: BLE001 - aggregate and report all at once
                errors.append(f"{path.name}[doc {i}] {doc.get('id', '<no id>')}: {exc}")

    if errors:
        raise AtlasError("Invalid attack cards:\n\n" + "\n\n".join(errors))

    duplicates = [cid for cid, n in Counter(c.id for c in cards).items() if n > 1]
    if duplicates:
        raise AtlasError(f"Duplicate attack card ids: {duplicates}")

    if check_injectors:
        missing = [
            f"{c.id} -> {INJECTOR_PACKAGE}.{c.injector}"
            for c in cards
            if c.simulated and importlib.util.find_spec(f"{INJECTOR_PACKAGE}.{c.injector}") is None
        ]
        if missing:
            raise AtlasError(
                "Cards marked simulated=true whose injector module does not exist:\n  "
                + "\n  ".join(missing)
            )

    return cards


@lru_cache(maxsize=1)
def atlas() -> tuple[AttackCard, ...]:
    """Cached atlas load. Injector checking is deferred to :func:`load_cards` callers."""
    return tuple(load_cards(check_injectors=False))


def simulated_cards() -> list[AttackCard]:
    """Cards that compile to a working injector - the families used for LOAO evaluation."""
    return [c for c in atlas() if c.simulated]


def by_id(card_id: str) -> AttackCard:
    for card in atlas():
        if card.id == card_id:
            return card
    raise KeyError(f"No attack card with id {card_id!r}")


def coverage() -> dict:
    """Summary statistics used by the README, the CLI and the Atlas dashboard view.

    Coverage is reported per axis so a reader can see where the atlas is thin rather than
    only seeing a single headline count.
    """
    cards = atlas()

    def _axis(attr: str, enum_cls) -> dict[str, int]:
        counts = Counter()
        for card in cards:
            for value in getattr(card, attr):
                counts[str(value)] += 1
        return {str(member): counts.get(str(member), 0) for member in enum_cls}

    return {
        "total_cards": len(cards),
        "simulated_cards": sum(1 for c in cards if c.simulated),
        "families": dict(Counter(c.family for c in cards)),
        "status": dict(Counter(str(c.status) for c in cards)),
        "rails": _axis("rails", Rail),
        "genai_enablers": _axis("genai_enablers", GenAIEnabler),
        "surfaces": _axis("surfaces", Surface),
        "monetization": _axis("monetization", Monetization),
        "phases": {
            str(p): sum(1 for c in cards for s in c.kill_chain if s.phase == p) for p in Phase
        },
        "distinct_observables": len({o for c in cards for o in c.observables}),
    }
