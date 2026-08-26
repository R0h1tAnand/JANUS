"""Generate the unstructured GenAI artefacts that accompany the payment events.

Fraud does not arrive as a row in a table. It arrives as a phone call, a message, or a
manipulated conversation with a support agent, and the payment is only the last step. This
module produces those artefacts so the defence can be evaluated on them as well.

Everything is drawn from the committed corpus in ``data/llm_corpus/`` and recombined under a
seeded RNG. That is the Tier-1 design in practice: an LLM authored the seed material at build
time, and at runtime the simulator produces far more variation than the corpus contains
without needing a key, a network connection or a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_corpus"

BANKS = ["your bank", "the issuer", "the card network", "customer services"]
BILLERS = ["the electricity board", "the gas provider", "the telecom operator", "the water utility"]


@lru_cache(maxsize=8)
def load_corpus(name: str) -> tuple[dict, ...]:
    """Load one JSONL corpus file, cached."""
    path = CORPUS_DIR / f"{name}.jsonl"
    if not path.exists():
        return ()
    return tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


@dataclass(frozen=True, slots=True)
class Artifact:
    """One generated piece of adversarial or benign content."""

    text: str
    is_scam: bool
    card: str | None
    channel: str
    language: str
    tells: tuple[str, ...] = ()
    #: Corpus record this artefact was built from. Carried so evaluation can split at the
    #: *seed* level: recombinations of the same seed share phrasing, so a random split over
    #: generated samples leaks fragments across train and test and reports a perfect score.
    source_id: str = ""


def _fill(template: str, rng: np.random.Generator) -> str:
    """Substitute the placeholder slots a lure template carries."""
    return (
        template.replace("{last4}", f"{int(rng.integers(1000, 9999))}")
        .replace("{amount}", f"Rs {int(rng.integers(5, 400)) * 100:,}")
        .replace("{bank}", str(rng.choice(BANKS)))
        .replace("{biller}", str(rng.choice(BILLERS)))
    )


def scam_call(rng: np.random.Generator, card: str | None = None) -> Artifact:
    """Assemble a scam call transcript from the corpus.

    Openings, escalations and asks are recombined *across* records of the same persona rather
    than replayed verbatim, so the classifier cannot succeed by memorising twenty fixed scripts.
    """
    rows = [r for r in load_corpus("scam_scripts") if card is None or r["card"] == card]
    if not rows:
        rows = list(load_corpus("scam_scripts"))
    base = rows[int(rng.integers(len(rows)))]
    same_persona = [r for r in rows if r["persona"] == base["persona"]] or [base]

    opening = base["opening"]
    escalation = same_persona[int(rng.integers(len(same_persona)))]["escalation"]
    ask = same_persona[int(rng.integers(len(same_persona)))]["ask"]
    parts = [opening, escalation, ask]
    if rng.random() < 0.4:
        parts.insert(2, "Please stay on the line while we complete this.")

    return Artifact(
        text=" ".join(parts),
        is_scam=True,
        card=base["card"],
        channel=base["channel"],
        language=base["language"],
        tells=tuple(base.get("tells", ())),
        source_id=base["id"],
    )


def smishing(rng: np.random.Generator, card: str | None = None) -> Artifact:
    rows = [r for r in load_corpus("smishing") if card is None or r["card"] == card]
    if not rows:
        rows = list(load_corpus("smishing"))
    row = rows[int(rng.integers(len(rows)))]
    return Artifact(
        text=_fill(row["text"], rng),
        is_scam=True,
        card=row["card"],
        channel="sms",
        language=row["language"],
        tells=tuple(row.get("tells", ())),
        source_id=row["id"],
    )


def prompt_injection(rng: np.random.Generator) -> Artifact:
    rows = list(load_corpus("prompt_injections"))
    row = rows[int(rng.integers(len(rows)))]
    return Artifact(
        text=row["text"], is_scam=True, card=row["card"],
        channel=row["vector"], language="en", tells=tuple(row.get("tells", ())),
        source_id=row["id"],
    )


def benign(rng: np.random.Generator) -> Artifact:
    rows = list(load_corpus("benign_conversations"))
    row = rows[int(rng.integers(len(rows)))]
    text = row["text"]
    # Light recombination so benign text is as varied as the scam text; otherwise the
    # classifier could separate the classes on surface diversity alone.
    if rng.random() < 0.3:
        text = text + " " + rows[int(rng.integers(len(rows)))]["text"]
    return Artifact(
        text=text, is_scam=False, card=None,
        channel=row["channel"], language=row["language"], source_id=row["id"],
    )


def generate_batch(n: int, *, scam_fraction: float = 0.5, seed: int = 0) -> list[Artifact]:
    """A labelled batch mixing scam and benign content, for training the text layer."""
    rng = np.random.default_rng(seed)
    out: list[Artifact] = []
    for _ in range(n):
        if rng.random() < scam_fraction:
            kind = rng.random()
            out.append(
                scam_call(rng) if kind < 0.5 else smishing(rng) if kind < 0.85
                else prompt_injection(rng)
            )
        else:
            out.append(benign(rng))
    return out


def corpus_stats() -> dict:
    return {
        name: len(load_corpus(name))
        for name in ("scam_scripts", "smishing", "prompt_injections", "benign_conversations")
    }
