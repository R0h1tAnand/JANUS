"""Save and load a trained defence bundle.

The dashboard cannot afford to retrain on startup, and a submission that only reproduces after
a ten-minute training run is not really reproducible. A bundle carries everything needed to
score an event: the supervised model, the novelty layer, the feature ordering, the calibrated
thresholds, and the provenance of the data it was fitted on.

Provenance is recorded deliberately. A model artefact with no record of which simulation seed
and which atlas revision produced it is unauditable, and model governance is one of the
real-world-feasibility points the walkthrough has to stand behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from janus.defend.novelty import NoveltyDetector
from janus.defend.supervised import TrainedDetector

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(slots=True)
class DefenceBundle:
    """Everything required to score a payment, plus how it came to exist."""

    detector: TrainedDetector
    novelty: NoveltyDetector | None
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata.get("trained_at", "unknown")


def save(
    detector: TrainedDetector,
    novelty: NoveltyDetector | None,
    *,
    metadata: dict | None = None,
    path: Path | None = None,
) -> Path:
    path = path or MODEL_DIR / "defence.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    from janus.identify.loader import coverage

    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_features": len(detector.features),
        "features": detector.features,
        "thresholds": {str(k): v for k, v in detector.thresholds.items()},
        "atlas": {k: coverage()[k] for k in ("total_cards", "simulated_cards")},
        **(metadata or {}),
    }
    joblib.dump(
        {"detector": detector, "novelty": novelty, "metadata": meta}, path, compress=3
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load(path: Path | None = None) -> DefenceBundle:
    path = path or MODEL_DIR / "defence.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained defence at {path}. Run: uv run janus defend train"
        )
    blob = joblib.load(path)
    return DefenceBundle(
        detector=blob["detector"], novelty=blob.get("novelty"), metadata=blob["metadata"]
    )


def exists(path: Path | None = None) -> bool:
    return (path or MODEL_DIR / "defence.joblib").exists()
