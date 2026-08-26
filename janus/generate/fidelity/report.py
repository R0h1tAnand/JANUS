"""Assemble the fidelity evidence into one reproducible report.

The report deliberately includes what does *not* match. A fidelity section that reports only
favourable numbers is worth nothing to a reader who cannot check it, and the residuals are the
most useful part of the output: they tell the next person what to fix.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from janus.generate.fidelity import discriminator, metrics
from janus.generate.fidelity.align import (
    COMPARABLE_COLUMNS,
    available_references,
    load_reference,
    project_janus,
)

REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"

#: Stated plainly in the output so no reader has to infer it.
KNOWN_LIMITATIONS = [
    "Comparison is restricted to amount, hour and weekday - the only fields both public "
    "reference datasets contain. Graph structure, device signals and session context are "
    "central to Janus's attacks but absent from Sparkov and PaySim, so they are defended on "
    "construction rather than measured here.",
    "PaySim's `step` is a simulation tick index, not a wall clock, so it contributes "
    "amount-distribution evidence only. An earlier revision compared hour-of-day against it "
    "and produced a spurious 0.98 discriminator AUC.",
    "Absolute amount levels are not comparable across economies (Sparkov is US card spend, "
    "PaySim is an unspecified currency), so the headline comparison uses standardised "
    "log-amount. Raw log_amount KS is reported alongside and is expected to be poor.",
    "Residual gap: synthetic log-amount skew is near 0 against Sparkov's -0.44. That skew "
    "comes from a US-specific category structure where grocery and fuel are the highest "
    "frequency, highest ticket and lowest variance categories. Matching it would require "
    "mis-stating Indian ticket sizes, so it is left in place and reported.",
]


def build(events: pd.DataFrame, *, seed: int = 0) -> dict:
    """Run every fidelity test against every available reference dataset."""
    refs = available_references()
    if not refs:
        raise FileNotFoundError(
            "No reference data found. Run: uv run python scripts/fetch_reference_data.py"
        )

    results = []
    for name in refs:
        syn = project_janus(events, segment=name)
        ref = load_reference(name)
        cols = COMPARABLE_COLUMNS[name]
        disc_features = [c for c in cols if c != "log_amount"]
        entry = metrics.summarise(syn, ref, name)
        entry["discriminator"] = discriminator.discriminator_auc(
            syn, ref, features=disc_features, seed=seed
        )
        entry["tstr"] = discriminator.tstr(syn, ref, seed=seed)
        entry["shape"] = {
            "synthetic_sd": round(float(syn.log_amount.std()), 3),
            "reference_sd": round(float(ref.log_amount.std()), 3),
            "synthetic_skew": round(float(syn.log_amount.skew()), 3),
            "reference_skew": round(float(ref.log_amount.skew()), 3),
        }
        results.append(entry)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_events": int(len(events)),
        "fraud_rate": round(float(events.is_fraud.mean()), 5),
        "references": results,
        "known_limitations": KNOWN_LIMITATIONS,
    }


def write_json(report: dict, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "fidelity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
