"""Download the public reference datasets Janus measures its fidelity against.

Two anchors, chosen because between them they cover both halves of the simulator:

* **Sparkov** (``fraudTrain.csv``) - synthetic-but-widely-used card transactions retaining real
  structure: merchant, category, amount, timestamp, cardholder and merchant geography. This is
  the anchor for card-rail realism and for amount/category/temporal distributions.
* **PaySim** - mobile-money transfers with type, amount and both parties' balances. Push-based,
  real-time and irrevocable, so it is the closest public proxy for UPI P2P and for the
  balance-retention behaviour that distinguishes a mule account from a normal one.

Neither requires authentication. Both are cached under ``data/raw/`` and gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "sparkov": ("dazzle-nu/CIS435-CreditCardFraudDetection", "fraudTrain.csv"),
    "paysim": ("theman10/paysim", "paysim.csv"),
}


def fetch(name: str) -> Path:
    repo, filename = SOURCES[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"{name}.csv"
    if target.exists():
        print(f"  {name}: cached ({target.stat().st_size / 1e6:.0f} MB)")
        return target
    print(f"  {name}: downloading from {repo} ...")
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    target.write_bytes(Path(path).read_bytes())
    print(f"  {name}: {target.stat().st_size / 1e6:.0f} MB -> {target}")
    return target


def main() -> int:
    print("Fetching Janus fidelity reference datasets")
    for name in SOURCES:
        try:
            fetch(name)
        except Exception as exc:  # noqa: BLE001 - report and continue; one anchor still works
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
