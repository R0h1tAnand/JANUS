<div align="center">

<img src="docs/janus-relief.png" alt="Relief carving of Janus, the two-faced Roman god of gateways" width="220"/>

# Project Janus

**A payment-fraud lab that plays both sides.**
Map how GenAI is changing fraud → simulate it at scale → detect it → let the attacker adapt, and go again.

`47 attack techniques` · `21 working simulators` · `measured fidelity` · `leave-one-attack-out evaluation` · `adversarial loop`

</div>

---

> Janus, the Roman god of gateways, was carved with two faces looking in opposite directions —
> one at what approaches, one at what departs. **You cannot guard a gate you have only ever
> stood behind.**

Built for the **Mastercard Innovation Challenge @ GFF 2026 — AI Defense Lab for Payment Security**.

## The thesis

Generative AI did not make fraud smarter. It made *persuasion* cheap, and persuasion was the
bottleneck.

Janus marks every step of every attack in its atlas for whether GenAI materially enables it.
The distribution is lopsided and it is the finding the rest of the system is built around:

| kill-chain phase | share of steps GenAI enables |
|---|---|
| pretext | **100%** |
| resource development | **95%** |
| trust building | **88%** |
| evasion | 61% |
| instrument | 12% |
| launder | 6% |
| **monetize** | **0%** |

Generative models have transformed the front of the kill chain and barely touched the back.
Moving money is still governed by the rail's own mechanics. **Detection has to shift left** —
toward the part of the chain that used to be too expensive for attackers to run at scale.

## What is here

**Identify.** 47 attack techniques as machine-readable YAML, spanning all 15 payment rails
(UPI P2P/P2M/collect/mandate/Lite, IMPS, NEFT, RTGS, AePS, card CNP/CP/token/ATM, wallet,
cross-border) and all 10 GenAI enabler types. Not documentation — the generator *compiles*
these into simulators, and `janus atlas validate` fails if a technique claims a simulator it
does not have. An **Ideator** searches the 21,000-cell product space of
`rail × enabler × surface × monetisation` for coherent combinations the atlas has missed.

**Generate.** An agent-based payment economy — contact graphs built by preferential attachment,
per-customer circadian rhythms, merchant affinity, mule supply — with 21 attack injectors that
manipulate that world rather than sampling from a separate distribution. 718,000 events in
about five seconds.

**Defend.** A point-in-time-correct feature store (63 features), gradient-boosted classifier,
unsupervised novelty layer trained on legitimate traffic only, graph features for mule
structures, a text layer for scam content, and a policy engine that decides in rupees rather
than in F1.

**Adapt.** A Red agent runs a black-box evolutionary search over the attacker's real degrees of
freedom to find variants that get the most **money** past the live detector. Blue retrains on
what Red finds, at a fixed false-positive rate so it cannot win by simply becoming
more aggressive.

## Results

Full generated report: **[RESULTS.md](RESULTS.md)** — every number read from `reports/`, never
hand-typed.

The headline is deliberately the uncomfortable one:

| | |
|---|---|
| Recall @ 0.1% FPR, families the model **has** seen | **88.8%** |
| Recall on families it has **never** seen (leave-one-attack-out) | **17.4%** |
| Generalisation gap | **+42.8 points** |
| Families above 50% recall when unseen | **2 of 21** |
| Families below 20% | **15 of 21** |

**Supervised fraud detection largely does not transfer to novel attack families.** That gap is
the actual finding, and it is why this repo also ships an unsupervised novelty layer and an
adversarial loop rather than stopping at a classifier and a good-looking AUC.

The two families that *do* generalise are instructive. SIM-swap account takeover (0.95) and
dormant-account reactivation both have **compositional** signatures — device rebinding
followed by a rapid sweep, sudden throughput on an account with long tenure but thin recent
history — assembled from signals other attack families already teach. The fifteen that fail
have idiosyncratic signatures nothing else in the atlas covers. That is a concrete design
lesson: a defence generalises to the extent its features are shared vocabulary rather than
per-attack special cases.

## Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (it fetches Python 3.13 itself) and
Node 20+. No API key, no GPU. The only network access is `make setup` pulling packages and one
dataset download for the fidelity measurement.

```bash
make setup      # uv sync + npm install
make all        # ~30 min: fetch reference data → generate → train → evaluate
                #          → fidelity → arena (fast profile) → regenerate docs
make serve      # API on :8000   (needs `make all` first — it loads the trained model)
make ui         # console on :5173, in a second shell → http://localhost:5173
```

`make all` deliberately leaves out the two slow evaluations. Run them when you want the
published numbers:

```bash
make loao       # leave-one-attack-out — trains one model per attack family, ~1h
make arena      # full-scale adversarial loop, ~12 min (make all uses a smaller profile)
make results    # regenerate RESULTS.md from whatever is in reports/
```

Or step by step, in dependency order:

```bash
uv run python scripts/fetch_reference_data.py   # public datasets for fidelity (~750 MB)
uv run janus generate run                       # simulate a payment world
uv run janus defend train                       # train and persist the defence
uv run janus generate fidelity                  # measure realism against real data
uv run janus defend evaluate                    # detection efficacy
uv run janus arena run --profile fast           # the Red/Blue loop
uv run janus status                             # what is built, what is still missing
```

Nothing to explore first? `uv run janus atlas matrix` renders the kill-chain matrix straight
from the atlas and needs no data at all, and `uv run janus atlas ideate` proposes attack vectors
the atlas does not yet cover.

`make check-ui` smoke-tests every console view for render errors (needs `make serve` and
`make ui` running, plus a local `chromium`). `make check-clone` verifies a fresh checkout
builds from scratch.

## Things this project refuses to do

Most of the engineering effort went into *not* reporting a flattering number.

- **Fidelity is measured, not claimed.** A discriminator tries to separate our synthetic
  payments from real public data (Sparkov, PaySim). It reaches AUC ≈ 0.72–0.77 — "moderate",
  not "indistinguishable". That number is reported, along with a written list of the four
  residuals we know about and chose not to tune away.
- **Features cannot see the future.** Rolling aggregates are recomputed on truncated history
  and must return bit-identical values. This test caught real leakage the first time it ran.
- **No attack gets its own private enum.** Legitimate traffic exercises every rail, channel and
  authentication method that fraud does. An early build had six perfect separators — device
  ages with disjoint ranges, an `api` channel only attackers used — that inflated ROC-AUC to
  0.9992 on pure artefact. Two guard tests now make that class of bug fail loudly.
- **The false-positive budget is a ceiling, not a target.** Isotonic calibration collapses
  ~119,000 distinct scores onto about 68 plateaus. An earlier build set thresholds on that
  calibrated output, so `score >= t` swept up an entire tied block of 96,000 events and the
  realised FPR ran at *twice* the requested budget — quietly inflating every recall figure that
  depended on it. Thresholds are now tie-aware and computed on the uncalibrated ranking score;
  the realised FPR is printed next to every target so the gap is visible.
- **A model that scores 0.9995 is assumed broken.** There is a test asserting it.
- **The economically optimal policy is rejected** when it challenges 16% of legitimate
  payments, because no bank ships that whatever the arithmetic says.

## Real-world feasibility

- **Scoring is CPU-only and measured**, not asserted. Single-event latency on the production
  path is **107 µs mean / 218 µs p99** — roughly 9,300 authorisations per second per core, on a
  laptop. (Scoring through the pandas/sklearn convenience wrapper instead costs ~7,700 µs, and
  essentially all of that is framework overhead a real service pays once at startup rather than
  once per payment. The API reports the production-path number and the code says why.)
  No LLM sits on the authorisation path, because none can survive payments latency and cost.
  LLMs run at *build* time for red-team content and threat ideation; the corpus they produce is
  committed to this repo, so the demo needs no key.
- **The schema is a rename from production, not an invention.** Every field maps to ISO 8583
  or ISO 20022 (`janus/generate/rails.py::ISO_MAPPING`).
- **Model governance is built in.** Trained bundles record their seed, feature ordering,
  thresholds and atlas revision.
- **Metrics are quoted at fixed false-positive rates**, because that is what a payments
  organisation actually buys.

## Layout

```
janus/identify/   attack atlas, signal registry, ideator, kill-chain matrix
janus/generate/   world model, rails, 21 injectors, artefacts, fidelity harness
janus/defend/     features, supervised, novelty, policy, explain, evaluate
janus/loop/       Red agent, arena
janus/api/        FastAPI + WebSocket
web/              React console
data/llm_corpus/  GenAI artefacts authored at build time (committed on purpose)
```

## Scope and ethics

This project generates fraud in order to detect it, so the boundary is stated explicitly in
**[docs/ETHICS.md](docs/ETHICS.md)**. In short: all payment data is synthetic and seeded, the
adversarial text samples carry no links, domains, numbers or real identities, and nothing here
synthesises voice, video or documents. The atlas *describes* those techniques because a defender
needs to know what they leave behind; the code produces none of them.

## Also here

- **[RESULTS.md](RESULTS.md)** — generated from `reports/`, never hand-typed
- **[docs/DEMO.md](docs/DEMO.md)** — three-minute demo script
- **[docs/Janus_Solution_Walkthrough.docx](docs/)** — the challenge write-up, also generated
- **[docs/CHALLENGE_BRIEF.txt](docs/CHALLENGE_BRIEF.txt)** — the original problem statement

## License

MIT — see [LICENSE](LICENSE).
