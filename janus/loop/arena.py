"""The closed loop: Red evolves, Blue retrains, repeat.

Each round:

1. **Red** searches for the attack configuration that gets the most rupees past the *currently
   deployed* detector.
2. Those evasive events are labelled and folded into Blue's training set - this is the step
   that turns the red team's output into the blue team's training ground, which is what the
   brief asks for.
3. **Blue** retrains and recalibrates its threshold at a fixed false-positive rate, so the
   comparison across rounds is like-for-like. Letting the FPR float would let Blue "win" by
   simply becoming more aggressive.
4. Everything is measured on a **held-out world Blue has never trained on**, so improvement
   reflects genuine learning rather than memorisation of Red's specific campaigns.

What to look for in the output: Red's value-through should fall round over round while the
false-positive rate stays flat. Where it does not - where Red finds a configuration Blue cannot
answer without spending friction - that is a real finding about the limits of the defence, and
it is reported rather than smoothed over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from janus.defend import supervised
from janus.defend.features import build_features
from janus.generate.legit import generate_legit
from janus.generate.rails import EVENT_COLUMNS
from janus.generate.simulate import load_injectors, simulate
from janus.generate.world import WorldConfig, build_world
from janus.loop.red import RedAgent, describe


@dataclass(slots=True)
class RoundResult:
    round: int
    red_evasion_rate: float
    red_value_through: float
    red_value_total: float
    blue_recall_at_fpr: float
    blue_roc_auc: float
    realised_fpr: float
    #: Fraud share of Blue's training set this round. Exposed because a replay buffer that
    #: quietly destroys the class balance is the easiest way to make this loop look like it is
    #: working while it is not - see the note in run_arena.
    train_base_rate: float
    n_train_events: int
    best_params: dict
    param_summary: str

    @property
    def value_through_pct(self) -> float:
        return self.red_value_through / max(self.red_value_total, 1.0)


@dataclass(slots=True)
class ArenaResult:
    rounds: list[RoundResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "round": r.round,
                "red_evasion_rate": round(r.red_evasion_rate, 4),
                "value_through_pct": round(r.value_through_pct, 4),
                "value_through_lakh": round(r.red_value_through / 1e5, 2),
                "blue_recall": round(r.blue_recall_at_fpr, 4),
                "blue_roc_auc": round(r.blue_roc_auc, 4),
                "realised_fpr": round(r.realised_fpr, 5),
                "train_base_rate": round(r.train_base_rate, 5),
                "n_train_events": r.n_train_events,
                "adaptation": r.param_summary,
            }
            for r in self.rounds
        ])

    def summary(self) -> dict:
        if not self.rounds:
            return {}
        first, last = self.rounds[0], self.rounds[-1]
        return {
            "rounds": len(self.rounds),
            "evasion_first": round(first.red_evasion_rate, 4),
            "evasion_last": round(last.red_evasion_rate, 4),
            "evasion_reduction": round(first.red_evasion_rate - last.red_evasion_rate, 4),
            "value_through_first_pct": round(first.value_through_pct, 4),
            "value_through_last_pct": round(last.value_through_pct, 4),
            "recall_first": round(first.blue_recall_at_fpr, 4),
            "recall_last": round(last.blue_recall_at_fpr, 4),
            "fpr_drift": round(last.realised_fpr - first.realised_fpr, 5),
            "base_rate_first": round(first.train_base_rate, 5),
            "base_rate_last": round(last.train_base_rate, 5),
        }


#: Preset profiles. The arena is a nested loop - every candidate in every generation of every
#: round triggers a fresh simulation plus a feature build - so cost grows as
#: rounds x generations x population x world size. The full profile is what the reported
#: numbers come from; the fast profile is what fits inside a live demo.
PROFILES: dict[str, dict] = {
    "full": {"rounds": 5, "generations": 3, "population": 10,
             "customers": 5000, "days": 30},
    "fast": {"rounds": 4, "generations": 2, "population": 6,
             "customers": 3000, "days": 25},
}


def run_arena(
    cfg: WorldConfig | None = None,
    *,
    rounds: int = 5,
    generations: int = 3,
    population: int = 10,
    target_fpr: float = 0.001,
    replay_fraud_ratio: float = 0.25,
    seed: int = 0,
    progress: bool = True,
) -> ArenaResult:
    """Run the adversarial loop and return per-round metrics."""
    # 5,000 customers over 30 days is the smallest world in which the fraud-rate calibration
    # actually converges (0.71%, calibration_achieved=True). Smaller worlds hit the
    # one-campaign-per-family floor and end up at 1-4% fraud, which makes every recall number
    # in the loop describe a portfolio no bank has.
    cfg = cfg or WorldConfig(n_customers=5000, n_merchants=400, days=30, seed=5)
    injectors = load_injectors()

    # Blue's initial training data: a baseline world with unmodified attacks.
    base = simulate(cfg)
    train_events = base.events.sort_values("ts", kind="stable").reset_index(drop=True)

    # A separate world, never trained on, that every round is measured against.
    holdout = simulate(replace(cfg, seed=cfg.seed + 777))
    holdout_events = holdout.events.sort_values("ts", kind="stable").reset_index(drop=True)
    X_holdout = build_features(holdout_events)
    y_holdout = holdout_events["is_fraud"].to_numpy()

    # Red's sandbox: its own world and legitimate base, distinct from both of the above.
    red_cfg = replace(cfg, seed=cfg.seed + 321)
    red_world = build_world(red_cfg)
    red_legit = generate_legit(red_world)

    result = ArenaResult()
    for round_idx in range(1, rounds + 1):
        if progress:
            print(f"  round {round_idx}/{rounds}: training Blue on "
                  f"{len(train_events):,} events ...", flush=True)

        X_train = build_features(train_events)
        y_train = train_events["is_fraud"].to_numpy()
        round_base_rate = float(y_train.mean())
        detector = supervised.train(X_train, y_train, seed=seed, n_estimators=300)

        scores_holdout = detector.rank_score(X_holdout)
        threshold = supervised.threshold_at_fpr(y_holdout, scores_holdout, target_fpr)
        blue_recall = float((scores_holdout[y_holdout == 1] >= threshold).mean())
        blue_auc = supervised.evaluate(
            detector, X_holdout, y_holdout, scores=scores_holdout
        )["roc_auc"]
        realised_fpr = float((scores_holdout[y_holdout == 0] >= threshold).mean())

        if progress:
            print(f"           Blue recall {blue_recall:.3f} @ fpr {realised_fpr:.4f}; "
                  f"Red searching ...", flush=True)

        red = RedAgent(red_world, red_legit, injectors, seed=seed + round_idx,
                       population=population)
        best = red.evolve(detector, threshold, generations=generations)

        result.rounds.append(RoundResult(
            round=round_idx,
            red_evasion_rate=best.evasion_rate,
            red_value_through=best.value_through,
            red_value_total=best.total_value,
            blue_recall_at_fpr=blue_recall,
            blue_roc_auc=blue_auc,
            realised_fpr=realised_fpr,
            train_base_rate=round_base_rate,
            n_train_events=int(len(train_events)),
            best_params=asdict(best.params),
            param_summary=describe(best.params),
        ))

        if progress:
            share = best.value_through / max(best.total_value, 1.0)
            print(f"           Red evaded {best.evasion_rate:.1%} of attempts, carrying "
                  f"{share:.1%} of attack value | {describe(best.params)}", flush=True)

        # Fold Red's discovered variant into Blue's training set for the next round.
        #
        # The base rate MUST be preserved here. An earlier version appended only Red's fraud
        # events, with no legitimate counterpart and no cap: Blue's training set went from 0.6%
        # fraud to 37% after one round and 54% after two, and its holdout recall duly collapsed
        # from 0.48 to 0.34. That looked like an interesting result about catastrophic
        # forgetting; it was just a broken replay buffer. Fraud detection is a rare-event
        # problem, and a retraining loop that quietly destroys the class balance is not
        # measuring adversarial robustness at all.
        #
        # So: cap how much evasive fraud any single round may contribute, and carry across
        # enough of Red's legitimate traffic to hold the base rate steady.
        if round_idx < rounds:
            evasive = red.best_attack_events(best.params)
            rng = np.random.default_rng(seed + round_idx)

            base_rate = float(train_events["is_fraud"].mean())
            current_fraud = int(train_events["is_fraud"].sum())

            pool_fraud = evasive[evasive.is_fraud == 1]
            n_fraud = min(len(pool_fraud), max(1, int(current_fraud * replay_fraud_ratio)))
            add_fraud = pool_fraud.sample(n=n_fraud, random_state=int(rng.integers(1 << 31)))

            pool_legit = evasive[evasive.is_fraud == 0]
            n_legit = min(
                len(pool_legit),
                int(n_fraud * (1 - base_rate) / max(base_rate, 1e-6)),
            )
            add_legit = pool_legit.sample(n=n_legit, random_state=int(rng.integers(1 << 31)))

            train_events = (
                pd.concat(
                    [train_events, add_fraud[EVENT_COLUMNS], add_legit[EVENT_COLUMNS]],
                    ignore_index=True,
                )
                .sort_values("ts", kind="stable")
                .reset_index(drop=True)
            )
            if progress:
                print(f"           replay: +{n_fraud:,} evasive fraud, +{n_legit:,} legitimate "
                      f"(base rate {float(train_events['is_fraud'].mean()):.3%})", flush=True)

    return result


def rounds_to_detect(result: ArenaResult, threshold: float = 0.5) -> int | None:
    """How many rounds Blue needed to push Red's evasion rate below ``threshold``.

    Reported as 'time to adapt to a novel family' - the metric a defender actually cares
    about when a new attack appears. ``None`` means Blue never got there.
    """
    for r in result.rounds:
        if r.red_evasion_rate < threshold:
            return r.round
    return None
