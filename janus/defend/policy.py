"""Risk score to action, with the economics made explicit.

A detector that outputs a probability has not yet done anything useful. The decision is which
of four actions to take, and each has a different cost:

* **allow** - costs nothing if right, costs the full transaction value if wrong
* **step_up** - costs a small amount of customer friction, and recovers most of the fraud
* **hold** - costs review headcount and a delayed payment
* **block** - costs a lost genuine transaction and a support call, if wrong

The optimal threshold therefore depends on rupees, not on F1. A model with worse F1 can be
worth more money if its errors fall in cheaper places, which is why this module exists and why
the walkthrough quotes a rupee figure rather than only a confusion matrix.

Cost constants are stated as assumptions, not facts. They are the parameters most worth
challenging, so they live in one visible place rather than being scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class Action(StrEnum):
    ALLOW = "allow"
    STEP_UP = "step_up"
    HOLD = "hold"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class CostModel:
    """Economic assumptions behind the operating point. All figures in INR."""

    #: Cost of challenging a legitimate customer: abandonment risk plus support load.
    step_up_friction: float = 12.0
    #: Cost of manually reviewing a held payment (analyst time).
    review_cost: float = 90.0
    #: Cost of wrongly blocking a legitimate payment: lost transaction plus relationship damage.
    false_block_cost: float = 450.0
    #: Share of fraud that a step-up challenge actually defeats. Not 1.0 - relay attacks and
    #: social-engineered victims pass step-up, which is the entire premise of VY-CARD-002.
    step_up_effectiveness: float = 0.62
    #: Share of fraud a manual review catches before release.
    review_effectiveness: float = 0.93


@dataclass(frozen=True, slots=True)
class Thresholds:
    step_up: float
    hold: float
    block: float


def decide(
    fraud_score: np.ndarray, novelty_score: np.ndarray, thresholds: Thresholds,
    *, novelty_weight: float = 0.25,
) -> np.ndarray:
    """Map blended risk onto actions.

    Novelty contributes a bounded uplift rather than an independent trigger. Given alone it
    would fire on every unusual-but-legitimate payment - a genuine first car purchase looks
    extremely novel - so it can escalate a borderline case but cannot by itself block.
    """
    blended = np.clip(fraud_score + novelty_weight * novelty_score * (1 - fraud_score), 0, 1)
    actions = np.full(len(blended), str(Action.ALLOW), dtype=object)
    actions[blended >= thresholds.step_up] = str(Action.STEP_UP)
    actions[blended >= thresholds.hold] = str(Action.HOLD)
    actions[blended >= thresholds.block] = str(Action.BLOCK)
    return actions


def evaluate_policy(
    actions: np.ndarray, y_true: np.ndarray, amounts: np.ndarray, costs: CostModel | None = None
) -> dict:
    """Rupee outcome of a policy, against the do-nothing baseline."""
    costs = costs or CostModel()
    fraud = y_true == 1
    legit = ~fraud

    exposure = float(amounts[fraud].sum())
    prevented = 0.0
    for action, effectiveness in (
        (Action.STEP_UP, costs.step_up_effectiveness),
        (Action.HOLD, costs.review_effectiveness),
        (Action.BLOCK, 1.0),
    ):
        hit = fraud & (actions == str(action))
        prevented += float(amounts[hit].sum()) * effectiveness

    friction = (
        float((legit & (actions == str(Action.STEP_UP))).sum()) * costs.step_up_friction
        + float((legit & (actions == str(Action.HOLD))).sum()) * costs.review_cost
        + float((legit & (actions == str(Action.BLOCK))).sum()) * costs.false_block_cost
    )

    counts = {str(a): int((actions == str(a)).sum()) for a in Action}
    return {
        "fraud_exposure": round(exposure, 2),
        "fraud_prevented": round(prevented, 2),
        "prevention_rate": round(prevented / exposure, 4) if exposure else 0.0,
        "friction_cost": round(friction, 2),
        "net_benefit": round(prevented - friction, 2),
        "legit_challenged_rate": round(
            float((legit & (actions != str(Action.ALLOW))).sum()) / max(int(legit.sum()), 1), 5
        ),
        "action_counts": counts,
    }


def sweep_thresholds(
    fraud_score: np.ndarray,
    novelty_score: np.ndarray,
    y_true: np.ndarray,
    amounts: np.ndarray,
    costs: CostModel | None = None,
    grid: np.ndarray | None = None,
    max_challenge_rate: float = 0.02,
) -> pd.DataFrame:
    """Find the best operating point subject to an operational friction ceiling.

    ``max_challenge_rate`` is not a modelling nicety - it is the binding real-world constraint.
    Optimising net rupees alone drives the step-up threshold to the floor, because a challenge
    costs a few rupees of friction and prevents thousands. The unconstrained optimum here
    challenged 16% of all legitimate payments, which is commercially unshippable regardless of
    what the arithmetic says: customers abandon, call centres flood, and the challenge itself
    stops working once it is routine.

    Two percent is a defensible ceiling for a real portfolio. The unconstrained frontier is
    still returned in full so the trade-off is inspectable rather than hidden.

    Returns one row per candidate threshold set, sorted by net rupee benefit, with rows that
    breach the ceiling marked and pushed below the feasible ones.
    """
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19)
    rows = []
    for step_up in grid:
        for hold_mult, block_mult in ((1.3, 1.8), (1.5, 2.5), (2.0, 3.0)):
            th = Thresholds(
                step_up=float(step_up),
                hold=float(min(step_up * hold_mult, 0.98)),
                block=float(min(step_up * block_mult, 0.99)),
            )
            result = evaluate_policy(
                decide(fraud_score, novelty_score, th), y_true, amounts, costs
            )
            rows.append({"step_up": th.step_up, "hold": th.hold, "block": th.block, **result})

    df = pd.DataFrame(rows)
    df["within_friction_budget"] = df.legit_challenged_rate <= max_challenge_rate
    return (
        df.sort_values(["within_friction_budget", "net_benefit"], ascending=[False, False])
        .reset_index(drop=True)
    )
