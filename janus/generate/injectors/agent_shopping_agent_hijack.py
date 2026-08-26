"""VY-AGENT-003 - compromised autonomous shopping agent.

Prospective rather than observed, and included for that reason: if delegated purchasing becomes
common, this is what the fraud looks like, and a defence designed only against today's families
will meet it with nothing.

The transaction carries the consumer's genuine card, genuine device and genuine history. What
differs is that no human was ever in the session - the agent selected a merchant nobody browsed,
and it purchases on a machine-regular cadence no shopper has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.injectors.base import (
    InjectionContext,
    Injector,
    blank_events,
    finalise,
    merchant_context,
    victim_context,
)


class ShoppingAgentHijack(Injector):
    card_id = "VY-AGENT-003"
    base_campaigns = 30

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        # Delegation skews to high-tenure users - the people who adopt agents first.
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns), tenure="high")
        n_camp = len(victims)

        frames = []
        for i in range(n_camp):
            v = int(victims[i])
            n = int(np.clip(rng.integers(4, 14) * ev.burst_scale, 2, 30))
            # Attacker-controlled listings: merchants the customer has no affinity with.
            known = set(w.merchants_of(v).tolist())
            candidates = np.array([m for m in rng.integers(0, w.n_merchants, n * 3) if m not in known])
            merch = candidates[:n] if len(candidates) >= n else rng.integers(0, w.n_merchants, n)

            cols = blank_events(n)
            cols.update(merchant_context(ctx, merch))
            cols.update(victim_context(ctx, np.full(n, v)))
            cols["payer_id"] = np.full(n, v)
            cols["payer_account"] = np.full(n, v)
            cols["amount"] = rng.lognormal(8.2, 0.7, n) * ev.amount_scale
            start = rng.uniform(0, max(1.0, w.cfg.days - 3))
            # Perfectly even cadence - the strongest tell that no human is driving.
            gap = rng.uniform(0.15, 0.9)
            offs = start + np.arange(n) * gap * ev.delay_scale
            secs = (offs * 86400 + rng.uniform(0, 300, n)).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["card_token"] * n
            cols["channel"] = ["api"] * n
            cols["auth_method"] = ["token_device"] * n
            cols["agent_initiated"] = np.ones(n, dtype=np.int8)
            cols["campaign_id"] = [f"AGENT003-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = ShoppingAgentHijack()
