"""VY-LAUND-001 - adaptive micro-structuring below reporting thresholds.

Naive structuring rounds down to just under a threshold and is trivially caught by looking for
amounts clustered at 49,000. This family does the opposite: it *samples split amounts from the
genuine population's own distribution*, so the marginal amount histogram is indistinguishable
from real traffic.

Which is the point. Amount-based detection cannot see this at all. What survives is aggregate
throughput against balance retention - the account moves a fortune and keeps none of it - and
that only exists as a feature if the defence aggregates per entity rather than per transaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.injectors.base import (
    InjectionContext,
    Injector,
    blank_events,
    finalise,
)


class AdaptiveStructuring(Injector):
    card_id = "VY-LAUND-001"
    base_campaigns = 18

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        # Fit the legitimate amount distribution and sample splits from it, rather than
        # from round numbers just under a limit.
        legit_amounts = ctx.legit["amount"].to_numpy()
        pool = legit_amounts[(legit_amounts > 500) & (legit_amounts < 45_000)]
        if len(pool) == 0:
            pool = np.array([5000.0])

        frames = []
        for i in range(n_camp):
            source = int(ctx.pick_mules(1)[0])
            n = int(np.clip(rng.integers(25, 90) * ev.burst_scale, 10, 200))
            targets = ctx.pick_mules(n)
            # Amounts drawn straight from observed legitimate traffic.
            amounts = rng.choice(pool, size=n) * ev.amount_scale
            start = rng.uniform(0, max(1.0, w.cfg.days - 4))
            # Irregular gaps sampled from an exponential - no periodicity to detect.
            offs = start + np.cumsum(rng.exponential(0.06 * ev.delay_scale, n))

            cols = blank_events(n)
            cols["payer_id"] = np.full(n, source)
            cols["payer_account"] = np.full(n, source)
            cols["payee_id"] = targets
            cols["payee_handle"] = [w.vpa(int(x)) for x in targets]
            cols["payee_account_age_days"] = w.cust_account_age[targets].astype(np.float32)
            cols["device_id"] = np.full(n, w.cust_device[source])
            cols["device_age_days"] = np.full(n, w.cust_device_age[source], dtype=np.float32)
            cols["ip_prefix"] = np.full(n, w.cust_ip_prefix[source])
            cols["amount"] = np.round(amounts, 2)
            secs = (offs * 86400).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = np.where(rng.random(n) < 0.3, "imps", "upi_p2p").tolist()
            cols["campaign_id"] = [f"LAUND001-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = AdaptiveStructuring()
