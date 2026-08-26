"""VY-UPI-008 - dormant-account reactivation for laundering capacity.

The most interesting family for feature design, because it is built specifically to defeat the
feature everyone reaches for first. Account age is the classic tenure proxy, and these accounts
are genuinely old - years old, clean history, real KYC. What they lack is *recent behavioural
history*, and that distinction is invisible to any model using account age alone.

If the detector catches this family, tenure is being measured correctly. If it misses it while
catching everything else, the tenure feature is naive.
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


class DormantReactivation(Injector):
    card_id = "VY-UPI-008"
    base_campaigns = 26

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        # Deliberately the oldest, least active accounts available.
        pool = np.flatnonzero(w.cust_account_age > 1800)
        if len(pool) < n_camp:
            pool = np.arange(w.n_customers)
        order = pool[np.argsort(w.cust_rate[pool])][: max(n_camp * 4, 40)]
        accounts = rng.choice(order, size=min(n_camp, len(order)), replace=False)

        frames = []
        for i, acct in enumerate(accounts):
            n = int(np.clip(rng.integers(6, 20) * ev.burst_scale, 3, 45))
            counterparties = ctx.pick_mules(n)
            # Reactivation is abrupt: the whole burst lands inside a couple of days.
            start = rng.uniform(0, max(1.0, w.cfg.days - 3))
            offs = start + np.sort(rng.uniform(0, 2.0 * ev.delay_scale, n))

            cols = blank_events(n)
            cols["payer_id"] = np.full(n, acct)
            cols["payer_account"] = np.full(n, acct)
            cols["payee_id"] = counterparties
            cols["payee_handle"] = [w.vpa(int(x)) for x in counterparties]
            cols["payee_account_age_days"] = w.cust_account_age[counterparties].astype(np.float32)
            # Device was rebound at reactivation - new hardware on a very old account.
            cols["device_id"] = np.full(n, 800_000 + int(acct))
            cols["device_age_days"] = np.clip(rng.gamma(1.4, 14.0, n), 0, 400).astype(np.float32)
            cols["ip_prefix"] = np.full(n, int(rng.integers(0, 4096)))
            cols["amount"] = rng.lognormal(9.6, 0.65, n) * ev.amount_scale
            secs = (offs * 86400 + rng.uniform(0, 24, n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = np.where(rng.random(n) < 0.35, "imps", "upi_p2p").tolist()
            cols["campaign_id"] = [f"UPI008-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = DormantReactivation()
