"""VY-SOC-002 - deepfake video CFO payment instruction.

Low volume, very high value. The detection problem is the inverse of the consumer scams: there
is no velocity to observe and no escalation, just a single payment that is out of pattern for
the *role* rather than for the population. High-income customers stand in for corporate
treasury operators, since the world models retail entities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.injectors.base import (
    InjectionContext,
    Injector,
    blank_events,
    finalise,
    place_in_window,
    victim_context,
)


class DeepfakeCfoBec(Injector):
    card_id = "VY-SOC-002"
    base_campaigns = 14

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n = ctx.campaigns(self.base_campaigns)

        # Target the top of the income distribution - the proxy for treasury authority.
        pool = np.flatnonzero((w.cust_income_mult >= 2.8) & (~w.cust_is_mule.astype(bool)))
        if len(pool) < n:
            pool = np.flatnonzero(~w.cust_is_mule.astype(bool))
        victims = rng.choice(pool, size=n, replace=False)
        mules = ctx.pick_mules(n)

        # Deliberately during business hours: urgency only works inside the working day.
        hour = np.mod(rng.normal(15.0, 1.6, n) * (1 - ev.hour_blend)
                      + ev.hour_blend * (11.0 + w.cust_circadian[victims]), 24.0)

        cols = blank_events(n)
        cols.update(victim_context(ctx, victims))
        cols["payer_id"] = victims
        cols["payer_account"] = victims
        cols["payee_id"] = mules
        cols["payee_handle"] = [f"acct{int(x):08d}" for x in mules]
        cols["payee_type"] = ["person"] * n
        # Beneficiary is a freshly incorporated entity: days old, not years.
        cols["payee_account_age_days"] = rng.uniform(2, 45, n).astype(np.float32)
        cols["amount"] = rng.lognormal(13.4, 0.7, n) * ev.amount_scale
        cols["ts"] = place_in_window(ctx, victims, hour)
        # Large-value corporate payments run over RTGS/NEFT, and some go cross-border.
        cols["rail"] = np.where(rng.random(n) < 0.25, "rtgs", "neft").tolist()
        cols["channel"] = ["web"] * n
        cols["auth_method"] = ["netbanking"] * n
        cols["payee_city"] = rng.choice(["Singapore", "Dubai", "Hong Kong", "Mumbai"], n).tolist()
        cols["campaign_id"] = [f"SOC002-{i:04d}" for i in range(n)]
        cols["payment_reference"] = rng.choice(
            ["confidential acquisition", "urgent vendor settlement", "project alpha advance"], n
        ).tolist()
        return finalise(cols, self.card_id)


INJECTOR = DeepfakeCfoBec()
