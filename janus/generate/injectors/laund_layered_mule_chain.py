"""VY-LAUND-002 - layered mule chain with randomised hop depth.

A chain rather than a star: value hops account to account with randomised depth, delay and split
ratio, so tracing rules that follow a fixed number of hops lose the trail before the exit node.

The feature that survives is structural - chain depth and the near-total absence of balance
retention at every intermediate node - and it is only computable on a transaction graph. This
family exists in the suite specifically to justify the graph layer earning its place.
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


class LayeredMuleChain(Injector):
    card_id = "VY-LAUND-002"
    base_campaigns = 24

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        frames = []
        for i in range(n_camp):
            depth = int(rng.integers(3, 8))
            chain = ctx.pick_mules(depth + 1)
            value = rng.lognormal(11.4, 0.6) * ev.amount_scale
            start = rng.uniform(0, max(1.0, w.cfg.days - 3))

            payers, payees, amounts, times = [], [], [], []
            t = start
            for hop in range(depth):
                # Each hop splits into two or three legs at non-round ratios.
                n_legs = int(rng.integers(2, 4))
                fracs = rng.dirichlet(np.ones(n_legs) * 3.0)
                leg_amounts = value * fracs * rng.uniform(0.95, 0.99)
                for k in range(n_legs):
                    payers.append(int(chain[hop]))
                    payees.append(int(chain[hop + 1]))
                    amounts.append(leg_amounts[k])
                    times.append(t + rng.uniform(0.002, 0.03) * ev.delay_scale)
                value = leg_amounts.sum()
                t += rng.uniform(0.01, 0.12) * ev.delay_scale

            n = len(payers)
            cols = blank_events(n)
            payer_arr = np.array(payers)
            cols["payer_id"] = payer_arr
            cols["payer_account"] = payer_arr
            cols["payee_id"] = np.array(payees)
            cols["payee_handle"] = [w.vpa(p) for p in payees]
            cols["payee_account_age_days"] = w.cust_account_age[np.array(payees)].astype(np.float32)
            cols["device_id"] = w.cust_device[payer_arr]
            cols["device_age_days"] = w.cust_device_age[payer_arr].astype(np.float32)
            cols["ip_prefix"] = w.cust_ip_prefix[payer_arr]
            cols["amount"] = np.round(np.array(amounts), 2)
            secs = (np.array(times) * 86400).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = np.where(rng.random(n) < 0.45, "imps", "upi_p2p").tolist()
            cols["campaign_id"] = [f"LAUND002-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = LayeredMuleChain()
