"""VY-UPI-007 - coordinated mule VPA farm with agentic orchestration.

Pure laundering structure, no victims in frame. Proceeds arrive at first-layer mules from many
unrelated payers, then leave within minutes, split across downstream accounts at deliberately
non-round ratios so that no fixed structuring rule matches.

The account-level view sees nothing remarkable. The graph view sees a node with near-zero
balance retention and a rapidly churning counterparty set, which is the whole point of building
graph features rather than only per-account aggregates.
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


class MuleFarm(Injector):
    card_id = "VY-UPI-007"
    base_campaigns = 22

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        frames = []
        for i in range(n_camp):
            collector = int(ctx.pick_mules(1)[0])
            n_in = int(np.clip(rng.integers(8, 26) * ev.burst_scale, 4, 60))
            payers = w.victims_matching(rng, n_in)
            n_in = len(payers)
            inflow = rng.lognormal(9.3, 0.7, n_in) * ev.amount_scale
            start = rng.uniform(0, max(1.0, w.cfg.days - 2))
            in_off = np.sort(rng.uniform(0, 0.35, n_in))

            # Fan out to downstream mules within minutes, at non-round split ratios.
            n_out = int(rng.integers(3, 8))
            downstream = ctx.pick_mules(n_out)
            splits = rng.dirichlet(np.ones(n_out) * 2.2)
            outflow = inflow.sum() * splits * rng.uniform(0.93, 0.985)
            out_off = in_off.max() + rng.uniform(0.005, 0.05, n_out) * ev.delay_scale

            n = n_in + n_out
            cols = blank_events(n)
            is_in = np.r_[np.ones(n_in, bool), np.zeros(n_out, bool)]
            cols["payer_id"] = np.r_[payers, np.full(n_out, collector)]
            cols["payer_account"] = cols["payer_id"]
            cols["payee_id"] = np.r_[np.full(n_in, collector), downstream]
            cols["payee_handle"] = [w.vpa(int(x)) for x in cols["payee_id"]]
            cols["payee_account_age_days"] = w.cust_account_age[cols["payee_id"]].astype(np.float32)
            cols["device_id"] = w.cust_device[cols["payer_id"]]
            cols["device_age_days"] = w.cust_device_age[cols["payer_id"]].astype(np.float32)
            cols["ip_prefix"] = w.cust_ip_prefix[cols["payer_id"]]
            cols["amount"] = np.r_[inflow, outflow]
            offs = start + np.r_[in_off, out_off]
            secs = (offs * 86400 + rng.uniform(0, 24, n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = np.where(is_in, "upi_p2p", "imps").tolist()
            cols["campaign_id"] = [f"UPI007-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = MuleFarm()
