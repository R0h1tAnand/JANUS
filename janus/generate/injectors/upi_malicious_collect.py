"""VY-UPI-001 - malicious collect-request framed as an inbound refund.

A pull request the victim approves believing they are being paid. The population-level shape is
the giveaway: one requester fires collect requests at hundreds of unrelated VPAs, most are
ignored or declined, and the few that succeed are approved within seconds of arriving - far
faster than someone who actually read the screen.

That low success ratio is the signal a PSP can act on before the victim ever taps approve.
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


class MaliciousCollect(Injector):
    card_id = "VY-UPI-001"
    base_campaigns = 12

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        requesters = ctx.pick_mules(n_camp)

        frames = []
        for i in range(n_camp):
            # Each requester blasts a large batch of VPAs.
            n_targets = int(np.clip(rng.integers(40, 160) * ev.burst_scale, 10, 400))
            victims = w.victims_matching(rng, n_targets)
            n = len(victims)
            cols = blank_events(n)
            cols.update(victim_context(ctx, victims))
            cols["payer_id"] = victims
            cols["payer_account"] = victims
            cols["payee_id"] = np.full(n, requesters[i])
            cols["payee_handle"] = [w.vpa(int(requesters[i]))] * n
            cols["payee_account_age_days"] = np.full(
                n, w.cust_account_age[requesters[i]], dtype=np.float32
            )
            # Refund-shaped amounts: plausible as a cashback, small enough not to alarm.
            cols["amount"] = rng.lognormal(7.4, 0.8, n) * ev.amount_scale
            cols["ts"] = place_in_window(ctx, victims, ctx.attack_hour(victims, 13.0))
            cols["rail"] = ["upi_collect"] * n
            # Most targets ignore or reject the request. Only ~9% approve, and that ratio is
            # itself the detectable artefact - a genuine biller's collect success rate is high.
            approve = rng.random(n) < 0.09
            cols["decision"] = np.where(approve, "approved", "declined").tolist()
            cols["is_fraud"] = approve.astype(np.int8)
            cols["campaign_id"] = [f"UPI001-{i:04d}"] * n
            cols["payment_reference"] = ["refund approval"] * n
            frames.append(finalise(cols, self.card_id))

        out = pd.concat(frames, ignore_index=True)
        out.loc[out.is_fraud == 0, "attack_id"] = ""
        return out


INJECTOR = MaliciousCollect()
