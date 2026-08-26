"""VY-UPI-002 - QR sticker substitution at merchant premises.

A physical-world attack with a clean digital signature: a *personal* VPA, days old, suddenly
receiving retail-shaped payments from dozens of unrelated payers who are all standing in the
same place. Real person-to-person payment graphs do not look like that; merchant graphs do,
but merchants have merchant accounts.

The mismatch between account type and traffic shape is what the detector should learn.
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
from janus.generate.world import CITY_NAMES


class QrSubstitution(Injector):
    card_id = "VY-UPI-002"
    base_campaigns = 18

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        collectors = ctx.pick_mules(n_camp)

        frames = []
        for i in range(n_camp):
            city = int(rng.integers(0, len(CITY_NAMES)))
            # Payers are drawn from one city: everyone who walked past the same shop.
            local = np.flatnonzero((w.cust_city == city) & (~w.cust_is_mule.astype(bool)))
            if len(local) < 12:
                local = np.flatnonzero(~w.cust_is_mule.astype(bool))
            n = int(np.clip(rng.integers(25, 90) * ev.burst_scale, 8, 250))
            payers = rng.choice(local, size=min(n, len(local)), replace=False)
            n = len(payers)

            cols = blank_events(n)
            cols.update(victim_context(ctx, payers))
            cols["payer_id"] = payers
            cols["payer_account"] = payers
            cols["payee_id"] = np.full(n, collectors[i])
            cols["payee_handle"] = [w.vpa(int(collectors[i]))] * n
            # Freshly created handle, rotated before it accumulates history.
            cols["payee_account_age_days"] = rng.uniform(1, 9, n).astype(np.float32)
            # Retail ticket sizes - this is the shape a real shop's takings have.
            cols["amount"] = rng.lognormal(5.9, 0.85, n) * ev.amount_scale
            cols["ts"] = place_in_window(ctx, payers, ctx.attack_hour(payers, 18.0))
            cols["rail"] = ["upi_p2m"] * n
            # Declared as a person-to-person payee despite merchant-shaped traffic.
            cols["payee_type"] = ["person"] * n
            cols["payee_city"] = [str(CITY_NAMES[city])] * n
            cols["campaign_id"] = [f"UPI002-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = QrSubstitution()
