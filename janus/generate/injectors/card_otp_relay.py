"""VY-CARD-002 - real-time OTP relay against 3-D Secure.

Every authentication control passes. The OTP was genuinely delivered to the genuine cardholder's
genuine handset and genuinely entered - just not by them, and not on their device. Under scheme
rules the liability shifts to the issuer, so the model has to be willing to decline a
transaction that authentication has already blessed.

The contradiction is the signal: successful step-up from a device that has never been seen,
at a merchant never used, converting immediately into resellable value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.injectors.base import (
    InjectionContext,
    Injector,
    attacker_device,
    blank_events,
    finalise,
    merchant_context,
    merchants_in,
    place_in_window,
)
from janus.generate.world import CITY_NAMES

CASHOUT_CATEGORIES = {"digital_goods", "electronics", "jewellery", "travel"}


class OtpRelay(Injector):
    card_id = "VY-CARD-002"
    base_campaigns = 70

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns))
        n_camp = len(victims)
        pool = merchants_in(w, CASHOUT_CATEGORIES)

        per = np.clip((rng.integers(1, 4, n_camp) * ev.burst_scale).round(), 1, 6).astype(int)
        total = int(per.sum())
        idx = np.repeat(np.arange(n_camp), per)
        merch = rng.choice(pool, size=total)
        offsets = np.concatenate([np.cumsum(rng.uniform(2, 11, k) * ev.delay_scale) for k in per])
        start = place_in_window(ctx, victims, ctx.attack_hour(victims, 19.0))

        cols = blank_events(total)
        cols.update(merchant_context(ctx, merch))
        cols.update(attacker_device(ctx, total))
        cols["payer_id"] = victims[idx]
        cols["payer_account"] = victims[idx]
        cols["amount"] = rng.lognormal(9.6, 0.7, total) * ev.amount_scale
        cols["ts"] = start[idx] + (offsets * 60).astype("timedelta64[s]")
        cols["rail"] = ["card_cnp"] * total
        cols["channel"] = ["web"] * total
        cols["auth_method"] = ["otp_3ds"] * total
        # Step-up was demanded and step-up succeeded. That is the whole problem.
        cols["step_up_required"] = np.ones(total, dtype=np.int8)
        cols["decision"] = ["approved"] * total
        cols["payer_city"] = CITY_NAMES[w.cust_city[victims[idx]]].tolist()
        cols["campaign_id"] = [f"CARD002-{i:04d}" for i in idx]
        return finalise(cols, self.card_id)


INJECTOR = OtpRelay()
