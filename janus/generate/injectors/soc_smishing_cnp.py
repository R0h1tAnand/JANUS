"""VY-SOC-003 - LLM-personalised smishing to card-not-present cashout.

The card is genuine and the 3-D Secure challenge genuinely succeeds, because the OTP is relayed
in real time. What does *not* match is everything around the authentication: an unseen device,
a merchant the cardholder has never used, high-value digital goods bought first rather than
after any relationship, and several attempts in quick succession before the issuer reacts.

This is the family that most punishes a detector which treats "3DS passed" as evidence of
legitimacy - the liability shift says the issuer is on the hook, and the model has to disagree
with the authentication result to save the money.
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

#: Categories that convert to cash fastest, and so are where stolen cards get spent.
RESALE_CATEGORIES = {"digital_goods", "electronics", "entertainment", "travel"}


class SmishingCnp(Injector):
    card_id = "VY-SOC-003"
    base_campaigns = 120

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns),
                                     age_bands=["25-35", "35-45"])
        n_camp = len(victims)
        merch_pool = merchants_in(w, RESALE_CATEGORIES)

        # Bursts of one to four purchases before the card is frozen.
        per = np.clip((rng.integers(1, 5, n_camp) * ev.burst_scale).round(), 1, 8).astype(int)
        total = int(per.sum())
        idx = np.repeat(np.arange(n_camp), per)
        merch = rng.choice(merch_pool, size=total)

        # The OTP relay window is minutes; delays here are in minutes, not hours.
        offsets = np.concatenate([np.cumsum(rng.uniform(1, 9, k) * ev.delay_scale) for k in per])
        start = place_in_window(ctx, victims, ctx.attack_hour(victims, 21.0))

        cols = blank_events(total)
        cols.update(merchant_context(ctx, merch))
        cols.update(attacker_device(ctx, total))
        cols["payer_id"] = victims[idx]
        cols["payer_account"] = victims[idx]
        cols["amount"] = rng.lognormal(8.9, 0.75, total) * ev.amount_scale
        cols["ts"] = start[idx] + (offsets * 60).astype("timedelta64[s]")
        cols["rail"] = ["card_cnp"] * total
        cols["channel"] = ["web"] * total
        cols["auth_method"] = ["otp_3ds"] * total
        cols["payer_city"] = CITY_NAMES[w.cust_city[victims[idx]]].tolist()
        # Issuers do already stop a share of these; the model must see the declines too,
        # otherwise "declined" becomes a spuriously clean fraud indicator.
        cols["decision"] = np.where(rng.random(total) < 0.18, "declined", "approved").tolist()
        cols["campaign_id"] = [f"SOC003-{i:04d}" for i in idx]
        return finalise(cols, self.card_id)


INJECTOR = SmishingCnp()
