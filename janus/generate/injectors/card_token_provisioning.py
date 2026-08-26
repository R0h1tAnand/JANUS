"""VY-CARD-003 - fraudulent token provisioning to an attacker wallet.

Tokenisation is a genuine security win, and this family exploits exactly that reputation. Once
a card is provisioned into a wallet, its transactions carry device-bound token credentials that
most risk models treat as strong evidence of cardholder presence.

So the fraud arrives wearing the highest-trust instrument on the rail. The only thing out of
place is the provisioning event itself: a brand-new device, in a city the card has no history
in, followed within hours by a contactless spend ramp that a genuine new wallet never shows.
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
)
from janus.generate.world import CITY_NAMES


class TokenProvisioning(Injector):
    card_id = "VY-CARD-003"
    base_campaigns = 40

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns))
        n_camp = len(victims)

        frames = []
        for i in range(n_camp):
            v = int(victims[i])
            n = int(np.clip(rng.integers(4, 12) * ev.burst_scale, 2, 25))
            merch = rng.choice(w.n_merchants, size=n)
            # Provisioning geography deliberately differs from the card's own history.
            far_city = int((w.cust_city[v] + rng.integers(3, len(CITY_NAMES) - 3)) % len(CITY_NAMES))
            start = rng.uniform(0, max(1.0, w.cfg.days - 2))
            # Spend ramps up over hours, not the gentle build a genuine wallet shows.
            offs = start + np.cumsum(rng.uniform(0.02, 0.25, n) * ev.delay_scale)

            cols = blank_events(n)
            cols.update(merchant_context(ctx, merch))
            cols["payer_id"] = np.full(n, v)
            cols["payer_account"] = np.full(n, v)
            cols["amount"] = np.sort(rng.lognormal(8.7, 0.8, n)) * ev.amount_scale
            secs = (offs * 86400).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["card_token"] * n
            cols["channel"] = ["pos"] * n
            cols["auth_method"] = ["token_device"] * n
            cols["device_id"] = np.full(n, 970_000 + i)
            cols["device_age_days"] = np.clip(rng.gamma(1.4, 14.0, n), 0, 400).astype(np.float32)
            cols["ip_prefix"] = np.full(n, int(rng.integers(0, 4096)))
            cols["payer_city"] = [str(CITY_NAMES[far_city])] * n
            cols["campaign_id"] = [f"CARD003-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = TokenProvisioning()
