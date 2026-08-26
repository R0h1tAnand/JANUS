"""VY-IDENT-001 - account takeover via AI-assisted SIM swap.

The sequence matters more than any single event. A port-out puts the SMS channel under attacker
control, so credential reset succeeds, a new device binds, and only then does money move. By
the time the payment appears, every authentication factor is legitimately the attacker's.

A detector that scores the payment in isolation has already lost. The window to act is the
gap between the credential change and the first transfer, which is why ``credential_change``
is its own signal family rather than a payment feature.
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
from janus.generate.world import CITY_NAMES


class SimSwapAto(Injector):
    card_id = "VY-IDENT-001"
    base_campaigns = 34

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        # Worth taking over: accounts with balance to sweep.
        pool = np.flatnonzero((w.cust_income_mult >= 1.0) & (~w.cust_is_mule.astype(bool)))
        n_camp = min(ctx.campaigns(self.base_campaigns), len(pool))
        victims = rng.choice(pool, size=n_camp, replace=False)

        frames = []
        for i, v in enumerate(victims):
            v = int(v)
            n = int(np.clip(rng.integers(3, 9) * ev.burst_scale, 2, 20))
            mules = ctx.pick_mules(n)
            start = rng.uniform(0, max(1.0, w.cfg.days - 1))
            # Everything happens inside a few hours, before the victim notices no signal.
            offs = start + np.cumsum(rng.uniform(0.01, 0.09, n) * ev.delay_scale)
            # Escalating sweep: test with a small one, then take the rest.
            amounts = rng.lognormal(8.4, 0.5) * np.cumprod(np.r_[1.0, rng.uniform(1.6, 3.0, n - 1)])

            cols = blank_events(n)
            cols["payer_id"] = np.full(n, v)
            cols["payer_account"] = np.full(n, v)
            cols["payee_id"] = mules
            cols["payee_handle"] = [w.vpa(int(x)) for x in mules]
            cols["payee_account_age_days"] = w.cust_account_age[mules].astype(np.float32)
            cols["amount"] = amounts * ev.amount_scale
            secs = (offs * 86400).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = np.where(rng.random(n) < 0.4, "imps", "upi_p2p").tolist()
            # Freshly bound device on an old account, in the wrong city.
            cols["device_id"] = np.full(n, 940_000 + i)
            cols["device_age_days"] = np.clip(rng.gamma(1.3, 10.0, n), 0, 400).astype(np.float32)
            cols["ip_prefix"] = np.full(n, int(rng.integers(0, 4096)))
            cols["payer_city"] = [str(CITY_NAMES[(w.cust_city[v] + 5) % len(CITY_NAMES)])] * n
            cols["campaign_id"] = [f"IDENT001-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = SimSwapAto()
