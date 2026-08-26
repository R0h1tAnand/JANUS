"""VY-CARD-001 - agentic card testing across BIN ranges.

The one family where the *decline* is the product. An agent is not trying to buy anything; it
is separating live cards from dead ones as cheaply as possible, so it fires micro-authorisations
at low-friction merchants and reads the response codes.

Three artefacts fall out of that and none of them are visible on a single transaction: many
distinct PANs behind one device fingerprint, a decline ratio far above any genuine shopper's,
and inter-arrival times that are machine-regular because nothing human is pacing them. The
Red agent can attack the last of these by widening its jitter, which is exactly the kind of
adaptation the closed loop is built to surface.
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
)

LOW_FRICTION = {"digital_goods", "entertainment", "telecom_recharge", "transport"}


class AgenticCardTesting(Injector):
    card_id = "VY-CARD-001"
    base_campaigns = 16

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        pool = merchants_in(w, LOW_FRICTION)

        frames = []
        for i in range(n_camp):
            n = int(np.clip(rng.integers(60, 260) * ev.burst_scale, 20, 600))
            # A contiguous block of cardholders stands in for a contiguous BIN range.
            start_pan = int(rng.integers(0, max(1, w.n_customers - n)))
            pans = np.arange(start_pan, start_pan + n) % w.n_customers
            merch = rng.choice(pool, size=n)

            cols = blank_events(n)
            cols.update(merchant_context(ctx, merch))
            cols.update(attacker_device(ctx, n))
            # One device fingerprint behind the whole batch.
            cols["device_id"] = np.full(n, 950_000 + i)
            cols["ip_prefix"] = np.full(n, int(rng.integers(0, 4096)))
            cols["payer_id"] = pans
            cols["payer_account"] = pans
            # Micro amounts: just enough to trigger an authorisation decision.
            cols["amount"] = np.round(rng.uniform(1, 40, n) * ev.amount_scale, 2)
            # Machine-regular pacing, jittered by delay_scale.
            base_gap = rng.uniform(4, 25) * ev.delay_scale
            gaps = np.cumsum(np.abs(rng.normal(base_gap, base_gap * 0.12, n)))
            start = rng.uniform(0, max(1.0, w.cfg.days - 1)) * 86400
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + (
                (start + gaps).astype(np.int64).astype("timedelta64[s]")
            )
            cols["rail"] = ["card_cnp"] * n
            cols["channel"] = ["api"] * n
            cols["auth_method"] = ["cvv_only"] * n
            # Most cards in a stolen batch are already dead.
            live = rng.random(n) < 0.13
            cols["decision"] = np.where(live, "approved", "declined").tolist()
            cols["is_fraud"] = np.ones(n, dtype=np.int8)
            cols["campaign_id"] = [f"CARD001-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = AgenticCardTesting()
