"""VY-UPI-005 - fake merchant onboarded with synthetic KYB documents.

Acceptance-side rather than payer-side: the payments themselves look ordinary, and the victims
are victims of some *other* scam whose proceeds are being collected here. What is anomalous is
the merchant - days old, taking money from payers scattered across the country with no
geographic coherence, settling out as fast as the aggregator allows, and never issuing a refund.

Included because a defence that only scores payers will never see this at all.
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


class FakeMerchantOnboarding(Injector):
    card_id = "VY-UPI-005"
    base_campaigns = 14

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        frames = []
        for i in range(n_camp):
            n = int(np.clip(rng.integers(40, 170) * ev.burst_scale, 10, 400))
            payers = w.victims_matching(rng, n)
            n = len(payers)
            # Synthetic merchant ids sit above the real merchant range.
            merch_id = 2_000_000 + i

            cols = blank_events(n)
            cols.update(victim_context(ctx, payers))
            cols["payer_id"] = payers
            cols["payer_account"] = payers
            cols["payee_id"] = np.full(n, merch_id)
            cols["payee_handle"] = [f"synthmerch{i:03d}@januspsp"] * n
            cols["payee_type"] = ["merchant"] * n
            cols["merchant_category"] = ["ecommerce"] * n
            cols["mcc"] = np.full(n, 5399)
            # Days old, not years - the single most actionable onboarding signal.
            cols["payee_account_age_days"] = rng.uniform(1, 22, n).astype(np.float32)
            cols["amount"] = rng.lognormal(7.9, 0.95, n) * ev.amount_scale
            cols["ts"] = place_in_window(ctx, payers, ctx.attack_hour(payers, 16.0))
            cols["rail"] = ["upi_p2m"] * n
            # Payers come from everywhere, unlike a real local merchant's customer base.
            cols["payee_city"] = CITY_NAMES[rng.integers(0, len(CITY_NAMES), n)].tolist()
            cols["campaign_id"] = [f"UPI005-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = FakeMerchantOnboarding()
