"""VY-CARD-009 - synthetic-identity credit bust-out.

The long game, and the family that most directly punishes a model for confusing *good history*
with *low risk*. A synthetic identity behaves impeccably for months - regular spend, full
repayment, never a late fee - precisely so that limits rise. Then every line is drawn down at
once and the identity ceases to exist.

The grooming phase is emitted as legitimate, because it genuinely is: no fraud has occurred
yet. That makes this the strictest test in the suite of whether the detector has learned
anything beyond "trusted customers are safe".
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

BUSTOUT_CATEGORIES = {"jewellery", "electronics", "travel", "digital_goods"}


class SyntheticBustout(Injector):
    card_id = "VY-CARD-009"
    base_campaigns = 28

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        # Thin-file identities: young accounts, sparse history.
        pool = np.flatnonzero(w.cust_account_age < 500)
        if len(pool) < n_camp:
            pool = np.arange(w.n_customers)
        identities = rng.choice(pool, size=min(n_camp, len(pool)), replace=False)
        bust_pool = merchants_in(w, BUSTOUT_CATEGORIES)

        frames = []
        for i, ident in enumerate(identities):
            ident = int(ident)
            # Phase 1 - impeccable behaviour across most of the window.
            n_good = int(rng.integers(18, 40))
            good_day = np.sort(rng.uniform(0, w.cfg.days * 0.82, n_good))
            good_amt = rng.lognormal(7.4, 0.55, n_good)
            good_merch = rng.choice(w.n_merchants, size=n_good)

            # Phase 2 - everything, everywhere, within a day or two.
            n_bust = int(np.clip(rng.integers(6, 16) * ev.burst_scale, 3, 30))
            bust_day = w.cfg.days * 0.9 + np.sort(rng.uniform(0, 1.5 * ev.delay_scale, n_bust))
            bust_amt = rng.lognormal(10.2, 0.6, n_bust) * ev.amount_scale
            bust_merch = rng.choice(bust_pool, size=n_bust)

            n = n_good + n_bust
            cols = blank_events(n)
            cols.update(merchant_context(ctx, np.r_[good_merch, bust_merch]))
            cols.update(attacker_device(ctx, n))
            cols["device_id"] = np.full(n, 960_000 + i)
            cols["device_age_days"] = np.full(n, 400.0, dtype=np.float32)
            cols["payer_id"] = np.full(n, ident)
            cols["payer_account"] = np.full(n, ident)
            cols["amount"] = np.r_[good_amt, bust_amt]
            secs = (np.r_[good_day, bust_day] * 86400 + rng.uniform(8, 22, n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["card_cnp"] * n
            cols["channel"] = ["web"] * n
            cols["auth_method"] = ["otp_3ds"] * n
            cols["is_fraud"] = np.r_[np.zeros(n_good, np.int8), np.ones(n_bust, np.int8)]
            cols["campaign_id"] = [f"CARD009-{i:04d}"] * n
            frames.append(finalise(cols, self.card_id))

        out = pd.concat(frames, ignore_index=True)
        out.loc[out.is_fraud == 0, "attack_id"] = ""
        return out


INJECTOR = SyntheticBustout()
