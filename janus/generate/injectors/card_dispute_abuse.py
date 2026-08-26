"""VY-CARD-004 - generative dispute evidence and first-party fraud.

First-party fraud inverts the usual assumption that the cardholder is the victim. The purchase
is entirely genuine - right card, right device, right person - and the fraud happens afterwards,
when the goods are kept and the money is clawed back with an LLM-written narrative and
generatively edited evidence.

Modelled as the refund leg flowing back to a cardholder whose dispute rate is far above the
population's. Nothing about the original purchase is detectable; the signal lives entirely in
the customer's history, which is why this family needs entity-level rather than event-level
features.
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
    merchants_in,
    victim_context,
)

DISPUTE_PRONE = {"electronics", "digital_goods", "travel", "jewellery"}


class DisputeAbuse(Injector):
    card_id = "VY-CARD-004"
    base_campaigns = 35

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        abusers = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns),
                                     age_bands=["18-25", "25-35", "35-45"])
        n_camp = len(abusers)
        pool = merchants_in(w, DISPUTE_PRONE)

        frames = []
        for i in range(n_camp):
            a = int(abusers[i])
            n = int(np.clip(rng.integers(3, 9) * ev.burst_scale, 2, 20))
            merch = rng.choice(pool, size=n)
            purchase = rng.lognormal(9.4, 0.75, n) * ev.amount_scale
            buy_day = np.sort(rng.uniform(0, max(2.0, w.cfg.days - 12), n))
            # The chargeback lands well after the purchase, inside the dispute window.
            refund_day = buy_day + rng.uniform(6, 11, n) * ev.delay_scale

            cols = blank_events(2 * n)
            cols.update(merchant_context(ctx, np.tile(merch, 2)))
            cols.update(victim_context(ctx, np.full(2 * n, a)))
            is_purchase = np.r_[np.ones(n, bool), np.zeros(n, bool)]
            cols["payer_id"] = np.where(is_purchase, a, np.tile(merch, 2) + 1_000_000)
            cols["payer_account"] = cols["payer_id"]
            cols["payee_id"] = np.where(is_purchase, np.tile(merch, 2) + 1_000_000, a)
            cols["amount"] = np.r_[purchase, purchase]
            secs = (np.r_[buy_day, refund_day] * 86400 + rng.uniform(9, 22, 2 * n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["card_cnp"] * (2 * n)
            cols["channel"] = ["web"] * (2 * n)
            cols["auth_method"] = ["otp_3ds"] * (2 * n)
            # Only the clawback is the loss; the purchase itself was genuine.
            cols["is_fraud"] = (~is_purchase).astype(np.int8)
            cols["campaign_id"] = [f"CARD004-{i:04d}"] * (2 * n)
            cols["payment_reference"] = ["purchase"] * n + ["dispute chargeback"] * n
            frames.append(finalise(cols, self.card_id))

        out = pd.concat(frames, ignore_index=True)
        out.loc[out.is_fraud == 0, "attack_id"] = ""
        return out


INJECTOR = DisputeAbuse()
