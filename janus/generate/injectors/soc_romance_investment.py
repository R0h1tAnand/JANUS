"""VY-SOC-004 - long-con romance investment.

The hardest consumer family to catch, because nothing about any single payment is anomalous.
The victim pays willingly, from their own device, at their normal hours, over weeks. Only the
*trajectory* gives it away: a counterparty that did not exist two months ago steadily absorbing
an increasing share of one person's liquidity, often funded by liquidating savings.

Detection therefore has to be cumulative rather than per-transaction, which is exactly the kind
of feature a per-event rule engine cannot express.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.generate.injectors.base import (
    InjectionContext,
    Injector,
    blank_events,
    finalise,
    victim_context,
)


class RomanceInvestment(Injector):
    card_id = "VY-SOC-004"
    base_campaigns = 45

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns),
                                     age_bands=["45-55", "55-70"])
        n_camp = len(victims)
        mules = ctx.pick_mules(max(1, n_camp // 3))
        assigned = rng.choice(mules, size=n_camp)

        rows, camp_ids = [], []
        for i in range(n_camp):
            # Six to eighteen deposits spread across most of the simulation window.
            n_tx = int(np.clip(rng.integers(6, 19) * ev.burst_scale, 3, 30))
            base = rng.lognormal(8.6, 0.6) * w.cust_income_mult[victims[i]] ** 0.5
            # Steady geometric escalation, not a spike - that is what makes it feel safe.
            amounts = base * np.cumprod(np.r_[1.0, rng.uniform(1.10, 1.45, n_tx - 1)])
            span = min(w.cfg.days - 2, 45 * ev.delay_scale)
            offsets = np.sort(rng.uniform(0, max(3.0, span), n_tx))
            rows.append((i, n_tx, amounts, offsets))
            camp_ids.extend([f"SOC004-{i:04d}"] * n_tx)

        total = sum(r[1] for r in rows)
        idx = np.repeat(np.arange(n_camp), [r[1] for r in rows])
        v, mule = victims[idx], assigned[idx]

        cols = blank_events(total)
        cols.update(victim_context(ctx, v))
        cols["payer_id"] = v
        cols["payer_account"] = v
        cols["payee_id"] = mule
        cols["payee_handle"] = [w.vpa(int(x)) for x in mule]
        cols["payee_account_age_days"] = w.cust_account_age[mule].astype(np.float32)
        cols["amount"] = np.concatenate([r[2] for r in rows]) * ev.amount_scale
        day_offsets = np.concatenate([r[3] for r in rows])
        hours = ctx.attack_hour(v, 21.5)
        secs = (day_offsets * 86400 + hours * 3600).astype(np.int64)
        cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
        cols["rail"] = np.where(rng.random(total) < 0.22, "imps", "upi_p2p").tolist()
        cols["campaign_id"] = camp_ids
        cols["payment_reference"] = rng.choice(
            ["trading topup", "margin", "investment", "portfolio addition", "withdrawal fee"], total
        ).tolist()
        return finalise(cols, self.card_id)


INJECTOR = RomanceInvestment()
