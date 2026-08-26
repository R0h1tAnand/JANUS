"""VY-SOC-008 - synthetic job-offer task scam.

Distinctive because the campaign *starts by paying the victim*. Small genuine commissions
arrive first, which is what earns the trust that makes the later deposits feel safe.

Those inbound credits are labelled ``is_fraud=0`` deliberately. They are not fraudulent debits
and a bank would never mark them as such - but they carry the campaign id, so the feature store
can learn the grooming signature (small credit in, much larger debit out to the same
counterparty) without the label itself giving the answer away.
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


class TaskScamGrooming(Injector):
    card_id = "VY-SOC-008"
    base_campaigns = 70

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        victims = w.victims_matching(ctx.rng, ctx.campaigns(self.base_campaigns),
                                     age_bands=["18-25", "25-35"])
        n_camp = len(victims)
        mules = ctx.pick_mules(max(1, n_camp // 5))
        assigned = rng.choice(mules, size=n_camp)

        frames = []
        for i in range(n_camp):
            v, mule = int(victims[i]), int(assigned[i])
            n_credit = int(rng.integers(2, 5))
            n_debit = int(np.clip(rng.integers(3, 8) * ev.burst_scale, 2, 12))
            start_day = rng.uniform(0, max(1.0, w.cfg.days - 12))

            # Phase 1: small commissions paid TO the victim.
            credit_amt = rng.uniform(120, 900, n_credit)
            credit_off = np.sort(rng.uniform(0, 4, n_credit))
            # Phase 2: escalating deposits FROM the victim.
            debit_amt = rng.uniform(1500, 4000) * np.cumprod(
                np.r_[1.0, rng.uniform(1.3, 2.0, n_debit - 1)]
            )
            debit_off = 4 + np.sort(rng.uniform(0, 7 * ev.delay_scale, n_debit))

            n = n_credit + n_debit
            cols = blank_events(n)
            cols.update(victim_context(ctx, np.full(n, v)))
            is_credit = np.r_[np.ones(n_credit, bool), np.zeros(n_debit, bool)]
            # On the credit legs the mule is the payer; on the debit legs the victim is.
            cols["payer_id"] = np.where(is_credit, mule, v)
            cols["payer_account"] = cols["payer_id"]
            cols["payee_id"] = np.where(is_credit, v, mule)
            cols["payee_handle"] = [w.vpa(int(x)) for x in cols["payee_id"]]
            cols["payee_account_age_days"] = w.cust_account_age[cols["payee_id"]].astype(np.float32)
            cols["amount"] = np.r_[credit_amt, debit_amt * ev.amount_scale]
            offs = np.r_[credit_off, debit_off] + start_day
            secs = (offs * 86400 + rng.uniform(9, 22, n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            # Only the victim's outbound deposits are the fraud loss.
            cols["is_fraud"] = (~is_credit).astype(np.int8)
            cols["campaign_id"] = [f"SOC008-{i:04d}"] * n
            cols["payment_reference"] = ["task commission"] * n_credit + ["task deposit"] * n_debit
            frames.append(finalise(cols, self.card_id))

        out = pd.concat(frames, ignore_index=True)
        # attack_id only labels the fraudulent legs; the grooming credits stay unattributed.
        out.loc[out.is_fraud == 0, "attack_id"] = ""
        return out


INJECTOR = TaskScamGrooming()
