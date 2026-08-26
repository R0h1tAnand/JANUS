"""VY-UPI-003 - autopay mandate abuse via deceptive consent.

The victim really did consent, so consent is not the control that failed. What failed is the
gap between the mandate *ceiling* and what the customer believed they were approving: a one
rupee verification debit against a five thousand rupee monthly authority.

The detectable artefact is the ratio, visible at mandate creation - before a single rupee of
loss - which makes this one of the few families a defender can stop pre-emptively.
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
    victim_context,
)


class MandateAbuse(Injector):
    card_id = "VY-UPI-003"
    base_campaigns = 10

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)
        # A fraudulent merchant is a real merchant record with an abnormal mandate portfolio.
        merchants = rng.choice(w.n_merchants, size=n_camp, replace=False)

        frames = []
        for i in range(n_camp):
            n_victims = int(np.clip(rng.integers(30, 120) * ev.burst_scale, 8, 300))
            victims = w.victims_matching(rng, n_victims)
            nv = len(victims)
            # One trivial verification debit, then monthly debits near the ceiling.
            n_debits = max(1, int(w.cfg.days // 30))
            n = nv * (1 + n_debits)
            idx = np.tile(victims, 1 + n_debits)
            seq = np.repeat(np.arange(1 + n_debits), nv)

            cols = blank_events(n)
            cols.update(victim_context(ctx, idx))
            cols.update(merchant_context(ctx, np.full(n, merchants[i])))
            cols["payer_id"] = idx
            cols["payer_account"] = idx
            ceiling = rng.uniform(2000, 9000, nv)
            first = np.ones(nv)
            recurring = np.tile(ceiling * rng.uniform(0.82, 0.99, nv), n_debits)
            cols["amount"] = np.r_[first, recurring] * ev.amount_scale
            day = np.where(seq == 0, rng.uniform(0, 3, n), seq * 30 + rng.uniform(0, 2, n))
            secs = (day * 86400 + rng.uniform(8, 20, n) * 3600).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["upi_mandate"] * n
            cols["auth_method"] = ["none"] * n
            # The verification debit is not itself the loss; the recurring pulls are.
            cols["is_fraud"] = (seq > 0).astype(np.int8)
            cols["campaign_id"] = [f"UPI003-{i:04d}"] * n
            cols["payment_reference"] = ["mandate debit"] * n
            frames.append(finalise(cols, self.card_id))

        out = pd.concat(frames, ignore_index=True)
        out.loc[out.is_fraud == 0, "attack_id"] = ""
        return out


INJECTOR = MandateAbuse()
