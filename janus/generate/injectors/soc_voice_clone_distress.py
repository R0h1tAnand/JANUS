"""VY-SOC-001 - cloned-voice relative distress call.

Signature this is meant to teach the detector: a low-tenure older victim, an inbound call
overlapping the session, then two to five *escalating* transfers to a payee they have never
paid, at an hour that is wrong for them - and on the receiving side, a mule collecting from
several unrelated victims within a short window.

Every one of those is individually weak. A first-time payee is normal; an evening transfer is
normal; a large amount is normal. The attack is only visible as a conjunction, which is why it
belongs in a learned model rather than a rule.
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


class VoiceCloneDistress(Injector):
    card_id = "VY-SOC-001"
    base_campaigns = 90

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        victims = w.victims_matching(rng, n_camp, age_bands=["55-70", "70+"], tenure="low")
        n_camp = len(victims)
        # Fewer mules than victims, so fan-in emerges rather than being asserted.
        mules = ctx.pick_mules(max(1, n_camp // 4))
        assigned = rng.choice(mules, size=n_camp)

        # Distress calls land in the evening, when the victim is alone and cannot verify.
        start_hour = ctx.attack_hour(victims, natural_hour=20.5)
        start_ts = place_in_window(ctx, victims, start_hour)

        rows, camp_ids = [], []
        for i in range(n_camp):
            n_tx = int(np.clip(rng.integers(2, 6) * ev.burst_scale, 1, 8))
            # Escalation: the first ask is small enough to feel safe, then it ratchets.
            base = rng.lognormal(9.1, 0.55) * w.cust_income_mult[victims[i]] ** 0.5
            amounts = base * np.cumprod(np.r_[1.0, rng.uniform(1.4, 2.3, n_tx - 1)])
            # Gaps grow as the victim hesitates and is talked round again.
            gaps = np.cumsum(rng.uniform(3, 14, n_tx) * ev.delay_scale)
            rows.append((i, n_tx, amounts, gaps))
            camp_ids.extend([f"SOC001-{i:04d}"] * n_tx)

        total = sum(r[1] for r in rows)
        cols = blank_events(total)
        idx = np.repeat(np.arange(n_camp), [r[1] for r in rows])
        v = victims[idx]
        mule = assigned[idx]

        cols.update(victim_context(ctx, v))
        cols["payer_id"] = v
        cols["payer_account"] = v
        cols["payee_id"] = mule
        cols["payee_handle"] = [w.vpa(int(x)) for x in mule]
        cols["payee_account_age_days"] = w.cust_account_age[mule].astype(np.float32)
        cols["amount"] = np.concatenate([r[2] for r in rows]) * ev.amount_scale
        cols["ts"] = np.concatenate(
            [start_ts[r[0]] + (r[3] * 60).astype("timedelta64[s]") for r in rows]
        )
        cols["rail"] = ["upi_p2p"] * total
        # The call is still live while the victim pays - the single strongest contextual signal,
        # and one a real PSP can observe through the handset SDK.
        cols["inbound_call_active"] = np.ones(total, dtype=np.int8)
        cols["campaign_id"] = camp_ids
        cols["payment_reference"] = rng.choice(
            ["hospital", "urgent", "emergency help", "police fine", "treatment advance"], total
        ).tolist()
        return finalise(cols, self.card_id)


INJECTOR = VoiceCloneDistress()
