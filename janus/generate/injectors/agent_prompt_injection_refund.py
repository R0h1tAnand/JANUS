"""VY-AGENT-001 - prompt injection of a merchant support agent holding payment tools.

An AI-native family with no human victim and no compromised credential. The support LLM has a
refund tool and a helpfulness bias, and the attacker supplies text - in chat, or smuggled into
an order note the agent reads later - that reframes an unwarranted payout as policy-compliant.

Two artefacts a payment system can actually see: refunds that exceed the original capture, and
the same beneficiary appearing across support cases that share no customer. Neither requires
understanding the conversation, which matters because the conversation is not on the rail.
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


class PromptInjectionRefund(Injector):
    card_id = "VY-AGENT-001"
    base_campaigns = 20

    def generate(self, ctx: InjectionContext) -> pd.DataFrame:
        w, rng, ev = ctx.world, ctx.rng, ctx.evasion
        n_camp = ctx.campaigns(self.base_campaigns)

        frames = []
        for i in range(n_camp):
            merch = int(rng.choice(w.n_merchants))
            n = int(np.clip(rng.integers(5, 18) * ev.burst_scale, 3, 40))
            # Payouts are spread across unrelated accounts to avoid a per-customer refund spike,
            # but they converge on a small set of beneficiaries the operator controls.
            claimants = w.victims_matching(rng, n)
            n = len(claimants)
            beneficiaries = ctx.pick_mules(max(1, n // 4))
            assigned = rng.choice(beneficiaries, size=n)

            cols = blank_events(n)
            cols.update(merchant_context(ctx, np.full(n, merch)))
            cols["payer_id"] = np.full(n, merch + 1_000_000)
            cols["payer_account"] = cols["payer_id"]
            cols["payee_id"] = assigned
            cols["payee_handle"] = [w.vpa(int(x)) for x in assigned]
            cols["payee_type"] = ["person"] * n
            cols["payee_account_age_days"] = w.cust_account_age[assigned].astype(np.float32)
            # Each payout sits just under a plausible human-review threshold.
            cols["amount"] = np.round(rng.uniform(1800, 4900, n) * ev.amount_scale, 2)
            start = rng.uniform(0, max(1.0, w.cfg.days - 2))
            offs = start + np.cumsum(rng.uniform(0.01, 0.4, n) * ev.delay_scale)
            secs = (offs * 86400).astype(np.int64)
            cols["ts"] = np.datetime64("2026-01-01T00:00:00") + secs.astype("timedelta64[s]")
            cols["rail"] = ["upi_p2m"] * n
            cols["channel"] = ["api"] * n
            cols["auth_method"] = ["none"] * n
            # The defining field: no human was in the loop when the money moved.
            cols["agent_initiated"] = np.ones(n, dtype=np.int8)
            cols["campaign_id"] = [f"AGENT001-{i:04d}"] * n
            cols["payment_reference"] = ["goodwill refund"] * n
            frames.append(finalise(cols, self.card_id))

        return pd.concat(frames, ignore_index=True)


INJECTOR = PromptInjectionRefund()
