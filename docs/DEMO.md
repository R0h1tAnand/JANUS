# Demo script

Three minutes, four beats. The through-line: **we built the attack first, and that is why the
defence numbers are worth believing.**

Setup, once:

```bash
make all          # ~15 min: generate, train, evaluate, fidelity, arena
make serve        # terminal 1
make ui           # terminal 2 → http://localhost:5173
```

---

## Beat 1 — the thesis (30s)

Open on **Identify → the kill-chain matrix**.

> "Generative AI didn't make fraud smarter. It made persuasion cheap — and persuasion was the
> bottleneck."

Point at the meters, left to right. Pretext 100%, resource development 95%, trust-building 88%
… monetise 0%.

> "We tagged every step of all 47 techniques for whether GenAI actually enables it. The front of
> the kill chain has been transformed. The back — moving the money — hasn't changed at all,
> because that's still governed by the rail. Which means detection has to shift left."

Click one cell — `SOC-001`, the cloned-voice distress call — to show the kill chain and the
observables a defender could see.

## Beat 2 — the world is real enough to matter (40s)

Switch to **Adapt → fidelity table**.

> "Every number we report comes from data we generated ourselves, so the first question is
> whether that data is worth anything. We don't assert it — we measure it. We train a classifier
> to separate our synthetic payments from real public payment data. If it can't, they're
> interchangeable."

> "It gets to about 0.77. That's 'moderate', not 'indistinguishable', and we report it that way,
> along with the four residuals we know about and chose not to tune away. One of them is a
> genuine finding: the reference set is US card data, and matching its amount skew would mean
> mis-stating Indian ticket sizes."

## Beat 3 — the defence, and where it breaks (60s)

Switch to **Defend → live console**. Let it stream for a few seconds.

> "Every payment is scored on CPU in about {latency}µs — the API measures that, we don't claim
> it. No language model on the authorisation path; nothing could survive the latency."

Click an escalated case.

> "The analyst gets a reason, not a score: money moving to a counterparty with no prior
> relationship, during an active inbound call. Those phrases come from the same signal registry
> the attack cards are written against — so every feature traces back to a technique that asked
> for it."

Now switch to **Adapt → leave-one-attack-out**. Pause here.

> "This is the number we'd most like to hide. Recall at a fixed 0.1% false-positive rate is 89%
> on attack families the model has seen. On families it has *never* seen — we retrain from
> scratch for each one, in a different world — it's about 20%."

> "Supervised fraud detection largely does not transfer to novel attacks. That's the finding.
> It's also the entire reason the rest of this system exists."

## Beat 4 — the loop is the answer (40s)

Stay on **Adapt → Red vs Blue**.

> "So we close the loop. Red treats the deployed model as a black box and evolves attack
> variants — amounts, pacing, how aged a mule account is — optimising for *rupees through*, not
> evasion rate. An attacker who evades everything by sending nothing isn't a threat."

Trace the red line down, the blue line up.

> "Blue retrains on what Red finds. Watch the dashed line — that's the false-positive rate. It
> has to stay flat, or Blue is just buying recall with customer friction. That's the honest
> version of this chart."

Close:

> "Build the attack, then build the defence. You can't guard a gate you've only ever stood
> behind."

---

## If asked

**"Why is your fidelity only 0.77?"** Because it's measured. We could have compared on a schema
that flattered us, or normalised away the differences. The report names the four residuals.

**"Why is unseen-family recall so low?"** Because that is what supervised detection does. Anyone
reporting 95% on novel attacks is either leaking labels or testing on families they trained on.
We shipped a novelty layer and an adversarial loop precisely because 20% isn't good enough.

**"What runs at inference?"** Gradient-boosted trees on CPU. LLMs run at build time — the corpus
they authored is committed to the repo, so this reproduces with no API key and no GPU.

**"Is this deployable?"** The schema maps field-by-field to ISO 8583 and ISO 20022. The binding
constraint isn't the model, it's the 2% ceiling on challenging legitimate customers, and the
policy engine optimises against that explicitly.
