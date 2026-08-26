# Scope and ethics

Janus generates fraud in order to detect it. That deserves an explicit statement of what is and
is not in this repository.

## What this produces

**Synthetic payment data.** Every customer, merchant, account, device and transaction is
generated from a seed. No record corresponds to a real person, a real institution, or a real
payment. Nothing here derives from a breach, a leak, or scraped personal data.

**Short adversarial text samples.** `data/llm_corpus/` contains brief scam openers, SMS lures
and prompt-injection payloads of the kind already documented in public fraud advisories from
the FBI IC3, the APWG, NPCI and RBI. They exist so a classifier has something to learn from.

## What it deliberately does not produce

- **No working infrastructure.** No links, no domains, no phone numbers, no payment
  identifiers, no delivery mechanism. The text samples are strings in a JSONL file with no path
  to a victim.
- **No real identities or brands.** No sample names a real person, bank, merchant or public
  figure. Institutional placeholders ("your bank", "the electricity board") are used instead.
- **No voice, video or document synthesis.** The atlas *describes* deepfake and document-forgery
  techniques because a defender needs to know they exist and what they leave behind. The code
  generates none of it, and ships no model or pipeline that could.
- **No evasion service.** The Red agent searches attack parameters against a detector that
  exists only inside this repository. It is not portable to a real fraud model, and its outputs
  are attack *descriptions* consumed by our own trainer.

## Why the attack side exists at all

A defence can only be evaluated against attacks. Every detection number in this project is
measured against something the system itself generated, and the honest ones — a 20% recall on
attack families the model has never seen — only exist because we built the attacks first and
refused to grade ourselves on the easy ones.

The asymmetry is the point: the attacker only needs one technique to work, so the defender has
to enumerate. Publishing the enumeration helps the defender far more than the attacker, who
already knows.

## If you build on this

The atlas and the simulator are useful for training and stress-testing fraud defences. They are
not a red-team toolkit and will not function as one. If you extend the generators toward real
delivery mechanisms — live messaging, real payment rails, actual media synthesis — that is a
different project with different obligations, and it should not carry this one's name.
