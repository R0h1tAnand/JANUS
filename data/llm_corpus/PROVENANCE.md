# LLM corpus provenance

Everything in this directory was authored by **Claude Opus 5** (`claude-opus-5`) during the
Vyuha build on **2026-08-26**, working from the attack cards in `vyuha/identify/atlas/`.

This is the Tier-1 layer of Vyuha's LLM strategy, and the reason it exists is worth stating
plainly: the generative model runs at **build time**, not at inference time. The corpus is
committed to the repository, so a reviewer can clone this repo with no API key, no network and
no GPU and still see genuine LLM-authored adversarial content driving the demo.

That is also the architecture a real payment system would use. Nothing calls a language model
per authorisation - the latency and cost are prohibitive - so LLMs belong in red-team content
generation and threat ideation, with CPU-bound models on the scoring path.

## Files

| file | records | purpose |
|---|---|---|
| `scam_scripts.jsonl` | voice/chat scam openers and pressure sequences | drives VY-SOC-001/002/004/005/008 |
| `smishing.jsonl` | SMS and chat lures | drives VY-SOC-003, VY-UPI-001/004 |
| `prompt_injections.jsonl` | payloads targeting merchant support agents | drives VY-AGENT-001 |

## Ethics and scope

These are **defensive artefacts**: short, generic lures of the kind already documented in
public fraud advisories, included so a detector can be trained and stress-tested against them.
They name no real person, institution, phone number, handle or brand, and they carry no
working infrastructure - no links, no payment identifiers, no delivery mechanism. They are
training data for the blue team, not an operational toolkit.

## Runtime use

`vyuha.generate.artifacts` samples and recombines these seeds with a seeded RNG, so the
simulator produces far more variation than the corpus contains while staying fully
reproducible from a seed.
