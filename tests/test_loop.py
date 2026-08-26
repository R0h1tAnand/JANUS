"""Closed-loop and text-layer tests."""

from __future__ import annotations

import numpy as np
import pytest

from janus.generate.artifacts import corpus_stats, generate_batch
from janus.generate.injectors.base import EvasionParams
from janus.generate.world import WorldConfig
from janus.loop.arena import rounds_to_detect, run_arena
from janus.loop.red import PARAM_NAMES, crossover, mutate, random_params

TINY = WorldConfig(n_customers=900, n_merchants=150, days=12, seed=3)


def test_evasion_params_stay_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = mutate(random_params(rng), rng, scale=5.0)
        for name in PARAM_NAMES:
            lo, hi = EvasionParams.BOUNDS[name]
            assert lo <= getattr(p, name) <= hi, name


def test_crossover_inherits_only_parent_values():
    rng = np.random.default_rng(1)
    a = EvasionParams(amount_scale=0.2, delay_scale=8.0)
    b = EvasionParams(amount_scale=2.4, delay_scale=0.3)
    child = crossover(a, b, rng)
    assert child.amount_scale in (a.amount_scale, b.amount_scale)
    assert child.delay_scale in (a.delay_scale, b.delay_scale)


def test_corpus_is_present_and_balanced():
    """The committed corpus is a deliverable; an empty one silently disables the text layer."""
    stats = corpus_stats()
    assert stats["scam_scripts"] >= 15
    assert stats["smishing"] >= 10
    assert stats["prompt_injections"] >= 10
    assert stats["benign_conversations"] >= 25, "need enough benign text, including hard negatives"


def test_generated_artifacts_carry_their_source():
    batch = generate_batch(200, seed=0)
    assert all(a.source_id for a in batch)
    assert len({a.source_id for a in batch}) > 20, "recombination should span many seeds"
    assert 0.3 < np.mean([a.is_scam for a in batch]) < 0.7


def test_hard_negatives_exist_in_the_corpus():
    """Without benign text that uses scam vocabulary, the text layer is keyword matching."""
    import json
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in (Path("data/llm_corpus/benign_conversations.jsonl"))
        .read_text().splitlines() if line.strip()
    ]
    hard = [r for r in rows if r.get("kind") == "hard_negative"]
    assert len(hard) >= 10
    vocabulary = " ".join(r["text"].lower() for r in hard)
    for word in ("urgent", "hospital", "refund", "kyc", "otp"):
        assert word in vocabulary, f"hard negatives should legitimately use {word!r}"


@pytest.mark.slow
def test_arena_runs_and_reports_every_round():
    result = run_arena(TINY, rounds=2, generations=1, population=4, progress=False)
    assert len(result.rounds) == 2
    for r in result.rounds:
        assert 0.0 <= r.red_evasion_rate <= 1.0
        assert r.red_value_total > 0
        assert 0.0 <= r.blue_recall_at_fpr <= 1.0
    summary = result.summary()
    assert summary["rounds"] == 2
    assert rounds_to_detect(result) in (None, 1, 2)


@pytest.mark.slow
def test_replay_buffer_preserves_the_class_balance():
    """Blue's training base rate must not drift as Red's discoveries are folded in.

    This is the single easiest way to make an adversarial loop look like it is teaching the
    defender something while actually breaking it. An earlier version appended only Red's fraud
    events: the training set went from 0.6% fraud to 37% after one round and 54% after two, and
    Blue's holdout recall fell from 0.48 to 0.34. Fraud detection is a rare-event problem, and a
    retraining loop that destroys the class balance is measuring nothing.
    """
    result = run_arena(TINY, rounds=3, generations=1, population=4, progress=False)
    assert len(result.rounds) == 3

    rates = [r.train_base_rate for r in result.rounds]
    assert rates[0] < 0.05, f"sanity: this must start as a rare-event problem, got {rates[0]}"
    assert max(rates) < rates[0] * 3, (
        f"training base rate drifted across rounds: {[f'{r:.3%}' for r in rates]}"
    )

    # Blue must not be buying recall with friction either.
    fprs = [r.realised_fpr for r in result.rounds]
    assert max(fprs) <= 0.0015, fprs


@pytest.mark.slow
def test_arena_holds_false_positive_rate_roughly_constant():
    """Blue must not buy recall by becoming indiscriminate.

    Every round recalibrates at a fixed target FPR, so realised FPR should barely move. If it
    climbs across rounds, the loop's apparent improvement is just rising friction.
    """
    result = run_arena(TINY, rounds=2, generations=1, population=4, progress=False)
    fprs = [r.realised_fpr for r in result.rounds]
    assert max(fprs) - min(fprs) < 0.005, fprs
