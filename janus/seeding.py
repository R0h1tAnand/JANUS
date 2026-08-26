"""Stable seed derivation.

Python's builtin ``hash()`` is randomised per process for str and bytes, so
``default_rng(hash(name) % 2**32)`` produces *different* streams on every run. That is exactly
what happened here: the same ``--seed 42`` generated 53,680 training rows with 422 fraud events
in one process and 53,767 rows with 506 in the next, which silently invalidated the claim that
every reported number is reproducible.

The determinism tests did not catch it because they compare two calls inside one process, where
``hash()`` is stable. Only a fresh interpreter exposes it.

Use :func:`derive_seed` for anything that turns a name into a random stream.
"""

from __future__ import annotations

import hashlib

MAX_SEED = 2**32


def derive_seed(*parts: object) -> int:
    """A stable 32-bit seed from any combination of values.

    Deterministic across processes, machines and Python versions - unlike ``hash()``. Floats
    are rounded before hashing so that values differing only in floating-point noise map to the
    same stream, which matters for the Red agent, where a candidate's fitness must be a pure
    function of its parameters.
    """
    tokens = []
    for part in parts:
        if isinstance(part, float):
            tokens.append(f"f:{part:.9g}")
        elif isinstance(part, (tuple, list)):
            tokens.append("(" + ",".join(
                f"f:{p:.9g}" if isinstance(p, float) else str(p) for p in part
            ) + ")")
        else:
            tokens.append(str(part))
    digest = hashlib.blake2b("|".join(tokens).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED
