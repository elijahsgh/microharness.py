# Always-on local coin-flip tool (`agents/default/tools/flip_coin.py`).
#
# Files placed in `<agent>/tools/*.py` are loaded eagerly by the `LocalTools`
# capability (`- LocalTools: ./tools` in `agent.yaml`). Unlike skills in
# `<agent>/skills/` (which are deferred until `load_capability` is called),
# public functions defined here are immediately available on every run from
# Step 1 as top-level native tool calls.

from __future__ import annotations

import random


def flip_coin(count: int = 1) -> dict[str, int | list[str]]:
    """Flip a fair two-sided coin one or more times, returning 'heads' or 'tails'.

    Args:
        count: Number of coin flips to perform (default: 1).
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    flips = [random.choice(["heads", "tails"]) for _ in range(count)]
    return {
        "count": count,
        "flips": flips,
        "heads": flips.count("heads"),
        "tails": flips.count("tails"),
    }
