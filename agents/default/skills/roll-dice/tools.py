# In-process Python tool approach for Agent Skills (`tools.py`).
#
# APPROACH 1: IN-PROCESS TOOL (`tools.py` or `tools/*.py`)
# --------------------------------------------------------
# Files named `tools.py` (or placed inside `tools/*.py`) in a skill folder are
# automatically discovered by `SkillsSpec` and registered as deferred Pydantic AI
# tools (`defer_loading=True`).
# - The tool stays hidden until the agent calls `load_capability(id="roll-dice")`.
# - Once loaded, the agent calls `roll_dice()` directly as a top-level native
#   tool call in-process with zero subprocess overhead and full schema validation.
# - Compare with `scripts/roll_dice.py`, which demonstrates Approach 2 (a
#   standalone CLI script executed via the agent's shell tool). A real skill
#   normally only needs one of these two approaches.

from __future__ import annotations

import random


def roll_dice(count: int = 1, sides: int = 6) -> dict[str, int | list[int]]:
    """Roll one or more dice in-process (default: a single 6-sided die from 1 to 6).

    Args:
        count: Number of dice to roll (default: 1).
        sides: Number of faces per die, generating random integers from 1 to `sides` (default: 6).
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if sides < 2:
        raise ValueError("sides must be at least 2")
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {
        "count": count,
        "sides": sides,
        "rolls": rolls,
        "total": sum(rolls),
    }
