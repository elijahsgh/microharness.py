#!/usr/bin/env python3
"""Standalone CLI script approach for Agent Skills (`scripts/roll_dice.py`).

APPROACH 2: STANDALONE CLI SCRIPT (`scripts/*.py`)
--------------------------------------------------
Scripts inside `scripts/` are NOT imported into the Microharness Python process
or registered as Pydantic AI function tools.
- Instead, when the agent loads this skill via `load_capability(id="roll-dice")`,
  it receives the `Skill directory: <path>` header and `SKILL.md` instructions,
  and executes `python3 scripts/roll_dice.py` via its `shell` tool.
- Use this approach when a skill wraps an external CLI, requires a separate
  virtual environment (`uv run --script`), or is written in another language
  (bash, node, etc.).
- Compare with `../tools.py`, which demonstrates Approach 1 (an in-process
  Pydantic AI tool). A real skill normally only needs one of these two approaches.
"""

from __future__ import annotations

import argparse
import json
import random


def roll(count: int = 1, sides: int = 6) -> dict[str, int | list[int]]:
    """Roll `count` dice with `sides` faces (1..sides) and return the results."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll one or more dice (default: 1d6, 1 to 6)."
    )
    parser.add_argument(
        "-n", "--count", type=int, default=1, help="Number of dice to roll (default: 1)"
    )
    parser.add_argument(
        "-s",
        "--sides",
        type=int,
        default=6,
        help="Number of sides per die (default: 6)",
    )
    args = parser.parse_args()
    print(json.dumps(roll(count=args.count, sides=args.sides)))


if __name__ == "__main__":
    main()
