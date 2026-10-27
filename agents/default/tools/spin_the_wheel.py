# CodeMode local tool example (`agents/default/tools/spin_the_wheel.py`).
#
# By default, `.py` files in `<agent>/tools/` stay as top-level native tools
# (like `flip_coin.py`). Setting `CODE_MODE = True` in a module routes its
# functions into Pydantic AI Harness's `CodeMode` (`run_code`) sandbox instead,
# where the agent calls them as async Python functions:
#   await spin_the_wheel(choices=["potato", "pizza", "sardine", "taco"])

from __future__ import annotations

import random

CODE_MODE = True


def spin_the_wheel(
    choices: list[str],
    spins: int = 1,
) -> dict[str, int | str | list[str]]:
    """Spin a wheel of user-provided choices and return the randomly selected winner(s).

    Example inside `run_code`:
        await spin_the_wheel(choices=["potato", "pizza", "sardine", "taco"])

    Args:
        choices: List of choices/segments on the wheel (e.g. ["potato", "pizza", "sardine", "taco"]).
        spins: Number of times to spin the wheel (default: 1).
    """
    cleaned = [str(c).strip() for c in choices if str(c).strip()]
    if not cleaned:
        raise ValueError("choices must contain at least one non-empty item")
    if spins < 1:
        raise ValueError("spins must be at least 1")
    results = [random.choice(cleaned) for _ in range(spins)]
    return {
        "choices": cleaned,
        "spins": spins,
        "winner": results[0] if spins == 1 else results,
        "results": results,
    }
