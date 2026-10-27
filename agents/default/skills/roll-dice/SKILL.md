---
name: roll-dice
description: >
  Roll one or more dice (generating a random number from 1 to 6 by default).
  Load this skill whenever the user asks to roll a die, roll dice, or pick a
  random dice outcome.
---

# Roll Dice Skill

Use this skill whenever the user asks to roll a die or dice. This reference skill includes both an **in-process Python tool** (`tools.py`) and a **standalone CLI script** (`scripts/roll_dice.py`).

## How to Roll Dice

1. **Approach 1 — In-Process Tool (`tools.py`, Preferred)**:
   - Call `roll_dice(count=1, sides=6)` (either directly or inside `run_code` via `await roll_dice(count=1, sides=6)`).
   - Returns a dictionary containing `count`, `sides`, `rolls` (list of integers from `1` to `sides`), and `total`.

2. **Approach 2 — Standalone CLI Script (`scripts/roll_dice.py`)**:
   - Run the script from the skill directory via the shell tool:
     ```bash
     python3 scripts/roll_dice.py --count 1 --sides 6
     ```

3. Report the rolled value(s) clearly to the user.
