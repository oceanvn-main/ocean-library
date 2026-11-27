"""Independent prescribed-input Dyna-Q reference (Python standard library only).

The implementation follows Sutton and Barto (2018), Chapter 8.  It deliberately
does not import Ocean code.  Randomness is represented by explicit U[0,1) draws
so MATLAB and Python consume exactly the same variates.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import platform
from pathlib import Path


NUM_STATES = 3
NUM_ACTIONS = 2
ALPHA = 0.5
GAMMA = 0.9
KAPPA = 0.1
TRANSITIONS = (
    (1, 1, 0.0, 2, False),
    (2, 1, 1.0, 3, True),
    (1, 2, 0.2, 1, False),
    (2, 2, -0.1, 1, False),
)
UNIFORM_DRAWS = (0.0, 0.74, 0.34, 0.99)


def pair_id(state: int, action: int) -> int:
    """MATLAB column-major linear index for a state-action table."""
    return state + (action - 1) * NUM_STATES


def target(q, reward: float, next_state: int, terminated: bool) -> float:
    return reward if terminated else reward + GAMMA * max(q[next_state - 1])


def run_dyna():
    q = [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)]
    model = {}
    eligible = []
    updates = []
    selected = []

    def update(origin, state, action, reward, next_state, terminated, real_step):
        old = q[state - 1][action - 1]
        y = target(q, reward, next_state, terminated)
        new = old + ALPHA * (y - old)
        q[state - 1][action - 1] = new
        updates.append({
            "origin": origin, "realStep": real_step,
            "state": state, "action": action, "oldQ": old,
            "target": y, "newQ": new,
            "q": [row[:] for row in q],
        })

    for real_step, transition in enumerate(TRANSITIONS, start=1):
        state, action, reward, next_state, terminated = transition
        update("real", *transition, real_step)
        pid = pair_id(state, action)
        if pid not in model:
            eligible.append(pid)
            count = 0
        else:
            count = model[pid]["count"]
        model[pid] = {
            "state": state, "action": action, "reward": reward,
            "nextState": next_state, "terminated": terminated,
            "count": count + 1,
        }
        position = min(int(UNIFORM_DRAWS[real_step - 1] * len(eligible)), len(eligible) - 1)
        selected_id = eligible[position]
        selected.append(selected_id)
        outcome = model[selected_id]
        update("planning", outcome["state"], outcome["action"],
               outcome["reward"], outcome["nextState"],
               outcome["terminated"], real_step)

    model_rows = [dict(pairID=pid, **model[pid]) for pid in eligible]
    bonuses = [KAPPA * math.sqrt(tau) for tau in (0, 1, 4, 9, 16)]
    return q, updates, selected, model_rows, bonuses


def run_priority_queue():
    heap = []
    generations = {}
    active = {}
    evolution = []
    operations = ((4, 0.5), (2, 0.5), (1, 0.9), (1, 0.4))
    for pid, priority in operations:
        generation = generations.get(pid, 0) + 1
        generations[pid] = generation
        active[pid] = (priority, generation)
        heapq.heappush(heap, (-priority, pid, generation))
        evolution.append({"operation": "push", "pairID": pid, "priority": priority})
    popped = []
    while active:
        while heap:
            neg_priority, pid, generation = heapq.heappop(heap)
            priority = -neg_priority
            if active.get(pid) == (priority, generation):
                del active[pid]
                popped.append({"pairID": pid, "priority": priority})
                evolution.append({"operation": "pop", "pairID": pid, "priority": priority})
                break
    return evolution, popped


def build_result():
    q, updates, selected, model, bonuses = run_dyna()
    queue_evolution, queue_pops = run_priority_queue()
    return {
        "reference": "Sutton and Barto (2018), Chapter 8 prescribed Dyna-Q",
        "pythonVersion": platform.python_version(),
        "parameters": {"numStates": NUM_STATES, "numActions": NUM_ACTIONS,
                       "alpha": ALPHA, "gamma": GAMMA, "kappa": KAPPA},
        "uniformDraws": UNIFORM_DRAWS,
        "selectedPairIDs": selected,
        "updates": updates,
        "finalQ": q,
        "model": model,
        "recencyBonuses": bonuses,
        "queueEvolution": queue_evolution,
        "queuePops": queue_pops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_result(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
