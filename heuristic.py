"""
Nearest-Car-Algorithm (NCA) baseline.

Score each feasible elevator by an "estimated time to pick up"
heuristic:

    score = max(0, busy_until - now)            # already-busy penalty
          + travel_time_from_current_floor_to_lobby
          + queue_length_penalty * len(queue)

The elevator with the lowest score is assigned the new call.

This is the strong-but-simple heuristic RL must beat in order
for the capstone to demonstrate value.
"""

from __future__ import annotations
import numpy as np

from elevator_env import (
    ElevatorEnv,
    ELEVATOR_FLOORS,
    SECONDS_PER_FLOOR,
    DOOR_AND_BOARD_SECONDS,
)


class NearestCarPolicy:
    """Classical NCA dispatch baseline (respects floor permissions)."""

    name = "nearest_car"

    def __init__(self, queue_penalty: float = 6.0):
        self.queue_penalty = queue_penalty

    def act(self, env: ElevatorEnv) -> int:
        call = env._current_call
        if call is None:
            return 0
        mask = env.action_mask()

        best_a, best_score = 0, float("inf")
        for i, ok in enumerate(mask):
            if not ok:
                continue
            ev = env.elevators[i]
            time_free = max(0.0, ev.busy_until - env.t)
            travel = abs(ev.floor - call.origin) * SECONDS_PER_FLOOR
            # how many stops are already queued up
            queue_penalty = self.queue_penalty * len(ev.queue)
            load_penalty = 1.5 * len(ev.load)
            score = time_free + travel + queue_penalty + load_penalty
            if score < best_score:
                best_a, best_score = i, score
        return best_a


class RoundRobinPolicy:
    """Trivial baseline: cycle through feasible elevators."""
    name = "round_robin"

    def __init__(self):
        self._i = 0

    def act(self, env: ElevatorEnv) -> int:
        mask = env.action_mask()
        for _ in range(len(mask)):
            self._i = (self._i + 1) % len(mask)
            if mask[self._i]:
                return self._i
        return 0


class RandomPolicy:
    """Uniform-random over feasible elevators."""
    name = "random"

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def act(self, env: ElevatorEnv) -> int:
        mask = env.action_mask()
        feasible = np.flatnonzero(mask)
        return int(self.rng.choice(feasible))
