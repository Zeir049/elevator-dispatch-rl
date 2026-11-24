"""
Elevator dispatch simulator (Gymnasium-style) for the morning-peak scenario.

Building
--------
* 10 floors (1..10), all boarding starts at floor 1 during the rush
* 4 elevators:
    - E0: odd floors only       -> {1, 3, 5, 7, 9}
    - E1: even floors, no 2nd   -> {1, 4, 6, 8, 10}
    - E2: all floors            -> {1..10}
    - E3: all floors            -> {1..10}
* Capacity: 15 passengers per car
* Travel: 2 s per floor, 4 s door+board lump per stop

MDP (event-driven dispatch)
---------------------------
* Decision point: a new hall call arrives at the lobby
  (a passenger appears wanting to go up).
* Action: which of the 4 elevators should serve this call?
  Action space = Discrete(4).
* Observation: a fixed-length vector that summarizes the system
  state at the moment the decision is required.
* Reward: negative cumulative wait time accrued between this
  decision and the next one (sums every waiting passenger's
  current wait, in seconds). An additional terminal penalty
  empties out any remaining wait at the end of the episode.

Arrival model
-------------
Piecewise non-homogeneous Poisson process fit to the 5-minute
bucket data the user supplied:

    08:40-45  -> 36 arrivals
    08:45-50  -> 56
    08:50-55  -> 75
    08:55-09:00 -> 105
    09:00-05  -> 73
    09:05-10  -> 34
    Total      -> 379

Episode = 1800 s (30 min). Destination is sampled uniformly
over floors {2..10}; override `destination_probs` in the
constructor for non-uniform mixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


NUM_FLOORS = 10
NUM_ELEVATORS = 4
CAPACITY = 15
SECONDS_PER_FLOOR = 2.0
DOOR_AND_BOARD_SECONDS = 4.0
EPISODE_SECONDS = 1800

ELEVATOR_FLOORS: List[set] = [
    {1, 3, 5, 7, 9},
    {1, 4, 6, 8, 10},
    set(range(1, NUM_FLOORS + 1)),
    set(range(1, NUM_FLOORS + 1)),
]

DEFAULT_ARRIVALS_PER_5MIN = [36, 56, 75, 105, 73, 34]


@dataclass
class Passenger:
    pid: int
    origin: int
    destination: int
    arrival_time: float
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None


@dataclass
class Elevator:
    eid: int
    floor: int = 1
    busy_until: float = 0.0
    load: List[Passenger] = field(default_factory=list)
    queue: List[int] = field(default_factory=list)
    assigned: List[Passenger] = field(default_factory=list)

    def allowed(self, fl: int) -> bool:
        return fl in ELEVATOR_FLOORS[self.eid]


class ElevatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        arrivals_per_5min: Optional[List[int]] = None,
        destination_probs: Optional[np.ndarray] = None,
        episode_seconds: int = EPISODE_SECONDS,
        seed: Optional[int] = None,
        record_trajectory: bool = False,
    ):
        super().__init__()
        self.arrivals_per_5min = arrivals_per_5min or DEFAULT_ARRIVALS_PER_5MIN
        self.episode_seconds = episode_seconds
        self.record_trajectory = record_trajectory

        if destination_probs is None:
            destination_probs = np.ones(NUM_FLOORS + 1)
            destination_probs[:2] = 0.0
            destination_probs = destination_probs / destination_probs.sum()
        self.destination_probs = destination_probs

        self.action_space = spaces.Discrete(NUM_ELEVATORS)
        obs_dim = NUM_ELEVATORS * 5 + 2 + NUM_FLOORS + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        self._np_random = np.random.default_rng(seed)
        self._current_call: Optional[Passenger] = None
        self._reset_state()

    # -- lifecycle ----------------------------------------------------- #

    def _reset_state(self):
        self.t: float = 0.0
        self.elevators: List[Elevator] = [Elevator(i) for i in range(NUM_ELEVATORS)]
        self.pending: List[Passenger] = []
        self.completed: List[Passenger] = []
        self.next_pid: int = 0
        self._arrival_times = self._generate_arrivals()
        self._arrival_idx: int = 0
        self._current_call = None
        # trajectory[eid] = list of (time, floor); hall_calls = (time, origin, dest)
        self.trajectory: List[List[Tuple[float, int]]] = [
            [(0.0, ev.floor)] for ev in self.elevators
        ]
        self.hall_calls: List[Tuple[float, int, int]] = []

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._reset_state()
        self._advance_to_next_call()
        return self._build_obs(), {}

    # -- arrivals ------------------------------------------------------ #

    def _generate_arrivals(self) -> List[Tuple[float, int]]:
        out = []
        bucket_seconds = 300.0
        for b_idx, n in enumerate(self.arrivals_per_5min):
            t0 = b_idx * bucket_seconds
            times = self._np_random.uniform(t0, t0 + bucket_seconds, size=n)
            for t in times:
                dest = int(self._np_random.choice(
                    np.arange(NUM_FLOORS + 1), p=self.destination_probs))
                out.append((float(t), dest))
        out.sort(key=lambda x: x[0])
        return out

    # -- step ---------------------------------------------------------- #

    def step(self, action: int):
        assert self._current_call is not None

        invalid = not self._is_valid_action(action)
        if invalid:
            action = self._fallback_action()

        passenger = self._current_call
        ev = self.elevators[action]
        self._enqueue(ev, passenger.origin)
        self._enqueue(ev, passenger.destination)
        ev.assigned.append(passenger)

        prev_t = self.t
        self._current_call = None
        self._advance_to_next_call()

        reward = self._interval_wait_reward(prev_t, self.t)
        if invalid:
            reward -= 5.0

        terminated = self._current_call is None and self.t >= self.episode_seconds
        truncated = False

        if terminated:
            for p in self.pending:
                reward -= max(0.0, self.episode_seconds - p.arrival_time)

        obs = self._build_obs()
        info = {
            "t": self.t,
            "invalid_action": invalid,
            "n_pending": len(self.pending),
            "n_completed": len(self.completed),
        }
        return obs, float(reward), terminated, truncated, info

    # -- masking ------------------------------------------------------- #

    def action_mask(self) -> np.ndarray:
        if self._current_call is None:
            return np.ones(NUM_ELEVATORS, dtype=bool)
        dest = self._current_call.destination
        return np.array(
            [dest in ELEVATOR_FLOORS[i] for i in range(NUM_ELEVATORS)],
            dtype=bool,
        )

    def _is_valid_action(self, a: int) -> bool:
        return bool(self.action_mask()[a])

    def _fallback_action(self) -> int:
        mask = self.action_mask()
        best, best_q = 0, math.inf
        for i, ok in enumerate(mask):
            if not ok:
                continue
            q = len(self.elevators[i].queue) + len(self.elevators[i].load)
            if q < best_q:
                best, best_q = i, q
        return best

    # -- simulator ----------------------------------------------------- #

    def _advance_to_next_call(self):
        while True:
            while (self._arrival_idx < len(self._arrival_times)
                   and self._arrival_times[self._arrival_idx][0] <= self.t):
                t_arr, dest = self._arrival_times[self._arrival_idx]
                self._arrival_idx += 1
                p = Passenger(pid=self.next_pid, origin=1,
                              destination=dest, arrival_time=t_arr)
                self.next_pid += 1
                self.pending.append(p)
                self._current_call = p
                if self.record_trajectory:
                    self.hall_calls.append((t_arr, p.origin, dest))
                return

            next_arrival = (
                self._arrival_times[self._arrival_idx][0]
                if self._arrival_idx < len(self._arrival_times)
                else math.inf
            )
            ev_busy_events = [
                ev.busy_until for ev in self.elevators
                if (ev.queue or ev.load) and ev.busy_until > self.t
            ]
            ev_ready_events = [
                self.t for ev in self.elevators
                if ev.queue and ev.busy_until <= self.t
            ]
            next_ev_event = min(ev_busy_events + ev_ready_events, default=math.inf)

            t_next = min(next_arrival, next_ev_event, self.episode_seconds)
            if t_next == math.inf or t_next >= self.episode_seconds:
                self.t = self.episode_seconds
                self._current_call = None
                return

            self.t = t_next
            for ev in self.elevators:
                if ev.busy_until <= self.t and ev.queue:
                    self._service_one_stop(ev)

    def _service_one_stop(self, ev: Elevator):
        target = ev.queue.pop(0)
        travel = 0.0 if target == ev.floor else abs(target - ev.floor) * SECONDS_PER_FLOOR
        start_t = max(ev.busy_until, self.t)
        arrived_at = start_t + travel
        if self.record_trajectory and target != ev.floor:
            # log the departure and arrival so plots show the move as a ramp
            self.trajectory[ev.eid].append((start_t, ev.floor))
            self.trajectory[ev.eid].append((arrived_at, target))
        ev.floor = target

        new_load = []
        for p in ev.load:
            if p.destination == target:
                p.dropoff_time = arrived_at
            else:
                new_load.append(p)
        ev.load = new_load

        kept_pending = []
        boarded_now = []
        for p in self.pending:
            if (p.origin == target
                    and p in ev.assigned
                    and len(ev.load) < CAPACITY):
                p.pickup_time = arrived_at
                ev.load.append(p)
                boarded_now.append(p)
                self.completed.append(p)
            else:
                kept_pending.append(p)
        self.pending = kept_pending
        for p in boarded_now:
            ev.assigned.remove(p)

        ev.busy_until = arrived_at + DOOR_AND_BOARD_SECONDS

    def _enqueue(self, ev: Elevator, floor: int):
        if not ev.queue or ev.queue[-1] != floor:
            ev.queue.append(floor)
        if ev.busy_until < self.t:
            ev.busy_until = self.t

    def _interval_wait_reward(self, t0: float, t1: float) -> float:
        total = 0.0
        for p in self.pending + self.completed:
            start = max(p.arrival_time, t0)
            end = min(p.pickup_time, t1) if p.pickup_time is not None else t1
            if end > start:
                total += end - start
        return -total

    # -- observation --------------------------------------------------- #

    def _build_obs(self) -> np.ndarray:
        feats = []
        for ev in self.elevators:
            feats.extend([
                ev.floor / NUM_FLOORS,
                len(ev.load) / CAPACITY,
                len(ev.queue) / NUM_FLOORS,
                max(0.0, ev.busy_until - self.t) / 60.0,
                float(ev.allowed(self._current_call.destination))
                if self._current_call else 0.0,
            ])

        if self._current_call is not None:
            feats.append(self._current_call.destination / NUM_FLOORS)
            feats.append(self._current_call.origin / NUM_FLOORS)
        else:
            feats.extend([0.0, 0.0])

        floor_counts = np.zeros(NUM_FLOORS, dtype=np.float32)
        for p in self.pending:
            floor_counts[p.destination - 1] += 1
        feats.extend((floor_counts / 10.0).tolist())

        phase = 2 * math.pi * self.t / self.episode_seconds
        feats.append(math.sin(phase))
        feats.append(math.cos(phase))

        return np.array(feats, dtype=np.float32)

    # -- metrics ------------------------------------------------------- #

    def metrics(self) -> Dict[str, float]:
        waits = []
        for p in self.completed:
            if p.pickup_time is not None:
                waits.append(p.pickup_time - p.arrival_time)
        unserved = len(self.pending)
        unserved_wait = [self.episode_seconds - p.arrival_time for p in self.pending]
        all_waits = waits + unserved_wait
        if not all_waits:
            return dict(mean_wait=0.0, p95_wait=0.0, max_wait=0.0,
                        served=0, unserved=unserved)
        return dict(
            mean_wait=float(np.mean(all_waits)),
            p95_wait=float(np.percentile(all_waits, 95)),
            max_wait=float(np.max(all_waits)),
            served=len(waits),
            unserved=unserved,
        )
