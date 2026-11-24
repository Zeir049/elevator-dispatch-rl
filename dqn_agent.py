"""
DQN agent with action masking (PyTorch).

Components
----------
* QNetwork : MLP feature trunk with an optional **dueling** head
             (separate value V(s) and advantage A(s,a) streams),
             outputs Q-values for each of the 4 elevator actions.
* ReplayBuffer / PrioritizedReplayBuffer : experience replay.
  - uniform sampling, or
  - **proportional prioritized** sampling with importance-sampling
    (IS) weights so high-TD-error transitions are revisited more often
    without biasing the gradient.
* DQNAgent : epsilon-greedy exploration with **action masking**
             (invalid elevators receive Q = -inf at selection time
             and a -5 reward penalty during training -- the latter
             is applied inside the environment). Supports
             **Double DQN** targets to curb value overestimation.

The three upgrades over the vanilla MVP -- Double DQN, dueling heads,
prioritized replay -- are each gated behind a flag so the agent can be
ablated back to plain DQN for comparison.

This is still meant to be read top-to-bottom in a capstone review:
one file, no dependencies beyond PyTorch + NumPy.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Q-network                                                                   #
# --------------------------------------------------------------------------- #

class QNetwork(nn.Module):
    """MLP Q-network with an optional dueling head.

    Vanilla mode: trunk -> Linear(hidden, n_actions).
    Dueling mode: trunk -> {V(s) [1], A(s,a) [n_actions]} recombined as
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).
    The mean-subtraction fixes the identifiability of V vs A and is the
    standard "average" aggregator from Wang et al. 2016.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        z = self.trunk(x)
        if self.dueling:
            v = self.value(z)
            a = self.advantage(z)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(z)


# --------------------------------------------------------------------------- #
# Replay buffers                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: float
    mask_next: np.ndarray


def _stack_batch(items):
    s = torch.tensor(np.stack([t.s for t in items]), dtype=torch.float32)
    a = torch.tensor([t.a for t in items], dtype=torch.long).unsqueeze(1)
    r = torch.tensor([t.r for t in items], dtype=torch.float32).unsqueeze(1)
    sn = torch.tensor(np.stack([t.s_next for t in items]), dtype=torch.float32)
    d = torch.tensor([t.done for t in items], dtype=torch.float32).unsqueeze(1)
    m = torch.tensor(np.stack([t.mask_next for t in items]), dtype=torch.bool)
    return s, a, r, sn, d, m


class ReplayBuffer:
    """Uniform-sample replay of (s, a, r, s', done, mask')."""

    prioritized = False

    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch: int):
        items = random.sample(self.buf, batch)
        s, a, r, sn, d, m = _stack_batch(items)
        # weights=1, indices=None keeps a uniform interface with the PER buffer
        weights = torch.ones((batch, 1), dtype=torch.float32)
        return s, a, r, sn, d, m, weights, None

    def update_priorities(self, indices, td_errors):
        pass

    def __len__(self):
        return len(self.buf)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (Schaul et al. 2016).

    Each transition i is sampled with probability
        P(i) = p_i^alpha / sum_j p_j^alpha,   p_i = |td_error_i| + eps
    and corrected with importance-sampling weights
        w_i = (1 / (N * P(i)))^beta,  normalized by max_i w_i.
    `beta` is annealed 0.4 -> 1.0 over training to remove the bias as the
    policy converges. New transitions enter at max priority so they are
    guaranteed at least one replay.
    """

    prioritized = True

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 50_000,
                 eps: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.eps = eps
        self.data: list[Optional[Transition]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self._sample_calls = 0

    def push(self, *args):
        self.data[self.pos] = Transition(*args)
        # new samples get the current max priority (or 1 if buffer is empty)
        max_prio = self.priorities[: self.size].max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _beta(self) -> float:
        frac = min(1.0, self._sample_calls / self.beta_steps)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def sample(self, batch: int):
        self._sample_calls += 1
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(self.size, batch, p=probs)
        items = [self.data[i] for i in idx]

        beta = self._beta()
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        s, a, r, sn, d, m = _stack_batch(items)
        return s, a, r, sn, d, m, weights_t, idx

    def update_priorities(self, indices, td_errors):
        prios = np.abs(td_errors) + self.eps
        self.priorities[indices] = prios

    def __len__(self):
        return self.size


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #

class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 20_000,
        target_sync: int = 500,
        device: str = "cpu",
        *,
        double: bool = True,
        dueling: bool = True,
        prioritized: bool = True,
        buffer_size: int = 50_000,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_sync = target_sync
        self.device = torch.device(device)
        self.double = double
        self.dueling = dueling
        self.prioritized = prioritized

        self.q = QNetwork(obs_dim, n_actions, dueling=dueling).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, dueling=dueling).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = (
            PrioritizedReplayBuffer(buffer_size)
            if prioritized else ReplayBuffer(buffer_size)
        )
        self.train_steps = 0

    # -- epsilon ------------------------------------------------------- #

    def _epsilon(self) -> float:
        frac = min(1.0, self.train_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # -- action selection --------------------------------------------- #

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        feasible = np.flatnonzero(mask)
        if not greedy and random.random() < self._epsilon():
            return int(np.random.choice(feasible))
        s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        qvals = self.q(s).cpu().numpy().squeeze(0)
        qvals_masked = np.where(mask, qvals, -np.inf)
        return int(np.argmax(qvals_masked))

    # -- env-side wrapper for evaluation ------------------------------- #

    def act(self, env) -> int:
        obs = env._build_obs()
        mask = env.action_mask()
        return self.select_action(obs, mask, greedy=True)

    # -- learning ------------------------------------------------------ #

    def update(self, batch_size: int = 64) -> float | None:
        if len(self.buf) < batch_size:
            return None
        s, a, r, sn, d, mn, w, idx = self.buf.sample(batch_size)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        sn = sn.to(self.device); d = d.to(self.device); mn = mn.to(self.device)
        w = w.to(self.device)

        q_pred = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                # Double DQN: ONLINE net picks the action, TARGET net scores it.
                q_next_online = self.q(sn).masked_fill(~mn, float("-inf"))
                next_actions = q_next_online.argmax(dim=1, keepdim=True)
                # guard fully-masked (terminal) rows: argmax of all -inf -> 0,
                # which is harmless because we zero those targets below.
                q_next_target = self.q_target(sn).masked_fill(~mn, float("-inf"))
                q_next_max = q_next_target.gather(1, next_actions)
            else:
                q_next_all = self.q_target(sn).masked_fill(~mn, float("-inf"))
                q_next_max = q_next_all.max(dim=1, keepdim=True).values

            # If no feasible action exists at sn (terminal), clamp to 0
            q_next_max = torch.where(
                torch.isfinite(q_next_max), q_next_max, torch.zeros_like(q_next_max)
            )
            target = r + (1.0 - d) * self.gamma * q_next_max

        td_error = q_pred - target
        # IS-weighted Huber loss (weights are all-ones for the uniform buffer)
        elementwise = F.smooth_l1_loss(q_pred, target, reduction="none")
        loss = (w * elementwise).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        if self.prioritized and idx is not None:
            self.buf.update_priorities(
                idx, td_error.detach().squeeze(1).cpu().numpy()
            )

        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # -- I/O ----------------------------------------------------------- #

    def save(self, path: str):
        torch.save(
            {
                "q": self.q.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions,
                    "dueling": self.dueling,
                    "double": self.double,
                    "prioritized": self.prioritized,
                },
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        # If the checkpoint was trained with a different head, rebuild to match
        # so old (vanilla) and new (dueling) checkpoints both load cleanly.
        ckpt_dueling = cfg.get("dueling", self.dueling)
        if ckpt_dueling != self.dueling:
            self.dueling = ckpt_dueling
            self.q = QNetwork(self.obs_dim, self.n_actions,
                              dueling=ckpt_dueling).to(self.device)
            self.q_target = QNetwork(self.obs_dim, self.n_actions,
                                     dueling=ckpt_dueling).to(self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q"])
