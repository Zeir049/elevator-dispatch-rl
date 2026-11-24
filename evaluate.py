"""
Evaluate DQN against the heuristic baselines.

Usage:
    python evaluate.py --ckpt dqn.pt --episodes 20
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from elevator_env import ElevatorEnv
from heuristic import NearestCarPolicy, RoundRobinPolicy, RandomPolicy
from dqn_agent import DQNAgent


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        return "cpu"
    return choice


def rollout(env: ElevatorEnv, policy, seed: int):
    env.reset(seed=seed)
    while env._current_call is not None:
        a = policy.act(env)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return env.metrics()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dqn.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv", default=None,
                    help="optional path to write the comparison table for plotting")
    args = ap.parse_args()

    device = resolve_device(args.device)
    env = ElevatorEnv(seed=0)

    rng = np.random.default_rng(42)
    policies = [
        RandomPolicy(rng=rng),
        RoundRobinPolicy(),
        NearestCarPolicy(),
    ]

    # Try to load DQN; skip silently if it's not there yet
    try:
        dqn = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            device=device,
        )
        dqn.load(args.ckpt)
        policies.append(dqn)
    except FileNotFoundError:
        print(f"(no checkpoint at {args.ckpt}; skipping DQN)")

    seeds = list(range(args.episodes))

    print(f"{'policy':<14} {'mean_wait':>10} {'p95_wait':>10} {'max_wait':>10} "
          f"{'served':>8} {'unserved':>9}")
    print("-" * 64)
    table = []
    for pol in policies:
        rows = [rollout(env, pol, s) for s in seeds]
        agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        table.append((pol.name, agg))
        print(f"{pol.name:<14} {agg['mean_wait']:>9.1f}s {agg['p95_wait']:>9.1f}s "
              f"{agg['max_wait']:>9.1f}s {agg['served']:>8.1f} {agg['unserved']:>9.1f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["policy", "mean_wait", "p95_wait", "max_wait",
                        "served", "unserved"])
            for name, agg in table:
                w.writerow([name, agg["mean_wait"], agg["p95_wait"],
                            agg["max_wait"], agg["served"], agg["unserved"]])
        print(f"comparison -> {args.csv}")


if __name__ == "__main__":
    main()
