"""
DQN training loop.

Run:
    python train.py --episodes 400 --out dqn.pt

    # pick a device explicitly (default: auto -> cuda if available)
    python train.py --episodes 400 --device cuda

    # ablate back to vanilla DQN for comparison
    python train.py --no-double --no-dueling --no-prioritized

Prints a per-episode summary line, writes the final Q-network
checkpoint to `--out`, and (optionally) a per-episode metrics CSV
to `--log-csv` for plotting with visualize.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import numpy as np
import torch

from elevator_env import ElevatorEnv
from dqn_agent import DQNAgent


def resolve_device(choice: str) -> str:
    """Map 'auto' to cuda when available, else honor the explicit choice."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA is unavailable; "
              "falling back to CPU. Install a CUDA build of torch to use the GPU.")
        return "cpu"
    return choice


def run_episode(env: ElevatorEnv, agent: DQNAgent, train: bool = True):
    obs, _ = env.reset()
    mask = env.action_mask()
    total_r = 0.0
    n_invalid = 0
    losses = []
    while True:
        a = agent.select_action(obs, mask, greedy=not train)
        obs2, r, term, trunc, info = env.step(a)
        mask2 = env.action_mask()
        done = float(term or trunc)
        if info["invalid_action"]:
            n_invalid += 1

        if train:
            # If terminal, mask is all-False (no decision to make).
            # That's fine -- the buffer handles it via the -inf -> 0 clamp.
            agent.buf.push(obs, a, r, obs2, done, mask2)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        total_r += r
        if term or trunc:
            break
        obs, mask = obs2, mask2

    metrics = env.metrics()
    return dict(total_r=total_r, mean_loss=np.mean(losses) if losses else 0.0,
                invalid=n_invalid, **metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dqn.pt")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--log-csv", default=None,
                    help="optional path to write per-episode metrics for plotting")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # algorithm ablation switches (all on by default)
    ap.add_argument("--no-double", dest="double", action="store_false")
    ap.add_argument("--no-dueling", dest="dueling", action="store_false")
    ap.add_argument("--no-prioritized", dest="prioritized", action="store_false")
    ap.set_defaults(double=True, dueling=True, prioritized=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    random.seed(args.seed)  # epsilon-greedy draws and uniform replay sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = ElevatorEnv(seed=args.seed)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        eps_decay_steps=15_000,
        device=device,
        double=args.double,
        dueling=args.dueling,
        prioritized=args.prioritized,
    )

    print(f"device={device}  obs_dim={env.observation_space.shape[0]} "
          f"n_actions={env.action_space.n}")
    print(f"algo: double={args.double} dueling={args.dueling} "
          f"prioritized={args.prioritized}")
    print(f"target arrivals/episode: {sum(env.arrivals_per_5min)}")

    csv_writer = csv_file = None
    if args.log_csv:
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["episode", "total_r", "mean_wait", "p95_wait", "max_wait",
             "served", "unserved", "invalid", "mean_loss", "epsilon"]
        )

    t0 = time.time()
    history = []
    for ep in range(1, args.episodes + 1):
        out = run_episode(env, agent, train=True)
        history.append(out)
        if csv_writer:
            csv_writer.writerow(
                [ep, out["total_r"], out["mean_wait"], out["p95_wait"],
                 out["max_wait"], out["served"], out["unserved"],
                 out["invalid"], out["mean_loss"], agent._epsilon()]
            )
        if ep % args.log_every == 0 or ep == 1:
            recent = history[-args.log_every:]
            wait = np.mean([h["mean_wait"] for h in recent])
            p95 = np.mean([h["p95_wait"] for h in recent])
            served = np.mean([h["served"] for h in recent])
            invalid = np.mean([h["invalid"] for h in recent])
            print(
                f"[ep {ep:4d}] eps={agent._epsilon():.3f} "
                f"R={np.mean([h['total_r'] for h in recent]):.0f}  "
                f"wait_mean={wait:5.1f}s  wait_p95={p95:5.1f}s  "
                f"served={served:.0f}  invalid/ep={invalid:.1f}  "
                f"loss={np.mean([h['mean_loss'] for h in recent]):.3f}"
            )

    agent.save(args.out)
    if csv_file:
        csv_file.close()
        print(f"metrics -> {args.log_csv}")
    print(f"saved -> {args.out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
