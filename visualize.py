"""
Visualization utilities for the elevator-RL project.

Three sub-commands, all writing PNG/GIF artifacts (no GUI needed):

    # 1. Learning curves from a training CSV (train.py --log-csv)
    python visualize.py curve --csv train_log.csv --out curve.png

    # 2. Policy comparison bar chart (evaluate.py --csv)
    python visualize.py compare --csv compare.csv --out compare.png

    # 3. Time-vs-floor trajectory of the cars for one episode,
    #    driven by the DQN checkpoint (or a heuristic baseline).
    python visualize.py trajectory --ckpt dqn.pt --seed 0 --out traj.png
    python visualize.py trajectory --policy nearest_car --out traj.png
    python visualize.py trajectory --ckpt dqn.pt --gif traj.gif   # animation

These make the learned behavior legible for a capstone presentation.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / file-only backend
import matplotlib.pyplot as plt

from elevator_env import ElevatorEnv, NUM_FLOORS, NUM_ELEVATORS

CAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CAR_LABELS = ["E0 (odd)", "E1 (even-no-2)", "E2 (all)", "E3 (all)"]


# --------------------------------------------------------------------------- #
# CSV loading                                                                 #
# --------------------------------------------------------------------------- #

def _load_csv(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]
            if k != "policy"}
    if "policy" in rows[0]:
        cols["policy"] = [r["policy"] for r in rows]
    return cols


def _smooth(y: np.ndarray, k: int = 11) -> np.ndarray:
    if len(y) < k or k < 2:
        return y
    # edge-pad so the moving average doesn't dip toward zero at the endpoints
    pad = k // 2
    ypad = np.pad(y, pad, mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


# --------------------------------------------------------------------------- #
# 1. Learning curve                                                           #
# --------------------------------------------------------------------------- #

def plot_curve(csv_path: str, out: str):
    d = _load_csv(csv_path)
    ep = d["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    def panel(ax, key, title, color):
        ax.plot(ep, d[key], color=color, alpha=0.25, lw=1)
        ax.plot(ep, _smooth(d[key]), color=color, lw=2, label="smoothed")
        ax.set_title(title); ax.set_xlabel("episode"); ax.grid(alpha=0.3)

    panel(axes[0, 0], "mean_wait", "Mean wait (s)", "#1f77b4")
    panel(axes[0, 1], "p95_wait", "p95 wait (s)", "#d62728")
    panel(axes[1, 0], "served", "Passengers served", "#2ca02c")
    panel(axes[1, 1], "mean_loss", "TD loss", "#9467bd")
    axes[0, 0].legend()
    fig.suptitle("DQN training curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- #
# 2. Policy comparison                                                        #
# --------------------------------------------------------------------------- #

def plot_compare(csv_path: str, out: str):
    d = _load_csv(csv_path)
    names = d["policy"]
    x = np.arange(len(names))
    metrics = [("mean_wait", "Mean wait (s)"),
               ("p95_wait", "p95 wait (s)"),
               ("max_wait", "Max wait (s)"),
               ("unserved", "Unserved")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, title) in zip(axes, metrics):
        bars = ax.bar(x, d[key], color="#4c72b0")
        # highlight the best (lowest) bar
        best = int(np.argmin(d[key]))
        bars[best].set_color("#55a868")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Policy comparison (lower is better; green = best)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- #
# 3. Trajectory                                                               #
# --------------------------------------------------------------------------- #

def _make_policy(args, env):
    if args.policy:
        from heuristic import NearestCarPolicy, RoundRobinPolicy, RandomPolicy
        return {"nearest_car": NearestCarPolicy,
                "round_robin": RoundRobinPolicy,
                "random": RandomPolicy}[args.policy]()
    from dqn_agent import DQNAgent
    agent = DQNAgent(obs_dim=env.observation_space.shape[0],
                     n_actions=env.action_space.n, device="cpu")
    agent.load(args.ckpt)
    return agent


def _rollout_trajectory(env, policy, seed):
    env.reset(seed=seed)
    while env._current_call is not None:
        a = policy.act(env)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return env


def plot_trajectory(args):
    env = ElevatorEnv(seed=args.seed, record_trajectory=True)
    policy = _make_policy(args, env)
    _rollout_trajectory(env, policy, args.seed)
    m = env.metrics()
    label = args.policy or "dqn"

    fig, ax = plt.subplots(figsize=(13, 6))
    for eid in range(NUM_ELEVATORS):
        ts = np.array([p[0] for p in env.trajectory[eid]])
        fl = np.array([p[1] for p in env.trajectory[eid]])
        ax.plot(ts, fl, color=CAR_COLORS[eid], lw=1.8,
                label=CAR_LABELS[eid], alpha=0.9)

    # hall calls as faint markers along the lobby line
    if env.hall_calls:
        ct = [c[0] for c in env.hall_calls]
        cd = [c[2] for c in env.hall_calls]
        ax.scatter(ct, cd, s=12, c="gray", alpha=0.35, marker="x",
                   label="hall calls (dest)")

    ax.set_xlabel("time (s)"); ax.set_ylabel("floor")
    ax.set_yticks(range(1, NUM_FLOORS + 1))
    ax.set_title(f"Car trajectories -- {label}  "
                 f"(mean wait {m['mean_wait']:.0f}s, p95 {m['p95_wait']:.0f}s, "
                 f"served {m['served']:.0f}, unserved {m['unserved']:.0f})")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")

    if args.gif:
        _animate_trajectory(env, label, args.gif)


def _animate_trajectory(env, label, out_gif, frames: int = 120):
    """Sweep a time cursor across the static plot and save a GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    t_max = env.episode_seconds
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, t_max); ax.set_ylim(0.5, NUM_FLOORS + 0.5)
    ax.set_yticks(range(1, NUM_FLOORS + 1))
    ax.set_xlabel("time (s)"); ax.set_ylabel("floor")
    ax.set_title(f"Car trajectories -- {label}")
    ax.grid(alpha=0.3)

    lines = [ax.plot([], [], color=CAR_COLORS[e], lw=1.8,
                     label=CAR_LABELS[e])[0] for e in range(NUM_ELEVATORS)]
    cursor = ax.axvline(0, color="k", lw=1, alpha=0.5)
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    series = []
    for e in range(NUM_ELEVATORS):
        ts = np.array([p[0] for p in env.trajectory[e]])
        fl = np.array([p[1] for p in env.trajectory[e]])
        series.append((ts, fl))

    def update(frame):
        t_now = t_max * frame / (frames - 1)
        for line, (ts, fl) in zip(lines, series):
            sel = ts <= t_now
            line.set_data(ts[sel], fl[sel])
        cursor.set_xdata([t_now, t_now])
        return lines + [cursor]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=20))
    print(f"saved -> {out_gif}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curve", help="plot training curves from train.py --log-csv")
    c.add_argument("--csv", required=True)
    c.add_argument("--out", default="curve.png")

    cmp = sub.add_parser("compare", help="bar chart from evaluate.py --csv")
    cmp.add_argument("--csv", required=True)
    cmp.add_argument("--out", default="compare.png")

    tr = sub.add_parser("trajectory", help="time-vs-floor car trajectories")
    tr.add_argument("--ckpt", default="dqn.pt")
    tr.add_argument("--policy", default=None,
                    choices=["nearest_car", "round_robin", "random"],
                    help="use a heuristic instead of the DQN checkpoint")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="trajectory.png")
    tr.add_argument("--gif", default=None, help="also save an animated GIF")

    args = ap.parse_args()
    if args.cmd == "curve":
        plot_curve(args.csv, args.out)
    elif args.cmd == "compare":
        plot_compare(args.csv, args.out)
    elif args.cmd == "trajectory":
        plot_trajectory(args)


if __name__ == "__main__":
    main()
