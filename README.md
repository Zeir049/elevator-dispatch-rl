# elevator-dispatch-rl

**A DQN that dispatches four elevators in a simulated 10-floor building during a morning rush, compared with a nearest-car heuristic** (capstone project).
The simulator is a small event-driven Python environment with a Gymnasium-style API. Lobby arrivals are generated from six 5-minute passenger counts (08:40–09:10, 379 passengers in total). Every time a passenger arrives, the agent picks the elevator that will serve them. The DQN is trained with 5 training seeds and compared with nearest-car, round-robin and random dispatching on the same 20 arrival seeds.

## What is implemented

| Component | File | Notes |
|---|---|---|
| Simulator | `elevator_env.py` | 10 floors, 4 cars with floor restrictions, capacity 15, event-driven: one decision per arriving passenger |
| Action masking | `elevator_env.py`, `dqn_agent.py` | `action_mask()` marks cars that can reach the passenger's floor; masked Q-values are set to `-inf` when acting and in the Bellman target |
| Baselines | `heuristic.py` | Nearest-car (NCA) score, round-robin, uniform random (all respect the floor restrictions) |
| DQN | `dqn_agent.py` | MLP Q-network; **Double DQN**, **dueling head** and **prioritized replay**, each with an on/off flag |
| Training | `train.py` | ε-greedy training loop, per-episode CSV log |
| Evaluation | `evaluate.py` | Greedy rollouts on seeds 0–19, mean / p95 / max wait, served / unserved |
| Multi-seed runs | `run_multiseed.sh`, `aggregate_seeds.py` | Trains both DQN variants with seeds 1–5, evaluates each checkpoint with `evaluate.py`, and aggregates the results to `compare_multiseed.csv` |
| Plots | `visualize.py` | Learning curves, comparison bars, time-vs-floor car trajectories (PNG/GIF) |

## Simulator

| Property | Value |
|---|---|
| Floors | 1–10, all passengers board at floor 1 (up-peak only) |
| Elevators | E0: odd floors `{1,3,5,7,9}` · E1: `{1,4,6,8,10}` · E2, E3: all floors |
| Capacity | 15 passengers per car |
| Timing | 2 s per floor, 4 s per stop for doors and boarding |
| Episode | 1800 s (30 min) |
| Arrivals | Fixed counts per 5-min bucket `[36, 56, 75, 105, 73, 34]`, arrival times uniform within each bucket |
| Destinations | Uniform over floors 2–10 |

Only the six bucket counts come from the morning-rush observation. They are hard-coded in `elevator_env.py`, and the raw observation data is not in this repo. Destinations and timings are assumptions.

> **TODO (author, fill in before publishing):** where and when the six counts were observed, and how they were collected — e.g. "Counted at the lobby of `<building>` on `<date(s)>` between 08:40 and 09:10 by `<method>`."

## MDP

**Decision.** `Discrete(4)`: which car serves the passenger who just arrived. Decisions are event-driven (one per passenger, so about 379 per episode), not one per simulator tick.

**Observation (34-dim).**

* Per car (×4): floor / 10, load / 15, queued stops / 10, seconds until free / 60, and whether the car can reach the current passenger's destination
* Current call: destination / 10, origin / 10
* Waiting passengers counted by destination floor (/ 10)
* `sin` / `cos` of the episode phase

**Reward.** Between two decisions at `t0 < t1`:

```
r = −Σ_passengers  (seconds spent waiting inside [t0, t1])
```

Summed over an episode, this is the total waiting time of all passengers. At the end of the episode, each passenger who is still waiting is charged their full wait (`1800 − arrival_time`) once more as an extra penalty. If a masked action ever reaches the environment, it is replaced by the feasible car with the shortest queue and the reward gets −5. With masking on, this never happened during training (the `invalid` column of `train_log.csv` is always 0).

**Agent** (`train.py` defaults). The network is a 2 × 128 ReLU MLP trained with Huber loss and Adam (lr 1e-3), with gradients clipped at norm 10 and γ = 0.99. Replay holds 50k transitions with batch size 64, and the target network is synced every 500 updates. ε decays linearly from 1.0 to 0.05 over 15k updates. Prioritized replay uses α = 0.6, with β annealed from 0.4 to 1.0.

## Results

### Main result: 5 training seeds

Both DQN variants were trained with the same recipe (500 episodes, `train.py` defaults) using training seeds 1–5. Each checkpoint was evaluated greedily on evaluation seeds 0–19. Each DQN cell is the **mean ± std across the 5 training seeds** of that run's 20-seed average (`compare_multiseed.csv`). The baselines don't depend on a training seed, so they have no spread here.

| Policy | Mean wait | p95 wait | Max wait | Served | Unserved |
|---|---|---|---|---|---|
| random | 104.7 s | 386.0 s | 484.5 s | 343.75 | 35.25 |
| round-robin | 70.0 s | 271.8 s | 361.2 s | 363.00 | 16.00 |
| nearest-car (NCA) | 31.4 s | 83.3 s | 163.3 s | 377.05 | 1.95 |
| DQN, vanilla | 27.0 ± 6.4 s | 68.3 ± 35.8 s | 130.2 ± 44.5 s | 377.95 ± 0.39 | 1.05 ± 0.39 |
| DQN, Double + Dueling + PER | 29.9 ± 8.9 s | 77.9 ± 40.1 s | 139.4 ± 50.0 s | 377.18 ± 1.08 | 1.82 ± 1.08 |

Per training seed (from `runs/<variant>_s<seed>_eval.csv`; NCA: 31.4 s mean, 83.3 s p95):

| Training seed | D+D+PER mean | D+D+PER p95 | Vanilla mean | Vanilla p95 |
|---|---|---|---|---|
| 1 | 24.7 s | 52.1 s | 23.9 s | 51.2 s |
| 2 | 44.7 s | 144.1 s | 23.0 s | 47.1 s |
| 3 | 23.8 s | 53.0 s | 23.7 s | 50.8 s |
| 4 | 24.5 s | 52.0 s | 38.4 s | 131.6 s |
| 5 | 31.6 s | 88.1 s | 26.2 s | 60.7 s |
| **runs better than NCA** | **3 / 5** | **3 / 5** | **4 / 5** | **4 / 5** |

Averaged over training seeds, both DQN variants have lower mean and p95 wait than nearest-car, but the margin is smaller than the spread between training seeds. Most runs beat NCA clearly, and one or two runs per variant are as bad as or worse than NCA. With 5 seeds, the Double + Dueling + PER variant is **not** better than vanilla DQN.

"Wait" is the time from arrival to boarding. Ride time is not counted. A passenger still waiting at 1800 s counts as unserved, and their wait is cut off at the end of the episode (served + unserved = 379 per episode).

### What changed and why

* **Before:** the README reported one training run per variant (`dqn.pt`, `dqn_vanilla.pt`, table below). That made the DQN look clearly better than nearest-car, and the enhanced variant look clearly better than vanilla.
* **Multi-seed runs added:** I retrained both variants with 5 training seeds and report mean ± std, so the comparison no longer rests on one possibly lucky run. The vanilla runs now also use 500 episodes, because the original vanilla checkpoint's episode count was not recorded.
* **Seeding fixed:** `train.py` now also seeds Python's `random` module, which drives the ε-greedy draws and uniform replay sampling. With a fixed seed, two CPU runs now produce identical training logs (checked on a 15-episode run with seed 1 and `OMP_NUM_THREADS=1`). `dqn.pt` was trained before this fix, so it cannot be regenerated bit-for-bit.
* **What it showed:** the single run was one of the good ones. The weak runs are not failed trainings. Enhanced seed 2 averages 28.3 s mean wait over its last 50 ε-greedy training episodes (`runs/enh_s2_train.csv`), but its greedy policy scores 44.7 s. `train.py` simply keeps the final network, with no validation-based checkpoint selection.

### Original single run

The first runs (`dqn.pt`, enhanced variant, 500 episodes; `dqn_vanilla.pt`, episode count not recorded), with the same evaluation (`compare.csv`, `compare_vanilla.csv`):

| Policy | Mean wait | p95 wait | Max wait | Served | Unserved |
|---|---|---|---|---|---|
| nearest-car (NCA) | 31.4 s | 83.3 s | 163.3 s | 377.05 | 1.95 |
| DQN, vanilla (`dqn_vanilla.pt`) | 28.3 s | 70.1 s | 160.3 s | 377.05 | 1.95 |
| DQN, Double + Dueling + PER (`dqn.pt`) | 25.7 s | 53.7 s | 112.3 s | 377.55 | 1.45 |

![policy comparison, single run](compare.png)

Training curves of `dqn.pt` (`train_log.csv`, ε-greedy, so noisier than greedy evaluation):

![training curves](curve.png)

Car trajectories of `dqn.pt` on seed 0 (grey crosses mark passengers' destinations at their arrival times):

![trajectories](trajectory.png)

## What I learned

1. **The reward can be the metric itself.** Charging the agent for every second every passenger spends waiting makes the episode return equal to minus the total waiting time (plus the end-of-episode charge for unserved passengers). That left only one extra penalty to reason about, instead of a set of shaping terms. The decision is also event-driven, so an episode is only about 379 steps long instead of 1800 ticks.
2. **Hard constraints belong in the action mask, not the reward.** Two cars cannot reach some floors. Masking those cars both when acting and in the target `max` meant the agent never picked an unreachable car in 500 training episodes. The −5 penalty exists only as a fallback.
3. **A simple heuristic is a strong baseline.** Random and round-robin are far behind. Averaged over 5 training seeds, the DQN beats nearest-car by only 1.5–4.4 s on mean wait, less than the 6–9 s spread between training seeds.
4. **One training run is not a result.** My first run made Double + Dueling + PER look like a clear win over both nearest-car and vanilla DQN. With 5 seeds, the variants overlap, and 1–2 runs per variant end up no better than the heuristic. I now seed every random number generator and report the spread across training seeds.

## Limitations / next steps

* **Few seeds, no checkpoint selection.** 5 training seeds are enough to show that the spread is large, but not to rank the two DQN variants. The final network is used as-is, so a policy that does well ε-greedy can still do badly when run greedily (enhanced seed 2). Keeping the best checkpoint by validation return would probably shrink the spread. I have not tried it.
* **Same traffic for training and evaluation.** Both use the same arrival profile, so robustness to other days or traffic patterns is untested.
* **Simplified building.** The simulator models up-peak traffic from the lobby only, with uniform destinations. Each car visits its queued stops in FIFO order, with no SCAN/collective control and no acceleration model, and every stop takes a fixed 4 s however many people board. A passenger who does not fit into a full car stays assigned to that car until it is next sent to the lobby.
* **Heuristic baseline.** The NCA score in `heuristic.py` is a simple hand-weighted rule (queue and load penalties), not a tuned commercial group-control algorithm.
* **No policy analysis** beyond the trajectory plot. I have not measured *what* the DQN does differently from NCA (for example, how it uses the two all-floor cars).
* Next steps: validation-based checkpoint selection, more training seeds, randomizing the arrival profile and timings, and a per-car multi-agent formulation.

## Run

Tested with Python 3.12, torch 2.9 (CPU), gymnasium 1.3, numpy 2.4 and matplotlib 3.10.

```bash
pip install -r requirements.txt

# evaluate the included checkpoints against the baselines (seconds on CPU)
python evaluate.py --ckpt dqn.pt --episodes 20 --csv compare.csv
python evaluate.py --ckpt dqn_vanilla.pt --episodes 20 --csv compare_vanilla.csv

# train from scratch (Double + Dueling + PER by default)
# (writes new files; the included dqn.pt / train_log.csv are left untouched)
python train.py --episodes 500 --seed 1 --out my_dqn.pt --log-csv my_train.csv
python train.py --episodes 500 --seed 1 --no-double --no-dueling --no-prioritized --out my_vanilla.pt --log-csv my_vanilla.csv

# multi-seed experiment: 5 seeds x {enhanced, vanilla}, run in parallel on CPU,
# then evaluate each checkpoint and write compare_multiseed.csv (needs bash, e.g. Git Bash on Windows)
bash run_multiseed.sh
python aggregate_seeds.py --runs runs --out compare_multiseed.csv   # re-aggregate only

# figures
python visualize.py curve      --csv train_log.csv --out curve.png
python visualize.py compare    --csv compare.csv   --out compare.png
python visualize.py trajectory --ckpt dqn.pt --seed 0 --out trajectory.png   # add --gif trajectory.gif for an animation
```

`--device auto` (the default) uses CUDA if a CUDA build of torch is installed. The network is small, so CPU is fine.
