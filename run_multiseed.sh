#!/usr/bin/env bash
# Train the enhanced (Double + Dueling + PER) and vanilla DQN with several
# training seeds, then evaluate every checkpoint with evaluate.py on the
# same 20 evaluation seeds (0-19). Outputs go to runs/.
# Runs are launched in parallel, one CPU thread each.
set -euo pipefail
SEEDS="${SEEDS:-1 2 3 4 5}"
EPISODES="${EPISODES:-500}"
mkdir -p runs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

for s in $SEEDS; do
  python train.py --episodes "$EPISODES" --seed "$s" --device cpu --log-every 100 \
    --out "runs/enh_s$s.pt" --log-csv "runs/enh_s${s}_train.csv" > "runs/enh_s$s.log" 2>&1 &
  python train.py --episodes "$EPISODES" --seed "$s" --device cpu --log-every 100 \
    --no-double --no-dueling --no-prioritized \
    --out "runs/van_s$s.pt" --log-csv "runs/van_s${s}_train.csv" > "runs/van_s$s.log" 2>&1 &
done
wait

for s in $SEEDS; do
  for v in enh van; do
    python evaluate.py --ckpt "runs/${v}_s$s.pt" --episodes 20 --device cpu \
      --csv "runs/${v}_s${s}_eval.csv" > /dev/null
  done
done

python aggregate_seeds.py --runs runs --out compare_multiseed.csv
