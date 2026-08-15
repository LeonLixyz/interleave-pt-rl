#!/bin/zsh
# Convert steps 200..1400 (every 200) of the four wave-2 RL runs to HF exports.
set -u
MILES_DIR="/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles"
CKPT_BASE="/pretrain-checkpoints/interleave_50m/pretrain/sft_injection_ablation_v1_20260801"
cd "$MILES_DIR" || exit 1
for run origin in \
    k16band-lr1e4-rl1500 tracep2k16 \
    rollband-lr1e4-rl1500 tracep2roll \
    e2w1band-lr1e4-rl1500 e2w1 \
    e3p2band-lr1e4-rl1500 e3p2; do
  for st in 200 400 600 800 1000 1200 1400; do
    name=$(printf "%s-s%04d" $run $st)
    if modal volume ls rl-reasoning-checkpoints interleave_50m/rl_hf/$name/ 2>/dev/null | grep -q config.json; then
      echo "skip $name (exists)"
      continue
    fi
    modal run --detach chess_rl_miles/scripts/modal_interleave.py \
      --action convert --run-name $run \
      --hf-checkpoint $CKPT_BASE/$origin/final \
      --output-name $name --step $st 2>&1 | grep SPAWNED | while read -r l; do echo "$name $l"; done
  done
done
echo "ALL CONVERTS SPAWNED"
