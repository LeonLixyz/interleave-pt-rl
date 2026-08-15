#!/bin/zsh
# Convert available 200-step checkpoints of the resumed control runs to HF,
# then launch stride-8 evals + held-out-loss batches. Idempotent via ledger.
set -u
S=/private/tmp/claude-501/-Users-leonli66-Desktop-Research-RL-Chess-RL/cf172269-6c26-448b-8a10-904c1e927526/scratchpad
LEDGER=$S/rl_eval_ledger.txt
MILES_DIR="/Users/leonli66/Desktop/Research/RL/Chess RL/chess-rl-miles"
EVAL_DIR="/Users/leonli66/Desktop/Research/RL/Chess RL"
CKPT_BASE="/pretrain-checkpoints/interleave_50m/pretrain/sft_injection_ablation_v1_20260801"
touch $LEDGER

typeset -A ORIGIN TAG
ORIGIN=(e3p2band-lr1e4-rl3000r2 e3p2)
TAG=(e3p2band-lr1e4-rl3000r2 E3P2R2)

converts=0
for run in e3p2band-lr1e4-rl3000r2; do
  latest=$(modal volume get chess-rl-miles-checkpoints "chess-rl-miles-interleave/$run/latest_checkpointed_iteration.txt" - 2>/dev/null | tr -dc '0-9')
  for st in 1600 1800 2000 2200 2400 2600 2800 3000; do
    [ -z "${latest:-}" ] && continue
    [ "$st" -gt "$latest" ] && continue
    name=$(printf "%s-s%04d" $run $st)
    if modal volume ls rl-reasoning-checkpoints interleave_50m/rl_hf/$name/ 2>/dev/null | grep -q config.json; then
      continue
    fi
    (cd "$MILES_DIR" && modal run --detach chess_rl_miles/scripts/modal_interleave.py \
      --action convert --run-name $run \
      --hf-checkpoint $CKPT_BASE/${ORIGIN[$run]}/final \
      --output-name $name --step $st >/dev/null 2>&1) && converts=$((converts+1)) && echo "convert spawned $name"
  done
done
echo "converts spawned: $converts"
[ "$converts" -gt 0 ] && sleep 420

for run in e3p2band-lr1e4-rl3000r2; do
  targets=""
  for st in 1600 1800 2000 2200 2400 2600 2800 3000; do
    name=$(printf "%s-s%04d" $run $st)
    if ! modal volume ls rl-reasoning-checkpoints interleave_50m/rl_hf/$name/ 2>/dev/null | grep -q config.json; then
      continue
    fi
    if ! grep -q "^$run $st\$" $LEDGER; then
      (cd "$EVAL_DIR" && modal run --detach Eval/modal_eval_clean.py \
        --action rl-point --arm "$run@$st" >/dev/null 2>&1) \
        && echo "$run $st" >> $LEDGER && echo "eval launched $run@$st"
    fi
    if ! grep -q "PTLOSS $run $st" $S/rl_ptloss_ledger.txt 2>/dev/null; then
      targets="${targets:+$targets,}../../rl_hf/$name:${TAG[$run]}@$st"
      echo "PTLOSS $run $st" >> $S/rl_ptloss_ledger.txt
    fi
  done
  if [ -n "$targets" ]; then
    (cd $S && modal run --detach eval_loss.py --targets "$targets" >> ptloss_${run}.out 2>&1) &
    echo "ptloss batch launched for $run"
  fi
done
wait
echo "SWEEP DONE"
