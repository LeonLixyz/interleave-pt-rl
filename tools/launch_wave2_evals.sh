#!/bin/zsh
# For every converted wave-2 curve point: launch stride-8 pass@k eval (rl-point)
# and per-run batched held-out-loss jobs. Idempotent via rl_eval_ledger.txt.
set -u
S=/private/tmp/claude-501/-Users-leonli66-Desktop-Research-RL-Chess-RL/cf172269-6c26-448b-8a10-904c1e927526/scratchpad
LEDGER=$S/rl_eval_ledger.txt
EVAL_DIR="/Users/leonli66/Desktop/Research/RL/Chess RL"
touch $LEDGER

typeset -A TAG
TAG=(k16band-lr1e4-rl1500 K16BAND rollband-lr1e4-rl1500 ROLLBAND \
     e2w1band-lr1e4-rl1500 E2W1BAND e3p2band-lr1e4-rl1500 E3P2BAND)

missing=0
for run in k16band-lr1e4-rl1500 rollband-lr1e4-rl1500 e2w1band-lr1e4-rl1500 e3p2band-lr1e4-rl1500; do
  targets=""
  for st in 200 400 600 800 1000 1200 1400; do
    name=$(printf "%s-s%04d" $run $st)
    if ! modal volume ls rl-reasoning-checkpoints interleave_50m/rl_hf/$name/ 2>/dev/null | grep -q config.json; then
      echo "MISSING export $name"
      missing=$((missing+1))
      continue
    fi
    if ! grep -q "^$run $st\$" $LEDGER; then
      (cd "$EVAL_DIR" && modal run --detach Eval/modal_eval_clean.py \
        --action rl-point --arm "$run@$st" >/dev/null 2>&1) \
        && echo "$run $st" >> $LEDGER && echo "eval launched $run@$st"
    fi
    if [ -z "$targets" ]; then
      targets="../../rl_hf/$name:${TAG[$run]}@$st"
    else
      targets="$targets,../../rl_hf/$name:${TAG[$run]}@$st"
    fi
  done
  if [ -n "$targets" ] && [ ! -f "$S/ptloss_${run}.out" ]; then
    (cd $S && modal run --detach eval_loss.py --targets "$targets" > ptloss_${run}.out 2>&1) &
    echo "ptloss batch launched for $run"
  fi
done
wait
echo "DONE missing=$missing"
