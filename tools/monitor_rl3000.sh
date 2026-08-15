#!/bin/zsh
# Poll the checkpoint tracker of both 3,000-step baseline RL runs; exit when both reach 3000.
set -u
RUNS=(e2w1band-lr1e4-rl3000r e3p2band-lr1e4-rl3000r2)
while true; do
  done_count=0
  for run in $RUNS; do
    step=$(modal volume get chess-rl-miles-checkpoints "chess-rl-miles-interleave/$run/latest_checkpointed_iteration.txt" - 2>/dev/null | tr -dc '0-9')
    echo "$(date '+%H:%M') $run step=${step:-none}"
    if [ -n "${step:-}" ] && [ "$step" -ge 3000 ]; then
      done_count=$((done_count + 1))
    fi
  done
  if [ "$done_count" -eq 2 ]; then
    echo "BOTH RUNS AT 3000"
    exit 0
  fi
  sleep 1800
done
