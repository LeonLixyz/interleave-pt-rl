# Our Miles fork

This is the Miles RL framework with our changes **already applied**. Run it as-is —
there is nothing to patch and no setup step.

- Upstream: https://github.com/radixark/miles.git
- Commit we forked from: `e20de26c94412301ba2a746e8d942220bad0d00d`

`our_changes.patch` is a record of the diff against that commit, kept only so the
changes can be replayed onto a newer upstream if we ever rebase. It is not applied at
runtime and does not need to be.

## What we changed

17 files modified:

- `miles/backends/experimental/fsdp_utils/actor.py`
- `miles/backends/experimental/fsdp_utils/update_weight_utils.py`
- `miles/backends/sglang_utils/arguments.py`
- `miles/backends/sglang_utils/sglang_engine.py`
- `miles/backends/training_utils/loss.py`
- `miles/backends/training_utils/loss_hub/advantages.py`
- `miles/backends/training_utils/loss_hub/losses.py`
- `miles/backends/training_utils/loss_hub/math_utils.py`
- `miles/ray/rollout.py`
- `miles/rollout/inference_rollout/inference_rollout_eval.py`
- `miles/rollout/inference_rollout/inference_rollout_train.py`
- `miles/rollout/sglang_rollout.py`
- `miles/router/router.py`
- `miles/utils/arguments.py`
- `miles/utils/tracking_utils/wandb_utils.py`
- `tests/fast/router/test_router.py`
- `train.py`

1 file added:

- `miles/rollout/env_reply_ordering.py`

The changes cover RL throughput (rollout batching, weight updates, SGLang engine
arguments), the loss and advantage path, environment-reply ordering, and router
behaviour. See `our_changes.patch` for the exact diff.

## How it is used

`chess/rl/` imports from this tree at runtime via `PYTHONPATH`. Only three
subpackages are used: `miles.rollout.*`, `miles.utils.*`, and
`miles.backends.sglang_utils.arguments`.
