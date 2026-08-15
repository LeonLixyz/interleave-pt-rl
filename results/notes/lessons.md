# Lessons and incidents

Things that cost time in this project. Kept out of `docs/` because none of it is
needed to read or launch the code — but all of it is worth knowing before trusting a
number.

---

## `<bos>` omission invalidated an entire generation of results

Chess prompts must start with `<bos>` (token id 0); the tokenizer does not add it.
Training data contains it, so a prompt without it is off-distribution: evaluations
returned ~0% format-valid across every arm, which read as "RL doesn't work".

The fix had to land in **two** files — `rollout.py` and `batched_rollout.py`. Only the
first was patched initially, and production uses `--batched-rollout`, so the first audit
still showed 0/2,048 prompts with `<bos>`. The eval results namespace is called
`ablation_pass16_clean_v2_bos` to mark the boundary.

## `modal run` without `--detach` silently kills runs

Modal stops the app when the local entrypoint returns, killing spawned GPU calls a few
seconds after launch with an empty `RemoteError('')`. Two full evaluation batches were
lost before this was diagnosed. Always `--detach`; verify with
`modal.FunctionCall.from_id(...).get(timeout=0)` raising `TimeoutError` (= still
running).

## Reward collapse with format intact means infrastructure, not policy

One run's mean reward fell 0.55 → 0.06 over ~20 updates while its format-valid rate
stayed at 99%. Well-formed answers scoring zero means the scorer or environment failed.
Training continued on ~40 mis-scored batches at lr 1e-4 and did real damage: held-out
PT loss 0.505 → 0.66, pass@1 40.6 → 38.7, and the checkpoint at that step was corrupt.
The segment was discarded and the run relaunched from the last good checkpoint.

Tell: **reward down + format flat → infrastructure. Reward down + format down → policy.**

## A converted checkpoint scoring at base-model level is a bad export

Two HF conversions produced checkpoints that evaluated at ~15% (base level) instead of
their true value. Both re-converted cleanly. Delete and retry; only suspect the source
checkpoint if the same step fails twice.

## The RL learning rate dominated everything

At 1e-5, RL added ~4 pass@1 and no coverage, and moved 0.2–0.4% of weight mass. At
1e-4 it tripled pass@1, moved 4%, and produced the first real coverage gain. Several
early conclusions ("RL only sharpens") were artifacts of the low learning rate. When a
method looks inert, check the step size before concluding anything.

## Save the steps you will want to branch from

`--save-interval 40` means only multiples of 40 exist, plus a forced save at
`--num-rollout`. Wanting to continue from step 1,500 of a 3,000-step run is impossible
unless 1,500 was a save point. The 1,500-update runs in this study were launched at
exactly 1,500 so that endpoint existed.

Related: the provenance guard refuses to reuse a run root with different semantics
(including a different `--num-rollout`), so a finished run cannot simply be relaunched
with a bigger target. Seed a new run root instead — `tools/seed_resume_root2.py`.

## Data-sampler state lives outside the checkpoint

Miles stores it as `rollout/global_dataset_state_dict_<step>.pt`, not inside
`iter_<step>/`. A seeded resume that copies only `iter_*` restarts the prompt stream at
the beginning of the epoch-0 shuffle. Harmless here (one epoch over ~32k prompts is
~125 updates, and each epoch reshuffles), but copy it and the continuation is
position-exact for free.

## bf16 parameter storage froze every RMSNorm scale

All 49 norm tensors sit at exactly 1.0 in every run: updates are smaller than one bf16
ULP at 1.0 and round away. Shared by all arms so comparisons hold, but absolute numbers
may improve with fp32 master weights. Not fixed.

## Miscellaneous

- `safetensors.numpy` cannot read bf16 — use `safetensors.torch` and cast to float64.
- On a torch tensor `.size` is a method; use `.numel()`.
- `PAD_RECORD` (int64 min) is negative, so "negative means SFT row" in a leg order
  array is only true after filtering it out.
- The W&B key in the Modal secret is expired; runs in this study uploaded nothing.
  Reward curves come from saved rollouts via `tools/rl_reward_curve.py`.
- Scratch directories are not storage: two pipeline scripts were deleted mid-session by
  tmp cleanup. Anything you would be annoyed to lose belongs in the repo.

## The provenance identity omitted the two things we varied most

`_run_provenance_identity` in the RL launcher hardcoded `"lr": 1e-5` and recorded the
default balanced dataset, regardless of what `--lr` and `--train-file` were actually
passed. Every 1e-4 run therefore carries a provenance record claiming 1e-5 and the
wrong dataset, and the identity hash could not distinguish them — the guard would have
allowed reusing a run root with a different learning rate or a different training set.

The launch command *is* recorded verbatim in the same file, so the truth is recoverable
from `initial_command` in each run's `run_provenance.json`; only the semantics block
was wrong.

Fixed 2026-08-12: `lr`, `kl_loss_type`, and the real `train_file` + sha256 are now part
of the identity (`training_data` replaces the old hardcoded `balanced_data` block).
Note this changes the identity hash, so existing run roots will not resume under the
new code — expected, since the semantics genuinely changed.

The same bug class hit `--dry-run`: the printed command was built without `train_file`,
`lr` or `kl_loss_type`, so it did not show what would actually run. Also fixed.
