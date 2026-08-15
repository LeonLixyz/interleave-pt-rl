# Chess RL: throughput work, and parity with the old verl setup

Notes on why the Miles-based RL runs faster than the original verl setup in
`Chess-RL`, and where the two configurations still differ. Not needed to run
anything — `docs/03-chess-rl.md` has the settings.

---

## 1. Batch geometry is now identical

Compared against `Chess-RL/verl/8_gpu_bash/run_multi_turn.sh`:

| Setting | verl (Chess-RL) | Miles (now) | Same? |
|---|---|---|---|
| prompts per update | `data.train_batch_size` 256 | `--rollout-batch-size` 256 | yes |
| samples per prompt | `rollout.n` 8 | `--n-samples-per-prompt` 8 | yes |
| trajectories per update | 2,048 | 2,048 | yes |
| optimizer steps per rollout batch | 1 (`ppo_mini_batch_size` 256 prompts = 2,048 samples) | 1 (`--global-batch-size` 2048) | yes |
| max prompt / response length | 512 / 2,560 | 512 / 2,560 | yes |
| learning rate | 1e-5 | 1e-5 default | yes |
| KL coefficient | 0.001 | 0.001 | yes |
| advantage estimator | grpo | grpo | yes |
| entropy coefficient | 0.0 | 0.0 | yes |
| KL estimator | `low_var_kl` | `low_var_kl` (since 2026-08-12; was `k1`) | yes |

## 2. The KL estimator (fixed 2026-08-12)

verl passed `kl_loss_type=low_var_kl`. The launcher originally passed nothing, so Miles
used its default `k1`. It now passes `--kl-loss-type low_var_kl` explicitly
(`KL_LOSS_TYPE_DEFAULT` in `modal_interleave.py`), so new runs match verl.
**Every result published in this study ran with `k1`.** With `r = log π_θ − log π_ref`
(`miles/miles/backends/training_utils/loss_hub/math_utils.py`):

| Estimator | Formula | Properties |
|---|---|---|
| `k1` (what we used) | `r` | unbiased, **signed** — negative on tokens where the policy is *less* likely than the reference, so per-token it can reward divergence; highest variance |
| `k2` | `r² / 2` | non-negative, biased |
| `k3` = `low_var_kl` (what verl used) | `e^{−r} − 1 + r` | non-negative, unbiased, lowest variance; `low_var_kl` additionally clamps to [−10, 10] |

At coefficient 0.001 the KL term is a small fraction of the loss, so this is unlikely
to explain any result — but it was a difference nobody chose. Pass `--kl-loss-type k1`
to reproduce the published runs exactly.

## 3. The config change that raised token throughput

verl micro-batched by **sequence count**: `ppo_micro_batch_size_per_gpu=16`, i.e. 16
sequences per micro-batch regardless of how long they are. With chess traces averaging
well under the 2,560-token cap, most micro-batches were far below what an H200 can
hold, and each update paid the fixed per-micro-batch overhead many times.

Miles micro-batches by **token budget**: `--max-tokens-per-gpu 131072`. It computes the
minimum number of micro-batches such that each holds at most that many tokens
(`miles/miles/backends/training_utils/data.py`, `get_minimum_num_micro_batch_size`),
then all-reduces to a DP-wide count. Short sequences pack together instead of wasting
the slot.

Two supporting changes in the same direction, both safe because the model is only 47M:

- `--no-gradient-checkpointing` (verl had `enable_gradient_checkpointing=True`) — no
  recompute in the backward pass.
- `--attn-implementation flash_attention_3`.

The profile enforces this shape: `small-model-h200` refuses anything other than 8×H200
with 192 GB host memory, and accepts only 65,536 or 131,072 for the token budget, so a
run cannot silently drift off the benchmarked configuration.

## 4. The code changes that made generation faster

These are in our Miles fork (`miles/OUR_CHANGES.md`, diff in `miles/our_changes.patch`)
plus our own rollout module. 468 added lines across 17 files.

**Sticky per-key routing** — `miles/miles/router/router.py`. The router previously sent
each request to the least-loaded worker. It now reads an `x-smg-routing-key` header and
pins that key to one worker for its lifetime; the first assignment is still load-aware
(hash-ordered choice among the currently least-loaded workers, so affinity cannot skew
one GPU), and a key remaps if its worker dies. `chess_rl_miles` sets the key per prompt
group, so all 8 samples of a prompt and every multi-turn `<call_env>` continuation land
on the same SGLang engine and hit its KV cache instead of re-prefilling the shared
prefix. For multi-turn chess rollouts — where each sample re-sends a growing prefix up
to 6 times — this is the dominant win.

**Token-id-only batched rollouts** — ours, `chess/rl/chess_rl_miles/batched_rollout.py`,
enabled by `--batched-rollout --sglang-token-id-only`. Requests carry token ids instead
of text, removing a detokenize/retokenize round trip on every turn.
`chess/rl/sitecustomize.py` monkey-patches SGLang's detokenizer manager to flatten
nested token ids so the path works.

**Weight-update session RPC** — `fsdp_utils/update_weight_utils.py` and
`sglang_engine.py` gain explicit `begin_weight_update` / `end_weight_update` around the
sync, instead of per-tensor traffic to each engine.

Not speed, but in the same patch: environment-reply ordering
(`miles/miles/rollout/env_reply_ordering.py`, new file), the CISPO policy loss and
advantage helpers, a separate `--eval-sglang-server-concurrency` guarded by a semaphore
so evaluation cannot starve training, and a colocate-mode default that disables the
SGLang prefill CUDA-graph backend to avoid an NVLS OOM.

**None of this is upstream.** A fresh clone of `radixark/miles` will be slower and will
lack the env-reply ordering fix. If you move to a newer upstream, replay
`miles/our_changes.patch`.

## 5. Is the token-budget micro-batching "matching verl"?

No — it is a deliberate departure, and it is the reason the runs are faster. What
matches verl is the **batch geometry** (256 prompts × 8 samples = 2,048 trajectories,
one optimizer step per rollout batch, the same length caps, lr and KL settings).
*How* that update is split into micro-batches is different by design: verl splits by
sequence count, Miles by token budget.

Micro-batch splitting is normally gradient-neutral, but under `token-mean` it is not
exactly so in this implementation. In
`miles/miles/backends/training_utils/loss.py`, the token-mean branch divides each
micro-batch's loss by *that micro-batch's* DP-global token count
(`global_num_tokens`), and `num_microbatches` is used only in the non-token-mean
branch. So the accumulated gradient is `Σᵢ Gᵢ/Nᵢ` rather than `(Σᵢ Gᵢ)/(Σᵢ Nᵢ)` — every
micro-batch carries equal weight regardless of how many tokens it holds.

The two agree exactly only when all micro-batches hold the same number of tokens.
Packing to a token budget makes them near-equal by construction, so Miles is *closer*
to a true global token-mean than verl's fixed-16-sequences split, where a micro-batch
of long traces and one of short traces would be weighted identically despite very
different token counts.

Two practical consequences:

- Changing `--max-tokens-per-gpu` (65,536 vs 131,072) changes the effective gradient
  slightly, not just speed. Keep it fixed within a comparison.
- Gradients are not bit-identical to verl's and were never going to be. The claim worth
  making is that the sampling distribution, per-update batch, and objective match — not
  that the two produce identical updates.
