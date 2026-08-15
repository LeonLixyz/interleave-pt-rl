# Interleaved v2r3 diagnostic contract

Frozen at: 2026-07-30T09:19:31-04:00

Contract schema:
`interleaved-v2r3-diagnostic-contract-v1`

Contract version:
`mix10b_sft90k_3072_v2r3_diagnostic_20260730`

## Scope and authorization

This is a diagnostic-only response-trajectory experiment. It cannot select an
SFT weight, write or satisfy a P1/Exp2 gate, authorize production pretraining,
or authorize RL training. Any later production decision requires a separately
frozen two-batch, prompt-disjoint gate.

V2r2 is closed as `unauthorizable_provenance_drift`. Its three step-2,000
seed-42 observations remain diagnostic-only:

| SFT loss weight | joint-valid rows | joint-valid groups | positives | variance groups | rollout source |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0 | 0 | 0 | 0 | `b2d45007…2673` |
| 96 | 1 | 1 | 0 | 0 | `b2d45007…2673` |
| 190.189290837 | 6 | 6 | 0 | 0 | `38f829c6…19f` |

The v2r2 rejection-writing call
`fc-01KYSFVK08PJ45H5KJH8E2DT9A` failed closed before writing a marker because
the evidence spanned those two source identities. No v2r2 rejection marker,
P1 gate, full P1, monolithic canary, or Exp2 was authorized.

The source difference is additive and runtime-neutral: the new 36-file,
349,922-byte `chess-rl-miles` manifest
`b2d45007c46c4bfac4f4b0074de31bfbf99db12552638b458f733846195c2673`
equals the old 34-file, 300,004-byte manifest
`38f829c60815cb8f7a07776561af51cc22a8b6740c431c59bd3d9847ff4c019f`
plus exactly:

- `chess_rl_miles/scripts/upload_interleaved_checkpoints.py`, 37,150 bytes,
  SHA-256 `74e6d03f4f0884d8249a0d5dd34839fff474f7b2de261ade198b9013ac028d44`;
- `tests/test_upload_interleaved_checkpoints.py`, 12,768 bytes, SHA-256
  `84b7727444b4cef09141122edbc0ce62ebbf8a4c48ad801bab7e43363398d589`.

No rollout, reward, data, training, or Miles runtime file changed.

V2r3 deliberately uses the later 36-file, 358,781-byte
`chess-rl-miles` source manifest
`d7d24d523a8c34577f7c1f01cf2e3855a9092d580cef918c319ad249d6d0f6b9`.
Relative to the v2r2 evidence source, this adds explicit diagnostic-only
deterministic-inference, true rollout-only plumbing, and persisted sibling
sampling-seed identity. It does not alter or retroactively repair the closed
v2r2 evidence.

## Frozen data, objective, order, and schedule

All four trajectories use the same cleaned v2r1 P1 mixed stream and exact
sample order:

- data artifact version
  `mix10b_sft90k_v2r1_clean_verify_gate`;
- manifest-set SHA-256
  `6f2cc9093b2515e0a6a3aedc56a0cfd597c6b0f76933c5dbcd69eefd22440a23`;
- manifest leg `p1`, P1 order file and shuffle seed 42;
- one globally shuffled PT+all-SFT stream, with no separate SFT stage;
- 3,072-token packed contexts, local batch 21 × 8 H200s, gradient
  accumulation 1;
- the unchanged globally normalized weighted PT/SFT token-CE objective;
- model/data seed 42, AdamW peak LR `1e-3`, weight decay 0.1, betas
  `[0.9, 0.95]`;
- one 9,920-update cosine arc, 5% warmup, LR floor `1e-5`, and no optimizer
  reset inside a trajectory;
- SDPA, no `torch.compile`, eight data workers per rank, and no remote
  tracker.

The only manipulated variable is `training.sft_loss_weight`.

## Four continuous trajectories and 12 snapshots

Each row is one fresh, continuous training call. A snapshot is not an
independent restart.

| SFT loss weight | Stop | Required snapshot steps |
| ---: | ---: | --- |
| 190.189290837 | 9,920 | 1,000; 2,000; 4,000; 6,000; 8,000; 9,920 |
| 256 | 2,000 | 1,000; 2,000 |
| 384 | 2,000 | 1,000; 2,000 |
| 768 | 2,000 | 1,000; 2,000 |

`training.save_interval=0` and `training.export_interval=0`. At each declared
step, the trainer atomically renames one immutable `snapshots/step_N`
directory containing:

- `resume/`: complete Accelerate model, optimizer, scheduler, scaler/RNG, and
  trainer/data cursor state;
- `hf/`: a clean `from_pretrained`-compatible HF model and tokenizer with the
  identical trainer state;
- `.complete.json`: the step, canonical trainer-state hash, and complete
  recursive relative-path/byte-count/SHA-256 inventories for both trees.

The launcher rehashes both complete inventories before accepting a snapshot.
It rejects `latest`, `final`, gaps, extra snapshot directories, state/HF
disagreement, or a changed `snapshot_steps` list. A failed call resumes only
from the newest authenticated contiguous immutable snapshot. Each completed
snapshot is committed to the Modal volume while training continues.
Completion means the exact declared inventory ends at `max_steps`; diagnostic
trajectories intentionally create no duplicate `latest` or `final`.

Every optimizer step also accumulates the globally reduced, unweighted PT and
SFT loss sums, token counts, and contributing-step counts. Each snapshot
stores both the cumulative values and the delta since the preceding declared
snapshot (or zero at step 0). Exact-prefix validation proves that every
current cumulative minus the preceding snapshot cumulative equals the
declared interval loss sums, token counts, and contributing-step counts; it
also proves chained start/end boundaries. Both PT and SFT mass must be
strictly positive in every interval. A resume revalidates the same boundary
and delta before accepting state.

## Frozen seed-42 rollout audits

All 12 clean-HF snapshots receive exactly one Miles canary rollout with:

- fixed balanced parquet SHA-256
  `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`;
- 256 prompt groups × 8 samples = 2,048 rows;
- seed 42 and prompt-set SHA-256
  `9ab746d0039bcc15d3573296cbe4503650a10b9b3248ffae1e7bb4121663b7c7`;
- no dynamic filter, one rollout only, and Miles
  `--debug-rollout-only`, so no actor optimizer or policy update is
  constructed or applied;
- SGLang deterministic inference enabled, with sibling sample seeds
  `rollout_seed + sibling_index` for indices 0 through 7. The custom batched
  path persists both the seed and sibling index, reuses them across multi-turn
  continuations, and the audit requires
  `sample_index = group_index * 8 + sibling_index`;
- the verified 8×H200 small-model path: 131,072 tokens/GPU, SGLang
  concurrency 128, no gradient checkpointing, 192 GB host memory;
- `chess-rl-miles` source
  `d7d24d523a8c34577f7c1f01cf2e3855a9092d580cef918c319ad249d6d0f6b9`
  and Miles source
  `9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`.

Each rollout provenance must bind the exact recursive HF snapshot manifest.
The final immutable diagnostic report must bind all four training calls, all
12 snapshot resume/HF identities, all 12 rollout calls and artifacts, and the
authenticated PT and SFT CE interval accumulators.

## Predeclared metrics

For every snapshot report:

- unweighted-objective PT and SFT token CE aggregated over the training-stream
  interval since the prior snapshot, with exact loss sums, token counts, and
  contributing-step counts. These are token-weighted pre-update batch-logit
  measurements, not held-out CE and not endpoint-checkpoint evaluation. They
  support only a stability/optimization diagnostic; a final-pretraining-
  performance claim requires a separately frozen endpoint benchmark or
  held-out evaluation;
- outputs containing `</T>`, outputs containing `<call_env>`, and rows with
  parsed moves;
- joint-valid protocol rows/groups, where the same row contains ordered
  `</T>` then `<call_env>` and a nonempty parsed move;
- positives, nonzero-variance groups, `p_protocol`,
  `p_solve_given_protocol`, `p_total`, and variance rate;
- status counts/rates for exactly `completed` and `truncated`; pending,
  aborted, failed, and unknown statuses fail closed. Binary reward is audited
  identically for both accepted statuses, and every positive row must have a
  joint-valid protocol parse;
- response length uses three authenticated quantities. Total Miles
  `response_length` must equal `model_token_count + env_token_count`.
  `effective_response_length` must equal both top-level and metadata
  `model_token_count`, while top-level and metadata `env_token_count` must
  agree. Total response, effective/model response, and environment-token
  lengths each receive mean, min, max, and nearest-rank p50/p90/p99. The
  2,560-token cap count/rate applies only to effective/model tokens, so a
  valid multi-turn response may have a larger total segment because it
  includes inserted environment replies;
- per-row deterministic sampling identity: top-level and metadata
  `sampling_seed` and `sampling_seed_sibling_index` must agree, seed must equal
  `42 + sibling_index`, every group must contain sibling indices 0 through 7
  exactly once, and `sample_index` must equal
  `group_index * 8 + sibling_index`;
- raw-move-without-protocol count/rate and matched-token count. A row is
  flagged when it lacks a joint-valid protocol parse and contains at least two
  whitespace-delimited exact LAN/UCI/castling move tokens. The frozen token
  grammar admits optional piece prefix, optional dash/capture separator,
  lowercase UCI or `=QRBN` promotion suffix, check/mate suffix, and both
  castlings; it rejects malformed forms such as `e7e8x`.

No threshold or result from this 12-snapshot grid authorizes production.

## Frozen implementation identities

- pretraining source-tree SHA-256:
  `490b7cd758fce7e0187204449071d82da3e1ff42687f41323740c756287a7065`;
- launcher SHA-256:
  `aa9d74d959e516aa6636e6cef5653d7ffeefd2700c0c0f68dd91825753932228`;
- trainer SHA-256:
  `44b4d3cf423850625c5c4982ceb42a265e1acc61ba529b707c241b841d3aa5a6`;
- shared v2r2 audit validator SHA-256:
  `0b1f46fb575b40eff6a60a5154f13f39c045cb0192b4500fba6260f8f7ff9962`;
- diagnostic metrics SHA-256:
  `46dd1fbb6d6d3258feaee01022942649afd1667294a837dc99b1275838b7c04a`;
- launcher tests SHA-256:
  `2f550e4e44a5c4f590e73cb9850b886219e49385011e7c94620d47c0e81340be`;
- trainer tests SHA-256:
  `b094de719ec773ff72c0a549a9168e832509642262caa3999530aabcac7c9785`;
- shared v2r2 audit tests SHA-256:
  `d116bf3bca2aa5e958c19c5e3d97ae00ba577d651478edbe18c2ea799c2ccd9e`;
- diagnostic tests SHA-256:
  `a7c9af09b279fe2a27a4dcc5221ead372fca38a231d2f2a59f798346dff52b7e`;
- batched rollout runtime SHA-256:
  `6bc79c18cd58f08f286c9ddbca39d0f80d159968697b6361345d568bf51ea2eb`;
- batched rollout tests SHA-256:
  `f6fe8bd87a45204d1c3a857c052e91e204786ef25d58190afec565f1abdbe1ba`;
- rollout launcher SHA-256:
  `9e8b572b74ae2e83df1171a392f9e85ac9ad08a436d8b058c40bd5bcddc209a7`;
- rollout wrapper SHA-256:
  `94432d9dfbb65cf1f02a4dbc2b1834587636745234a37670efbf667945fd8ce5`;
- rollout launcher tests SHA-256:
  `2a80cd10d7b0895ff2a43c4050106c3d1aeb32a7a8d49d4c06d47d5f93acbdc7`;
- rollout wrapper tests SHA-256:
  `24b8dde216b9f3eb61346ee96f30803b76b55a2015fcdb541ef1109ddfad83c5`.

Verification passed: the complete local pretraining suite has 138 tests plus
3 subtests passing; the focused Miles rollout/provenance suite has 48 tests
passing. A broader Miles suite has 98 tests passing when the single
Ray-dependent routing module is excluded because Ray is not installed in the
local venv. Modal dry runs completed without GPU submission under apps
`ap-UrBWiokgRb5Zpxg5I03gmh` and `ap-4ztqKWmRPXpAJJR7ir6sKv`; the latter
printed both
`--sglang-enable-deterministic-inference` and `--debug-rollout-only`.

Launch is blocked until an independent science review approves these exact
identities and the frozen registry/ledger entry.
