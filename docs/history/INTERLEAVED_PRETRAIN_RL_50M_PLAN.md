# 50M Interleaved Pretraining and RL Experiment

Status: implementation and smoke-test preparation  
Owner: Pre-to-Post-2  
Date: 2026-07-29

Completed setup:

- balanced RL parquet uploaded to the Modal data volume;
- remote download checksum verified against the local source;
- Miles launcher default changed to the balanced parquet;
- dataset SHA verification and explicit `train_file`/`rollout_seed` arguments
  added;
- focused launcher tests passing (21 tests);
- arbitrary-HF optimized Miles handoff app deployed as
  `chess-interleave-rl` with balanced-data checksum enforcement, same-leg
  retry/resume, 40-step checkpointing, and raw-FSDP-to-HF conversion;
- focused handoff and existing launcher tests passing (30 tests).

## 1. Research questions

This experiment tests whether inserting RL between two pretraining stages:

1. improves the model obtained after the second pretraining stage; and
2. improves the final RL policy when pretraining data and RL updates are
   controlled.

Experiments 1--3 are the controlled comparison. Experiment 4 is an explicitly
FLOP-unbounded behavioral study and must not be used as an equal-compute arm.

## 2. Model and tokenizer

- Label: 50M Qwen.
- Actual parameter count with the required SFT tokenizer: 47,245,312.
- Architecture: 12 layers, hidden size 512, 8 query heads, 4 KV heads,
  head dimension 128, intermediate size 1536, Q/K norm enabled.
- Tokenizer: `LanTokenizerSFT`, vocabulary size 85.
- Keep vocabulary 85 through pretraining, RL, resumed pretraining, and final RL.
- Model maximum position length: 3072.

The 47.245M count is intentional. "50M" remains the experiment-size label.

## 3. Training data

### 3.1 Pretraining

The Modal directory retains a stale `20b` name but contains the current 54B
corpus:

- Provenance alias: `pretrain_v1_54b`
- HF dataset: `chess-pre-to-post/pretrain_v1_20b`
- HF revision:
  `07dd1b7090ca5f0fb05ef624c26b20bff19483c8`
- Modal source:
  `rl-reasoning-training-data:/pretrain_v1_20b`
- Immutable shard set: `raw.0000.npy` through `raw.47089.npy`
- Shards: 47,090
- Exact source tokens: 53,970,293,905
- Flat Modal filename/size manifest SHA-256:
  `07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa`
- Training sample seed: 42
- Requested training budget: exactly 10B pretraining tokens

Deterministically sample 10B tokens without replacement from this full
53.97B-token source and freeze the selected-shard/offset manifest. Do not reuse
the old 24,775-shard historical subset.

Pretraining tokens will be concatenated across shard boundaries and packed
into full 3072-token examples. Shard-tail tokens must carry into the next shard
rather than being dropped, repeated, or padded independently.

Each full pretraining record consumes a 3073-token source window:
`input_ids = tokens[0:3072]` and the already next-token-aligned
`labels = tokens[1:3073]`. The trainer computes cross-entropy over all 3072
positions without shifting the labels a second time. Adjacent records use a
3072-token stride and therefore share only the causal boundary token. This is
what makes the accounting below exactly 5B predicted pretraining tokens per
leg rather than 3071 targets per nominally full record.

The target topology is 8 H200s, batch 21/GPU, and gradient accumulation 1.
The global batch is 168 examples with capacity for 516,096 context positions
per optimizer update. This should fit comfortably for a 47.245M model; verify
it with the memory/throughput canary rather than enabling accumulation by
default.

### 3.2 Original SFT data

- Dataset: `chess-pre-to-post/sft_v1_200m_90k`
- Revision:
  `97f60746dd253b4e130beeb5e66f39e9d42ef25c`
- Files: 180 JSON files
- Rows: 77,717 physically present at the pinned revision. The repository name
  and nominal shard ranges say “90k”, but a full source audit found that 157
  of 180 JSON shards are partial; this is not downstream filtering.
- Machine-readable audit: [`SFT_PINNED_SOURCE_AUDIT.json`](./SFT_PINNED_SOURCE_AUDIT.json)
- Response field:
  `cot_by_method.trajectory_sep.cot_format_no_labels`
- Prompt field: `pgn`
- Exposure: every valid row exactly once
- SFT shuffle seed: fixed in the generated manifest
- Maximum sequence length: 3072

SFT examples are not concatenated together: each row remains one masked
example and is right-padded within the unified mixed batch, up to 3072. Prompt
tokens and environment replies following `<call_env>` are masked. Model-owned
reasoning, moves, `</T>`, and `<call_env>` remain supervised.

The preprocessing job must fail rather than silently omit a row with a missing
response or a sequence longer than 3072. If such rows exist, record their
count and resolve the context/truncation policy before launching production.

### 3.3 RL data

- Local source:
  `train_v4_dataset_balanced_multi_turn.parquet`
- Rows: 53,225
- SHA-256:
  `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`
- Modal destination:
  `chess-rl-miles-data:/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet`

The launcher must select this path explicitly and verify its checksum. It must
not fall back to `train_v4_easy_skewed_multi_turn.parquet`.

## 4. Pretraining optimizer and schedules

- Optimizer: AdamW
- Peak learning rate: `1e-3`
- Weight decay: `0.1`
- Betas: `[0.9, 0.95]`
- Warmup: 5% of the applicable schedule
- Schedule: cosine decay
- Final learning rate: `1e-5`, equal to the RL learning rate
- Model/data seed: 42
- Mixed precision: BF16 autocast with FP32-loaded/master parameters

There is no separate SFT stage or SFT-only optimizer event. Build one unified
sample-level manifest per leg:

- 1,627,604 full 3072-token pretraining records;
- one pretraining record with 512 valid tokens and masked padding;
- 38,858 individually masked SFT records in `P1` and 38,859 in `P2`; and
- 97 (`P1`) / 96 (`P2`) all-ignore padding records to make each manifest
  divisible by 168.

Shuffle the union of these records deterministically. Consequently, normal
training batches can contain both pretraining and SFT examples.

Each `P1`/`P2` leg contains exactly 5B real pretraining tokens, its
38,858/38,859-row deterministic SFT half, 1,666,560 manifest records including
padding, and 9,920 optimizer updates. The complete run contains exactly 10B
pretraining tokens, all 77,717 physically present SFT rows, and 19,840 optimizer
updates.

Scheduler lengths:

- each `P1`/`P2` arc: 9,920 updates, 496 warmup updates;
- Experiment 2 monolithic arc: 19,840 updates, 992 warmup updates.

Loss must be normalized by the global number of valid target tokens across all
DDP ranks. A local-rank mean would change the SFT/pretraining weighting when
different ranks receive different numbers of masked tokens.

Experiment 2 consumes the exact concatenation of the same two manifests.
Exp 1, Exp 2, and Exp 3 therefore see identical base tokens and SFT examples;
only the optimizer/scheduler arcs and the RL interruption differ.

### 4.1 V1 result and versioned-v2 rescue protocol

The original contract above is now called **v1**. Its globally normalized
raw-token mean was implemented as written; its completed artifacts, manifests,
cache, checkpoints, and evaluations are immutable evidence and must not be
overwritten or relabeled as v2.

The completed P1 endpoint established that v1 is not RL-feasible. The official
23,680-row B1--B5 endpoint evaluation was exactly zero on every chess metric.
The unfiltered Miles run then produced 38,912 trajectories across rollout IDs
0--18 with zero parseable positives and all prompt groups having zero reward
variance. An audit ruled out a prompt-frame, tokenizer-ID, masking, or
next-token-alignment mismatch:

- RL prompts correctly end in `<T>`, and 38,858/38,858 P1 SFT rows supervise
  exactly one `</T>` token.
- P1 supervises 93,647 `<call_env>` tokens, with at least one in every SFT row.
- P2 has the corresponding 38,859 `</T>` and 93,707 `<call_env>` targets.
- P1 rollout 0 had 2,037/2,048 nonempty generations but zero `</T>`, zero
  `<call_env>`, and only one `<sep>`; the model generated ordinary LAN move
  continuations rather than the SFT interaction frame.

The operative v1 failure is objective dilution. The exact post-mask target
accounting is:

| Stage | PT targets | SFT targets | SFT share of raw-token loss | PT:SFT target ratio |
| --- | ---: | ---: | ---: | ---: |
| P1 | 5,000,000,000 | 29,229,010 | 0.581182721% | 171.062926866 |
| P2 | 5,000,000,000 | 29,121,125 | 0.579049983% | 171.696663505 |
| Full P1+P2 | 10,000,000,000 | 58,350,135 | 0.580116363% | 171.379209320 |

Thus each SFT row is exposed only once while 99.42% of supervised targets come
from pretraining. In P1, `</T>` is only one of every approximately 129,426
total targets and `<call_env>` only one of every approximately 53,704 targets.
The existing pure-SFT reference learned the format with the same masking and
85-token tokenizer, so a separate SFT stage is not a missing software
dependency; v1 simply assigned too little loss mass to the mixed SFT task.

The pinned field named
`cot_by_method.trajectory_sep.cot_format_no_labels` also contains an upstream
data defect. It still embeds strings such as `<verify> <+3>` and
`<verify> <-3>`. Because the fixed tokenizer intentionally excludes reward
tokens, each pair becomes two supervised `<unk>` targets. Across all 77,717
rows, 2,933,691 such pairs produce exactly 5,867,382 `<unk>` labels, 10.06% of
the SFT targets. This defect alone did not prevent the pure-SFT reference from
learning the interaction frame, but it must not be carried into a corrected
run.

The rescue is a separately named and content-addressed **v2**, with these
pre-registered changes:

1. Before tokenization, remove every numeric verifier-score pair from the
   selected response field using
   `\s*<verify>\s*<[+-]?(?:\d+(?:\.\d+)?|\.\d+)>`, normalize whitespace,
   and leave all chess moves, `<T>`, `</T>`, `<sep>`, and `<call_env>`
   unchanged. After that substitution, fail closed on any residual literal
   `<verify>` tag: an unpaired or non-numeric verifier annotation must never
   pass silently into tokenization. Do not expand the vocabulary or alter the
   47,245,312-parameter architecture.
2. Rebuild a new SFT cache and dependent manifests under a v2 path. Fail unless
   all 77,717 rows remain, no cleaned response contains `<verify>`, no
   supervised response target is `<unk>`, every row has exactly one supervised
   `</T>`, every row has at least one supervised `<call_env>`, and the two SFT
   halves remain disjoint with an exact 77,717-row union. Removing the exact
   unknown-token pairs leaves 26,289,598 supervised SFT targets in P1,
   26,193,155 in P2, and 52,482,753 in the full stream; the rebuilt-cache audit
   must reproduce those counts.
3. Keep the requested training form: one sample-level shuffled PT+SFT stream,
   the same PT target counts, one SFT exposure, no oversampling, no SFT-only
   optimizer event, and no extra SFT stage.
4. Replace the implicit raw-token task mixture with an explicit weighted token
   objective:

   `L = (sum(PT CE) + w_sft * sum(SFT CE)) /
   (N_PT + w_sft * N_SFT)`.

   The sums and weighted denominator must be reduced globally across all DDP
   ranks. The primary v2 setting gives PT and cleaned SFT equal integrated loss
   mass. Because cleaning removes the 5,867,382 bad targets, use the
   **post-cleaning** ratios: `190.189290837` for P1, `190.889566377` for P2,
   and `190.538785189` for the monolithic full run. The `171.062926866`,
   `171.696663505`, and `171.379209320` ratios in the v1 table remain provenance
   for the uncleaned cache, not v2 coefficients. Log unweighted PT CE,
   unweighted SFT CE, both target counts, the coefficient, and the resulting
   effective task shares at every reporting interval.
5. Give every v2 cache, manifest, checkpoint, HF export, RL run, evaluation,
   and dashboard row a new identity. V1 remains queryable as the raw-token
   mixture result.

No full v2 retrain may launch before a structure canary passes. Run 500 P1
updates from the same random initialization and first 500 deterministic mixed
manifest batches, with the full 9,920-step P1 LR schedule (do not rescale the
schedule to 500). Then run the fixed 2,048-prompt RL-style rollout audit. The
gate requires all of the following:

- the cache assertions above and finite, globally normalized weighted loss;
- nonzero emitted counts for both `</T>` and `<call_env>`;
- nonzero parsed `extracted_moves` and nonzero positive trajectories; and
- at least one prompt group with nonzero reward variance, so both U and D can
  form a meaningful update.

If any item is zero, stop and diagnose; do not launch the 10B v2 matrix. If the
gate passes, preserve the canary report and hashes, approve the v2 contract,
and rerun the controlled matrix from scratch. The original v1 endpoints remain
part of the final results rather than being replaced.

The v2r1 implementation was independently audited before relaunch. The frozen
source-tree SHA-256 is
`8b8cea9bdba2408209a5abd942ee24cd5c179dc9899c97f1edfc0cb3080832ff`.
The audited protocol binds the cleaned-cache identity and exact P1/P2 target
counts into each manifest, applies the SFT coefficient in a globally reduced
weighted numerator and denominator, records PT/SFT losses, counts, and
effective shares, and rejects resume or completion artifacts whose configured
weight or provenance differs. Production remains fail-closed behind the exact
500-update, 8×H200/local-batch-21/GA1 P1 canary and its authenticated
2,048-trajectory Miles rollout gate; implementation or data preparation alone
does not approve a full run.

One minor operational residual risk was accepted by the audit: data preparation
and cache reuse fully hash the large flat cache files, but ordinary per-rank
runtime loads do not repeat those expensive full-file SHA-256 scans. Runtime
loads still verify the content-addressed metadata, file presence and byte
lengths, offsets, counts, and manifest/cache identity. A cache must first pass
the full preparation/reuse hash checks; the runtime optimization is not
permission to accept an unauthenticated cache.

### 4.2 V2r2 immutable staged-gate contract

This contract was frozen at **2026-07-30T07:42:58-04:00**, before inspecting
or acting on the remaining weight-32/weight-96 grid outcomes. It is a new
identity, `mix10b_sft90k_3072_v2r2_staged_gate_20260730`; the rejected v2r1
gate and all v2r1 artifacts remain immutable evidence. V2r2 reuses the exact
authenticated v2r1 cleaned data/cache/manifests and the same 47,245,312
parameter architecture, optimizer, LR schedule, topology, data order, and
weighted globally normalized objective. It changes only the staged
authorization rule and, after selection, the frozen SFT-loss coefficient.

The integrity gate is unchanged and permits no training when any cache hash,
target count, manifest identity, topology, provenance, finite-loss, or global
normalization assertion fails.

The **P1 protocol gate** is evaluated only at update 2,000. Updates 500 and
1,000 remain diagnostics and cannot authorize full pretraining. Each audit is
exactly 2,048 completed samples in 256 prompt groups of eight. A joint valid
protocol row must contain `</T>` before `<call_env>` in the same output and
have a nonempty parser-produced `extracted_moves`. Each audit must have at
least 32 joint valid-protocol rows spanning at least 16 prompt groups. The
candidate must pass twice: the seed-42 primary audit and a fresh disjoint
256x8 confirmation audit. Confirmation selection is outcome-independent: use
the smallest integer seed at least 43 whose exact Miles-selected prompt
fingerprints are disjoint from the primary batch, and authenticate the seed,
prompt-set hashes, and zero intersection in the gate marker.

Eligible coefficients are exactly `32`, `96`, and `190.189290837`. Select the
smallest coefficient whose update-2,000 P1 checkpoint passes both protocol
audits. Report unweighted PT CE, unweighted SFT CE, and:

- `p_protocol = joint_valid_protocol_rows / 2048`;
- `p_solve_given_protocol = positive_samples / joint_valid_protocol_rows`;
- `p_total = positive_samples / 2048`; and
- `variance_rate = nonzero_variance_groups / 256`.

Zero chess reward does not by itself block full pretraining at this stage.
Full P1 is authorized only for the selected coefficient. Full Exp2 is not yet
authorized: it additionally requires a fresh selected-coefficient,
update-2,000 canary using the monolithic 19,840-step optimizer/LR schedule and
full P1+P2 manifest, followed by the same primary plus disjoint-confirmation
protocol gate.

Miles RL is separately blocked until a complete full endpoint passes two
fresh disjoint 256x8 audits. **Each** audit must contain at least eight
positive samples and at least eight prompt groups with nonzero reward
variance. A dynamic-filter launch additionally must fill all 256 accepted
nonzero-variance groups within at most 8,192 attempted prompt groups. Official
B1--B5 evaluation remains an outcome metric, not a substitute for this
operational RL-feasibility gate.

Every gate is fail-closed, self-hashed, and binds the contract version, source
and data identities, candidate HF manifest, successful Modal call IDs, exact
rollout artifacts, seeds, prompt fingerprints, thresholds, and measured
counts. No v2r1 marker can authorize a v2r2 stage, and no manual interpretation
of a partial/overlapping batch is permitted.

## 5. Controlled experiment matrix

Each experiment is run with the unfiltered (`U`) and dynamically filtered
(`D`) RL sampler, for six controlled arms total.

### Experiment 1: interleaved RL

`P1 -> RL 1500 -> P2 -> RL 1500`

- First RL data-shuffle seed: 42
- Second RL data-shuffle seed: 43
- The dataset is not divided into disjoint halves.
- After first RL, load the clean RL endpoint weights into `P2`.
- Start `P2` with a fresh AdamW optimizer and fresh cosine schedule.
- Start second RL with a fresh RL optimizer and a reference equal to the `P2`
  endpoint.
- Number the two RL legs locally as 0--1500 and offset the second leg by +1500
  only in analysis and dashboards.

### Experiment 2: monolithic pretraining control

`P1 + P2 under one cosine -> RL 3000`

- One optimizer and one cosine schedule span the complete mixed-data stream.
- RL is one continuous 3000-update run.

### Experiment 3: two-cosine control

`P1 -> fresh optimizer/cosine -> P2 -> RL 3000`

- Carry model weights across the midpoint.
- Reset both AdamW state and the cosine schedule.
- Do not insert RL at the midpoint.
- This matches Experiment 1's two pretraining optimizer/scheduler arcs.

## 6. RL recipe

Use the optimized Miles/SGLang path already validated in the r6 runs:

- Hardware: 8 H200 GPUs per run
- Prompts per rollout update: 256
- Samples per prompt: 8
- Trajectories per update: 2048
- Global batch: 2048
- Optimizer updates per rollout update: 1
- Algorithm: GRPO, not CISPO
- Loss aggregation: token mean
- Optimizer: AdamW
- LR: constant `1e-5`
- Betas: `[0.9, 0.999]`
- Epsilon: `1e-8`
- Weight decay: `0.01`
- KL coefficient: `0.001`, low-variance KL
- Temperature/top-p: `1.0/1.0`
- Prompt/response/context limits: `512/2560/3072`
- SGLang concurrency: 128
- Batched token-ID fast path: enabled
- Miles dynamic filter: disabled for `U`; enabled for `D`

Under this topology:

- RL 3000 = 768,000 prompt draws and 6,144,000 trajectories.
- RL 1500 = 384,000 prompt draws and 3,072,000 trajectories.

Retries may resume only inside the same RL leg. A later leg must not restore
the earlier leg's optimizer, scheduler, RNG, or rollout cursor.

## 7. RL filtering ablation

### U: no filtering

Use all eligible rows from the immutable 53,225-row balanced parquet. This is
the primary same-distribution comparison.

### D: Miles dynamic nonzero-variance filtering

For each on-policy prompt group, let `success_count` be the number of successful
binary-reward trajectories among the `n` siblings. Keep the group only when:

`0 < success_count < n`

Therefore, the dynamic filter drops both:

- all-failure groups (`success_count == 0`, mean reward 0); and
- all-success groups (`success_count == n`, mean reward 1).

With the fixed optimized Miles recipe, `n = 8`. Groups with 1--7 successes are
kept. Miles draws replacement prompt groups until it has a complete accepted
training batch.

This is deliberately checkpoint/policy/sample dependent: a PuzzleId is not
permanently deleted and can be accepted or rejected on a later draw. Log the
numbers of attempted and accepted groups plus separate all-zero and all-one
drop rates.

Do not describe this as filtering `pass@n == 1`. Standard `pass@n` is
`int(success_count > 0)`, so mixed groups with 1--7 successes also have
`pass@n == 1` and are retained. The precise name is dynamic nonzero-reward-
variance filtering.

An offline `pass@16` or "remove only fully solved prompts" dataset would be a
different third condition and is not part of the two-setting core experiment.

## 8. Experiment 4: FLOP-unbounded positive-rollout transfer

Experiment 4 asks whether using successful intermediate-RL behavior improves
the final model. It is not part of the equal-FLOP claim.

Use the first RL-1500 stage to create a positive corpus:

- require `score == 1`;
- require completed, legal, structurally valid trajectories;
- require exactly one `</T>`;
- require at least one `<call_env>`;
- enforce the 2560 response limit;
- preserve exact token IDs and loss masks where possible;
- select one successful trajectory uniformly per prompt group with a fixed
  extraction seed;
- deduplicate only exact `(prompt, response)` hashes;
- preserve RL-step and difficulty metadata.

Run three methods:

1. **Hard SFT**
   - Student initialization: pre-RL `P1` checkpoint.
   - Transfer objective: masked hard cross-entropy on positive trajectories.
   - Then run `P2`, followed by RL 1500.

2. **Soft distillation**
   - Same student initialization, examples, batches, ordering, and updates.
   - Frozen teacher: RL-1500 endpoint.
   - Objective: full-vocabulary forward KL on model-owned response positions.
   - Then run the identical `P2`, followed by RL 1500.

3. **Scratch replay**
   - Use the RL-1500 model only to generate positive data.
   - Discard its weights.
   - Start the exact 47,245,312-parameter model from a seed-pinned random
     initialization; loading RL-1500 or other weights is forbidden.
   - Take every non-padding record from the authenticated `P2` PT/SFT
     manifest, add every extracted positive replay record, and apply one
     seed-pinned PCG64 sample-level shuffle. Positive rows use their exact
     rollout token IDs and response loss masks without re-tokenization.
   - Do not drop or replace any original PT/SFT record. Pad the combined order
     once to the fixed 8 x 21 global batch. Therefore
     `total_steps = ceil((P2_real_records + positive_rows) / 168)`.
   - Preserve the P2 manifest's warmup/cosine arc (currently 9,920 updates) and
     its `1e-5` endpoint without stretching it. Any replay-induced updates
     beyond that baseline remain at exactly `1e-5`. Replay examples are still
     interspersed across the unified shuffle; this is an LR-time tail, not a
     replay-only phase.
   - Full resume binds the combined-manifest hash, P2/replay checksums, data
     cursor, topology, scheduler tail, shuffle seed, and model-init seed.
   - Finish with RL 1500.

Report base-pretrain tokens, original-SFT processed tokens, positive-rollout
processed tokens, padding, teacher forward passes, RL trajectories, and total
estimated FLOPs separately.

### 8.1 Exp 4 launcher and immutable artifact contract

The checked-in launcher is
`chess_reasoning/modal_scripts/launch_exp4_interleave.py`. It is deliberately
deployed as a stable fail-closed app, but no extraction or GPU function may be
invoked until both first RL-1500 arms, their immutable provenance, all positive
rollouts, and their clean HF exports exist. Its production topology is fixed to
8 H200s, local batch 21, global batch 168, and gradient accumulation 1.

Exp4 v1 is frozen to the same explicit SDPA / no-`torch.compile` production
contract as P1. The launcher refuses runtime drift and authenticates P1's
config snapshot plus clean-HF trainer state against upstream pretraining source
tree SHA-256
`98db54b40e6af5bbbbca526b890c5cf19a96924c08c0c3e92cf0ea7edc6aba49`.
The inactive model-config compatibility value remains `2.8.3`, but the
`flash-attn` runtime package is not installed: SDPA is the active backend and
the trainer state must record no FlashAttention runtime.

Stable launcher deployment (definition only; zero invoked tasks at deploy):

- Modal app: `chess-50m-interleaved-exp4`
- App ID: `ap-Kx7JfX9BdbfEpMsj8zecZL`
- Deployment:
  `https://modal.com/apps/modal-labs/leon-dev/deployed/chess-50m-interleaved-exp4`
- Exp4 source-tree SHA-256:
  `1f1fe5896d95a151b5e8b25764eae08b1b06133b59d2c13cf1f9a3722add93d7`
- GPU train-function timeout: Modal's 48-hour maximum; this does not relax
  any readiness, provenance, or content-addressing gate.

Every clean-HF checkpoint identity uses the same file-set/fingerprint algorithm
as the endpoint evaluator. It validates the exact Qwen3 architecture and
requires safetensors weights, `interleaved_training_state.json`,
`tokenizer.py`, and `vocab.json`. It hashes the sorted unique top-level files
matching `model*.safetensors`; `config.json`, `generation_config.json`,
`model.safetensors.index.json`, and `interleaved_training_state.json`; plus
`tokenizer*`, `vocab*`, `merges*`, `special_tokens_map.json`,
`added_tokens.json`, `sentencepiece*`, and `spiece*`. For each file the digest
stream is its UTF-8 POSIX-relative path, NUL, exact file bytes, NUL. Logs,
completion markers, and arbitrary JSON are excluded. `complete.json` records
this exact final-HF SHA-256.

Positive extraction is separate for `U` and `D`. Before selecting examples it
requires exactly 1,500 paired `rollout_N.jsonl` /
`rollout_N.summary.json` files from the corresponding
`all_attempts_positive` directory. The content address binds the SHA-256 and
size of every JSONL and summary, the RL-1500 HF policy checkpoint content, the
filter setting, extraction seed, and source tree. A source mutation therefore
creates a new output identity rather than overwriting the old corpus.
The launcher re-hashes the complete rollout inventory, RL provenance bundle,
and teacher checkpoint after extraction and again after copying provenance,
before it atomically publishes the completion marker.

Extraction also fails unless the RL run root contains its immutable
`run_provenance.json` and every append-only
`provenance/launch_<command-sha16>.json`. It validates the exact U/D run
identity, profile/topology, RL semantics, balanced-data hash, every P1 origin
HF file, mounted chess-rl-miles/Miles source manifests, image/packages, and
commands. The root and all applicable launch documents are hashed into the
positive-corpus address and copied byte-for-byte under that corpus. A later
Exp4 stage therefore does not depend on mutable external “current code”
claims.

Each hard-SFT, soft-KL, or scratch-replay endpoint lives below:

`/checkpoints/interleave_50m/exp4/positive-rollout-transfer-v1-20260730/{u|d}/{method}/{full_plan_sha256}/`

The full plan hash binds the replay JSONL, replay manifest, replay artifact
manifest, P1 weights when applicable, frozen teacher weights for soft KL, P2
manifest, exact optimizer/topology settings, and source tree. Hard-SFT and
soft-KL save full model/optimizer/RNG/cursor state every 200 updates and refuse
cross-plan resume. Their transfer endpoint then enters an ordinary fresh P2
optimizer/cosine arc. Scratch replay forbids weights-only initialization and
uses the strict combined-manifest resume contract described above.

The original proposal did not specify the positive-transfer optimizer length
or LR. The approved fail-closed Exp4-v1 clarification is: hard-SFT and soft-KL
each make exactly one deterministic pass over the selected positive corpus at
a constant `1e-5`, using seed 42 and identical examples/order/batches. This is
an explicit post-plan clarification, not an originally specified
hyperparameter; that provenance string is embedded in every method-plan
manifest.

Read-only dry runs:

```bash
cd chess_reasoning
modal run modal_scripts/launch_exp4_interleave.py \
  --action extract --filter-setting U --dry-run

modal run modal_scripts/launch_exp4_interleave.py \
  --action train --method soft-kl --filter-setting D \
  --replay-path /checkpoints/REPLAY.jsonl \
  --replay-manifest-path /checkpoints/REPLAY.manifest.json \
  --dry-run
```

After RL1 exists, run `extract`, then the CPU-only `validate` action, before
submitting the same arguments with `--action train`. The launcher has no
implicit “latest replay” lookup: downstream stages must name the exact
content-addressed replay paths returned by extraction.

### 8.2 Exp4 unattended dependency controller

The root-level `exp4_autopilot.py` is outside the frozen Exp4 app source tree.
Its default invocation is a one-shot, read-only Modal report. Execution is
disabled unless `registry.exp4.review.status` is `approved`, the registry
records the exact controller SHA-256, and the same digest is supplied with
`--approved-controller-sha`. The checkpoint-publication migration candidate is
SHA-256
`a325a45b9fe1cfd511dc465e0aedcbe2fe34665249aacc69ec5d2f8e488a62d8`;
it remains pending independent root review.

For each U/D source, positive extraction remains blocked until all four gates
hold together:

1. the raw E1 RL1 tracker is at least 1,500 and the exact step-1,500 model
   directory exists;
2. `run_provenance.json` and every append-only launch record authenticate the
   exact U/D run, seed 42, 8-H200 optimized profile, balanced-data hash, P1
   origin, source manifests, runtime, and command;
3. rollout IDs 0 through 1,499 each have both an all-attempt positive JSONL
   and summary, with no temporary file; and
4. the clean step-1,500 HF conversion plus matching handoff manifest exists.

The only remote functions the controller can call are the pinned deployed
`extract_positive`, `validate_method`, and `train_method` functions on
`chess-50m-interleaved-exp4`, followed by pinned `train_hf` on
`chess-interleave-rl`. The latter always uses the content-addressed Exp4 final
HF endpoint, 1,500 updates, seed 43, save interval 40, the corresponding U/D
filter, the verified 131,072-token/GPU profile, and resumable Miles state.
Inline Miles eval stays disabled; the external evaluator discovers and queues
every durable step-40 checkpoint from the live registry feed.

The controller hydrates and checks the exact deployed function revision before
every spawn, not only the stable Modal app ID. Frozen function IDs are
`fu-QsFalvVx0u8sr0BLW7q9s3` (extract),
`fu-znHC5moS3ODGa8FpsghXZN` (validate),
`fu-h6CzrZ0rX88nTpOO8JMCfp` (Exp4 train), and
`fu-WQtDLWXkoYvMjVaFSqt1iR` (final RL). A same-name app redeployment therefore
disables submission until its new revision is reviewed.

The controller also fail-closes on the exact shared `registry.fixed_rl`
production contract: eight H200s, 131,072 tokens/GPU, SGLang concurrency 128,
gradient checkpointing disabled, 192 GB host memory, balanced-data hash,
source-tree SHA-256
`38f829c60815cb8f7a07776561af51cc22a8b6740c431c59bd3d9847ff4c019f`,
Miles source-tree SHA-256
`9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`,
incremental Modal-volume checkpoint publication every 5 seconds gated on the
tracker, model metadata, RNG state, and checkpoint metadata readiness markers,
the digest-pinned `radixark/miles` runtime image, the exact runtime package
versions and installed-package-set hash, and the two recorded profile-gate call
IDs. It passes concurrency 128 explicitly and binds the complete contract into
every final-RL invocation and live-feed provenance expectation. Any missing,
extra, or changed `fixed_rl` field disables all Exp4 submission until a new
review.

Immediately before each final-RL spawn, the controller streams the actual
content-addressed Exp4 HF endpoint from the Modal checkpoint volume and
recomputes the launcher's exact selected-file fingerprint (including the exact
47.245M architecture check). It refuses the spawn unless that observed digest
equals `complete.json` / the immutable invocation's `final_hf_sha256`. This
closes the same-path checkpoint mutation window.

Before any remote spawn, the controller atomically reserves an immutable
invocation contract in `registry.exp4.orchestration.calls`. A crash or
ambiguous spawn response becomes `submission_uncertain` and is never retried
automatically. Known terminal failures have a fixed three-attempt cap per
content identity at the controller-submission layer; the deployed functions'
own resumable Modal retry policy remains separately recorded. On restart under
the exclusive process lock, any orphaned `submitting` intent is converted to
non-retryable `submission_uncertain`. One process lock plus the pre-spawn
intent removes the duplicate-submission window.

Before each of the six final RL spawns, the same atomic registry write
populates the arm's nested RL record and `registry.exp4.final_rl_runs` with
its non-null content-addressed run ID, exact origin HF and SHA-256, plan hash,
target 1,500, effective offset 1,500, save/eval cadence 40, raw endpoint,
expected provenance, status, and call stage. The returned Modal call ID is
then attached to both records before it is reported as submitted, giving the
evaluator and dashboard a machine-readable live discovery feed.

Every Exp4 poll and every post-submission mutation is also published through
the pinned `chess-rl-live-dashboard/publish_live_control_plane` revision and
accepted only when the returned canonical registry SHA-256 matches the local
registry. Publication failure is visible and retried on the next poll; it never
silently advances a different evaluator feed.

## 9. Evaluation

Save/evaluate RL checkpoints every 40 effective RL updates. For Experiment 1,
offset second-leg checkpoint steps by +1500 in the result table.

Required result columns:

- experiment
- filter setting
- model/checkpoint identifier
- phase
- effective RL step
- pass@1
- average reward
- B3 average
- B4 average
- B3--B4 pooled/mean result

Also evaluate at the end of each pretraining stage:

- held-out pretraining loss/perplexity;
- held-out masked SFT loss/accuracy;
- chess benchmark pass@1 and average reward.

Publish the final tables and training/evaluation status on the existing Modal
dashboard endpoint.

## 10. Reproducibility and safety requirements

- Freeze the exact `chess_reasoning`, `chess-rl-miles`, and Miles source
  snapshots used for launch.
- Give every arm and stage a unique model ID and checkpoint path.
- Record dataset revisions, checksums, manifest hashes, seeds, and effective
  configs beside every checkpoint.
- Save clean Hugging Face checkpoints at every pretrain-to-RL transition.
- Save complete optimizer/scheduler/RNG/data-cursor state for in-stage retry.
- Refuse cross-arm or cross-stage auto-resume.
- Run a small Modal canary covering:
  - mixed pretrain and SFT updates;
  - checkpoint save/reload;
  - clean HF export;
  - one optimized Miles RL update;
  - positive-rollout extraction;
  - one hard-SFT and one KL-distillation update.

## 11. Launch order

1. Stage and checksum the balanced RL parquet.
2. Build deterministic pretrain/SFT manifests and validate all 77,717 rows
   physically present in the pinned source revision.
3. Implement the HF mixed trainer and explicit Miles dataset/seed options.
4. Run the Modal canary.
5. Launch the shared `P1` and Experiment 2 pretraining roots.
6. Launch unfiltered controlled RL arms as dependencies become ready.
7. Launch dynamically filtered controlled arms as dependencies become ready.
8. Extract positive rollout corpora from RL-1500.
9. Launch Experiment 4 hard-SFT, KL, and scratch-replay arms.
10. Evaluate every 40 RL steps and update the dashboard.
