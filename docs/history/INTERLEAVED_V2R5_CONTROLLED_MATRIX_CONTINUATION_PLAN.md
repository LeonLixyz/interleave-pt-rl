# Interleaved v2r5 controlled-matrix continuation plan

Status: **draft; scientific requirements frozen here, execution not authorized**

Drafted: 2026-07-30 14:03 EDT

Proposed version: `mix10b_sft90k_3072_v2r5_controlled_20260730`

This document defines the minimum continuation needed to make the clean-v2
comparison scientifically interpretable. It does not approve a Modal GPU
launch, mutate the core registry, replace a prior artifact, or authorize a
downstream stage. Before execution, an independent review must freeze the
remaining implementation and artifact identities listed in section 11 in a
self-hashed launch contract.

The behavioral and protocol thresholds below were inherited from contracts
frozen before the v2r4a rollout outcomes:

- the 2026-07-30 07:42 EDT v2r2 staged-gate contract; and
- the v2r4/v2r4a contracts, whose scientific rules were frozen before their
  new prompt batches ran.

No threshold in this plan was selected from a v2r4a reward result.

## 1. Scientific scope

The controlled clean-v2 matrix is:

| Experiment | Pretraining trajectory | RL trajectory per filter |
| --- | --- | --- |
| E1-v2 | P1 5B -> RL1 -> P2 5B | 1,500 + 1,500 updates |
| E2-v2 | one monolithic P1+P2 10B cosine | 3,000 updates |
| E3-v2 | P1 5B -> fresh optimizer/cosine -> P2 5B | 3,000 updates |

Each experiment has an unfiltered `U` arm and a Miles dynamic
nonzero-reward-variance `D` arm, for six controlled RL arms. E1-U and E1-D
have separate P2 trajectories because their RL1 endpoint weights differ.

Experiment 4 remains a separate FLOP-unbounded positive-rollout-transfer
study. It is downstream of E1 RL1 and is not part of the six-arm controlled
matrix.

This remains a behavioral comparison, not an equal-FLOP comparison. In
particular, E1 inserts RL compute before P2. E1 can therefore answer whether
the interruption helps the later model under the proposed procedure, but it
cannot isolate an RL benefit at equal total compute.

## 2. Existing v1 E2/E3 are not clean-v2 controls

The completed/current E2 and E3 artifacts in
`INTERLEAVED_CORE_REGISTRY.json` belong to
`mix10b_sft90k_3072_v1_20260730`:

- E2-v1 monolithic endpoint:
  `/checkpoints/interleave_50m/pretrain/mix10b_sft90k_3072_v1_20260730/exp2_monolithic/final`,
  endpoint fingerprint
  `22ce8af7277d0c2fb1e1e603fb686f6a947cdd79bfd94aaa01fecdff86079a0b`;
- E3-v1 two-cosine endpoint:
  `/checkpoints/interleave_50m/pretrain/mix10b_sft90k_3072_v1_20260730/p2/exp3-two-cosine-control-from-p1-from-d8315ae0645b/final`,
  endpoint fingerprint
  `d7f6be3ced127707f365b9aec7da72c07894f630726c25fd1349dccdc5a26efc`.

They used the v1 uncleaned SFT cache and raw-token-mean objective. The clean-v2
P1 instead removes verifier-score annotations and uses an explicit weighted
PT/SFT objective. Those changes alter both targets and loss mass. Therefore:

- E2-v1 and E3-v1, including any RL derived from them, remain immutable v1
  evidence;
- they must be labeled `v1_noncomparable_to_controlled_v2`;
- they must not be pooled with, relabeled as, or substituted for E2-v2 or
  E3-v2; and
- the clean-v2 controlled comparison requires fresh E2-v2 and E3-v2
  pretraining plus fresh downstream RL.

This plan does not cancel, delete, overwrite, or otherwise mutate any v1 run.

## 3. Fixed model, data, and stream identities

### 3.1 Model and tokenizer

All controlled-v2 arms use the same 47,245,312-parameter Qwen3 model:

- 12 layers;
- hidden size 512;
- 8 query heads and 4 KV heads;
- head dimension 128;
- intermediate size 1,536;
- Q/K norm enabled;
- tied input/output embeddings;
- `LanTokenizerSFT`, vocabulary size 85;
- model/context length 3,072; and
- SDPA with no `torch.compile`.

The current reference config is
`chess_reasoning/config/configs/interleaved_50m/base_3072.yaml`, SHA-256
`3ec2303cca8ada094124be8d36c380640b0a2cb8fa6001dc3b1d08d20d46a518`.
That file hash is provenance, not launch authorization: the final contract
must separately pin the resolved config, generated HF `config.json`, exact
tokenizer files, and the common step-0 model-weight identity.

E2-v2 must start from byte-identical model weights to the step-0
initialization that produced the clean-v2 P1 trajectory. Seed equality alone
is insufficient if code or library versions differ. If that exact step-0
weight identity cannot be reconstructed and authenticated, P1 and E2-v2 must
both be retrained from one newly frozen common initialization.

### 3.2 Base pretraining source

- HF dataset: `chess-pre-to-post/pretrain_v1_20b`
- Revision:
  `07dd1b7090ca5f0fb05ef624c26b20bff19483c8`
- Exact source tokens: 53,970,293,905
- Source shards: `raw.0000.npy` through `raw.47089.npy`
- Flat Modal filename/size manifest SHA-256:
  `07ae91cded540a00e9b6554d1d54ed46310715b7fd68e3520a64b7f5967f99aa`
- Selection seed: 42
- Controlled budget: exactly 10,000,000,000 predicted PT tokens

Source chunks are concatenated across shard boundaries and packed into
3,072-token training records. A full record uses a 3,073-token source window:
3,072 inputs and the already aligned next 3,072 labels. There is no
trainer-side second shift. P1 and P2 each contain exactly 5B predicted PT
tokens.

### 3.3 Cleaned SFT source

- HF dataset: `chess-pre-to-post/sft_v1_200m_90k`
- Revision:
  `97f60746dd253b4e130beeb5e66f39e9d42ef25c`
- Physical rows: 77,717
- Prompt field: `pgn`
- Response field:
  `cot_by_method.trajectory_sep.cot_format_no_labels`
- P1/P2 row split: 38,858 / 38,859, disjoint exact union

Before tokenization, numeric verifier-score pairs are removed with:

`\s*<verify>\s*<[+-]?(?:\d+(?:\.\d+)?|\.\d+)>`

The cache must fail closed on any residual `<verify>` tag, supervised
`<unk>`, missing row, missing supervised `</T>`, missing supervised
`<call_env>`, or sequence overflow. Exact cleaned supervised targets are:

- P1: 26,289,598;
- P2: 26,193,155;
- full: 52,482,753.

SFT rows stay individually masked and padded; they are not concatenated with
one another.

### 3.4 Existing clean-v2 data artifact

- Artifact version:
  `mix10b_sft90k_v2r1_clean_verify_gate`
- Modal root:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate`
- Embedded manifest-set SHA-256:
  `6f2cc9093b2515e0a6a3aedc56a0cfd597c6b0f76933c5dbcd69eefd22440a23`
- `manifest_set.json` file SHA-256:
  `d2d741998a258ed1367587f922df07e0d7a2b46d906a965208c781e1380feb6e`
- P1 metadata/order SHA-256:
  `b3a67af83912a6f82290b23ff7463b22e9cb9cad6403e9d2a54c783d588a55ba` /
  `68fc5a3934ea677f31365d998f67380e7b5d2fa12f7b5bbbed9756a3f8bd9ac4`
- P2 metadata/order SHA-256:
  `2536c129a5bbd04c082533b9a4ffed2d318723ea8ac3dec6b85583f217691eed` /
  `95d70dfc4474be1b8b301196875db511f9c9d63036332ca6913dd0504d7c17b8`
- SFT cache metadata SHA-256:
  `48b30362e729603798a14daa2c9f42e484fd68942a689c763d18095e8f3baeac`
- SFT cache input/label/offset SHA-256:
  `c8c75b6eec58c6d9943a799d04f3e054221f4e2207873b521e5b8eae548bb8a8`,
  `7bb6b16fdd6a7fe1b1e0702f21e9535334421a5c12d074848f60f8d76d357373`,
  and
  `0c6f777a79ae8f0d397f1e623724e30137fa5c89060efed1ba24e5ce48c83701`.

Each leg is one deterministic sample-level shuffle of PT records and its SFT
half. There is no SFT-only stage and no SFT oversampling:

- 1,627,604 full PT records plus one 512-valid-token residual record;
- 38,858 P1 or 38,859 P2 SFT rows;
- 97 P1 or 96 P2 all-ignore divisibility-padding records;
- 1,666,560 manifest records; and
- 9,920 optimizer updates at global batch 168.

E2-v2 consumes the exact `P1 || P2` manifest concatenation. E3-v2 consumes
P1 through the authenticated shared P1 endpoint and then the exact P2
manifest. No arm may rebuild, reshuffle, omit, replace, or append a data row.

### 3.5 RL data

- Dataset:
  `train_v4_dataset_balanced_multi_turn.parquet`
- Rows: 53,225
- SHA-256:
  `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`
- Modal path:
  `/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet`

Every launch names this path and verifies this digest. Falling back to the
62,290-row easy-skewed dataset is a hard failure.

## 4. Fixed pretraining objective, topology, and schedules

The clean-v2 loss is:

`L = (sum(PT CE) + w_sft * sum(SFT CE)) /
     (N_PT + w_sft * N_SFT)`.

Both numerator terms and the weighted denominator are globally reduced over
all DDP ranks. The run logs unweighted PT CE, unweighted SFT CE, PT/SFT target
counts, the coefficient, and effective task shares.

Fixed coefficients:

- P1: `190.189290837`;
- P2: `190.889566377`;
- monolithic P1+P2: `190.538785189`.

These coefficients give equal integrated PT/SFT loss mass within their
respective streams. Their small numeric difference is fixed by the different
clean target denominators; it is not a tuned arm-specific hyperparameter.

Common topology and optimizer:

- 8 H200 GPUs;
- local batch 21 per GPU, global batch 168;
- gradient accumulation 1;
- BF16 autocast with FP32-loaded/master weights;
- AdamW, peak LR `1e-3`, weight decay `0.1`, betas `[0.9, 0.95]`;
- gradient norm cap `1.0`;
- 5% warmup, cosine decay to `1e-5`.

Schedules:

- each P1/P2 arc: 9,920 updates, 496 warmup updates;
- E2 monolithic arc: 19,840 updates, 992 warmup updates;
- E2 has no optimizer or scheduler reset at step 9,920;
- E3 and each E1 branch load only model weights at the P1/RL1 boundary and
  start P2 with a fresh AdamW optimizer, scheduler, RNG contract, and
  9,920-step cosine arc.

## 5. Fixed clean-v2 P1 origin

The only full 5B P1 snapshot proposed for the original E1/E3 transition is
the weight-`190.189290837` step-9,920 snapshot from training call
`fc-01KYSKQESVN6S1SXSRXFTAHM4M`:

- HF path:
  `/pretrain-checkpoints/interleave_50m/pretrain/mix10b_sft90k_3072_v2r3_diagnostic_20260730/p1_w4067c60eaba84b1e/snapshots/step_9920/hf`
- Recursive HF identity:
  `d0c013bf51c17691ef9bdf5e5d65561912471ef949a161f80b4aa818da96c4fd`
- HF directory-manifest SHA-256:
  `49fe6fe87d78ba58ebd96cf154567bd1526b6c12a4193809652b875a7af5d186`
- Endpoint-evaluator fingerprint:
  `9a89d52a60b87b0f27108e5b08e33395757e374a4b59a592babb9435edb4b1c8`
- Training-state SHA-256:
  `b042157d06bb7b89b1a69cb190cbde1e5d17455e76eda9f8c15639d90d4c05b7`

This identity is fixed, but its transition to E1 RL1 remains contingent on
the immutable v2r4a report. Steps 6,000 and 8,000 are diagnostics and may not
replace step 9,920 in the controlled matrix.

## 6. E2-v2 monolithic canary and full trajectory

E2-v2 must be fresh. It may not reuse E2-v1 weights, optimizer state, data
cursor, or RL artifacts.

### 6.1 Step-2,000 canary

Before a full 19,840-update E2-v2 trajectory is allowed:

1. Start from the exact common step-0 model weights.
2. Use the clean-v2 `P1 || P2` manifest, monolithic coefficient
   `190.538785189`, and the full 19,840-step schedule. The canary must not
   rescale a cosine schedule to 2,000 steps.
3. Stop after update/cursor 2,000 and atomically preserve full model,
   optimizer, scheduler, scaler, RNG, and data-cursor state plus a clean HF
   export.
4. Require all data, topology, global-loss, finite-loss, target-count,
   coefficient, source, and snapshot-identity checks to pass.
5. Run two strict rollout-only protocol audits. Each audit has exactly 256
   unique prompt groups, eight siblings per group, and 2,048 rows.

The two prompt sets must be disjoint from one another, from the v2r3 seed-42
diagnostic set, and from both v2r4/v2r4a A/B sets. Before the canary training
call starts, the final launch contract must pin an outcome-independent
hash-ranking rule, exact source-row inventory, parquets, prompt fingerprints,
prompt-set hashes, rollout seeds, sibling/sample seeds, and zero-intersection
proofs.

Audits use deterministic inference, no dynamic filter, no policy update, no
prompt replacement, no dataset wrap, and no automatic retry. Every audit
must independently contain:

- exactly 2,048 authenticated completed/truncated rows in 256 groups of 8;
- at least 32 joint-valid protocol rows; and
- those joint-valid rows spanning at least 16 prompt groups.

A joint-valid row has `</T>` before `<call_env>` in the same output and a
nonempty parser-produced `extracted_moves`. These thresholds are copied from
the pre-v2r4a v2r2 contract. Positive reward is reported but is not required
at the step-2,000 protocol gate.

Pooling cannot rescue a failed audit. Any audit failure blocks the full E2-v2
trajectory. A replacement requires a new written, pre-outcome amendment with
fresh identities.

### 6.2 Continuation to 19,840

If both audits pass, the preferred production action is to resume the exact
authenticated step-2,000 model/optimizer/scheduler/RNG/cursor state and
continue the same trajectory to update 19,840. A weights-only restart is
forbidden.

If operational reasons require replaying from step 0, the replay must use the
same immutable initialization, source/runtime, data order, and deterministic
state and reproduce the already authenticated step-2,000 weight and state
fingerprints before advancing. Otherwise it is a new experiment version.

Completion requires exact cursor 19,840, LR floor `1e-5`, a complete clean HF
export, and full provenance. The endpoint then enters section 8's endpoint
and RL-feasibility gates before either E2-v2 RL arm can launch.

## 7. E3-v2 two-cosine P2

E3-v2 P2 must:

1. load model weights from the exact section 5 step-9,920 HF identity;
2. discard P1 optimizer, scheduler, RNG, and cursor state;
3. consume only the exact clean-v2 P2 manifest;
4. use SFT coefficient `190.889566377`;
5. run a fresh 9,920-update AdamW/cosine arc with 496 warmup updates,
   peak `1e-3`, and floor `1e-5`; and
6. save complete resumable state and a content-addressed clean HF endpoint.

It may not reuse E3-v1, E2-v1, E2-v2, or either E1 RL1 endpoint. The final
launch contract must pin a new E3-v2 run ID and output root so that no
cross-version resume is possible.

E3-v2 P2 can be prepared independently of E1 RL1 because it is the
no-midpoint-RL control. It still requires an independently approved v2r5
implementation contract and the exact P1 identity checks above. If E1 cannot
complete, E3-v2 remains a valid endpoint artifact but the intended E1 versus
E3 controlled comparison is incomplete.

After E3-v2 P2 completes, it must pass section 8 before E3-U-v2 or E3-D-v2
RL can launch.

## 8. Endpoint and RL-feasibility gates

Every pretraining endpoint that will initialize RL must pass a fail-closed
gate. This applies to:

- the P1 step-9,920 endpoint before E1 RL1, through v2r4a;
- each E1-U/E1-D post-RL1 P2 endpoint before its RL2;
- the E2-v2 monolithic endpoint before E2 U/D RL; and
- the E3-v2 P2 endpoint before E3 U/D RL.

For endpoints other than P1 step 9,920, the gate requires:

1. finite held-out PT CE/perplexity/accuracy on the immutable 4,096-record,
   12,582,912-target holdout with SHA-256
   `c6f1ed19085c43987775e2013c3dd9a687b04138ec199dc583c1b382a0b4df02`;
2. exact completion of the official 23,680-row B1--B5 evaluation, with at
   least one positive row overall;
3. two fresh, mutually disjoint 256-prompt × 8-sibling rollout-only audits;
4. at least 8 positive samples in each audit; and
5. at least 8 nonzero-reward-variance prompt groups in each audit.

The two audit manifests for each endpoint must be selected by an
outcome-independent, arm/stage-domain-separated hash rule and frozen before
the endpoint metrics are read. They must be disjoint from prior
selection/gating prompt sets. Exact prompt intersection, row inventory,
generation status, binary rewards, protocol parses, and artifact hashes are
audited independently. Pooling cannot rescue one failed batch.

The P1 step-9,920 v2r4a gate is already a separately frozen, stronger
two-batch gate. If its final immutable report authorizes E1 RL1, no third P1
audit is required.

Before a `D` production run, a no-update dynamic-filter preflight must also
fill all 256 accepted nonzero-variance prompt groups within at most 8,192
attempted prompt groups. The dynamic filter retains only groups with 1--7
successes among 8 siblings; it drops both all-failure and all-success groups.
Failure to fill the batch blocks only that `D` branch and cannot be bypassed
by switching datasets or accepting a partial batch.

The 8-positive, 8-variance, and 8,192-attempt thresholds are inherited
verbatim from the pre-v2r4a v2r2 contract. They are operational feasibility
gates, not claims about official benchmark quality.

## 9. E1 downstream dependency gates

### 9.1 RL1

Only a terminal, independently authenticated v2r4a report with
`analysis.authorization.pass=true` and `eligible_step=9920` may authorize two
new, versioned E1 RL1 calls:

- E1-U-v2: 1,500 updates, seed 42, dynamic filter disabled;
- E1-D-v2: 1,500 updates, seed 42, dynamic filter enabled.

Both must start from the exact section 5 HF bytes. The terminated v1 E1 calls
cannot resume or supply optimizer/RNG/rollout state.

Each RL1 run must reach exactly 1,500 policy updates and publish:

- every raw rollout and all-attempt positive artifact for rollout IDs
  0--1,499;
- complete append-only launch and run provenance;
- durable checkpoints every 40 steps plus an exact step-1,500 endpoint;
- a content-addressed clean HF step-1,500 export; and
- the external checkpoint evaluations required by section 10.

A failed `D` feasibility/acceptance gate does not block the `U` branch.

### 9.2 E1 P2

E1-U P2 and E1-D P2 are distinct runs. A branch may start only after its own
RL1 endpoint and provenance are complete and independently authenticated.
It loads only that branch's exact step-1,500 HF model weights and then uses:

- the exact clean-v2 P2 manifest;
- coefficient `190.889566377`;
- a fresh AdamW optimizer and fresh 9,920-step/496-warmup cosine;
- no RL optimizer, reference-model, scheduler, rollout cursor, or RNG state.

Each P2 endpoint must pass section 8 before its corresponding RL2 starts.

### 9.3 RL2

Each surviving branch starts a fresh RL optimizer and a reference model equal
to its exact P2 endpoint:

- 1,500 local updates;
- rollout seed 43;
- the same U/D filter setting as its RL1;
- effective analysis/dashboard step offset `+1500`.

RL2 may resume only within its own branch and leg. It may not restore RL1
optimizer, scheduler, reference, RNG, or rollout cursor.

### 9.4 Experiment 4

Positive-rollout extraction and all Exp4 methods remain blocked until the
corresponding E1 RL1 has all 1,500 paired rollout/summary artifacts, immutable
provenance, step-1,500 raw checkpoint, and clean HF export. Exp4 has its own
content-addressed extraction/training contracts and is never pooled with the
controlled six-arm compute comparison.

## 10. Fixed RL recipe and evaluation

The semantic RL recipe is:

- optimized Miles/SGLang path;
- 8 H200 GPUs per run;
- 256 prompts/update × 8 samples = 2,048 trajectories/update;
- one GRPO optimizer update per rollout update;
- token-mean loss;
- AdamW LR `1e-5`, betas `[0.9, 0.999]`, epsilon `1e-8`,
  weight decay `0.01`;
- KL coefficient `0.001`, low-variance KL;
- temperature/top-p `1.0/1.0`;
- prompt/response/context caps `512/2560/3072`;
- 131,072 tokens/GPU;
- SGLang concurrency 128;
- gradient checkpointing disabled;
- host memory 192 GB;
- balanced dataset and checksum from section 3.5.

The currently recorded verified implementation provenance is:

- chess-rl-miles source-tree SHA-256
  `38f829c60815cb8f7a07776561af51cc22a8b6740c431c59bd3d9847ff4c019f`;
- Miles source-tree SHA-256
  `9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`;
- runtime image
  `radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e`.

Those digests are reference provenance for the verified recipe. The v2r5
launch contract must pin the exact new source bundle actually mounted. If it
differs because of additive gate/finalizer work, review must prove that
training-relevant bytes are unchanged or approve the successor identity.

Save and evaluate every 40 effective RL updates. Also evaluate the exact final
step even when it is not divisible by 40. The primary final-RL comparison is
the exact step-3,000 endpoint; intermediate checkpoints are learning-curve
evidence and may not be cherry-picked as the final model.

Required result columns are:

- experiment/version and filter;
- exact model/checkpoint identity;
- phase and effective RL step;
- official B1--B5 Pass@1 and average reward;
- B3 and B4 values; and
- B3--B4 pooled/mean value.

At every completed pretraining endpoint, also report held-out PT
loss/perplexity/accuracy and official B1--B5 results. The 4,096-row P2 SFT
subset used to evaluate P1 was later consumed during P2, so it must not be
called a held-out final-SFT test. Final SFT generalization requires a new
external, pre-frozen dataset; otherwise SFT metrics are explicitly
training-set/descriptive only.

Comparisons are made within filter setting:

- E1-U versus E2-U/E3-U, and separately E1-D versus E2-D/E3-D;
- final-pretraining endpoints before the final RL leg; and
- final effective RL step 3,000.

All models use the same official benchmark rows and evaluator identity.
Paired row-level uncertainty/tests and multiplicity handling must be frozen
before reading the new endpoint results. With one model/data seed, results
are a controlled single-seed case study; a population-level claim requires
pre-registered additional seeds.

## 11. Identities and implementation still required before launch

The following are intentionally not invented by this draft. A final
self-hashed v2r5 execution contract and independent audit must freeze them
before any GPU submission:

1. Exact common step-0 model weights and recursive/file fingerprints, plus
   proof that the existing P1 trajectory derives from those bytes. If that
   proof is unavailable, freeze a fresh common initialization and retrain the
   affected P1/E2 roots.
2. Resolved E2-v2 and E3-v2 configs, run IDs, output roots, snapshot names,
   completion-marker schemas, and cross-version resume rejection rules.
3. Exact pretraining launcher/trainer source-tree manifest, container digest,
   Python/package lock, Accelerate/torch/transformers versions, DDP launch
   command, environment, and generated HF/tokenizer/config file identities.
4. A tested E2 monolithic stop-at-2,000/full-state-resume path that preserves
   the 19,840-step optimizer/scheduler semantics, plus an exact state
   fingerprint equivalence test.
5. Exact E2 canary audit A/B prompt manifests, parquets, seeds, ordered
   fingerprints, disjointness proof, rollout command, source manifests,
   runtime contract, artifact validator, and one-call/no-retry ledger.
6. Exact E3-v2 P2 origin-to-output plan hash and proof that only model weights
   cross the P1/P2 boundary.
7. Exact endpoint-evaluator bundle and result namespaces for E2-v2, E3-v2,
   and both E1 P2 endpoints; outcome-independent endpoint feasibility audit
   manifest generation and all resulting hashes.
8. Exact production RL source bundle/function revision, commands, U/D run
   identities, reference-policy identity, save/final-HF conversion code,
   checkpoint publication schema, and retry limits.
9. Exact official evaluation source/data fingerprints, paired-analysis plan,
   family of comparisons, multiplicity correction, and dashboard/table
   publication identity.
10. A machine-readable dependency ledger whose only state transitions are
    backed by immutable success artifacts and independent checks. No
    `latest` lookup or manual status interpretation may authorize a stage.

Required prelaunch verification includes focused unit tests, full
source/data rehashes, CPU-only Modal dry runs, one minimal mixed-training
save/resume/export canary, one optimized Miles rollout/update canary, and an
independent read-only review. Any implementation or identity drift after
freeze requires a new versioned amendment.

## 12. Launch DAG after final approval

The allowed dependency order is:

1. Freeze and independently audit the v2r5 execution contract and common
   initialization identity.
2. Run the fresh E2-v2 monolithic prefix to step 2,000.
3. Run and independently audit both disjoint E2 protocol batches.
4. If both pass, resume the exact E2 state to step 19,840.
5. In parallel after its own contract approval, run E3-v2 P2 from the exact
   step-9,920 P1 weights.
6. If and only if the v2r4a report authorizes step 9,920, launch new E1-U/D
   RL1 calls.
7. Evaluate and apply section 8 to each E2/E3 endpoint before its U/D RL.
8. After each E1 RL1 endpoint authenticates, run its branch-specific P2;
   evaluate/gate that endpoint; then run RL2.
9. Evaluate every durable RL checkpoint at effective 40-step cadence and
   publish the complete table, never only a selected checkpoint.
10. Treat Exp4 as a separate downstream DAG after its RL1 source branch is
    complete.

Independent branches may run concurrently after their own dependencies pass.
Failure of one filter branch does not authorize a recipe change and does not
erase evidence from another branch.
