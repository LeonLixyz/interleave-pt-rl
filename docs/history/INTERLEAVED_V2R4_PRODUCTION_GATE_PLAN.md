# Interleaved v2r4 production gate contract

Frozen at: 2026-07-30 13:25 EDT

Contract schema:
`interleaved-v2r4-production-gate-contract-v1`

Contract version:
`v2r4_production_gate_20260730`

## Purpose and authorization boundary

This gate tests whether the corrected clean weighted P1 trajectory is usable
for the original interleaved experiment and measures how behavior changes
near its endpoint.

The authenticated weight-190.189290837 checkpoints at steps 6,000, 8,000,
and 9,920 are evaluated. Only step 9,920 consumed the complete planned 5B P1
stream. Therefore:

- step 9,920 is the only checkpoint eligible to authorize a replacement of
  the original 5B-P1-to-RL1 transition;
- steps 6,000 and 8,000 are comparative early-stop diagnostics;
- selecting an early checkpoint would create a separately versioned
  early-stop experiment and is not authorized by this contract;
- this gate cannot relabel or revive the two terminated E1 calls;
- it cannot directly authorize P2, RL2, or Exp4.

If and only if step 9,920 passes every absolute gate below, one immutable
approval may authorize exactly two new versioned RL1 calls: U unfiltered and
D with the verified Miles dynamic nonzero-variance filter, each for 1,500
updates from the exact authenticated step-9,920 HF checkpoint and the already
frozen optimized 8-H200 profile. Later P2 and Exp4 stages must satisfy their
own new-version dependency gates after those RL1 endpoints exist.

## Candidate identities

All three snapshots come from the same continuous training call
`fc-01KYSKQESVN6S1SXSRXFTAHM4M`, SFT loss weight `190.189290837`, and the
same cleaned P1 stream.

| Step | Recursive snapshot HF identity | HF directory manifest | Endpoint evaluator fingerprint | Original 5B P1 eligible |
| ---: | --- | --- | --- | --- |
| 6,000 | `5df40e4794193a490297e19837ea5d8ec49326329ab405e58234b67519862425` | `3285baeb7c6ca4de2a320522906b031f6538c75106cb13fddc68194c96d23d70` | `17acd19dd1e89390c609a3f0f6c72ab543b8869f2d2ffd10528c8fe84cb20690` | no |
| 8,000 | `d17a709df6debd483932e3e38214a91a1ec1f62814dd73dd6cad1f51a9b6070e` | `13fde44ba75511e8cd7d23a9e73db507bade841fcc682dc2851261690e918758` | `e1006a970b5b7c9c9e5aefdbae3c716740e69970c0bcb4bb32b4cbab7af43634` | no |
| 9,920 | `d0c013bf51c17691ef9bdf5e5d65561912471ef949a161f80b4aa818da96c4fd` | `49fe6fe87d78ba58ebd96cf154567bd1526b6c12a4193809652b875a7af5d186` | `9a89d52a60b87b0f27108e5b08e33395757e374a4b59a592babb9435edb4b1c8` | yes |

Every launch must rehash the complete HF directory and match the declared
identity before allocating useful GPU work.

## Frozen prompt construction

Source:

- `/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet`
- 53,225 rows
- SHA-256
  `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`

The exact candidate tokenizer must reproduce:

- 68 source rows longer than 512 prompt tokens;
- 53,157 eligible rows;
- 53,157 unique prompt fingerprints.

A prompt fingerprint is SHA-256 of compact, sorted-key canonical JSON over
the exact `input`, `FEN`, `PuzzleId`, and `ground_truth` strings.

The old v2r3 diagnostic set contains 256 unique prompt fingerprints and has
sorted-set hash
`9ab746d0039bcc15d3573296cbe4503650a10b9b3248ffae1e7bb4121663b7c7`.
Those prompts are removed before selection.

Rank the remaining 52,901 rows ascending by:

`SHA256("interleaved-v2r4-production-gate-20260730" + NUL + fingerprint)`

Batch A is ranks 0–1,023. Batch B is ranks 1,024–2,047. The immutable
self-hashed prompt manifest is:

- embedded manifest SHA-256:
  `8ce046f9a560c7227ad33cc5f2baecc79d210e6f703c73e895469c1d566c6af5`;
- file SHA-256:
  `a01bb692dd2f129c2463df91aab7006e4762a0ac55e6471f0708ef4db34ba126`;
- file size: 596,884 bytes.

| Batch | Parquet SHA-256 | Bytes | Prompt-set SHA-256 | Rollout seed | Epoch-0 prompt-order SHA-256 |
| --- | --- | ---: | --- | ---: | --- |
| A | `9002d22fd567a91de9d7a3a7ba2119d0a5e812a74d473d82dce2508c2eefd01d` | 1,064,180 | `8d2f389ba1df4aa1594d8abb894941723158b4c4d072e1b54a9681ac8a7b89a2` | 1,567,877,051 | `502dd02b274ef964b49d7e8b8fc187d12d9b8e27f90d5df6de03a7091d1c55d4` |
| B | `1f9031efe2ea071c18d4beccb0c6394d1c3e10a4d962bfc2773ac5ad20d3c79e` | 1,059,160 | `ceabf3581a9ea0bfbbe61430c22d24849d459aad03e17cbac356ee2c71ca9d74` | 923,570,888 | `663d0a51dfab347900dab5fe32bd55adc5b98d424306cd4108ab605e4b6c6eb2` |

The manifest must authenticate 1,024 unique rows in each batch and prove
`A ∩ B = A ∩ diagnostic = B ∩ diagnostic = ∅`. It also binds every source
row index, full-row hash, prompt-token count, epoch permutation, and each
rollout quarter's ordered fingerprint inventory.

## Exact rollout execution

Launch all six cells before inspecting or aggregating reward outcomes:

`{step 6000, step 8000, step 9920} × {batch A, batch B}`

Each cell is one 8-H200 call with:

- `num_rollout=4`;
- exactly four pulls of 256 prompt groups;
- eight siblings per group;
- exactly 1,024 groups and 8,192 rows total;
- no dynamic filter, partial rollout, policy update, eval, checkpoint save,
  resume, dataset wrap, aborted-sample requeue, or prompt replacement;
- deterministic inference with one unique sampling seed per trajectory:
  `batch_rollout_seed + global_sample_index`, indices 0–8,191;
- 131,072 tokens/GPU, SGLang concurrency 128, 192 GB host memory, and no
  gradient checkpointing;
- zero automatic retries and exactly one FunctionCall ID per cell.

The gate-specific source must fail immediately if any generation task raises.
It must not silently draw replacement prompts. Each of the four JSONL files
is written atomically and must contain exactly 2,048 rows, its exact declared
256-prompt quarter, global group indices 0–1,023, global sample indices
0–8,191, eight sibling indices, exact seeds, and only completed/truncated
statuses. A cell success marker authenticates shape and hashes but deliberately
does not aggregate reward metrics before the six-cell terminal barrier.
Any failed cell remains failed. A replacement requires a separately frozen
amendment; the sole launcher may not silently resubmit it.

Nominal rollout allocation is six nodes / 48 H200s, 49,152 trajectories, and
approximately 25–45 minutes wall time when capacity is available.

## Endpoint validation

All three candidates receive the following independently authenticated
endpoint evaluations:

1. Existing immutable PT holdout:
   - schema `interleaved-pt-heldout-v1`;
   - holdout hash
     `c6f1ed19085c43987775e2013c3dd9a687b04138ec199dc583c1b382a0b4df02`;
   - 4,096 records and 12,582,912 next-token targets from 32 whole source
     shards absent from training;
   - finite token CE, perplexity, and token accuracy are required.
2. Exact production B1–B5:
   - 23,680 rows must complete with the frozen evaluator identity;
   - overall Pass@1/average reward and every benchmark, including the B3–B4
     average, are reported;
   - step 9,920 must have at least one positive row overall.
3. P2 SFT validation held out at P1:
   - select exactly 4,096 unique cleaned P2 SFT codes by a frozen canonical
     hash rule;
   - prove they are disjoint from every P1 SFT code;
   - compute response-masked, unweighted loss sum, supervised-target count,
     CE, and token accuracy on each candidate;
   - bind the exact cleaned cache, P1/P2 manifests, code inventory, model,
     runtime, and output hashes.
   - the frozen selection hash is
     `99d20a1ee7dad9ab88ab5de2dfe0df50cc9d9e076636cf41252fbb1db2ea371e`;
     its cache-shape hash is
     `6b8b068a1d02480d9c0a9933c19a534bb64eb15fe16e9ae7a313f4ea66c4d5c5`;
     the denominator is 2,759,776 supervised targets over 3,560,000 aligned
     positions in exactly 4,096 rows.

The first P2 evaluator namespace
`v2r4_p2_sft_at_p1_20260730` is quarantined: all three simultaneously
launched calls failed before model scoring with the same missing
`python-chess` import. It exposed no candidate metric and cannot be promoted
or reused. The production rerun uses the fresh namespace
`v2r4_p2_sft_at_p1_20260730_v2`, app
`chess-interleave-v2r4-p2-sft-eval-v2`, pinned `chess==1.11.2`, corrected
full-position cache-aligned logits/labels, zero retries, and the exclusive
workspace ledger
`/Users/leonli66/Desktop/Research/RL/Chess RL/INTERLEAVED_V2R4_P2_SFT_V2_LAUNCH_LEDGER.json`.
The external production
contract must pin its exact evaluator bundle before those three v2 GPU calls
are launched.

The frozen v2 identities are:

- runner byte SHA-256:
  `28a0d9ed54bdb9fd8a2cc54db9ae9d07353bfaed79e821ab8021934d587fc0ea`;
- pure evaluator byte SHA-256:
  `d2130127ac50fc35644472263f22207f71a14796c6281669427814bdb57a9bf3`;
- evaluator bundle SHA-256:
  `65551135a0eb3eac5e1b65447a499c5849c154ec75f16db762903198eb2bf920`;
- evaluator runtime-contract SHA-256:
  `b4a69fc7c15df54b0ac8d4b89ba6d2fe9ec169568becfedf2615e47260a7bf3d`.

The no-GPU dependency preflight self-hash is
`90d2897d8821ee6f13c96d86e9bfd1099005288dbf29cdd41c5106485cf26269`;
the data/checkpoint/root preflight self-hash is
`47bcf7b78a3123961505a045548cc9be2840b776cf82ad32b5bc07167d1b4530`.

The P2 SFT subset is legitimate validation for these P1 snapshots, but it is
not a final SFT test. Once used for selection and later consumed during P2,
it cannot support a final-generalization claim.

## Absolute step-9,920 pass rule

Step 9,920 passes behavior only if it independently satisfies all of the
following in both A and B; pooling cannot rescue a failed batch:

- exact 1,024-group / 8,192-row authenticated inventory;
- at least 8,184 completed rows (99.9%), with all remaining accepted rows
  explicitly truncated and no failed/aborted/pending/unknown rows;
- at least 16 prompt groups with any positive sibling, equivalent to a
  one-sided Wilson 95% lower bound of at least 1% for solve@8;
- at least 16 prompt groups with nonzero sibling reward variance, equivalent
  to the same one-sided lower bound of at least 1%;
- at least 443 joint-valid protocol rows, equivalent to a one-sided Wilson
  95% lower bound of at least 5% for protocol-row rate.

Additionally, all three endpoint jobs for step 9,920 must be complete and
finite, and its exact B1–B5 evaluation must contain at least one positive.

No threshold was read from either new prompt batch; the thresholds were
frozen from the prior diagnostic's operational feasibility question.

## Paired comparisons and claims

After all six rollout cells and all endpoint calls are terminal and
authenticated:

- primary paired outcome for comparisons is prompt-level solve@8:
  `Y=1` iff any of eight siblings has binary reward 1;
- run exact two-sided paired McNemar comparisons for all three candidate
  pairs: 6,000 versus 8,000, 6,000 versus 9,920, and 8,000 versus 9,920;
- pool A and B only after both batches are complete, preserve the per-batch
  estimates, and apply Holm family-wise correction at 0.05 across all three
  pooled contrasts;
- require effect directions to agree in A and B before describing a result
  as replicated;
- a behavioral winner may be reported only when it has the unique largest
  pooled solve@8 rate, Holm-significant wins against both alternatives, and
  a positive direction separately in A and B; otherwise the comparison is
  explicitly inconclusive with no selection;
- sample-row reward rate, protocol rate, solve given protocol, variance rate,
  and other pairwise tests are secondary/descriptive.

This 2,048-prompt design has approximately 71.2% power at the conservative
`alpha=1/60` multiplicity allocation for the observed v2r3
step-6,000-versus-step-9,920 effect. Roughly 2,560 prompt groups per
checkpoint would raise that power to approximately 82.5%. The present grid
is therefore underpowered for unrestricted three-candidate selection.
No result from this grid may select step 6,000 or 8,000 for the original
experiment. A powered early-stop selector requires a new contract and
pre-result PT plus P2-SFT noninferiority margins; the PT and P2-SFT CEs in
this gate are descriptive because no such margins were frozen.

The rollout binary-reward rate is not called official benchmark Pass@1.
Official Pass@1 is reserved for the exact B1–B5 evaluator.

## Fail-closed implementation and launch gate

Pinned implementation identities at freeze:

- `chess_rl_miles/gate_data_source.py`:
  `6a6c8daf44cd1c93b61f92437f2c2d4524daa514aa566033896f2b204d7f55e9`;
- `chess_rl_miles/batched_rollout.py`:
  `72a61ac79d3cf6db36a2576195f38175645d8902a3a2131e602b89f3ecc2c9d6`;
- `chess_rl_miles/io.py`:
  `0c28931bbec8ed86b6562dd2c4aa8505dada25c00253f90afbfab82bbe5914c6`;
- `scripts/run_chess_miles.py`:
  `c3a5503ff1f2bf39b5f200ef9da12032f194681ec0146db60ea1a53766bf03f5`;
- `scripts/modal_interleave.py`:
  `8eeff0ed079f42d86c3a8c1c0c4bda2aa4f3a1cb69097866f935dec608bf93d3`;
- complete `chess-rl-miles` source manifest:
  `8740d3a25a2a684ba8b6333f0dd7619b975214df95611de6180870e1d0b11491`;
- unchanged Miles source manifest:
  `9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`.

The immutable aggregate analyzer is
`Eval/v2r4_gate_analysis.py`, SHA-256
`2ddd308c5221ce07dd1dce04f5649f0a05b60f7d819a5cb9e5f71411a6f05615`.
Its focused tests pass 4/4 and enforce the six exact inventories, binary
rewards, joint-valid protocol rule, paired prompt order, all three exact
McNemar comparisons, Holm correction, unique-winner rule, endpoint
finiteness, step-9,920 authorization boundary, and report self-hash.

The exact PT/B1–B5 endpoint evaluator bundle is
`80caa51691611ad89a2496e2ca89f1c4039777d1d1a9b663fc550a26cff585f0`.
Its exclusive six-call launch ledger is
`INTERLEAVED_V2R4_ENDPOINT_LAUNCH_LEDGER.json`, embedded SHA-256
`e150ec1f66670301a789f736a1522ae96e8abf490d94cd43ab2d98978ffad032`.

The gate uses an external immutable, self-hashed runtime contract. Its sole
two-digest binding module is excluded from the project source-tree manifest
to avoid a circular digest; every other source/config file is included. The
runtime contract binds the plan, both endpoint evaluators, exact project and
Miles source manifests, pinned image digest/profile, candidates, prompts,
semantics, and all six authorized cells.

Before launch, all of the following must pass:

1. an independent read-only audit reproduces both prompt parquets and all
   manifest hashes;
2. an independent code review approves the strict no-replacement source,
   task-exception behavior, atomic writes, exact command, and artifact
   validator;
3. a no-GPU Modal dry run prints the exact frozen command;
4. both prompt parquets and their manifest are uploaded and re-downloaded
   from the target Modal volume with identical bytes;
5. one CPU-only preflight authenticates the runtime contract and proves all
   six canonical run roots absent before any GPU cell is spawned;
6. the endpoint evaluator and P2-SFT validation implementations pass focused
   tests and bind their exact source identities.
7. one local controller atomically creates an `O_EXCL` launch ledger, records
   the preflight call/result, launches the six cells once in frozen order, and
   records each unique FunctionCall ID immediately.

Any call-count, prompt, order, seed, status, source, data, model, runtime, or
hash drift fails closed. There are no automatic retries. Any manual
replacement call requires a separately frozen amendment. The final aggregate
report is immutable, self-hashed, and must bind every successful Modal call,
every artifact hash, both per-batch absolute gates, and the paired McNemar/Holm
analysis.
