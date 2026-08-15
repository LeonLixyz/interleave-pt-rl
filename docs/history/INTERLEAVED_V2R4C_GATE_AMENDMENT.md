# Interleaved v2r4c fresh-prompt production-gate amendment

Freeze state: `FROZEN_SUCCESSOR`

Freeze snapshot version: `v2r4c-source-freeze-s3`

Frozen source snapshot: `2026-07-30T15:46:05-04:00`

Independent full-transitive successor science source gate: `PASS`

This successor freeze supersedes the source-freeze-s2 amendment with SHA-256
`8d3dfaa3e464bc565a6a47b052230253e73da418cc2ea5d92924a909e8c0dcee`
and its invalid local-only runtime contract, which omitted the substantive
base analyzer binding, with canonical SHA-256
`e251b91b9d79c316169825f4c9dfeae3dc26c923b959049dc5d1b4a08e17c995`
and file SHA-256
`1e9414d8296206892bfa5e22ddbdea4b74f9dd2f95e4f39956ede1175b7b4eb8`.
That contract was never staged, uploaded, or used for a launch, and the
contract binding remains its all-zero fail-closed placeholder. Source-freeze
s3 binds the complete analyzer chain
`v2r4c_gate_analysis.py -> v2r4b_gate_analysis.py -> v2r4_gate_analysis.py`
in the plan, builder, launcher, and finalizer.

For completeness, source-freeze s2 had superseded the source-freeze-s1
amendment with SHA-256
`fee6e426688c80efa22a8e28d871b184f4a8dedb7625cf8dcd90bb90d4cd9c05`
and the invalid local-only runtime contract with canonical SHA-256
`9fa99dee7a61be7cad418d6e40fdbea37e6d8e0121bd1e5605a207f382bf828e`
and file SHA-256
`5f3b81666879b7998f49c6596353af81af27e560b7b0a6c793da11d204bbd325`.
That contract was never staged, uploaded, or used for a launch, and the
contract binding was reset to its all-zero fail-closed placeholder. The only
successor correction canonicalizes the quarantined pre-barrier exposure tuple
as a JSON list in `_contract_static()`, making the static contract exactly
equal to its JSON round trip. It changes no disclosure value or order,
scientific semantic, prompt, seed, checkpoint, threshold, outcome barrier, or
authorization rule.

Contract schema:
`interleaved-v2r4c-production-gate-amendment-v1`

Contract version:
`v2r4c_production_gate_20260730`

## Scope and authorization boundary

This amendment supersedes the unfrozen v2r4b draft and replaces the complete
quarantined v2r4a rollout grid. It keeps the frozen v2r4 scientific question,
candidate checkpoints, endpoint evidence, absolute thresholds, paired tests,
and step-9,920-only authorization rule. It changes only:

1. the operational runtime fixes required after v2r4a failed;
2. the direction-neutral analyzer correction required by the predeclared
   independent chess-reward and protocol outcomes; and
3. fresh prompt sets and seeds, selected without using any rollout outcome and
   disjoint from all diagnostic and quarantined prompts.

No v2r4a artifact may enter a v2r4c estimate. No early checkpoint may
authorize the original experiment. Only a complete authenticated v2r4c report
whose step-9,920 A and B cells both pass may authorize exactly two replacement
E1 RL1 launches: U and D, 1,500 updates each.

## Complete v2r4a quarantine and outcome disclosure

The immutable quarantine report is
`INTERLEAVED_V2R4A_TERMINAL_QUARANTINE_REPORT.json`:

- canonical self-hash:
  `576bf2fb346666f8b9da2d3df5563d9e29e65b60ce1a66c07b74bf6318428947`;
- file SHA-256:
  `b800d3f8289cf1e4d2ef1320efb4185ab62761c576f3dd05b863d11c8d694970`.

The terminal v2r4a vector was `S,F,F,S,F,S` in canonical order
`6000/A, 6000/B, 8000/A, 8000/B, 9920/A, 9920/B`. The three failures were
caused by the destructive Miles rollout-health monitor killing busy SGLang
engines after a 20-second heartbeat stall. The three successes were also
quarantined because they emitted prohibited positive-attempt aggregates,
default outcome metrics, and unredacted outcome logs before the all-cell
barrier. The complete old grid has no authorization value.

The disclosure is time-separated:

- Before all six calls were terminal, no aggregate, pass rate, variance,
  prompt-level outcome, positive-stream file, or rollout JSONL content was
  manually inspected. Exactly two individual rewards leaked through process
  logs: step-6,000/B reward 0 and step-8,000/A reward 0.
- After all six calls were terminal, the frozen check-only finalizer
  necessarily materialized outcome-bearing rows before failing. Diagnosis
  inspected step-6,000/A `rollout_1.jsonl` line 388, sample 2,435,
  `PuzzleId=4DXgm`, score 1, output `Qh5f7# <call_env>`, with joint protocol
  validity false. No complete six-cell aggregate was computed.
- That post-barrier row directly motivated only the direction-neutral analyzer
  repair below. It did not alter checkpoints, thresholds, endpoint evidence,
  prompt eligibility, or statistical tests. To avoid any prompt-level
  contamination claim, v2r4c excludes every old diagnostic and v2r4 A/B
  prompt.

The launcher authenticates every quarantined root against the report's exact
recursive file count, bytes, and content-hash inventory, and authenticates each
old success marker by byte hash and self-hash.

## Fresh prompt construction

Source:

- `/data/chess-rl-data/train_v4_dataset_balanced_multi_turn.parquet`;
- 53,225 rows;
- SHA-256
  `bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`.

The authenticated tokenizer leaves 53,157 unique prompts at or below 512
tokens. Exclude the exact union of:

- the 256-prompt v2r3 diagnostic set;
- quarantined v2r4 batch A, 1,024 prompts;
- quarantined v2r4 batch B, 1,024 prompts.

The disjoint exclusion union contains 2,304 prompts and has sorted-set SHA-256
`ca0b9a0a52d7f3dd1f33527e93cc500e21aeda2c069d9691bc3850cb4d3e2827`.

Rank the remaining 50,853 rows ascending by:

`SHA256("interleaved-v2r4c-fresh-production-gate-20260730" + NUL + fingerprint)`

with fingerprint as the tie-break. Batch A is ranks 0–1,023, batch B is
1,024–2,047, and the canary is 2,048–2,303. Seeds are derived without outcome
input as:

`1 + uint64_be(SHA256(salt + NUL + "rollout-seed" + NUL + label)[:8]) mod 2e9`.

The final self-hashed prompt manifest is:

- path:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r4c_production_gate_20260730/prompt_batches_manifest.json`;
- embedded SHA-256:
  `83f4718b829b955cb000908c2ecbb9052883d14404114cbca4ecd42988659056`;
- file SHA-256:
  `261a313c687bb328d3306301c5477705f6bd8f1b5334d8bccd657102ecfdce60`;
- bytes: 814,493.

| Set | Rows | Parquet SHA-256 | Bytes | Prompt-set SHA-256 | Seed | Epoch-0 order SHA-256 |
| :--- | ---: | --- | ---: | --- | ---: | --- |
| A | 1,024 | `c5f6f208f348b079ee476ecf99c4fad14bba4210ac31c850ed0ce9d801caab61` | 1,072,820 | `b15170a3799027e0ac37af842ed9915eb78a682d2963820988f46edf5cb96e4f` | 1,138,054,401 | `245910389323f01312de5bbf5dfccb5fd435462b278d5cdcaf1c5dcd57db9414` |
| B | 1,024 | `230911d2fb7ddc7a331cfac5b3ae3ebd8149270fe138d6beca683742a3a6541f` | 1,066,692 | `e9c536e23eb1d829b1afbe1841c4163384663dce0a4e27d4e6db24eb28ce5d40` | 893,756,028 | `f736b739d7d69dd7b36b2f7a85b5f3de82becb02087de1e818e3bf9c1b4f78f5` |
| Canary | 256 | `8c714172b9f9b90348673b92705f3c4ddbded404a8ced5c22e38be031e14accb` | 286,673 | `bac5f41fbed4e41ea511203078a9195c613d0eb04a682a7ea076db56158a8663` | 13,477,620 | `1ae1bdece961311cd3fe57c07d4f4a4eb05826d6a7e193c8d37585347f17b73b` |

The manifest proves all six intersections among A, B, canary, and the prior
exclusion union are zero. An independent reconstruction verified source-row
equality, full-row hashes, tokenizer lengths, rank slices, pyarrow-24 bytes,
seed derivation, permutations, rollout quarters, and nonoverlapping
trajectory-seed ranges.

## Runtime corrections

Every canary and grid cell uses one 8-H200 node and the same strict runtime:

1. Miles fault tolerance disabled;
2. router and destructive rollout health probes suppressed with interval
   `1e18`;
3. zero Modal retries and no prompt replacement;
4. two-hour subprocess timeout and three-hour function timeout;
5. strict exact-once source with no wrap, requeue, partial rollout, update,
   eval, save, or resume;
6. zero-pending success tail with no unnecessary abort RPC;
7. prompt, response, label, and reward redacted from strict sample logs;
8. W&B disabled;
9. Miles pass-rate logging omitted;
10. positive-attempt stream and default reward metric path suppressed;
11. no outcome-named strict metric;
12. recursive regular-file allowlist rejecting every undeclared file,
    temporary, symlink, validation, log, MLflow, positive, or summary
    artifact.

Successful grid cells contain exactly four training JSONLs with 2,048 rows
each plus one intent, one root provenance file, and one launch provenance
file before their success marker is written. Pre-barrier success markers omit
raw byte counts and contain only identities, shape, and approved content
hashes. They never contain reward aggregates.

## Direction-neutral analyzer correction

The frozen plan defined chess reward and joint protocol validity as independent
row facts. The v2r4a analyzer accidentally imposed the unstated implication
`positive reward => joint protocol valid`. v2r4c counts them independently and
adds a positive × protocol cross-tab. The protocol threshold remains 443 rows
per cell. Every reward threshold, endpoint rule, McNemar comparison, Holm
correction, and authorization rule remains unchanged.

The frozen v2r4c wrapper is `Eval/v2r4c_gate_analysis.py`. It delegates the
substantive corrected computation to the audited
`Eval/v2r4b_gate_analysis.py` and changes only report identity plus the
fresh-prompt disclosure.

## Canary and six-cell launch

The canary is one rollout of 256 prompts × 8 siblings using the step-6,000
checkpoint. It is operational only. It passes on exact prompt order, sample
indices, top-level and metadata seeds, completed/truncated status, artifact
hash, recursive allowlist, and absence of prohibited telemetry. No reward or
output field is accessed by its validator.

The exclusive canary ledger must self-authenticate exactly one preflight call,
one canary FunctionCall, one success self-hash, and the exact contract. The
grid preflight independently reauthenticates the remote canary marker and raw
shape, then requires that success hash to match the local ledger.

Only after that may the launcher spawn all six cells:

`{step 6000, step 8000, step 9920} × {fresh batch A, fresh batch B}`.

The grid launcher persists each FunctionCall ID immediately, records any
partial spawn failure as `launch_failed`, and cannot be rerun at the canonical
ledger path. Any failed cell fails the entire grid. There is no automatic
retry, replacement, partial estimator, or successful-cell carry-forward.

## Unchanged endpoint evidence and pass rule

The authenticated checkpoints and endpoint evaluator bundles are unchanged:

- PT/B1–B5 evaluator:
  `80caa51691611ad89a2496e2ca89f1c4039777d1d1a9b663fc550a26cff585f0`;
- P2-SFT-at-P1 evaluator:
  `65551135a0eb3eac5e1b65447a499c5849c154ec75f16db762903198eb2bf920`.

Step 9,920 must independently pass in both A and B:

- exactly 1,024 groups and 8,192 rows;
- at least 8,184 completed rows;
- at least 16 solve@8 groups;
- at least 16 nonzero-sibling-variance groups;
- at least 443 joint-valid protocol rows;
- all step-9,920 endpoints complete and finite;
- exact B1–B5 contains at least one positive.

Early checkpoints are diagnostic only.

## Frozen implementation identities

The independently audited frozen values are:

- `chess_rl_miles/batched_rollout.py`:
  `3a75a9338b8665f3707fa66b135ecbd35df8e9cea264832e6ae83ab2c397296b`;
- `chess_rl_miles/io.py`:
  `b00ee01f946427916b6cfd12fa9fbbe6387475a872a55b7c67699be0cd95ad81`;
- `chess_rl_miles/gate_data_source.py`:
  `6a6c8daf44cd1c93b61f92437f2c2d4524daa514aa566033896f2b204d7f55e9`;
- `scripts/run_chess_miles.py`:
  `858f3ea4120f6669ca2e77eb9c9e1f8c4f00ec6e2b2cf6f54091e380bbfb3b3c`;
- `scripts/modal_interleave.py`:
  `1f37eb0879d987fe2a2ec6bc9a37604bf74fb384f461a09209b3d8a1f3dea6e5`;
- `scripts/modal_v2r4c_gate.py`:
  `d0d635283a95752e4a05266191c63d633b2eb28ebea9bc8088c41ad298407aa7`;
- `tests/test_modal_v2r4c_gate.py`:
  `d1b1cd7317285d98952156ed967b8be96ab90014444d7c2744346029cd435ffe`;
- substantive `Eval/v2r4_gate_analysis.py` base dependency:
  `16c9dfd5cc421b196344b773998629cd707f4fadaa405bdba9efaf806d876e6f`;
- corrected `Eval/v2r4b_gate_analysis.py` dependency:
  `ad0071df90a50d99802b5998038539131dde8226890673a9798765c89a023933`;
- `Eval/v2r4c_gate_analysis.py`:
  `00ff347e4accb8927751bf02f4702df7777f7f621db199cc29ffa9dcc25addc5`;
- `Eval/finalize_v2r4c_gate.py`:
  `3a2371949cd335e06a576cd713debc8b775e12fcbe2e56feec8f8a6159fe2dcd`;
- reused `Eval/finalize_v2r4_gate.py` dependency:
  `711003060a730c5ba58dfdb1e20e6cd6b07c99a3f3b958e86efd896ae5fef2bb`;
- `Eval/tests/test_finalize_v2r4c_gate.py`:
  `bffb8ca03fa49152e1c9e5285099ea971fd230e37c967d469f62ed7c5ca0d0de`;
- `build_v2r4c_runtime_contract.py`:
  `06abaf41b596bf002396eca4e2607d8602e8ae3d9eeece6891f1540a2e7fc51d`;
- full `chess-rl-miles` manifest excluding only
  `chess_rl_miles/v2r4c_contract_binding.py`: manifest SHA-256
  `2e970537315d4364bb5f2a5f0ffed01a158bf5bfc7a34d017464a6d62b74eb1b`,
  41 files, 497,374 bytes;
- unchanged Miles manifest: manifest SHA-256
  `9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`,
  679 files, 3,721,726 bytes;
- validated relevant test count: 178 passed, excluding only
  `chess-rl-miles/tests/test_rollout_routing.py`, whose local collection
  requires Ray; Ray is present in the bound Modal image.

Any change to a frozen source, test, launcher, analyzer, finalizer, source
manifest, or this amendment invalidates the independent source-gate verdict
and requires a new audit and runtime contract.

The runtime contract must bind this plan, final prompt manifest, immutable
quarantine report, source manifests, the analyzer and its corrected and
substantive base dependencies, the finalizer and its reused dependency and
test source, endpoint evaluators, candidate checkpoints, canary, and exact six
grid cells. Both contract digests must be nonzero before any exclusive launch
ledger can be created.

## Outcome barrier and authorization

No process may inspect, aggregate, compare, or publish a v2r4c grid reward
outcome until all six FunctionCalls are terminal. Shape-only status polling is
allowed. Only then may the frozen finalizer authenticate the exact ledger call
IDs and success markers, read all six raw JSONLs, apply the frozen analyzer,
and write one immutable self-hashed report.

No report means no authorization. A failed or incomplete report means no
authorization. A complete report whose step-9,920 checks fail means no
authorization. Only a complete passing report may authorize the separately
ledgered replacement E1 RL1 U/D launches.
