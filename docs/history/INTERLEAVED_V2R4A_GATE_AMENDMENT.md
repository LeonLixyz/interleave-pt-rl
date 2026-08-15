# Interleaved v2r4a production-gate amendment

Frozen at: 2026-07-30 13:40 EDT

Contract schema:
`interleaved-v2r4a-production-gate-amendment-v1`

Contract version:
`v2r4a_production_gate_20260730`

## Scope

This is a minimal, non-outcome-adaptive execution amendment to the frozen
v2r4 production-gate plan. Every scientific choice, candidate checkpoint,
prompt, prompt order, rollout seed, sibling count, absolute threshold,
endpoint evaluation, comparison, and authorization boundary remains exactly
as frozen in:

- `INTERLEAVED_V2R4_PRODUCTION_GATE_PLAN.md`, SHA-256
  `ec01e2639b532081f5fb928f7996b3b7a40d71377e1cd74ac2f45fa73864f229`;
- original runtime contract canonical SHA-256
  `3127bdd6dfca62b34813e3fe938300d5d44c8d7ac253bf4a65836f4b2fc1ffd3`;
- original runtime-contract file SHA-256
  `9bc355fb7ac89dda15cbf4d0c1a4767a3ac5e314e6c800c398f3e9062de02f29`.

The sole scientific effect of this amendment is to make the already-declared
sample-index deterministic seed environment visible to Ray workers before
the Ray head starts. No threshold or analysis rule was changed after seeing
a rollout reward.

## Quarantined v2r4 launch

The original exact-once launch ledger is
`INTERLEAVED_V2R4_GATE_LAUNCH_LEDGER.json`, embedded SHA-256
`8367979ea65f37d5bcf921cda3c3bbf465e39cc15bf20107b23ea68d9d3b980b`,
file SHA-256
`632ea875c1403f9681c973f8074094f0d206ec6ffa67787f827756d69976608e`.
It owns these six unique, terminal-failure FunctionCall IDs:

| Step | Batch | FunctionCall |
| ---: | :---: | --- |
| 6,000 | A | `fc-01KYT16H4M5MRHCV3FNSHQ3WGQ` |
| 6,000 | B | `fc-01KYT16H9WPNCGJW90KT646ABA` |
| 8,000 | A | `fc-01KYT16HCD9WCV77NM08HFWMRK` |
| 8,000 | B | `fc-01KYT16HF1SC79V08FBJRRVXZ9` |
| 9,920 | A | `fc-01KYT16HH87GZRVQER47J4RPJR` |
| 9,920 | B | `fc-01KYT16HMCKPS13RT36RRC4VM7` |

All six failed at `RolloutManager.__init__`, before the strict data source
could return a prompt. The exact error was:

`strict exact-once rollout source rejected configuration: deterministic seed mode must equal sample-index`

The adapter had set
`CHESS_RL_MILES_DETERMINISTIC_SEED_MODE=sample-index` only in the later Miles
subprocess environment. Ray workers inherit the Ray-head environment, so the
strict source correctly failed closed.

An independent recursive volume audit proved every failed root contains only
the intent, run provenance, one provenance launch JSON, and empty rollout/log
directories. Across all six roots there are:

- zero rollout JSONL files;
- zero success markers;
- zero reward or outcome artifacts;
- zero sampled prompts and zero exposed candidate outcomes.

The failed roots, ledger, and Modal calls remain immutable and quarantined.
They cannot be promoted, overwritten, resumed, retried, or counted as gate
data. Because no outcome or prompt sample was exposed, reusing the exact
frozen prompts, orders, and seeds creates no outcome-adaptive selection.

## Minimal implementation correction

`_runtime_env` now accepts only the optional literal seed mode
`sample-index`. The v2r4a worker calls it with that value before
`_cleanup_runtime` and `_start_ray_head`; non-gate training remains unchanged.
The CPU-only preflight starts a real local Ray head with `num_gpus=0` and
requires a remote Ray worker to return exactly:

```json
{
  "artifact_root": "/rl-checkpoints/chess-rl-miles-interleave/v2r4a-ray-env-preflight",
  "gpu_allocated": false,
  "seed_mode": "sample-index"
}
```

The preflight must also authenticate the new runtime contract and prove all
six fresh v2r4a roots absent. The exact-once launcher must compare this live
Ray result before spawning any H200 call.

Fresh identities are mandatory:

- contract version `v2r4a_production_gate_20260730`;
- runtime-contract path
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r4a_production_gate_20260730/runtime_contract.json`;
- run roots
  `v2r4a-gate-w190-s{6000,8000,9920}-batch-{a,b}`;
- exclusive ledger
  `INTERLEAVED_V2R4A_GATE_LAUNCH_LEDGER.json`;
- W&B group `v2r4a_production_gate`;
- zero automatic function retries and one unique FunctionCall per cell.

## Frozen implementation identities

- `chess_rl_miles/gate_data_source.py`:
  `6a6c8daf44cd1c93b61f92437f2c2d4524daa514aa566033896f2b204d7f55e9`;
- `chess_rl_miles/batched_rollout.py`:
  `72a61ac79d3cf6db36a2576195f38175645d8902a3a2131e602b89f3ecc2c9d6`;
- `chess_rl_miles/io.py`:
  `0c28931bbec8ed86b6562dd2c4aa8505dada25c00253f90afbfab82bbe5914c6`;
- `scripts/run_chess_miles.py`:
  `c3a5503ff1f2bf39b5f200ef9da12032f194681ec0146db60ea1a53766bf03f5`;
- corrected `scripts/modal_interleave.py`:
  `68b295f82af5a9dbf6c3e2143d732b554c8c97dce01b8932d9874c0b962edf99`;
- corrected focused test:
  `chess-rl-miles/tests/test_modal_interleave.py`,
  `772143c45b6e0a9180411f5ec798712dbdbd1d2758c4085a609ea6704d662f50`;
- complete `chess-rl-miles` source manifest excluding only the two-digest
  binding module:
  `e7cd51377a676ba9e070bdad079c00f02277b12a798c5611aba99d19d355bb65`,
  38 files and 422,629 bytes;
- unchanged Miles source manifest:
  `9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`;
- contract builder:
  `7658af6c4cec4571018818d62d64e0d8682ea3333884071e882307fd125f82ee`.

The focused rollout suite passes 89/89. The amended analyzer
`Eval/v2r4_gate_analysis.py`, SHA-256
`16c9dfd5cc421b196344b773998629cd707f4fadaa405bdba9efaf806d876e6f`,
passes 4/4 and emits only
`v2r4a_production_gate_20260730` under schema
`interleaved-v2r4a-production-gate-report-v1`.

## Reused immutable endpoint evidence

The candidate checkpoints, prompt batches, prompt manifest, runtime profile,
and all scientific semantics are inherited byte-for-byte from the original
contract. The endpoint jobs do not need to be repeated because the models,
data, and evaluator implementations are unchanged.

The exact PT/B1–B5 evaluator bundle is
`80caa51691611ad89a2496e2ca89f1c4039777d1d1a9b663fc550a26cff585f0`;
its six-call ledger embedded SHA-256 is
`e150ec1f66670301a789f736a1522ae96e8abf490d94cd43ab2d98978ffad032`.
All six endpoint results were independently authenticated.

The exact P2-SFT evaluator bundle is
`65551135a0eb3eac5e1b65447a499c5849c154ec75f16db762903198eb2bf920`;
its corrected v2 three-call ledger embedded SHA-256 is
`3440c2429be81d4bd3157dacd3ea3b81d0d9bb8bec9e13032b8b5e019baa77dc`.
All three P2 results were independently authenticated.

The final v2r4a report must bind every reused endpoint FunctionCall,
persisted success file, result self-hash, raw chess artifact, and the
quarantined no-outcome v2r4 failure ledger in addition to the six new
rollout cells.

## Authorization boundary

The absolute per-batch thresholds, all-three-pair McNemar/Holm analysis,
limited-power interpretation, and step-9,920-only authorization boundary are
identical to the original plan. Only an immutable, self-hashed v2r4a report
with all six new cells terminal and authenticated may authorize exactly two
new versioned E1 RL1 calls. Any v2r4a failure remains failed and requires a
new written amendment; there are no silent retries or replacement prompts.
