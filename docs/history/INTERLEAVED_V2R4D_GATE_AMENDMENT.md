# Interleaved v2r4d import-safe production-gate amendment

Freeze state: `FROZEN_AFTER_REMOTE_IMPORT_AND_SOURCE_AUDIT`

Contract schema:
`interleaved-v2r4d-production-gate-runtime-contract-v1`

Contract version:
`v2r4d_production_gate_20260730`

## Purpose

v2r4d is the operational successor to v2r4c. It changes no scientific
question, checkpoint, prompt row, seed, rollout setting, threshold,
comparison, or authorization rule. It exists because the v2r4c direct-file
Modal launcher did not add `/root/chess-rl-miles` to `sys.path` before its
first package import.

The v2r4c failure occurred in CPU container import, before the preflight
function body. It produced no child FunctionCall, GPU root, prompt read,
rollout, model response, reward, metric, JSONL, or result. v2r4c is terminally
quarantined and must never be relaunched.

## Immutable predecessor incident

The self-hashed incident report is
`INTERLEAVED_V2R4C_PREFLIGHT_INCIDENT_REPORT.json`:

- embedded report SHA-256:
  `2a2eaeb867d73366b69377e6f965a925eff0c3db8bd8bae881ec4d035e3ec5f0`;
- file SHA-256:
  `8c12073b996486318c7df5b107669c4cb3d55952e32b2cb22a3ca68a855a3168`;
- bytes: 4,508.

It binds:

- stopped app `ap-ScCyVCCvkygnaNd1uNHq0f`;
- terminal cancelled preflight
  `fc-01KYT9E8019J10C6FST75VNVP8`, with `children=[]`;
- twelve identical import-container failures;
- v2r4c contract canonical SHA-256
  `d80e77b2e80149342983a5d37ed90cc3c2b74e058c51a108a950435171198940`
  and file SHA-256
  `7478421ad104e538e7a38c01a7f1d0695d726965bbb21dad5a7e1a9e9d20196f`;
- final failed launch-ledger file SHA-256
  `27e9a093cba15a31129a00f99049db0b6ca4e554079c17f25552880375342fde`,
  state `preflight_failed`, and `calls=[]`;
- exact absence of the v2r4c canary and six grid roots;
- zero v2r4c result/outcome artifacts.

The exact historical v2r4c launcher and test remain preserved in parallel:

- `chess_rl_miles/scripts/modal_v2r4c_gate.py`:
  `d0d635283a95752e4a05266191c63d633b2eb28ebea9bc8088c41ad298407aa7`,
  54,524 bytes;
- `tests/test_modal_v2r4c_gate.py`:
  `d1b1cd7317285d98952156ed967b8be96ab90014444d7c2744346029cd435ffe`,
  11,749 bytes.

## Prompt reuse

Prompt reuse is authorized only because the incident report proves zero
prompt or outcome exposure. v2r4d reuses the exact immutable v2r4c prompt
artifacts, rather than drawing another outcome-dependent sample:

- manifest schema:
  `interleaved-v2r4c-fresh-prompt-batches-v1`;
- manifest version: `v2r4c_production_gate_20260730`;
- manifest embedded SHA-256:
  `83f4718b829b955cb000908c2ecbb9052883d14404114cbca4ecd42988659056`;
- manifest file SHA-256:
  `261a313c687bb328d3306301c5477705f6bd8f1b5334d8bccd657102ecfdce60`;
- batch A parquet SHA-256:
  `c5f6f208f348b079ee476ecf99c4fad14bba4210ac31c850ed0ce9d801caab61`;
- batch B parquet SHA-256:
  `230911d2fb7ddc7a331cfac5b3ae3ebd8149270fe138d6beca683742a3a6541f`;
- canary parquet SHA-256:
  `8c714172b9f9b90348673b92705f3c4ddbded404a8ced5c22e38be031e14accb`.

Rows, prompt-set hashes, deterministic seeds, epoch-0 order hashes, and all
six zero-intersection proofs remain exactly those frozen in
`INTERLEAVED_V2R4C_GATE_AMENDMENT.md`.

Every v2r4d CPU preflight must:

1. authenticate the staged incident report byte-for-byte and by self-hash;
2. reauthenticate the c prompt manifest and all three parquet files;
3. prove all seven v2r4c GPU roots still absent;
4. prove no result-volume path containing `v2r4c` exists;
5. prove the v2r4d canary and all six v2r4d grid roots are absent for a canary
   preflight, or authenticate the v2r4d canary for a grid preflight.

If any predecessor root or result appears, prompt reuse is revoked and this
version must fail closed.

## Import correction and pre-ledger probe

The v2r4d launcher inserts the first existing path from:

1. `Path(__file__).resolve().parent / "chess-rl-miles"`;
2. `/root/chess-rl-miles`;

into `sys.path` before every `chess_rl_miles` import.

Two tests bind this correction:

- static source order requires bootstrap insertion before the first package
  import;
- an isolated `python -I` Modal-layout simulation succeeds with the
  bootstrap and fails with `ModuleNotFoundError` when the bootstrap is
  removed.

Before any exclusive launch ledger is created, one no-volume, no-GPU,
no-contract import probe must run remotely and return the mounted project and
package paths plus the exact launcher source digest. This probe is operational
only and cannot read prompts or outcomes.

That probe completed in Modal app `ap-DFsqe6twFQbMsqEK1FlHQR`. Its immutable
self-hashed report is `INTERLEAVED_V2R4D_IMPORT_PROBE_REPORT.json`:

- embedded report SHA-256:
  `2994a5660f8cb08dedcb128c1bc5d4638114142ecd4d0534a219daea7c74ee84`;
- file SHA-256:
  `5f183b9e9a2b6c0943db5e2e537ac5f5f5ca8e633c6d4d40690377bfc3a9deb1`;
- bytes: 799;
- remote launcher path: `/root/modal_v2r4d_gate.py`;
- remote launcher SHA-256:
  `1bdcb15aeef7c0e1419df0b5393ab369850000976e1e81d82800272e18428419`;
- mounted project path: `/root/chess-rl-miles`;
- imported package source:
  `/root/chess-rl-miles/chess_rl_miles/scripts/modal_interleave.py`;
- project present on `sys.path`: true;
- volumes mounted: false;
- GPU requested: false.

## Fresh operational namespace

- Modal app: `chess-interleave-v2r4d-gate`;
- runtime contract:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r4d_production_gate_20260730/runtime_contract.json`;
- staged incident:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r4d_production_gate_20260730/v2r4c_preflight_incident_report.json`;
- staged import-probe report:
  `/data/50m_interleaved_mix10b_sft90k_v2r1_clean_verify_gate/v2r4d_production_gate_20260730/import_probe_report.json`;
- binding exclusion:
  `chess_rl_miles/v2r4d_contract_binding.py`;
- canary run:
  `v2r4d-runtime-canary-s6000-seed13477620`;
- grid runs:
  `v2r4d-gate-w190-s{6000,8000,9920}-batch-{a,b}`;
- canary ledger:
  `INTERLEAVED_V2R4D_CANARY_LAUNCH_LEDGER.json`;
- grid ledger:
  `INTERLEAVED_V2R4D_GATE_LAUNCH_LEDGER.json`;
- finalized report:
  `INTERLEAVED_V2R4D_PRODUCTION_GATE_REPORT.json`;
- intent/success markers use `_V2R4D_*`.

No v2r4c operational name, ledger, contract path, marker, or binding may be
reused.

## Frozen scientific computation

The nested analysis remains the exact frozen chain:

`Eval/v2r4c_gate_analysis.py -> Eval/v2r4b_gate_analysis.py -> Eval/v2r4_gate_analysis.py`.

The nested analysis retains the v2r4c report identity. Only the outer
authenticated final report has the v2r4d identity and additionally binds the
predecessor incident. The direction-neutral reward/protocol correction,
paired tests, Holm correction, endpoint evidence, and all absolute thresholds
are unchanged.

The complete v2r4a quarantine remains unchanged and fully authenticated.

## Frozen source and validation identities

The post-audit, pre-contract source identities are:

- launcher `chess_rl_miles/scripts/modal_v2r4d_gate.py`:
  `1bdcb15aeef7c0e1419df0b5393ab369850000976e1e81d82800272e18428419`;
- launcher tests `tests/test_modal_v2r4d_gate.py`:
  `0823b9ff8d5a9048e10be397b6fc7b0cc838ef8bde8c893b816b3755ea854586`;
- finalizer `Eval/finalize_v2r4d_gate.py`:
  `75195157fe21efb74440b2560720a7a919a211e416a97d96bc3616bafc769ea5`;
- finalizer tests `Eval/tests/test_finalize_v2r4d_gate.py`:
  `d4ad5df1ac52746ca64125f3c018b190573562140c773947b351ae8a04b5b54b`;
- contract builder `build_v2r4d_runtime_contract.py`:
  `c2ccd0209d92819e96ad66dbbc0d7486b1b00429321b59903205bca0cec7aeb9`;
- nested analyzer `Eval/v2r4c_gate_analysis.py`:
  `00ff347e4accb8927751bf02f4702df7777f7f621db199cc29ffa9dcc25addc5`;
- corrected analyzer dependency `Eval/v2r4b_gate_analysis.py`:
  `ad0071df90a50d99802b5998038539131dde8226890673a9798765c89a023933`;
- base analyzer dependency `Eval/v2r4_gate_analysis.py`:
  `16c9dfd5cc421b196344b773998629cd707f4fadaa405bdba9efaf806d876e6f`;
- frozen endpoint/P2 evidence dependency `Eval/finalize_v2r4_gate.py`:
  `711003060a730c5ba58dfdb1e20e6cd6b07c99a3f3b958e86efd896ae5fef2bb`.

The normalized `chess-rl-miles` source manifest excludes only
`chess_rl_miles/v2r4d_contract_binding.py` and has 44 files, 580,742 bytes,
and SHA-256
`70940e4d39e7c35dc0c8b7e7226f6a835da8517b0ff41f17b2f574f60a876292`.
The normalized Miles manifest has 679 files, 3,721,726 bytes, and SHA-256
`9aeb9274e3c4d38ed2d2bdf80cd37df6b5ad3cb0202775de443bf1c6d60b9f0d`.

The all-zero pre-contract binding has SHA-256
`5bca03bc4d145bb78641af680630ffa0aae146981ee8a9b77040777e9ab28709`.
After contract construction, only its two zero digest placeholders may be
changed. The complete local suite passed 267 tests; only
`chess-rl-miles/tests/test_rollout_routing.py` was excluded locally because it
requires Ray, which is present in the bound Modal image.

## Runtime and authorization

Each canary/grid cell retains the verified v2r4c runtime: one 8-H200 node,
Miles fault tolerance off, destructive router-health probes suppressed, zero
Modal retries, strict no-wrap/no-requeue source, deterministic sample-index
seeding, W&B disabled, outcome logs redacted, positive-attempt artifacts
disabled, exact recursive allowlist, two-hour subprocess timeout, and
three-hour function timeout.

The canary remains 256 prompts × 8 siblings at step 6,000 and is operational
only. After an authenticated canary, launch exactly the six cells
`{6000,8000,9920} × {A,B}` once.

No grid outcome may be read until all six FunctionCalls return successful
terminal results. Step 9,920 must pass independently in both A and B under the
unchanged thresholds. Only that complete result can authorize exactly the two
replacement E1 RL1 launches, U and D, for 1,500 updates each. Early
checkpoints remain diagnostic only.

## Release sequence

1. independently authenticate the incident and preserved c sources;
2. keep the v2r4d binding all-zero and verify no d contract/ledger/root exists;
3. pass the isolated import regression and the full launcher/finalizer suites;
4. run the no-volume/no-GPU remote import probe and bind its launcher digest;
5. freeze plan/source identities and independently review them twice;
6. build one self-hashed d contract, patch only the d binding, and rerun all
   tests;
7. create-only stage the incident, import-probe report, and contract to the d
   data namespace, redownload, and verify exact bytes;
8. run one CPU preflight and one canary under the exclusive d canary ledger;
9. independently authenticate the canary;
10. run the grid preflight and launch all six cells exactly once;
11. finalize only after the six-cell terminal barrier, publish the authenticated
    report, and apply the unchanged authorization rule.
