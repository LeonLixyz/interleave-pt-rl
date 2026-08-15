# SFT-injection ablation: mixed-no-reweight vs staged, × PT budget × SFT amount

Status: **draft spec — for approval before build/launch**
Drafted: 2026-07-31
Data source: `Pre-to-Post-2/200M_SFT_dataset@fd343bd2` (clean; 0 `<verify>`), PT `pretrain_v1_20b@07dd1b7`.
Model: same 47,245,312-param Qwen3, vocab 85, ctx 3072, 8×H200, gbatch 168.

## 1. The 3 knobs → 8 checkpoints

| Knob | Values |
| --- | --- |
| PT budget | 5B / 10B |
| SFT amount | half (~38.9k rows) / full (77.7k rows) |
| SFT injection | **mixed, no reweight** (SFT interleaved, loss weight = 1.0) / **staged** (pure PT → separate SFT stage) |

### Group A — mixed, NO reweight (loss weight = 1.0, plain token-mean)

| Arm | PT | SFT | ≈ updates | SFT share of loss | Reuses |
| --- | --- | --- | --- | --- | --- |
| A1 `mixNoRW-5B-half`  | 5B  | half | ~9,920  | ~0.52% | P1 manifest (retrain, weight=1) |
| A2 `mixNoRW-5B-full`  | 5B  | full | ~10,150 | ~1.04% | **new manifest** |
| A3 `mixNoRW-10B-half` | 10B | half | ~19,610 | ~0.26% | **new manifest** |
| A4 `mixNoRW-10B-full` | 10B | full | ~19,840 | ~0.52% | E2 manifest (retrain, weight=1) |

### Group B — staged / decoupled (PT-only → SFT-only stage)

| Arm | PT base | SFT stage | ≈ PT updates | ≈ SFT updates |
| --- | --- | --- | --- | --- |
| B1 `staged-5B-half`  | PT@5B  | half | (shared) | ~232 |
| B2 `staged-5B-full`  | PT@5B  | full | (shared) | ~463 |
| B3 `staged-10B-half` | PT@10B | half | (shared) | ~232 |
| B4 `staged-10B-full` | PT@10B | full | (shared) | ~463 |

(SFT stage = 1 epoch over the rows at constant/decayed LR; exact counts fixed at manifest build.)

## 2. Shared checkpoints

```
Group B — ONE pure-PT run, snapshot at 5B and 10B:
    PT-run (→10B) ── snapshot ──► PT@5B ──► +SFT(half) → B1
                    │                   └─► +SFT(full) → B2
                    └────────────► PT@10B ─► +SFT(half) → B3
                                        └──► +SFT(full) → B4

Group A — no sharing (SFT mixed from step 0 → 4 independent runs).
   (A1 == 5B midpoint of A4 ONLY if A4 uses a matched 5B→5B schedule; otherwise separate.)
```

- **Shared:** `PT@5B`, `PT@10B` (both from one PT run) feed all 4 staged arms, and are free PT-only baselines.
- **Training jobs:** 4 (Group A) + 1 PT + 4 SFT-stages (Group B) = **9 runs → 8 final checkpoints** (+2 PT-only baselines).

## 3. Evaluation — NO RL training (decided 2026-08-01)

No RL runs (neither 8 nor 16). Instead, each of the 8 final checkpoints gets **one
generation eval over the full 53,225-row balanced parquet** (`bcf131d8…`; difficulty-
balanced 800–2500, ~12k rows per band), **n=16 samples/prompt**, temp 1.0, RL caps
(512/2560/3072). ≈852k generations per checkpoint. Report per checkpoint, overall and
by difficulty band:

- pass@1, pass@8, pass@16;
- protocol-valid rate (`</T>` before `<call_env>`, parseable moves);
- variance rate (prompts with 1–15/16 successes — the would-RL-have-signal metric);
- held-out PT loss / perplexity / token accuracy.

RL training decisions are deferred until this table exists.

### Frozen recipe decisions (2026-08-01)
- Group B staged LR: PT base cosine **1e-3 → 1e-4**; SFT stage 1 epoch cosine
  **1e-4 → 1e-5** (continuous handoff). Note: PT-5B/PT-10B endpoints are therefore
  less annealed than Group A endpoints when read as PT-only baselines.
- "Half" SFT = existing seed-42 P1 half (38,858 rows).
- Wave 1 = all 6 runs in parallel (48 H200s).

## 4. What this tests

- **A vs B**: does mixing dilute SFT, and does a separate SFT stage fix it without reweighting?
- **half vs full SFT**, **5B vs 10B PT**: sensitivity to SFT amount and PT budget.
- **vs existing E2** (mixed **with** ~190× reweight): 3-way — mixed-reweighted / mixed-plain / staged.

⚠️ Expectation: Group A (no reweight, SFT 0.26–1.04% of loss) likely reproduces the **v1 dilution failure** (RL-infeasible). Group B avoids dilution (SFT gets its own 100%-loss stage). That contrast is the point.

## 5. Infra deltas needed before launch (smaller than expected)

1. **No-reweight is native.** `interleaved_hf_trainer.py:718` documents `sft_loss_weight=1.0` as the raw token-mean (default). Group A = existing trainer with weight 1.0. No new trainer code.
2. **Group A data:** build **2 new mixed manifests** (`5B+full`, `10B+half`); reuse P1/E2 manifests for A1/A4.
3. **Group B — mostly wiring:** `sft_trainer.py` already loads a PT checkpoint (`_load_pretrained_weights`) and does SFT-only. Need: (a) a **PT-only leg manifest** (SFT records removed) to make the shared `PT@5B`/`PT@10B`, (b) config the SFT stage to init from those checkpoints on clean half/full SFT.
4. New content-addressed identities + fail-closed manifest hashes per arm.

## 6. Build order

1. Freeze this spec.
2. Build + validate the pure-PT manifest and the 2 new mixed manifests (deterministic, CPU).
3. Wire the SFT-only staged trainer; canary 1 update save/resume/export.
4. Launch the shared PT run (→10B, snapshot 5B/10B) + the 4 mixed runs.
5. Run the 4 SFT stages off the PT snapshots.
6. RL (U) each of the 8; eval every 40; publish the comparison table.
