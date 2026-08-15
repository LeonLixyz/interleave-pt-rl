# Revised 20B interleaved pretraining/RL experiment

Status: implementation and launch authorized on 2026-07-30.

Experiment version: `mix20b_sft77k_once_3072_v1_20260730`

## Fixed model and data

- Model: 47,245,312-parameter Qwen3-style model, context 3,072.
- Pretraining source: pinned 53.970B-token `pretrain_v1_20b` corpus.
- Total selected pretraining targets: exactly 20B.
- SFT source: all 77,717 physical rows, cleaned and masked with the existing
  audited cache.
- SFT is exposed exactly once. Seed 42 splits it into 38,858 P1 rows and
  38,859 P2 rows.
- Each leg is one deterministic sample-level PT+SFT shuffle. E2 consumes the
  exact `P1 || P2` sequence so all controlled arms see the same examples and
  within-leg order.

## Fixed pretraining

- Eight H200 GPUs, batch 21/GPU, global batch 168, gradient accumulation 1.
- AdamW, peak LR `1e-3`, 5% warmup, cosine decay to `1e-5`.
- Sequence length 3,072; pretraining tokens are concatenated and packed.
- P1/P2 each contain 10B pretraining targets and run for 19,608 updates.
- E2 uses one 39,216-update cosine arc.
- E3 uses two 19,608-update arcs with a fresh optimizer/scheduler at P2.
- The cleaned SFT objective retains equal integrated PT/SFT loss mass:
  P1 weight `380.37858167325345`, P2 weight
  `381.7791327543398`, and monolithic weight
  `381.0775703782155`.

## Experiment DAG

1. E1-U/D: P1 10B + half SFT -> 1,500 RL -> P2 10B + remaining
   SFT -> 1,500 RL.
2. E2-U/D: 20B + all SFT with one cosine -> 3,000 RL.
3. E3-U/D: P1 -> P2 without midpoint RL, two cosine arcs -> 3,000 RL.
4. E4: positive-rollout SFT, distillation, and scratch-retraining variants;
   this is compute-unbounded and downstream of E1 RL1.

RL uses the pinned 53,225-row balanced multi-turn parquet. Each controlled
endpoint branches into unfiltered and Miles dynamic-filter arms. RL saves
every 40 updates and uses the verified optimized Miles/SGLang path.

## Exact record accounting

| Stream | PT records | SFT rows | Padding | Updates |
| --- | ---: | ---: | ---: | ---: |
| P1 | 3,255,209 | 38,858 | 77 | 19,608 |
| P2 | 3,255,209 | 38,859 | 76 | 19,608 |
| E2 (`P1 || P2`) | 6,510,418 | 77,717 | 153 | 39,216 |

## Launch order

1. Build and validate the versioned 20B manifests.
2. Run a one-update mixed PT+SFT canary.
3. Launch P1 and E2 concurrently.
4. When P1 finishes, launch E1 RL1-U/D and E3 P2.
5. Continue each dependency branch only from its authenticated endpoint.

All artifacts, checkpoints, RL run names, ledgers, and controller locks use a
new 20B identity. The old 10B registry and running old autopilot are not
modified.
