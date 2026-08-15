# Overview

## The question

Standard practice is pretrain first, then post-train (SFT, then RL). This project asks
whether **interleaving** helps: does putting an RL round *between* pretraining stages
produce a better model than doing all the pretraining first?

Two model lines were built to answer it:

- **Chess**, 47M parameters — the controlled workhorse. Cheap enough to run a full
  matrix of arms with matched budgets.
- **Math**, 1.5B parameters — the scale check, with pretrain / anneal / SFT / RL
  anchors released as `pre-to-post-olmo/Math-Models`.

## The answer, in short

**Do not interleave in weight space. Do interleave in data space.**

1. **Pretraining overwrites RL, at any dose.** Four second-leg pretraining runs — from
   no RL, from weak RL, and from an RL model that had tripled its pass@1 — all landed
   at ~15% pass@1 with identical held-out losses. A pretraining leg moves weights ~50%
   in relative L2; RL moves them 0.2–4%. At equal compute the interleaved model ends
   *below* the pretrain-then-RL control.

2. **RL knowledge survives as data.** Distill the RL model into verified traces, mix
   those into the next pretraining leg, and the ability carries through. This scales
   with dosage and, unlike the RL model itself, *restores* sampling diversity: every
   trace-pretrained model has higher pass@16 than the RL model that generated its
   traces.

3. **The traces are free.** An RL run already writes every rollout to disk. Harvesting
   its verified-correct trajectories costs no extra generation and beats harvesting via
   a separate evaluation pass.

4. **The loop compounds.** RL → harvest → pretrain → RL took a 10.2% model to 51.4%
   pass@1, against 45–48% for conventional pretrain-then-RL at the same total budget.
   Applied again on top of the 10B models it reached **54.6%** pass@1 and 65.7%
   pass@16 — the best numbers in the study.

5. **The RL learning rate dominated everything.** At 1e-5 (the historical default) RL
   added ~4 points of pass@1 and no coverage; at 1e-4 it tripled pass@1 and produced
   the first real coverage growth. Several early conclusions about "RL only sharpens"
   were artifacts of the low learning rate.

Full tables and curves: `https://modal-labs-leon-dev--interleave-results-web.modal.run`

## Chess model lineage

Everything descends from one pretraining corpus and one SFT set:

```
5B pretrain (+ half the SFT rows mixed in)            10.2% pass@1
├── + 5B pretrain (second half)                       15.1%   ← the control
├── + RL(1e-5)                                        13.8–14.7%
└── + RL(solvable-only, 1e-4, 1500)                   36.3%
    ├── → 5B pretrain from these weights              14.8%   ← washes out
    ├── → traces (k per puzzle) → 5B pretrain         21.6% (k=1) … 34.8% (k=16)
    └── → rollout harvest (1.59M traces) → 5B pretrain 38.4%
         └── → RL(own solvable-only, 1e-4, 1500)      51.4%

10B pretrain, one run                                 14.8%
└── → RL(solvable-only, 1e-4, 3000)                   48.5%  (format degraded to 87%)
    └── → rollout harvest → 5B fresh pretrain         42.7%
         └── → RL(own solvable-only, 1e-4, 1500)      54.6%  ← best in study
```

The 10B two-run variant (optimizer reset at midpoint) tracks the one-run variant within
noise at every stage, which is itself a result: splitting the cosine schedule changes
nothing.

## Vocabulary used throughout

Defined once here, used consistently in the other documents and on the results pages.

| Term | Meaning |
|---|---|
| **leg** | one pretraining run over a fixed slice of the token selection |
| **mixed stream** | pretraining text and SFT rows shuffled into one record order and trained with one objective (not alternating batches) |
| **all** | RL data setting: every drawn prompt trains |
| **dynamic filter** | RL data setting: keep only prompt groups with some successes and some failures; redraw the rest (on-policy) |
| **solvable-only** | RL data setting: an offline pre-filter to the puzzles the starting checkpoint solved 1–15 times out of 16, fixed for the run |
| **verified traces** | model outputs that scored correct and parsed as well-formed, kept for use as training data |
| **rollout harvest** | verified traces taken from an RL run's own saved training rollouts (free) as opposed to a separate evaluation pass |
| **wash-out** | the observed erasure of RL weight changes by a subsequent pretraining leg |

Avoided deliberately: "band" (say *solvable-only*), and any name that encodes an
experiment id rather than what the run did.

## Caveats that apply to every number

- One seed per cell. Only one earlier ablation arm was replicated under a reseed.
- Pretraining stores parameters in bf16, which froze all RMSNorm scales at exactly 1.0
  in every run. Shared by all arms, so comparisons hold; absolute numbers may improve
  with fp32 master weights.
- The second pretraining leg re-warms the learning rate to 1e-3. A low-peak
  continuation — which might preserve more RL signal — was never tested.
- The 10B loop chains consumed 15B pretraining tokens against 10B everywhere else.
  They demonstrate the loop keeps working; they are not a matched comparison.
