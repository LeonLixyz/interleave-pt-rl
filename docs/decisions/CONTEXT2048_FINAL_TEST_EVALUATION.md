# Context-2048 final held-out evaluation decision

Status: **accepted and immutable for the final FP32-master v13 comparison**  
Decision date: 2026-08-15  
Evaluation version: `context2048-fp32-master-v13-final-b1b5-n16-v2-20260815`

Sealed production-contract SHA-256:
`f69b1b791ba3c2e35bd62901c679a2e6499a4c0ebd247a0064f92ea51001a501`.

## Decision

Evaluate the five final RL checkpoints at committed update 1,500 on the held-out
B1--B5 chess benchmark. Use the native 2,048-token rollout contract and the
online RL scorer. Report both aggregate and per-bucket results, including
unbiased pass@k for k=1 through 16.

This protocol is frozen. Any change to a dataset, checkpoint, tokenizer,
generation setting, scorer, seed mapping, prompt admission rule, or metric
requires a new evaluation version and output namespace.

## Checkpoints

| Key | Final RL run | Committed checkpoint SHA-256 |
|---|---|---|
| `vocab81_expand85_sft3` | `ctx2048-fp32masterv13-vocab81pt-expand85-sft3-filtered-lr1e5-rl1500-r3` | `76130c6b3e6990264422c8de2d42985466963c7423c88a408f4771c3970fb466` |
| `vocab85_sft3` | `ctx2048-fp32masterv13-vocab85pt-sft3-filtered-lr1e5-rl1500-r3` | `9967929d610e86b4343d6a434ee1eeac056aa498d827a1a705403f3de18834e3` |
| `mixed_sft1` | `ctx2048-fp32masterv13-mixed-sft1-filtered-lr1e5-rl1500-r3` | `cc9b82f2f378cd954a7369ba4458170b245cf6e69e09590515589863f5e2712b` |
| `mixed_sft3_fresh_adam` | `ctx2048-fp32masterv13-mixed-sft3-filtered-lr1e5-rl1500-r3` | `8e32bacc7a2aa7856a25c678919bc70858716493246573a5cd26517578da40fc` |
| `mixed_sft3_continued_adam` | `ctx2048-fp32masterv13-mixed-pt-plus-sft3-continue-adam36848-filtered-lr1e5-rl1500` | `8029879eb82aeb2f07c63a9c3e0e6423fbca8dc58d808752b5ba69abb870dedf` |

Every source must have an authenticated Miles `COMMITTED.json` with iteration
and global step 1,500 and rollout id 1,499. Conversion validates the complete
checkpoint payload before creating an immutable FP32 Hugging Face export. vLLM
then loads that export explicitly in BF16 for inference.

The two `mixed_sft3` checkpoints share the same pre-RL model. Their controlled
difference is the RL optimizer initialization: fresh Adam versus parent FP32
Adam moments and per-parameter step continued from 36,848.

## Held-out benchmark

| Bucket | Raw prompts | Parquet SHA-256 |
|---|---:|---|
| B1 | 310 | `3ac5df0af21b395c23f864dd75b6a64335e3fe681c2b774f1485b276c6893c78` |
| B2 | 299 | `9b315fe82a676b9b817ae77f96f7987be04ab34ec18513e3d42544896a133c3f` |
| B3 | 267 | `8e41e0cf7c17babf6ae9a17a5b51607eef5674788dd09042e7dbbf90a945a5b9` |
| B4 | 288 | `9583e4f6621ffee456eefc3e9d9de15800ec24226d20b882ff4805e82c4a985b` |
| B5 | 320 | `927d62a4994d39e61ffb6719f85961ba14dbd55f365c539477fe6db72288c5cc` |

The union contains 1,484 raw prompts. Exact serialized prompt comparison finds
zero overlap in every bucket with the 53,225-row
`train_v4_dataset_balanced_multi_turn.parquet` source (SHA-256
`bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30`)
from which the RL training cohort was constructed.

The tokenizer-specific production admission rule drops four overlength prompts,
leaving 1,480 evaluated prompts per checkpoint.

## Frozen generation contract

| Field | Value |
|---|---|
| prompt tokenization | checkpoint tokenizer, `add_special_tokens=False` |
| BOS | prepend exactly one explicit BOS; reject a source prompt containing BOS |
| prompt cap | 512 tokens including BOS |
| response budget | 1,536 model-generated tokens; environment replies do not count |
| total context | exactly 2,048 tokens, no extra margin |
| environment calls | at most 6 |
| sampling | temperature 1.0, top-p 1.0 |
| samples | 16 per admitted prompt |
| base seed | 0 |
| request seed | deterministic hash of bucket, row, sample slot, and generation round |
| precision | authenticated FP32 export; BF16 vLLM inference |
| scorer | `chess_rl_miles.reward._score_sample` |
| decoded response | `skip_special_tokens=False`, matching online RL |

The request seed excludes the checkpoint identity. Therefore all five models use
the same seed for a given bucket, row, sample slot, and multi-turn generation
round.

## Frozen metrics

For each bucket and for the five-bucket union, retain the 0--16 win histogram
and compute:

- unbiased pass@k for every k from 1 through 16;
- format-valid sample rate;
- all-zero prompt count and percentage;
- all-sixteen prompt count and percentage;
- mixed-outcome prompt count;
- exact raw generation records with score, format flag, token count, environment
  call count, and response text.

The unbiased estimator is:

```text
1 - C(16 - wins, k) / C(16, k)
```

averaged over prompts.

The final report must explicitly show continued-Adam minus fresh-Adam deltas for
pass@1--16, format rate, and all-zero percentage.

## Execution and immutability

Implementation: `chess/eval/modal_eval_context2048_final_test.py`  
Pure contract helpers: `chess/eval/context2048_eval_core.py`  
Results volume: `chess-rl-eval-results-r6`

Production results live only under:

```text
/results/context2048-fp32-master-v13-final-b1b5-n16-v2-20260815/
```

The dispatcher writes the durable evaluation ledger before spawning workers.
It refuses a second production launch when that ledger exists. Each worker also
claims its checkpoint/profile output before conversion and will not overwrite a
running, failed, or authenticated successful result.

Required order:

1. run input inspection;
2. run focused unit tests;
3. run the real H200 canary (8 B1 prompts, 2 samples each);
4. launch exactly five production workers once;
5. authenticate all five success markers and the merged ledger;
6. publish the aggregate and B1--B5 comparison.

The accepted v2 canary completed on H200 in 84.765 seconds. It evaluated eight
B1 prompts with two samples each and wrote authenticated summary SHA-256
`3bf3f006ce7dde363397557ce663f3f57356d20e2f02ffb7a887cb1365518476`.
Its Modal function-call id is `fc-01M03ENZ14YB6ZDZYWQ9D99WSK`.

The packaged source identities exercised by that canary are:

| Source | SHA-256 |
|---|---|
| evaluator | `68a5f4da2069d935695ee545e0d5bf20b73ddf0e17759edc646f8be6157fa3d8` |
| pure helpers | `ec272147bd01821dbfa53610359c5245eba535baa5d93e7b7f51bb302a9e54fe` |
| FSDP-to-HF converter | `c0b2dabedff27297cd9a9ce5831cf7cd17a70511dbc818d289d762d7b54e9067` |
| online reward | `1a6065c58f0cf8c775112815c90930a87bde205e484a78572f8a0e54eb2bc5c0` |
| chess move utilities | `c4680a3e736f9cb7e2abaa5460e5f29f48c68603e924b68a7474fc3cf8c256a0` |
| rollout implementation | `8414927087039693df11525a827553db5f0b6b4a2660d18ac43e0f338bd376e9` |

## Production execution record

The v2 production dispatcher was launched exactly once in Modal environment
`leon-dev`. Its application run is `ap-gWVlvhJgZ4HUsotQRQpupI` and its durable
dispatcher call is `fc-01M03EV0K4J7XCE53J0X5177X3`.

The sealed worker mapping is:

| Checkpoint key | Modal function-call id |
|---|---|
| `vocab81_expand85_sft3` | `fc-01M03EV4QXBJSZ4566CZ3Y0DHQ` |
| `vocab85_sft3` | `fc-01M03EV5DXB3CQA7K089M08RWC` |
| `mixed_sft1` | `fc-01M03EV66F62A9MC2XEX8HDQDH` |
| `mixed_sft3_fresh_adam` | `fc-01M03EV721QAHBN58RXWD11R5Y` |
| `mixed_sft3_continued_adam` | `fc-01M03EV832D5HH5HXVM6Y8A5XA` |

These ids are evidence of the only authorized production execution. A pending
or running call must never be replaced. Final results are accepted only after
all five worker success markers and the merged ledger authenticate their JSON
and gzip artifacts.

## Final held-out results

All five workers completed and authenticated. The merged ledger reached
`complete` at `2026-08-15T19:50:14.278706+00:00`. Across the five models, the
evaluation admitted 1,480 prompts per model and scored 118,400 trajectories in
total.

### Aggregate metrics

| Model | Format valid | All-zero | All-sixteen | Mixed outcomes |
|---|---:|---:|---:|---:|
| 81→85 + SFT×3 | 99.9029% | 43.0405% (637) | 19.5946% (290) | 553 |
| Native-85 + SFT×3 | 99.9367% | 42.0270% (622) | 20.4054% (302) | 556 |
| Mixed PT + SFT×1 | 99.8775% | 43.9189% (650) | 15.0676% (223) | 607 |
| Mixed PT + SFT×3, fresh Adam | 99.8775% | 42.3649% (627) | 20.0000% (296) | 557 |
| Mixed PT + SFT×3, continued Adam | 99.7973% | 41.9595% (621) | 18.7162% (277) | 582 |

Aggregate unbiased pass@k percentages:

| Model | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 81→85 + SFT×3 | 37.9561 | 43.8328 | 46.8126 | 48.7521 | 50.1651 | 51.2700 | 52.1770 | 52.9484 | 53.6221 | 54.2227 | 54.7667 | 55.2659 | 55.7289 | 56.1622 | 56.5709 | 56.9595 |
| Native-85 + SFT×3 | 39.2399 | 45.1211 | 48.0524 | 49.9740 | 51.3938 | 52.5135 | 53.4331 | 54.2102 | 54.8812 | 55.4705 | 55.9951 | 56.4670 | 56.8949 | 57.2855 | 57.6436 | 57.9730 |
| Mixed PT + SFT×1 | 36.0051 | 42.3986 | 45.5466 | 47.5722 | 49.0437 | 50.1951 | 51.1416 | 51.9470 | 52.6499 | 53.2750 | 53.8394 | 54.3549 | 54.8305 | 55.2731 | 55.6883 | 56.0811 |
| Mixed PT + SFT×3, fresh Adam | 38.4417 | 44.1582 | 47.1152 | 49.0982 | 50.5739 | 51.7417 | 52.7051 | 53.5244 | 54.2371 | 54.8683 | 55.4350 | 55.9496 | 56.4210 | 56.8559 | 57.2593 | 57.6351 |
| Mixed PT + SFT×3, continued Adam | 38.0405 | 43.9465 | 46.9872 | 49.0520 | 50.6109 | 51.8567 | 52.8890 | 53.7666 | 54.5272 | 55.1963 | 55.7920 | 56.3279 | 56.8137 | 57.2573 | 57.6647 | 58.0405 |

Aggregate 0--16 win histograms:

| Model | Counts for wins 0, 1, ..., 16 |
|---|---|
| 81→85 + SFT×3 | `[637, 92, 36, 35, 26, 26, 32, 24, 25, 26, 29, 24, 24, 31, 39, 84, 290]` |
| Native-85 + SFT×3 | `[622, 78, 51, 31, 32, 27, 27, 24, 15, 19, 28, 32, 34, 30, 47, 81, 302]` |
| Mixed PT + SFT×1 | `[650, 93, 40, 40, 22, 30, 28, 28, 27, 23, 35, 32, 32, 28, 45, 104, 223]` |
| Mixed PT + SFT×3, fresh Adam | `[627, 89, 49, 32, 32, 29, 24, 25, 25, 22, 19, 21, 27, 33, 30, 100, 296]` |
| Mixed PT + SFT×3, continued Adam | `[621, 89, 56, 39, 35, 26, 26, 22, 19, 21, 25, 16, 31, 36, 44, 97, 277]` |

### B1--B5 pass@k

#### 81→85 + SFT×3

| Bucket | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 78.9976 | 84.6726 | 86.8767 | 88.1460 | 89.0071 | 89.6513 | 90.1634 | 90.5869 | 90.9466 | 91.2582 | 91.5322 | 91.7759 | 91.9944 | 92.1916 | 92.3701 | 92.5325 |
| B2 | 62.4371 | 71.2975 | 75.3002 | 77.6687 | 79.2539 | 80.4035 | 81.2878 | 81.9991 | 82.5914 | 83.0984 | 83.5425 | 83.9389 | 84.2989 | 84.6309 | 84.9413 | 85.2349 |
| B3 | 34.8549 | 43.7016 | 48.4704 | 51.6446 | 53.9841 | 55.8147 | 57.3016 | 58.5399 | 59.5897 | 60.4916 | 61.2751 | 61.9620 | 62.5696 | 63.1117 | 63.6002 | 64.0449 |
| B4 | 10.1916 | 15.1684 | 18.3953 | 20.7170 | 22.4987 | 23.9382 | 25.1524 | 26.2139 | 27.1683 | 28.0451 | 28.8635 | 29.6359 | 30.3708 | 31.0743 | 31.7509 | 32.4042 |
| B5 | 3.1445 | 4.7656 | 5.8253 | 6.6374 | 7.3176 | 7.9184 | 8.4672 | 8.9799 | 9.4664 | 9.9330 | 10.3842 | 10.8233 | 11.2528 | 11.6745 | 12.0898 | 12.5000 |

#### Native-85 + SFT×3

| Bucket | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 80.5195 | 85.7819 | 87.8096 | 88.9500 | 89.7079 | 90.2682 | 90.7115 | 91.0786 | 91.3924 | 91.6671 | 91.9120 | 92.1332 | 92.3353 | 92.5216 | 92.6948 | 92.8571 |
| B2 | 63.9052 | 72.8020 | 76.5065 | 78.6721 | 80.1495 | 81.2415 | 82.0868 | 82.7616 | 83.3135 | 83.7748 | 84.1686 | 84.5114 | 84.8154 | 85.0895 | 85.3398 | 85.5705 |
| B3 | 37.8277 | 46.6355 | 51.2393 | 54.2577 | 56.4716 | 58.2105 | 59.6419 | 60.8613 | 61.9266 | 62.8746 | 63.7292 | 64.5067 | 65.2187 | 65.8739 | 66.4794 | 67.0412 |
| B4 | 11.7378 | 17.1661 | 20.6290 | 23.2086 | 25.2788 | 27.0049 | 28.4788 | 29.7597 | 30.8889 | 31.8957 | 32.8015 | 33.6220 | 34.3685 | 35.0494 | 35.6707 | 36.2369 |
| B5 | 2.3828 | 4.0156 | 5.2243 | 6.1657 | 6.9229 | 7.5463 | 8.0692 | 8.5153 | 8.9011 | 9.2388 | 9.5371 | 9.8027 | 10.0407 | 10.2552 | 10.4492 | 10.6250 |

#### Mixed PT + SFT×1

| Bucket | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 79.4643 | 86.3663 | 88.8625 | 90.2505 | 91.1785 | 91.8663 | 92.4078 | 92.8505 | 93.2219 | 93.5397 | 93.8156 | 94.0581 | 94.2735 | 94.4670 | 94.6429 | 94.8052 |
| B2 | 58.7458 | 68.9150 | 73.3617 | 75.9785 | 77.7397 | 79.0312 | 80.0381 | 80.8614 | 81.5610 | 82.1739 | 82.7236 | 83.2252 | 83.6883 | 84.1191 | 84.5218 | 84.8993 |
| B3 | 29.9860 | 39.3071 | 44.2957 | 47.5478 | 49.9121 | 51.7635 | 53.2893 | 54.5912 | 55.7282 | 56.7370 | 57.6413 | 58.4576 | 59.1981 | 59.8720 | 60.4869 | 61.0487 |
| B4 | 8.7761 | 13.3333 | 16.3645 | 18.5961 | 20.3438 | 21.7685 | 22.9634 | 23.9877 | 24.8815 | 25.6738 | 26.3865 | 27.0370 | 27.6394 | 28.2056 | 28.7456 | 29.2683 |
| B5 | 2.4414 | 4.0339 | 5.1685 | 6.0493 | 6.7813 | 7.4194 | 7.9935 | 8.5208 | 9.0120 | 9.4745 | 9.9134 | 10.3326 | 10.7355 | 11.1250 | 11.5039 | 11.8750 |

#### Mixed PT + SFT×3, fresh Adam

| Bucket | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 80.1136 | 85.6683 | 88.0380 | 89.5232 | 90.5871 | 91.4037 | 92.0602 | 92.6067 | 93.0739 | 93.4812 | 93.8406 | 94.1601 | 94.4452 | 94.6997 | 94.9269 | 95.1299 |
| B2 | 62.0176 | 71.0543 | 75.1091 | 77.5583 | 79.2269 | 80.4528 | 81.4017 | 82.1657 | 82.8004 | 83.3415 | 83.8133 | 84.2328 | 84.6117 | 84.9581 | 85.2768 | 85.5705 |
| B3 | 35.8146 | 44.0574 | 48.6022 | 51.7241 | 54.0647 | 55.9161 | 57.4382 | 58.7267 | 59.8425 | 60.8263 | 61.7069 | 62.5057 | 63.2390 | 63.9201 | 64.5599 | 65.1685 |
| B4 | 10.4965 | 15.0232 | 17.9306 | 20.0574 | 21.7421 | 23.1473 | 24.3606 | 25.4338 | 26.3997 | 27.2800 | 28.0895 | 28.8389 | 29.5358 | 30.1858 | 30.7927 | 31.3589 |
| B5 | 3.6328 | 5.3724 | 6.5921 | 7.5407 | 8.3237 | 8.9923 | 9.5745 | 10.0884 | 10.5469 | 10.9598 | 11.3348 | 11.6779 | 11.9939 | 12.2865 | 12.5586 | 12.8125 |

#### Mixed PT + SFT×3, continued Adam

| Bucket | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 | k=13 | k=14 | k=15 | k=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 80.0933 | 86.0958 | 88.3529 | 89.6932 | 90.6253 | 91.3216 | 91.8658 | 92.3059 | 92.6710 | 92.9801 | 93.2454 | 93.4758 | 93.6775 | 93.8555 | 94.0138 | 94.1558 |
| B2 | 61.3465 | 69.8993 | 73.6829 | 75.9851 | 77.5858 | 78.7839 | 79.7220 | 80.4783 | 81.1010 | 81.6222 | 82.0646 | 82.4451 | 82.7762 | 83.0677 | 83.3263 | 83.5570 |
| B3 | 34.5506 | 43.0774 | 47.7936 | 51.0374 | 53.4742 | 55.3978 | 56.9690 | 58.2876 | 59.4203 | 60.4140 | 61.3019 | 62.1081 | 62.8498 | 63.5393 | 64.1854 | 64.7940 |
| B4 | 10.4530 | 15.7085 | 19.2316 | 21.9114 | 24.0931 | 25.9440 | 27.5540 | 28.9760 | 30.2435 | 31.3798 | 32.4021 | 33.3245 | 34.1588 | 34.9158 | 35.6054 | 36.2369 |
| B5 | 3.5156 | 5.2604 | 6.5329 | 7.5388 | 8.3708 | 9.0815 | 9.7037 | 10.2593 | 10.7636 | 11.2271 | 11.6574 | 12.0596 | 12.4369 | 12.7917 | 13.1250 | 13.4375 |

### B1--B5 outcome diagnostics

| Model | Bucket | Evaluated | Format valid | All-zero | All-sixteen | Mixed outcomes |
|---|---|---:|---:|---:|---:|---:|
| 81→85 + SFT×3 | B1 | 308 | 99.8985% | 7.4675% (23) | 53.5714% (165) | 120 |
| 81→85 + SFT×3 | B2 | 298 | 99.9161% | 14.7651% (44) | 31.8792% (95) | 159 |
| 81→85 + SFT×3 | B3 | 267 | 99.8596% | 35.9551% (96) | 10.4869% (28) | 143 |
| 81→85 + SFT×3 | B4 | 287 | 99.9347% | 67.5958% (194) | 0.6969% (2) | 91 |
| 81→85 + SFT×3 | B5 | 320 | 99.9023% | 87.5000% (280) | 0.0000% (0) | 40 |
| Native-85 + SFT×3 | B1 | 308 | 99.9797% | 7.1429% (22) | 57.1429% (176) | 110 |
| Native-85 + SFT×3 | B2 | 298 | 99.9581% | 14.4295% (43) | 29.8658% (89) | 166 |
| Native-85 + SFT×3 | B3 | 267 | 99.9532% | 32.9588% (88) | 13.1086% (35) | 144 |
| Native-85 + SFT×3 | B4 | 287 | 99.9347% | 63.7631% (183) | 0.6969% (2) | 102 |
| Native-85 + SFT×3 | B5 | 320 | 99.8633% | 89.3750% (286) | 0.0000% (0) | 34 |
| Mixed PT + SFT×1 | B1 | 308 | 99.9391% | 5.1948% (16) | 46.7532% (144) | 148 |
| Mixed PT + SFT×1 | B2 | 298 | 99.8322% | 15.1007% (45) | 21.1409% (63) | 190 |
| Mixed PT + SFT×1 | B3 | 267 | 99.8127% | 38.9513% (104) | 5.2434% (14) | 149 |
| Mixed PT + SFT×1 | B4 | 287 | 99.9564% | 70.7317% (203) | 0.6969% (2) | 82 |
| Mixed PT + SFT×1 | B5 | 320 | 99.8438% | 88.1250% (282) | 0.0000% (0) | 38 |
| Mixed PT + SFT×3, fresh Adam | B1 | 308 | 99.9391% | 4.8701% (15) | 56.4935% (174) | 119 |
| Mixed PT + SFT×3, fresh Adam | B2 | 298 | 99.8532% | 14.4295% (43) | 27.8523% (83) | 172 |
| Mixed PT + SFT×3, fresh Adam | B3 | 267 | 99.9064% | 34.8315% (93) | 11.9850% (32) | 142 |
| Mixed PT + SFT×3, fresh Adam | B4 | 287 | 99.8258% | 68.6411% (197) | 1.7422% (5) | 85 |
| Mixed PT + SFT×3, fresh Adam | B5 | 320 | 99.8633% | 87.1875% (279) | 0.6250% (2) | 39 |
| Mixed PT + SFT×3, continued Adam | B1 | 308 | 99.7362% | 5.8442% (18) | 51.6234% (159) | 131 |
| Mixed PT + SFT×3, continued Adam | B2 | 298 | 99.8322% | 16.4430% (49) | 29.1946% (87) | 162 |
| Mixed PT + SFT×3, continued Adam | B3 | 267 | 99.8127% | 35.2060% (94) | 10.4869% (28) | 145 |
| Mixed PT + SFT×3, continued Adam | B4 | 287 | 99.8040% | 63.7631% (183) | 0.3484% (1) | 103 |
| Mixed PT + SFT×3, continued Adam | B5 | 320 | 99.8047% | 86.5625% (277) | 0.6250% (2) | 41 |

Per-bucket 0--16 win histograms:

| Model | Bucket | Counts for wins 0, 1, ..., 16 |
|---|---|---|
| 81→85 + SFT×3 | B1 | `[23, 8, 6, 4, 3, 5, 3, 3, 5, 4, 9, 6, 5, 10, 12, 37, 165]` |
| 81→85 + SFT×3 | B2 | `[44, 14, 6, 8, 9, 4, 11, 9, 6, 10, 11, 8, 6, 11, 19, 27, 95]` |
| 81→85 + SFT×3 | B3 | `[96, 19, 14, 15, 8, 7, 9, 3, 10, 6, 6, 7, 11, 5, 6, 17, 28]` |
| 81→85 + SFT×3 | B4 | `[194, 30, 8, 6, 4, 9, 7, 8, 3, 3, 3, 2, 1, 3, 1, 3, 2]` |
| 81→85 + SFT×3 | B5 | `[280, 21, 2, 2, 2, 1, 2, 1, 1, 3, 0, 1, 1, 2, 1, 0, 0]` |
| Native-85 + SFT×3 | B1 | `[22, 8, 4, 4, 2, 4, 4, 3, 4, 5, 5, 11, 5, 5, 12, 34, 176]` |
| Native-85 + SFT×3 | B2 | `[43, 11, 7, 7, 11, 7, 6, 4, 6, 6, 13, 5, 18, 12, 20, 33, 89]` |
| Native-85 + SFT×3 | B3 | `[88, 24, 14, 9, 5, 8, 9, 10, 4, 6, 7, 9, 6, 9, 11, 13, 35]` |
| Native-85 + SFT×3 | B4 | `[183, 26, 19, 7, 9, 7, 5, 4, 1, 2, 3, 6, 4, 4, 4, 1, 2]` |
| Native-85 + SFT×3 | B5 | `[286, 9, 7, 4, 5, 1, 3, 3, 0, 0, 0, 1, 1, 0, 0, 0, 0]` |
| Mixed PT + SFT×1 | B1 | `[16, 8, 5, 7, 2, 4, 7, 1, 4, 7, 8, 8, 11, 11, 17, 48, 144]` |
| Mixed PT + SFT×1 | B2 | `[45, 18, 9, 5, 6, 13, 8, 10, 7, 9, 10, 12, 14, 10, 17, 42, 63]` |
| Mixed PT + SFT×1 | B3 | `[104, 24, 17, 9, 7, 5, 10, 8, 12, 2, 15, 6, 6, 4, 11, 13, 14]` |
| Mixed PT + SFT×1 | B4 | `[203, 24, 6, 14, 7, 6, 1, 8, 2, 4, 0, 5, 1, 3, 0, 1, 2]` |
| Mixed PT + SFT×1 | B5 | `[282, 19, 3, 5, 0, 2, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0]` |
| Mixed PT + SFT×3, fresh Adam | B1 | `[15, 10, 9, 5, 2, 5, 8, 3, 5, 2, 3, 5, 7, 11, 11, 33, 174]` |
| Mixed PT + SFT×3, fresh Adam | B2 | `[43, 14, 9, 4, 14, 6, 7, 8, 9, 8, 9, 6, 13, 10, 10, 45, 83]` |
| Mixed PT + SFT×3, fresh Adam | B3 | `[93, 26, 10, 15, 5, 12, 4, 8, 7, 8, 2, 4, 4, 10, 8, 19, 32]` |
| Mixed PT + SFT×3, fresh Adam | B4 | `[197, 26, 14, 4, 7, 4, 4, 5, 3, 4, 3, 3, 3, 2, 1, 2, 5]` |
| Mixed PT + SFT×3, fresh Adam | B5 | `[279, 13, 7, 4, 4, 2, 1, 1, 1, 0, 2, 3, 0, 0, 0, 1, 2]` |
| Mixed PT + SFT×3, continued Adam | B1 | `[18, 7, 6, 6, 3, 5, 4, 6, 4, 2, 4, 3, 9, 15, 18, 39, 159]` |
| Mixed PT + SFT×3, continued Adam | B2 | `[49, 11, 10, 8, 12, 4, 7, 5, 7, 9, 7, 8, 12, 10, 16, 36, 87]` |
| Mixed PT + SFT×3, continued Adam | B3 | `[94, 26, 12, 9, 13, 11, 8, 4, 5, 5, 8, 2, 8, 9, 7, 18, 28]` |
| Mixed PT + SFT×3, continued Adam | B4 | `[183, 29, 20, 15, 3, 3, 4, 6, 3, 4, 4, 3, 2, 1, 3, 3, 1]` |
| Mixed PT + SFT×3, continued Adam | B5 | `[277, 16, 8, 1, 4, 3, 3, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2]` |

### Continued Adam compared with fresh Adam

The following values are continued-Adam minus fresh-Adam, in percentage
points.

| Metric | Delta |
|---|---:|
| pass@1 | -0.4012 |
| pass@2 | -0.2117 |
| pass@3 | -0.1280 |
| pass@4 | -0.0461 |
| pass@5 | +0.0370 |
| pass@6 | +0.1150 |
| pass@7 | +0.1839 |
| pass@8 | +0.2423 |
| pass@9 | +0.2901 |
| pass@10 | +0.3280 |
| pass@11 | +0.3570 |
| pass@12 | +0.3783 |
| pass@13 | +0.3927 |
| pass@14 | +0.4015 |
| pass@15 | +0.4054 |
| pass@16 | +0.4054 |
| format-valid rate | -0.0802 |
| all-zero percentage | -0.4054 |

The fresh optimizer is higher at pass@1 by 0.4012 percentage points. The
continued optimizer crosses above it at pass@5 and is higher at pass@16 by
0.4054 percentage points. It also has 0.4054 percentage points fewer all-zero
prompts, while its format-valid rate is 0.0802 percentage points lower. These
are small descriptive differences; this single evaluation does not provide
uncertainty estimates.

### Result integrity

| Model | Worker call | Summary SHA-256 |
|---|---|---|
| 81→85 + SFT×3 | `fc-01M03EV4QXBJSZ4566CZ3Y0DHQ` | `c028e434f57328f9cbeb0fdf595a3e2e68452d09f391b89d21fe9185f30338e5` |
| Native-85 + SFT×3 | `fc-01M03EV5DXB3CQA7K089M08RWC` | `9280bd6e9908d5fa1fa93b2dcfff12981b1955058ec46528e36a6101d8bf1b8b` |
| Mixed PT + SFT×1 | `fc-01M03EV66F62A9MC2XEX8HDQDH` | `e9e3ed123f98a47a395d50280923928fe12c2d952d9e54a2039da0e026c80f2f` |
| Mixed PT + SFT×3, fresh Adam | `fc-01M03EV721QAHBN58RXWD11R5Y` | `1a123e101392b7021d4ef00d583525a9d83f79a32c94c06077fe1e1f7fe45818` |
| Mixed PT + SFT×3, continued Adam | `fc-01M03EV832D5HH5HXVM6Y8A5XA` | `59541bca9ca99b9e744ecb932a0c0d41d5b3da01243547d4fef228f8fc59c743` |

The merged ledger SHA-256 is
`bc171e5452a3d319cba914e677db73772c5f031e5fc1edcf24fb456bfa402ba5`.
Each model evaluated 1,480 admitted prompts (1,484 raw rows minus four
overlength prompts) and 23,680 trajectories. The result set contains five
authenticated worker summaries, their raw gzip artifacts, and the committed
merged ledger.

Recent worker logs contain `illegal san` messages from expected scoring of
invalid model responses. Four workers also emitted a multiprocessing
`KeyboardInterrupt` during interpreter teardown after their success artifacts
and ledger entries had committed. Every exact worker call returned its success
result, so these post-success teardown messages do not invalidate the
evaluation.

The v1 namespace contains only a failed pre-inference canary. It exposed a
Transformers 4.51 compatibility bug where the converter assigned a
non-serializable `torch.dtype` to a model-specific config field. The converter
now writes the canonical string `float32`; a regression test covers this case.
No v1 production dispatcher or production worker was launched.

## Rejected alternatives

- Do not use the older 2,560-response/3,072-context B1--B5 harness. Its context
  contract does not match these policies.
- Do not call the 53,225-row balanced-source evaluation a held-out test. The RL
  training prompts were derived from that source.
- Do not report only pass@1 or a thresholded "at least one win" statistic.
- Do not relaunch a pending or running evaluation call.
