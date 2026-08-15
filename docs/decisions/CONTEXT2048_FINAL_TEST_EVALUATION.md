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
