# Context-2,048 PT + SFT + trace + RL matrix

Status: accepted 2026-08-15. The seven 5B-total experiment graphs are frozen in
`experiments/context2048_pt_sft_trace_rl_v1/5b/`. Production may start only
after the configuration, data, W&B, and two-process GPU gates pass.

## Meaning of PT checkpoints

Every PT stage uniformly mixes ordinary SFT records into the selected PT
records. There is no separate ordinary SFT stage in this matrix.

“Reset to PT(5B)” means reset to the complete checkpoint produced by 5B PT
target tokens plus all 233,151 ordinary SFT exposures. “Reset to PT(2.5B)”
means reset to the complete first-stage checkpoint produced by 2.5B PT target
tokens plus the first 116,575 ordinary SFT exposures. A reset never means a
PT-only checkpoint.

The same rule applies to the planned 10B-total matrix: “Reset to PT(10B)”
means the complete checkpoint produced by 10B PT target tokens plus all
233,151 ordinary SFT exposures. It does not mean a PT-only checkpoint.

The SFT source has 77,717 rows. Exactly three exposure copies are used, for
233,151 exposures total. Two-stage PT divides them into 116,575 and 116,576
exposures. Stable deterministic placement uniformly mixes them into PT while
preserving the relative order inside the PT and SFT streams.

## Seven 5B-total experiments

| # | Experiment graph | PT learning rate | Trace learning rate | RL learning rate |
|---:|---|---|---:|---:|
| 1 | PT(5B + SFT×3) → RL(3,000) | one schedule | — | 1e-5 |
| 2 | PT(2.5B + first SFT half) → PT(2.5B + second SFT half) → RL(3,000) | fresh schedule in each PT stage | — | 1e-5 |
| 3 | PT(5B + SFT×3) → RL(1,500); reset to the complete PT checkpoint → shuffled successful-trace training → RL(1,500) | one schedule | 1e-5 | 1e-5 |
| 4 | PT(5B + SFT×3) → RL(1,500); reset to the complete PT checkpoint → chronological successful-trace training → RL(1,500) | one schedule | 1e-5 | 1e-5 |
| 5 | PT(2.5B + first SFT half) → RL(1,500); reset to that complete PT checkpoint → shuffled successful-trace training → PT(2.5B + second SFT half) → RL(1,500) | fresh schedule in each PT stage | 1e-5 | 1e-5 |
| 6 | PT(2.5B + first SFT half) → RL(1,500); reset to that complete PT checkpoint → chronological successful-trace training → PT(2.5B + second SFT half) → RL(1,500) | fresh schedule in each PT stage | 1e-5 | 1e-5 |
| 7 | PT(2.5B + first SFT half) → RL(1,500); reset to that complete PT checkpoint → PT(2.5B + second SFT half + chronological successful traces, uniformly mixed) → RL(1,500) | fresh schedule in each PT stage; traces share the second PT schedule | — | 1e-5 |

Every PT schedule uses a fresh AdamW optimizer, 5% linear warmup to 1e-3,
then cosine decay to 1e-5. Every RL stage uses a constant 1e-5 learning rate.
Separate trace-supervised stages use a fresh AdamW optimizer and constant 1e-5
learning rate. Each experiment has exactly 5B PT target tokens and 3,000 RL
updates in total.

## Successful RL traces

Trace examples come only from the first 1,500-update RL stage in the same
experiment. A trace is accepted only if reward is 1, the move and format are
valid, it contains exactly one closing think token and at least one
`<call_env>`, and the prompt-response pair is not duplicated. Saved token IDs
and loss masks are reused; traces are not regenerated or retokenized.

Chronological order is `(rollout_id, prompt_occurrence, sample_index)`.
Shuffled order uses seed 42. A separate trace stage makes one supervised pass.

## Tokenizer, masks, and precision

- Vocabulary size: 85. Mapping SHA-256:
  `f0366c5dc44ada849282959e67b172da79264c0b9336707c03648c430ccf0651`.
- Special IDs: `<bos>=0`, `<eos>=1`, `<unk>=2`, `<T>=81`, `</T>=82`,
  `<sep>=83`, `<call_env>=84`.
- Every sequence contains exactly one explicit BOS. The tokenizer does not add
  BOS. For packed PT, BOS is context and not a prediction target.
- PT tokens are supervised. SFT/trace prompts and environment replies are
  masked. Reasoning and move tokens are supervised.
- Master parameters, gradients after reduction, Adam moments, resumable
  checkpoints, and Hugging Face exports are FP32. Forward/backward and
  in-memory inference are BF16.

## Shared RL contract

RL uses seed 42, deterministic sample-index inference, low-variance KL
coefficient 0.001, 256 prompts × 8 samples per update, context 2,048,
131,072 tokens per GPU, and SGLang concurrency 128. The common 28,419-prompt
parquet SHA-256 is
`024d423c1438a1bfc0ed24898732a842dc13542082e3ecefdb62f4d0f43cbcbc`.

## Configuration and launch authority

- Shared contract: `experiments/context2048_pt_sft_trace_rl_v1/shared.yaml`
- Config resolver and validator: `chess/experiments/context2048_matrix_config.py`
- Initial shared-parent launcher:
  `chess/pretrain-sft/modal_scripts/modal_context2048_pt_matrix.py`

The two initial PT workers are shared because experiments 1/3/4 use one
bit-identical 5B parent and experiments 2/5/6/7 use one bit-identical 2.5B
first-stage parent. Downstream stages branch from those authenticated exports.

## Current initial-parent gate and launch evidence

The active production version is
`context2048_pt_sft_trace_rl_fp32_master_v5_20260815` in Modal app
`chess-context2048-configured-pt-matrix-v5` (`ap-qRtR2ublLtJiXCAM6JZ7uf`).

- Source-tree SHA-256:
  `242927295fb5eaf2534cf653a95ca1c897de8d3c22f4256b6ebff531cd1cc08a`
- Initial manifest-set SHA-256:
  `b1b2b8e8d14ff731b5fd5924e98fed04ee47bd8428d01324adff32885deaf96d`
- PT selection SHA-256:
  `104c07f90209cae6d96a86cf7e6b4ecf341271f5abeb3019f239b78a72eb681c`
- W&B write/read gate SHA-256:
  `38d33f18cc629b7eeee19c0e09aac2343713782d80f01b9ec1d4d76d886d11b1`
- Canary calls:
  `fc-01M03W1ANWK93A8D991ZFC1JH9` (5B parent) and
  `fc-01M03W1ASGJEGEKF9NDBD2FSK1` (2.5B parent). The 5B gate passed with
  gate SHA-256
  `a0a9655c01e2faacbf7e99d79a362687adc7e0006879f217a81b964bed433cdf`.
  The 2.5B gate passed with gate SHA-256
  `6f791b2fe5da36629afdee2380918c39e622c0104fadd686191cfef7c07524ce`.

The v1 container-layout check, v2 deterministic-CUDA check, and v3
random-initialization identity check failed before any optimizer update. They
did not create production claims. The v4 gate reached real training, but its
ordinary uniform production order did not guarantee both PT and SFT examples
in every rank's first local batch. It therefore failed the gate before any
production claim. V5 keeps the production manifests unchanged and adds a
separate authenticated two-update canary manifest that alternates PT and SFT
inside every rank-local batch. Failed-version roots must not be used as
experiment inputs.

The two shared initial parents launched once on 2026-08-15:

| Parent | Steps | Worker | Claim SHA-256 | Execution SHA-256 |
|---|---:|---|---|---|
| 5B PT targets + all SFT×3 | 20,895 | `fc-01M03WF0P3RNXYENX90AXFAK2D` | `83429200aec1e14d4325d326dd98a19ab2a671a8e40a2b4b4113d8c71e0bd89b` | `6164df8e9360f75062062f1d60deb60bc1648aada6e037e66d76d94aeeff44e6` |
| first 2.5B PT targets + first SFT half | 10,448 | `fc-01M03WF2S16GRK386WJ5PCRSWQ` | `eee4fddcf54bd2eab6af90cca76533f428d4dc08c12e416ba60e572734adf96c` | `fbebe92f37a3edee48f5b6d79d4fe313db4c33b763809a8db4a4284ae68c57cb` |

The mode-0600 local recovery records are under
`chess/pretrain-sft/.launch-recovery/context2048_pt_sft_trace_rl_fp32_master_v5_20260815/`
and are excluded from Git. These exact calls are the only authorized workers.
