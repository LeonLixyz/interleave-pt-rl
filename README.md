# interleave-pt-rl

Consolidated code for the interleaved pretraining ↔ RL research program: two model
lines (chess 47M, math 1.5B), each with pretraining, SFT, RL, and evaluation.

This tree contains **code only**. Corpora, tokenized caches, and checkpoints live on
Modal volumes and Hugging Face — see `docs/08-data-and-artifacts.md` for the exact
locations and hashes.

```
interleave-pt-rl/
├── docs/                  ← read this first: code, config, launch — one doc per stage
├── chess/
│   ├── pretrain-sft/      pretraining + SFT trainers, configs, tokenizer, Modal launchers
│   ├── rl/                chess_rl_miles: GRPO launcher, rollout, reward, provenance
│   └── eval/              held-out and pass@k evaluators; see docs/04-chess-eval.md
├── math/                  pretraining, anneal, SFT, RL, eval for the math line
│   └── external/          first-party deps pulled in from sibling repos (reward fn, SFT yamls)
├── miles/                 the Miles RL framework, our changes already applied
├── tools/                 measurement + analysis scripts (curves, losses, trace harvest)
├── results/               measurement caches + research notes (findings, lessons)
└── dashboard/             the two published result pages + their deploy scripts
```

## Where things run

Everything trains on Modal. Nothing in this tree runs locally except the analysis
scripts in `tools/`.

| Stage | Entry point | Hardware |
|---|---|---|
| chess pretraining (mixed PT+SFT) | `chess/pretrain-sft/scripts/train/train_interleaved_hf.py` | 8×H200 |
| chess SFT (separate stage) | `chess/pretrain-sft/scripts/train/run_sft.py` | 2×GPU |
| chess RL | `chess/rl/chess_rl_miles/scripts/modal_interleave.py` | 8×H200 |
| chess final held-out eval | `chess/eval/modal_eval_context2048_final_test.py` | 1×H200 per checkpoint |
| math pretraining / anneal | `math/train.py` | 8×H200 (or 4-node) |
| math SFT | `math/sft.py` | 8×H200 |
| math RL | `math/rl_train.py` | 8×H200 |

## Experiment configs and launchers

Every experiment is launched from one of these. Nothing else starts training.

| Line | Launcher | Config it reads |
|---|---|---|
| chess context-2,048 PT/SFT/trace/RL matrix | `chess/pretrain-sft/modal_scripts/modal_context2048_pt_matrix.py` for the shared initial PT parents; downstream launchers consume the same resolved graph | `experiments/context2048_pt_sft_trace_rl_v1/shared.yaml` + one experiment YAML under `5b/` |
| chess pretraining (all published results) | `chess/pretrain-sft/modal_scripts/launch_sft_injection_ablation.py` | `config/configs/interleaved_50m/base_3072.yaml` + explicit overrides in `_run_v2r1_leg` |
| chess pretraining (original 50M DAG) | `chess/pretrain-sft/modal_scripts/launch_50m_interleaved.py` | same base config |
| chess pretraining (20B fork) | `chess/pretrain-sft/modal_scripts/launch_50m_interleaved_20b.py` | same base config |
| chess SFT stage | `chess/pretrain-sft/modal_scripts/launch_sft_interleave.py` | `config/configs/qwen_multiturn_sft/sft_interleave_3072.yaml` |
| chess RL | `chess/rl/chess_rl_miles/scripts/modal_interleave.py` | `chess/rl/config/chess_multiturn.yaml` (env rules) + `build_train_command` in the launcher (all hyperparameters) |
| chess final held-out eval | `chess/eval/modal_eval_context2048_final_test.py` | frozen constants plus `docs/decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md` |
| math pretraining / anneal | `math/train.py`, fanned out by `math/launch_anneals.py` | `math/train_inner_mix.py` (Python, not YAML) |
| math SFT | `math/sft.py` | `math/external/sft-configs/olmo_sft_1b_numinamath.yaml` |
| math RL | `math/rl_train.py` | hydra overrides inside the file (no YAML of its own) |

New context-2,048 experiments are defined by validated YAML graphs. The launcher
still restates resolved values as explicit trainer overrides and records the resolved
config hash, so a config edit cannot silently change a run. Older experiments retain
their historical Python launchers. The math side has no YAML for pretraining and RL;
the Python defaults are the config. The per-stage documents in `docs/` reproduce
those values in tables.

## Three things that will bite you

1. **`modal run --detach`, always.** Without `--detach`, Modal stops the app when the
   local entrypoint returns and silently kills the spawned GPU call. Runs appear to
   start and then vanish.
2. **Chess prompts must start with `<bos>`.** The tokenizer does not add it. Missing
   `<bos>` in a rollout or eval prompt destroys scores — this was the historical cause
   of "RL doesn't work" in this project. The fix must be present in *both*
   `rollout.py` and `batched_rollout.py`.
3. **`math/` was never under version control.** The original `math-pretraining`
   directory has no git repo and no remote. The copy here is the only backup.

## Provenance of this consolidation

Copied 2026-08-11 from the working tree at `~/Desktop/Research/RL/Chess RL`.
Sources were left untouched. Where a component was a git checkout, the upstream
origin and commit are recorded next to the copy:

| Component here | Copied from | Upstream |
|---|---|---|
| `chess/pretrain-sft/` | `chess_reasoning/` | `github.com/jy-evangeline/chess_reasoning` (working tree was ahead of last commit) |
| `chess/rl/` | `chess-rl-miles/` | not a git repo |
| `chess/eval/` | `Eval/` | not a git repo |
| `chess/eval/reward_function/` | `Eval/pre2post-chess/rl/verl/` | `pavelslab-nyu/pre2post-chess@40f04428` |
| `math/` | `math-pretraining/` | **no git — this copy is the backup** |
| `math/external/` | `pretrain-rl-scaling/`, `pre2post-LM-SFT/` | `pavelslab-nyu/pretrain-rl-scaling`, LLaMA-Factory fork |
| `miles/` | `miles/` (working tree, our patches applied) | `radixark/miles@e20de26c9` + `miles/our_changes.patch` |

External frameworks that are **not** included here and must be installed or pinned:
OLMo-core (math pretraining), verl 0.9 (math RL), LLaMA-Factory fork (math SFT),
SGLang (chess RL inference). Versions in `docs/08-data-and-artifacts.md`.
