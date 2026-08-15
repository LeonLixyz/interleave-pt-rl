# RL checkpoint evaluation

Launched on Modal on 2026-07-26 (America/New_York).

- Modal app: `ap-RDpnfGIVOdxuFWwRoWbec8`
- Watcher call: `fc-01KYGRRH79QTQK12WR0QR2YX5W`
- Results volume: `chess-rl-eval-results-r6`
- Worker limit: 4 concurrent H200 containers
- Source revision: `pavelslab-nyu/pre2post-chess@40f04428`
- HF checkpoint prefix: `miles_sglang_grpo_r6`

The watcher polls Hugging Face every 120 seconds and queues newly uploaded
checkpoints until it has seen the complete 20-step grids:

- `Pre-to-Post-2/rl_C6p5e18_32m_alpha0.200_beta0.013`: steps 20–4000
- `Pre-to-Post-2/rl_C6p5e18_32m_alpha0.400_beta0.013`: steps 20–4000
- `Pre-to-Post-2/rl_C6p5e18_410m_alpha0.750_beta0.148`: steps 20–3000
- `Pre-to-Post-2/rl_C6p5e18_410m_alpha1.000_beta0.148`: steps 20–3000

Production evaluation uses B1–B5, multi-turn thinking, vLLM, response length
2560, temperature 1, 16 samples, and seed 0. Four overlength raw prompts are
filtered, leaving 1,480 prompts and exactly 23,680 output rows per checkpoint.
A checkpoint is marked successful only after the subprocess exits zero,
`metrics.json` exists, and the JSONL row count is exact.

Preflight results:

- 32M step 20, B1 full settings: 4,928/4,928 rows in 431.5 seconds;
  mean reward 0.321, first-move score 0.452, legality 0.808.
- 410M step 20, B1 full settings: 4,928/4,928 rows in 623.9 seconds;
  mean reward 0.580, first-move score 0.690, legality 0.913.
- First complete production result: 32M alpha 0.200 step 80,
  23,680/23,680 rows in 1,648.9 seconds (27m 29s). B1–B5 reward
  `mean@16`: 0.3894, 0.2433, 0.1196, 0.0409, 0.0154.

The local launcher is `modal_eval_all_rl_ckpts.py`. Useful commands:

```bash
cd "/Users/leonli66/Desktop/Research/RL/Chess RL/Eval"
modal run modal_eval_all_rl_ckpts.py --mode status
modal app logs ap-RDpnfGIVOdxuFWwRoWbec8 --timestamps
```

Results are namespaced on the Volume as:

```text
/results/v1/<run>/global_step_<step>/production_<fingerprint>/
  _QUEUED.json
  _RUNNING.json
  _SUCCESS.json or _FAILED.json
  output/eval/eval.log
  output/eval/generations/0.jsonl
  output/eval/generations/metrics.json
```
