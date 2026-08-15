# Documentation

What the code is, what it is configured with, and how to launch it. Nothing else —
research findings, performance comparisons, and incident history live in
`results/notes/`.

Every document has the same three sections:

1. **Code** — which file does what
2. **Config** — the exact defaults, and where they are defined
3. **Launch** — the commands

| Doc | Stage |
|---|---|
| [01-chess-pretraining.md](01-chess-pretraining.md) | chess pretraining with SFT mixed into the stream |
| [02-chess-sft.md](02-chess-sft.md) | chess SFT as a separate stage |
| [03-chess-rl.md](03-chess-rl.md) | chess RL (GRPO, Miles + SGLang) |
| [04-chess-eval.md](04-chess-eval.md) | chess pass@k evaluation |
| [05-math-pretraining.md](05-math-pretraining.md) | math pretraining and annealing |
| [06-math-sft.md](06-math-sft.md) | math SFT |
| [07-math-rl.md](07-math-rl.md) | math RL (GRPO, verl) |
| [08-data-and-artifacts.md](08-data-and-artifacts.md) | where data, checkpoints and external dependencies live |

Accepted experiment protocols that must remain immutable are recorded under
`decisions/`. The current native context-2048 final test protocol is
[`decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md`](decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md).

`history/` holds the original experiment plans and launch ledgers, unedited.

## Before you launch anything

Three rules that apply to every stage:

1. **Use `modal run --detach`** for anything that spawns GPU work. Without it Modal
   kills the run seconds after the local entrypoint returns.
2. **Chess prompts must start with `<bos>`** (token id 0). The tokenizer does not add
   it. This affects rollouts and evaluation, not training data.
3. **Run the canary first** where a stage has one. It executes a single step on the
   real topology and catches manifest, cache and hardware errors in about a minute.
