# Eval data (not stored in this repo)

`modal_eval_clean.py` needs the 4 benchmark shards here (51 MB total):

    eval_train_v4_balanced_shard0.parquet
    eval_train_v4_balanced_shard1.parquet
    eval_train_v4_balanced_shard2.parquet
    eval_train_v4_balanced_shard3.parquet

Sources, in order of preference:

1. The original working tree: `~/Desktop/Research/RL/Chess RL/Eval/test_data/`
2. Modal volume `chess-rl-miles-data`, path `/chess-rl-data/` (the unsharded
   `train_v4_dataset_balanced_multi_turn.parquet`, sha256
   bcf131d88cd9916f6ec12a12f8e85901a05e0dabc5d5a1fdebd8ea0b0ae86c30, plus the
   per-arm solvable-only datasets).

Symlink rather than copy if you want one physical copy:

    ln -s "$HOME/Desktop/Research/RL/Chess RL/Eval/test_data"/*.parquet .

Or point the evaluator elsewhere with `CHESS_EVAL_DATA_ROOT=/path/to/shards`.

The B1–B5 benchmark files (`test_B*_multi_turn.parquet`, 1.6 MB) are the held-out
suite for `modal_eval_context2048_final_test.py`. They remain external to this
repository and are packaged into that evaluator's immutable Modal image. The
file names, row counts, and SHA-256 hashes are pinned in the evaluator and in
`docs/decisions/CONTEXT2048_FINAL_TEST_EVALUATION.md`.
