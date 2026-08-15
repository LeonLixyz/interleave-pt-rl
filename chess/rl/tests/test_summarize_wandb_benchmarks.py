import pytest

from chess_rl_miles.scripts.summarize_wandb_benchmarks import resolve_run_path, summarize_run


class _FakeRun:
    id = "run123"
    name = "cache-canary"
    path = ["entity", "project", id]

    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, *, keys, page_size):
        assert page_size == 1000
        return ({key: row[key] for key in keys} for row in self.rows if all(key in row for key in keys))


def test_resolve_run_path_accepts_ids_paths_and_urls():
    assert resolve_run_path("abc", entity="team", project="chess") == "team/chess/abc"
    assert resolve_run_path("team/chess/abc", entity=None, project=None) == "team/chess/abc"
    assert (
        resolve_run_path("https://wandb.ai/team/chess/runs/abc/overview", entity=None, project=None)
        == "team/chess/abc"
    )


def test_resolve_run_path_rejects_bare_id_without_scope():
    with pytest.raises(ValueError, match="requires both"):
        resolve_run_path("abc", entity=None, project=None)


def test_summarize_run_skips_warmup_and_deduplicates_steps():
    rows = [
        {
            "rollout/step": 1,
            "perf/rollout_time": 10,
            "rollout/prefix_cache_hit_rate": 0.1,
            "rollout/avg_cached_tokens_per_sample": 10,
            "rollout/response_len/mean": 100,
        },
        {
            "rollout/step": 2,
            "perf/rollout_time": 20,
            "rollout/prefix_cache_hit_rate": 0.4,
            "rollout/avg_cached_tokens_per_sample": 40,
            "rollout/response_len/mean": 190,
        },
        {
            "rollout/step": 2,
            "perf/rollout_time": 22,
            "rollout/prefix_cache_hit_rate": 0.5,
            "rollout/avg_cached_tokens_per_sample": 50,
            "rollout/response_len/mean": 200,
        },
        {
            "rollout/step": 3,
            "perf/rollout_time": 30,
            "rollout/prefix_cache_hit_rate": 0.7,
            "rollout/avg_cached_tokens_per_sample": 70,
            "rollout/response_len/mean": 300,
        },
    ]

    summary = summarize_run(_FakeRun(rows), warmup_steps=1)

    assert summary["skipped_steps"] == [1.0]
    assert summary["used_steps"] == [2.0, 3.0]
    assert summary["metrics"]["perf/rollout_time"] == {"mean": 26.0, "count": 2}
    assert summary["metrics"]["rollout/prefix_cache_hit_rate"] == pytest.approx(
        {"mean": 0.6, "count": 2}
    )
    assert summary["metrics"]["rollout/avg_cached_tokens_per_sample"] == {"mean": 60.0, "count": 2}
    assert summary["metrics"]["rollout/response_len/mean"] == {"mean": 250.0, "count": 2}


def test_summarize_run_keeps_available_metrics_when_one_is_missing():
    rows = [
        {"rollout/step": 1, "perf/rollout_time": 10},
        {"rollout/step": 2, "perf/rollout_time": 20},
    ]

    summary = summarize_run(
        _FakeRun(rows),
        metrics=("perf/rollout_time", "rollout/prefix_cache_hit_rate"),
        warmup_steps=1,
    )

    assert summary["metrics"]["perf/rollout_time"] == {"mean": 20.0, "count": 1}
    assert summary["metrics"]["rollout/prefix_cache_hit_rate"] == {"mean": None, "count": 0}
