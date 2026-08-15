from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Eval import modal_rl_dashboard as dashboard


class _FakeVolume:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def listdir(self, path: str, *, recursive: bool = False):
        assert path == dashboard.ENDPOINT_NAMESPACE
        assert recursive is True
        return [SimpleNamespace(path=item) for item in self.paths]


def _write_result(
    root: Path,
    *,
    endpoint_id: str,
    checkpoint_sha256: str,
    component: str,
    fingerprint: str,
    metrics: dict,
) -> tuple[str, dict]:
    relative = (
        f"{dashboard.ENDPOINT_NAMESPACE}/{endpoint_id}/"
        f"{checkpoint_sha256}/{component}_{fingerprint[:12]}/_SUCCESS.json"
    )
    value = {
        "schema": dashboard.ENDPOINT_RESULT_SCHEMA,
        "schema_version": 1,
        "state": "complete",
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "endpoint_id": endpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "component": component,
        "eval_fingerprint": fingerprint,
        "metrics": metrics,
        "duration_seconds": 12.5,
        "finished_at": "2026-07-30T11:00:00+00:00",
    }
    if component == "chess":
        value["expected_rows"] = dashboard.EXPECTED_ROWS
        value["actual_rows"] = dashboard.EXPECTED_ROWS
    value["result_hash"] = dashboard.canonical_json_sha256(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return relative, value


def _chess_metrics() -> dict:
    benchmarks = {
        name: {"pass_at_1": value, "avg_reward": value}
        for name, value in zip(
            dashboard.ENDPOINT_BENCHMARKS,
            (0.1, 0.2, 0.3, 0.4, 0.5),
            strict=True,
        )
    }
    return {
        "pass_at_1": 0.3,
        "avg_reward": 0.3,
        "b3_avg": 0.3,
        "b4_avg": 0.4,
        "b3_b4_avg": 0.35,
        "benchmarks": benchmarks,
        "pass_at_1_semantics": "binary_reward_mean@16_fallback",
    }


def test_endpoint_summary_reads_only_verified_success_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = "1" * 64
    loss_path, loss_result = _write_result(
        tmp_path,
        endpoint_id="e2-final",
        checkpoint_sha256=checkpoint,
        component="losses",
        fingerprint="a" * 64,
        metrics={
            "heldout_pretrain_loss": 0.48,
            "heldout_pretrain_perplexity": 1.62,
            "heldout_pretrain_token_accuracy": 0.81,
            "heldout_pretrain_target_tokens": 12_582_912,
            "masked_sft_status": "unavailable_no_heldout",
        },
    )
    chess_path, chess_result = _write_result(
        tmp_path,
        endpoint_id="e2-final",
        checkpoint_sha256=checkpoint,
        component="chess",
        fingerprint="b" * 64,
        metrics=_chess_metrics(),
    )
    monkeypatch.setattr(dashboard, "EVAL_MOUNT", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "eval_volume",
        _FakeVolume([loss_path, chess_path]),
    )

    summary = dashboard._collect_endpoint_evaluations()
    endpoint = summary["endpoints"]["e2-final"]

    assert summary["aggregate"] == {
        "expected": 3,
        "complete": 1,
        "partial": 0,
        "missing": 2,
    }
    assert summary["errors"] == []
    assert endpoint["state"] == "complete"
    assert endpoint["checkpoint_sha256"] == checkpoint
    assert endpoint["loss"]["metrics"]["heldout_pretrain_loss"] == 0.48
    assert endpoint["chess"]["metrics"]["pass_at_1"] == 0.3
    assert endpoint["chess"]["metrics"]["benchmarks"]["B5"] == {
        "pass_at_1": 0.5,
        "avg_reward": 0.5,
    }
    assert endpoint["chess"]["metrics"]["b3_b4_avg"] == 0.35
    assert endpoint["result_hashes"] == {
        "losses": loss_result["result_hash"],
        "chess": chess_result["result_hash"],
    }


def test_endpoint_summary_rejects_tampered_result_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative, _ = _write_result(
        tmp_path,
        endpoint_id="p1",
        checkpoint_sha256="2" * 64,
        component="losses",
        fingerprint="c" * 64,
        metrics={
            "heldout_pretrain_loss": 0.5,
            "heldout_pretrain_perplexity": 1.65,
            "heldout_pretrain_token_accuracy": 0.8,
            "heldout_pretrain_target_tokens": 12_582_912,
            "masked_sft_status": "unavailable_no_heldout",
        },
    )
    marker = tmp_path / relative
    value = json.loads(marker.read_text())
    value["metrics"]["heldout_pretrain_loss"] = 999
    marker.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(dashboard, "EVAL_MOUNT", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "eval_volume",
        _FakeVolume([relative]),
    )

    summary = dashboard._collect_endpoint_evaluations()

    assert summary["endpoints"]["p1"]["state"] == "missing"
    assert any("endpoint result hash mismatch" in item for item in summary["errors"])


def test_endpoint_summary_retains_last_verified_table_on_listing_limit() -> None:
    previous = {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": "2026-07-30T11:00:00+00:00",
        "endpoints": {
            endpoint_id: {"endpoint_id": endpoint_id, "state": "complete"}
            for endpoint_id in dashboard.ENDPOINT_SUMMARY_SPECS
        },
        "aggregate": {
            "expected": 3,
            "complete": 3,
            "partial": 0,
            "missing": 0,
        },
        "errors": [],
    }
    current = {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": "2026-07-30T12:00:00+00:00",
        "endpoints": {
            endpoint_id: {"endpoint_id": endpoint_id, "state": "missing"}
            for endpoint_id in dashboard.ENDPOINT_SUMMARY_SPECS
        },
        "aggregate": {
            "expected": 3,
            "complete": 0,
            "partial": 0,
            "missing": 3,
        },
        "errors": [
            "endpoint result listing: ResourceExhaustedError: rate limit"
        ],
    }

    retained = dashboard._retain_endpoint_summary_on_listing_failure(
        current,
        previous,
    )

    assert retained["aggregate"]["complete"] == 3
    assert retained["aggregate"]["missing"] == 0
    assert retained["stale"] is True
    assert retained["last_success_generated_at"] == previous["generated_at"]
    assert retained["generated_at"] == current["generated_at"]
    assert retained["errors"] == []
    assert retained["warnings"] == current["errors"]
    assert retained["cache_status"] == "listing_backoff"
    assert retained["next_list_after"]


def test_endpoint_summary_does_not_mask_marker_validation_failure() -> None:
    previous = {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": "2026-07-30T11:00:00+00:00",
        "endpoints": {},
        "aggregate": {
            "expected": 3,
            "complete": 3,
            "partial": 0,
            "missing": 0,
        },
        "errors": [],
    }
    current = {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": "2026-07-30T12:00:00+00:00",
        "endpoints": {},
        "aggregate": {
            "expected": 3,
            "complete": 0,
            "partial": 0,
            "missing": 3,
        },
        "errors": ["endpoint_v1/p1: ValueError: result hash mismatch"],
    }

    assert (
        dashboard._retain_endpoint_summary_on_listing_failure(
            current,
            previous,
        )
        is current
    )


def test_durable_final_overrides_stale_running_modal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage"
    final = output / "final"
    final.mkdir(parents=True)
    (final / "config.json").write_text("{}", encoding="utf-8")
    (final / "model.safetensors").write_bytes(b"weights")
    metric = {
        "schema": "interleaved-local-metrics-v1",
        "step": 10,
        "manifest_hash": "d" * 64,
        "runtime_provenance": {
            "attention_backend": "sdpa",
            "torch_compile_mode": "none",
            "torch_version": "2.9.0+cu128",
            "transformers_version": "4.57.0",
            "flash_attention_version": None,
            "data_num_workers": 8,
        },
        "metrics": {
            "train/loss": 0.48,
            "train/lr": 1e-5,
            "train/global_valid_tokens": 500_000,
            "train/token_positions_per_second": 1_000_000,
            "train/manifest_cursor": 10,
        },
    }
    (output / "metrics.jsonl").write_text(
        json.dumps(metric) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "PRETRAIN_MOUNT", tmp_path)
    monkeypatch.setattr(
        dashboard,
        "_resolve_pretrain_output",
        lambda spec: "stage",
    )
    monkeypatch.setattr(
        dashboard,
        "_latest_call_record",
        lambda registry, stage_id: {
            "call_id": "fc-stale",
            "modal_state": "running",
            "status": "submitted",
        },
    )

    result = dashboard._pretrain_stage_status(
        {},
        {
            "stage_id": "exp2",
            "label": "Exp 2",
            "target": 10,
            "call_stage": "exp2",
            "registry_status": "complete",
        },
    )

    assert result["state"] == "complete"
    assert result["modal_state"] == "success"
    assert result["modal_state_source"] == "durable_final"
    assert result["reported_modal_state"] == "running"


def test_dashboard_html_renders_endpoint_summary_table() -> None:
    assert 'id="interleave-endpoints"' in dashboard.DASHBOARD_HTML
    assert "Pretraining endpoint evaluations" in dashboard.DASHBOARD_HTML
    assert "B3–B4" in dashboard.DASHBOARD_HTML
    assert "result_hashes" in dashboard.DASHBOARD_HTML
    assert 'id="interleave-artifacts"' in dashboard.DASHBOARD_HTML
    assert "Artifact publication" in dashboard.DASHBOARD_HTML
    assert "artifact_publication" in dashboard.DASHBOARD_HTML
