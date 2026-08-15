from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from Eval import modal_rl_dashboard as dashboard
from Eval.interleave_live_schema import build_live_feed


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict:
    return json.loads((ROOT / "INTERLEAVED_CORE_REGISTRY.json").read_text())


def test_v2r3_contract_normalizes_four_trajectories_and_twelve_audits() -> None:
    registry = _registry()
    source = registry["v2r3_diagnostic_contract"]

    value = dashboard._validate_v2r3_diagnostic_contract(registry)

    assert value["present"] is True
    assert value["authorization"] == {
        "scope": "diagnostic_only",
        "production": False,
        "p1": False,
        "exp2": False,
        "rl": False,
        "statement": (
            "Diagnostic-only and non-authorizing: no production, P1, "
            "Exp2, or RL launch authorization."
        ),
    }
    assert len(value["trajectories"]) == 4
    assert [row["sft_loss_weight"] for row in value["trajectories"]] == [
        190.189290837,
        256.0,
        384.0,
        768.0,
    ]
    assert len(value["rollouts"]) == 12
    assert value["aggregate"]["rollout_launched"] == len(
        source["rollout_calls"]
    )
    assert value["aggregate"]["expected_snapshot_count"] == 12
    assert value["metric_semantics"] == {
        "scope": "diagnostic_only",
        "official_evaluation": False,
        "display_label": "diagnostic positives / rate",
    }
    encoded = json.dumps(value, sort_keys=True)
    assert '"pass_at_1"' not in encoded
    assert '"avg_reward"' not in encoded


def test_v2r3_inspected_metrics_have_explicit_diagnostic_semantics() -> None:
    value = dashboard._validate_v2r3_diagnostic_contract(_registry())
    inspected = [
        row for row in value["rollouts"] if row["inspector"] is not None
    ]

    assert inspected
    for row in inspected:
        metrics = row["inspector"]
        assert metrics["state"] == "inspected_pass"
        assert 0 <= metrics["diagnostic_positive_rate"] <= 1
        assert metrics["rollout_rows"] == 2_048
        assert (
            metrics["status_counts"]["completed"]
            + metrics["status_counts"]["truncated"]
            == 2_048
        )
        assert "not an official evaluation" in metrics["metric_semantics"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda contract: contract.__setitem__(
                "production_authorized", True
            ),
            "production_authorized must be false",
        ),
        (
            lambda contract: contract["rollout_launch_progress"].__setitem__(
                "launched",
                contract["rollout_launch_progress"]["launched"] + 1,
            ),
            "rollout_launch_progress.launched is inconsistent",
        ),
        (
            lambda contract: contract["rollout_calls"][0].__setitem__(
                "snapshot_marker_sha256", "not-a-hash"
            ),
            "must be a lowercase SHA-256 digest",
        ),
    ),
)
def test_v2r3_contract_fails_closed_on_malformed_fields(
    mutation,
    message: str,
) -> None:
    registry = copy.deepcopy(_registry())
    mutation(registry["v2r3_diagnostic_contract"])

    with pytest.raises(ValueError, match=message):
        dashboard._validate_registry(registry)


def test_v2r3_contract_rejects_inconsistent_inspected_rate() -> None:
    registry = copy.deepcopy(_registry())
    inspected = next(
        row
        for row in registry["v2r3_diagnostic_contract"]["rollout_calls"]
        if row["status"] == "inspected_success"
    )
    inspected["inspector"]["p_total"] = 0.5

    with pytest.raises(ValueError, match="p_total is inconsistent"):
        dashboard._validate_v2r3_diagnostic_contract(registry)


def test_v2r3_final_report_requires_immutable_authenticated_metadata() -> None:
    registry = copy.deepcopy(_registry())
    contract = registry["v2r3_diagnostic_contract"]
    contract["final_report"] = {
        "status": "immutable_authenticated",
        "immutable": True,
        "path": contract["report_path"],
        "schema": dashboard.V2R3_DIAGNOSTIC_REPORT_SCHEMA,
        "record_count": 12,
        "audit_call_id": "fc-test-final-audit",
        "reported_at": "2026-07-30T15:00:00+00:00",
        "report_sha256": "f" * 64,
    }

    value = dashboard._validate_v2r3_diagnostic_contract(registry)

    assert value["final_report"] == {
        "state": "immutable_authenticated",
        "path": contract["report_path"],
        "immutable": True,
        "schema": dashboard.V2R3_DIAGNOSTIC_REPORT_SCHEMA,
        "report_sha256": "f" * 64,
        "record_count": 12,
        "audit_call_id": "fc-test-final-audit",
        "reported_at": "2026-07-30T15:00:00+00:00",
    }

    contract["final_report"]["report_sha256"] = "tampered"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        dashboard._validate_v2r3_diagnostic_contract(registry)


def test_live_control_plane_overlay_exposes_normalized_v2r3_api() -> None:
    registry = _registry()
    feed = build_live_feed(
        registry,
        {"stages": {}, "aggregate": {}, "errors": []},
        generated_at="2026-07-30T15:00:00+00:00",
    )
    snapshot = {
        "interleave": {
            "live_sync": {"generated_at": "2026-07-30T14:00:00+00:00"},
            "registry": {},
            "pretraining": {},
            "aggregate": {},
            "errors": [],
        }
    }

    value = dashboard._overlay_live_control_plane(snapshot, feed)
    diagnostics = value["interleave"]["v2r3_diagnostics"]

    assert diagnostics["present"] is True
    assert len(diagnostics["trajectories"]) == 4
    assert len(diagnostics["rollouts"]) == 12
    assert diagnostics["authorization"]["production"] is False


def test_dashboard_html_has_separate_non_authorizing_v2r3_tables() -> None:
    html = dashboard.DASHBOARD_HTML

    assert 'id="interleave-v2r3-note"' in html
    assert 'id="interleave-v2r3-trajectories"' in html
    assert 'id="interleave-v2r3-rollouts"' in html
    assert 'id="interleave-v2r3-report"' in html
    assert "DIAGNOSTIC ONLY · NON-AUTHORIZING" in html
    assert "never official Pass@1 or benchmark results" in html
    assert "diagnostic_positive_rate" in html

