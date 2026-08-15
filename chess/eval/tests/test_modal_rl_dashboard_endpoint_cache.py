from __future__ import annotations

from datetime import datetime, timedelta, timezone

from Eval import modal_rl_dashboard as dashboard


def _endpoint_summary(
    generated_at: datetime,
    *,
    errors: list[str] | None = None,
) -> dict:
    endpoints = {
        endpoint_id: {
            "endpoint_id": endpoint_id,
            **spec,
            "state": "complete",
            "checkpoint_sha256": endpoint_id,
            "loss": {},
            "chess": {},
            "result_hashes": {"losses": "a" * 64, "chess": "b" * 64},
            "superseded_checkpoint_count": 0,
        }
        for endpoint_id, spec in dashboard.ENDPOINT_SUMMARY_SPECS.items()
    }
    return {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": generated_at.isoformat(),
        "endpoints": endpoints,
        "aggregate": {
            "expected": len(endpoints),
            "complete": len(endpoints),
            "partial": 0,
            "missing": 0,
        },
        "errors": list(errors or []),
        "warnings": [],
    }


def test_complete_verified_endpoint_summary_is_cached_for_fifteen_minutes() -> None:
    generated_at = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
    now = generated_at + timedelta(minutes=14)

    cached = dashboard._cached_endpoint_evaluations(
        _endpoint_summary(generated_at),
        now=now,
    )

    assert cached is not None
    assert cached["aggregate"] == {
        "expected": 3,
        "complete": 3,
        "partial": 0,
        "missing": 0,
    }
    assert cached["cache_status"] == "verified_cache"
    assert cached["served_from_cache_at"] == now.isoformat()
    assert cached["next_list_after"] == (
        generated_at
        + timedelta(seconds=dashboard.ENDPOINT_COMPLETE_CACHE_SECONDS)
    ).isoformat()


def test_expired_or_marker_invalid_endpoint_summary_is_not_reused() -> None:
    generated_at = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
    expired = _endpoint_summary(generated_at)
    invalid = _endpoint_summary(
        generated_at,
        errors=["endpoint_v1/p1: ValueError: result hash mismatch"],
    )

    assert dashboard._cached_endpoint_evaluations(
        expired,
        now=generated_at + timedelta(minutes=15),
    ) is None
    assert dashboard._cached_endpoint_evaluations(
        invalid,
        now=generated_at + timedelta(seconds=1),
    ) is None


def test_listing_rate_limit_retains_verified_table_as_warning_with_backoff() -> None:
    generated_at = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
    failed_at = generated_at + timedelta(minutes=16)
    previous = _endpoint_summary(generated_at)
    listing_error = (
        "endpoint result listing: ResourceExhaustedError: "
        "VolumeListFiles rate limit exceeded"
    )
    current = {
        "schema_version": 1,
        "namespace": dashboard.ENDPOINT_NAMESPACE,
        "generated_at": failed_at.isoformat(),
        "endpoints": {
            endpoint_id: {
                "endpoint_id": endpoint_id,
                **spec,
                "state": "missing",
            }
            for endpoint_id, spec in dashboard.ENDPOINT_SUMMARY_SPECS.items()
        },
        "aggregate": {
            "expected": 3,
            "complete": 0,
            "partial": 0,
            "missing": 3,
        },
        "errors": [listing_error],
        "warnings": [],
    }

    retained = dashboard._retain_endpoint_summary_on_listing_failure(
        current,
        previous,
        now=failed_at,
    )

    assert retained["aggregate"]["complete"] == 3
    assert retained["errors"] == []
    assert retained["warnings"] == [listing_error]
    assert retained["stale"] is True
    assert retained["cache_status"] == "listing_backoff"
    assert retained["last_success_generated_at"] == generated_at.isoformat()
    assert retained["next_list_after"] == (
        failed_at
        + timedelta(seconds=dashboard.ENDPOINT_LISTING_BACKOFF_SECONDS)
    ).isoformat()

    cached = dashboard._cached_endpoint_evaluations(
        retained,
        now=failed_at + timedelta(minutes=4),
    )
    assert cached is not None
    assert cached["cache_status"] == "listing_backoff"
    assert cached["errors"] == []
    assert cached["warnings"] == [listing_error]


def test_malformed_endpoint_aggregate_cannot_enter_cache() -> None:
    generated_at = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
    malformed = _endpoint_summary(generated_at)
    malformed["aggregate"]["complete"] = 2

    assert dashboard._cached_endpoint_evaluations(
        malformed,
        now=generated_at + timedelta(seconds=1),
    ) is None
