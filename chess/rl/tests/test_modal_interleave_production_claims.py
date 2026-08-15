from __future__ import annotations

import fcntl
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest import mock

from chess_rl_miles.scripts import modal_interleave as launcher


class AtomicStore:
    """Thread-safe local model of Modal Dict.put(skip_if_exists=True)."""

    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def get(self, key, default=None):
        with self.lock:
            return self.values.get(key, default)

    def put(self, key, value, *, skip_if_exists=False):
        with self.lock:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True


def _token(number: int) -> str:
    return f"{number:064x}"


def _identity() -> dict[str, object]:
    return {
        "schema": "test-production-identity-v1",
        "run_root": "/rl-checkpoints/run",
        "checkpoint": "a" * 64,
        "data": "b" * 64,
        "lr": "1e-5",
        "kl": "low_var_kl",
        "seed": 42,
        "batch": 2048,
        "context": 2048,
        "token_budget": 131072,
        "wandb": {"project": "test", "group": "run"},
        "source": "c" * 64,
        "app": launcher.APP_NAME,
        "image": "im-test",
    }


def _terminal_evidence(
    call_id: str,
    exception: Exception | None = None,
) -> dict[str, object]:
    failure = exception or RuntimeError(
        launcher.PRODUCTION_RL_TRAINING_TERMINAL_MARKER
    )
    get_result = mock.Mock(side_effect=failure)
    evidence = launcher._authoritative_production_call_evidence(
        get_result,
        function_call_id=call_id,
        allow_success=False,
    )
    get_result.assert_called_once_with(timeout=0)
    return evidence


def _claim_and_attempt(store: AtomicStore):
    acquired = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=_identity(),
    )
    claim = acquired["claim"]
    attempt = launcher._acquire_production_attempt(
        store,
        run_name="run",
        claim=claim,
        generation=0,
        dispatcher_function_call_id="fc-DISPATCH",
        recovery_evidence=None,
    )["attempt"]
    return claim, attempt


def test_atomic_production_claim_has_exactly_one_winner():
    store = AtomicStore()

    def acquire(number: int):
        return launcher._acquire_production_claim(
            store,
            run_name="run",
            launch_token=_token(number),
            launch_identity=_identity(),
            claimed_at=f"2026-08-13T00:00:{number:02d}+00:00",
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(acquire, range(1, 17)))
    assert sum(row["outcome"] == "acquired" for row in results) == 1
    assert len({row["claim"]["claim_sha256"] for row in results}) == 1


def test_claim_is_never_stolen_by_age_and_wrong_token_cannot_execute():
    store = AtomicStore()
    first = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=_identity(),
        claimed_at="2000-01-01T00:00:00+00:00",
    )
    second = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(2),
        launch_identity=_identity(),
        claimed_at="2099-01-01T00:00:00+00:00",
    )
    assert first["outcome"] == "acquired"
    assert second["outcome"] == "existing_claim"
    assert second["claim"] == first["claim"]
    launcher._acquire_production_attempt(
        store,
        run_name="run",
        claim=first["claim"],
        generation=0,
        dispatcher_function_call_id="fc-DISPATCH",
        recovery_evidence=None,
    )
    with pytest.raises(RuntimeError, match="recovery token does not match"):
        launcher._begin_claimed_production_worker(
            store,
            run_name="run",
            launch_token=_token(2),
            expected_identity=_identity(),
            generation=0,
            function_call_id="fc-WORKER",
        )


def test_same_function_call_retry_is_allowed_but_different_call_is_rejected():
    store = AtomicStore()
    _claim_and_attempt(store)
    first = launcher._begin_claimed_production_worker(
        store,
        run_name="run",
        launch_token=_token(1),
        expected_identity=_identity(),
        generation=0,
        function_call_id="fc-WORKER",
    )
    retry = launcher._begin_claimed_production_worker(
        store,
        run_name="run",
        launch_token=_token(1),
        expected_identity=_identity(),
        generation=0,
        function_call_id="fc-WORKER",
    )
    assert first["new_binding"] is True
    assert retry["new_binding"] is False
    with pytest.raises(RuntimeError, match="different Modal FunctionCall"):
        launcher._begin_claimed_production_worker(
            store,
            run_name="run",
            launch_token=_token(1),
            expected_identity=_identity(),
            generation=0,
            function_call_id="fc-OTHER",
        )


def test_concurrent_worker_calls_allow_exactly_one_training_owner():
    store = AtomicStore()
    _claim_and_attempt(store)

    def begin(call_id: str):
        try:
            launcher._begin_claimed_production_worker(
                store,
                run_name="run",
                launch_token=_token(1),
                expected_identity=_identity(),
                generation=0,
                function_call_id=call_id,
            )
        except RuntimeError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        owners = list(pool.map(begin, [f"fc-WORKER{i}" for i in range(16)]))
    assert sum(owners) == 1


def test_recovery_close_wins_paused_worker_interleaving():
    """Worker reads the attempt, pauses; recovery closes; worker cannot train."""

    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "resumable"},
        },
        hash_field="recovery_sha256",
    )
    decision = launcher._close_or_observe_generation_for_recovery(
        store,
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        recovery_evidence=recovery,
    )
    assert decision["outcome"] == "closed"
    with pytest.raises(RuntimeError, match="closed for recovery"):
        launcher._begin_claimed_production_worker(
            store,
            run_name="run",
            launch_token=_token(1),
            expected_identity=_identity(),
            generation=0,
            function_call_id="fc-DELAYEDWORKER",
        )


def test_completed_checkpoint_close_rejects_delayed_worker():
    """Completion must close an unbound generation before returning."""

    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "complete", "checkpoint_step": 1500},
        },
        hash_field="recovery_sha256",
    )
    decision = launcher._close_or_observe_generation_for_recovery(
        store,
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        recovery_evidence=recovery,
    )
    assert decision["outcome"] == "closed"
    assert decision["resolution"]["recovery_evidence"]["checkpoint"][
        "state"
    ] == "complete"
    with pytest.raises(RuntimeError, match="closed for recovery"):
        launcher._begin_claimed_production_worker(
            store,
            run_name="run",
            launch_token=_token(1),
            expected_identity=_identity(),
            generation=0,
            function_call_id="fc-DELAYEDWORKER",
        )


def test_next_generation_evidence_binds_resolution_hash():
    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "resumable"},
        },
        hash_field="recovery_sha256",
    )
    decision = launcher._close_or_observe_generation_for_recovery(
        store,
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        recovery_evidence=recovery,
    )
    bound = launcher._bind_recovery_evidence_to_generation_resolution(
        decision["resolution"]["recovery_evidence"],
        resolution=decision["resolution"],
    )
    assert bound["generation_resolution_sha256"] == decision["resolution"][
        "resolution_sha256"
    ]
    next_attempt = launcher._new_production_attempt(
        run_name="run",
        claim=claim,
        generation=1,
        dispatcher_function_call_id="fc-NEXTDISPATCH",
        recovery_evidence=bound,
    )
    assert next_attempt["recovery_evidence"] == bound


def test_recovery_resolution_rejects_wrong_dispatcher_function_call():
    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "resumable"},
        },
        hash_field="recovery_sha256",
    )
    valid = launcher._new_production_generation_resolution(
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        decision="recovery_closed",
        function_call_id="fc-DISPATCH",
        recovery_evidence=recovery,
    )
    tampered_core = {
        key: value
        for key, value in valid.items()
        if key != "resolution_sha256"
    }
    tampered_core["function_call_id"] = "fc-OTHERDISPATCH"
    tampered = launcher._production_self_hash(
        tampered_core,
        hash_field="resolution_sha256",
    )
    with pytest.raises(RuntimeError, match="recovery closure evidence drifted"):
        launcher._validate_production_generation_resolution(
            tampered,
            run_name="run",
            claim=claim,
            attempt=attempt,
            generation=0,
        )


def test_worker_wins_inverse_toctou_interleaving_recovery_cannot_close():
    """Worker binds during recovery checks; recovery must observe the worker."""

    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    worker = launcher._resolve_generation_for_worker(
        store,
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        function_call_id="fc-WORKER",
    )
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "resumable"},
        },
        hash_field="recovery_sha256",
    )
    decision = launcher._close_or_observe_generation_for_recovery(
        store,
        run_name="run",
        claim=claim,
        attempt=attempt,
        generation=0,
        recovery_evidence=recovery,
    )
    assert decision["outcome"] == "worker_bound"
    assert decision["resolution"]["function_call_id"] == "fc-WORKER"
    assert decision["resolution"]["resolution_sha256"] == worker[
        "resolution_sha256"
    ]


def test_worker_and_recovery_resolution_race_has_one_immutable_winner():
    store = AtomicStore()
    claim, attempt = _claim_and_attempt(store)
    barrier = threading.Barrier(2)
    recovery = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-production-recovery-evidence-v1",
            "prior_attempt_sha256": attempt["attempt_sha256"],
            "terminal_call": _terminal_evidence("fc-DISPATCH"),
            "checkpoint": {"state": "resumable"},
        },
        hash_field="recovery_sha256",
    )

    def worker():
        barrier.wait()
        try:
            result = launcher._resolve_generation_for_worker(
                store,
                run_name="run",
                claim=claim,
                attempt=attempt,
                generation=0,
                function_call_id="fc-WORKER",
            )
        except RuntimeError:
            return "lost"
        return result["decision"]

    def closer():
        barrier.wait()
        result = launcher._close_or_observe_generation_for_recovery(
            store,
            run_name="run",
            claim=claim,
            attempt=attempt,
            generation=0,
            recovery_evidence=recovery,
        )
        return result["resolution"]["decision"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        worker_future = pool.submit(worker)
        closer_future = pool.submit(closer)
        results = [worker_future.result(), closer_future.result()]
    observed = store.get(launcher._production_resolution_key("run", 0))
    assert observed["decision"] in {"worker_bound", "recovery_closed"}
    assert all(result in {"lost", observed["decision"]} for result in results)


def test_claim_rejects_any_launch_semantic_drift():
    store = AtomicStore()
    launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=_identity(),
    )
    for field, changed in (
        ("lr", "1e-4"),
        ("kl", "k1"),
        ("seed", 43),
        ("batch", 1024),
        ("context", 3072),
        ("token_budget", 65536),
        ("checkpoint", "f" * 64),
        ("data", "e" * 64),
        ("source", "d" * 64),
        ("image", "im-other"),
    ):
        drifted = {**_identity(), field: changed}
        with pytest.raises(RuntimeError, match="different production RL semantics"):
            launcher._acquire_production_claim(
                store,
                run_name="run",
                launch_token=_token(2),
                launch_identity=drifted,
            )


def test_local_recovery_token_is_durable_before_claim_and_tamper_evident(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        launcher,
        "LOCAL_PRODUCTION_LAUNCH_RECOVERY_ROOT",
        tmp_path,
    )
    path = launcher._write_local_production_recovery_record(
        run_name="run",
        launch_token=_token(7),
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert launcher._read_local_production_recovery_record("run")[
        "launch_token"
    ] == _token(7)
    with pytest.raises(RuntimeError, match="already exists"):
        launcher._write_local_production_recovery_record(
            run_name="run",
            launch_token=_token(8),
        )
    path.write_text(path.read_text().replace(_token(7), _token(8)))
    with pytest.raises(RuntimeError, match="self hash drifted"):
        launcher._read_local_production_recovery_record("run")


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (None, "completed successfully"),
        (TimeoutError(), "still pending"),
        (launcher.modal.exception.OutputExpiredError(), "output expired"),
        (launcher.modal.exception.ServiceError("transient"), "ambiguous"),
        (launcher.modal.exception.InternalFailure("internal"), "ambiguous"),
        (launcher.modal.exception.ExecutionError("decode"), "ambiguous"),
        (ValueError("unknown"), "unknown exception type"),
    ],
)
def test_terminal_recovery_rejects_success_pending_expired_and_ambiguous(
    outcome, message
):
    get_result = mock.Mock(
        return_value="ok" if outcome is None else mock.DEFAULT,
        side_effect=outcome,
    )
    with pytest.raises(RuntimeError, match=message):
        launcher._authoritative_production_call_evidence(
            get_result,
            function_call_id="fc-PRIOR",
            allow_success=False,
        )
    get_result.assert_called_once_with(timeout=0)


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (
            RuntimeError(launcher.PRODUCTION_RL_TRAINING_TERMINAL_MARKER),
            "application_failure",
        ),
        (
            RuntimeError(launcher.PRODUCTION_RL_DISPATCHER_TERMINAL_MARKER),
            "application_failure",
        ),
        (
            launcher.modal.exception.FunctionTimeoutError("timeout"),
            "function_timeout",
        ),
        (
            launcher.modal.exception.RemoteError("terminated"),
            "remote_terminal_failure",
        ),
    ],
)
def test_terminal_recovery_accepts_authoritative_unsuccessful_result(
    exception, category
):
    evidence = _terminal_evidence("fc-PRIOR", exception)
    assert evidence["result_category"] == category
    assert evidence["function_call_id"] == "fc-PRIOR"
    assert "message" not in evidence
    launcher._validate_production_terminal_call_evidence(
        evidence,
        expected_function_call_id="fc-PRIOR",
    )


def test_terminal_inspectors_use_get_not_call_graph(monkeypatch):
    failed = SimpleNamespace(
        get=mock.Mock(
            side_effect=RuntimeError(
                launcher.PRODUCTION_RL_TRAINING_TERMINAL_MARKER
            )
        ),
        get_call_graph=mock.Mock(side_effect=AssertionError("must not be used")),
    )
    monkeypatch.setattr(
        launcher.modal.FunctionCall,
        "from_id",
        mock.Mock(return_value=failed),
    )
    launcher._inspect_terminal_unsuccessful_production_call("fc-PRIOR")
    failed.get.assert_called_once_with(timeout=0)
    failed.get_call_graph.assert_not_called()


def test_checkpoint_is_authenticated_only_after_terminal_worker_poll(monkeypatch):
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    recovery = source[
        source.index("        if recovery and not same_dispatcher:") :
        source.index("    if current is None or generation != int(current[\"generation\"]):")
    ]
    worker_branch = recovery[
        recovery.index('elif resolution["decision"] == "worker_bound":') :
        recovery.index("            else:", recovery.index('elif resolution["decision"]'))
    ]
    assert worker_branch.index("_inspect_terminal_completed_production_call(") < (
        worker_branch.index("base.ckpt_vol.reload()")
    ) < worker_branch.index("_authenticated_production_recovery_checkpoint(")


def test_clean_success_with_partial_checkpoint_is_not_recoverable():
    success = launcher._authoritative_production_call_evidence(
        mock.Mock(return_value={"ok": True}),
        function_call_id="fc-WORKER",
        allow_success=True,
    )
    assert success["result_category"] == "success"
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "worker returned success before the exact" in source


def test_dispatcher_retries_are_disabled_and_replay_is_fail_closed():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    prefix = source[: source.index("def dispatch_production_train(")]
    decorator = prefix[prefix.rfind("@app.function(") :]
    assert "retries=0" in decorator
    assert "dispatcher replay found its unbound generation" in source


def test_authenticated_recovery_checkpoint_binds_semantics_and_provenance(
    tmp_path, monkeypatch
):
    identity = {
        "semantics": {
            "target_updates": 1500,
            "model_id": launcher.CONTEXT2048_MODEL_ID,
            "rollout_seed": 42,
            "dynamic_filter": False,
            "save_interval": 40,
            "eval_interval": 0,
            "deterministic_inference": True,
            "resume_if_available": True,
            "lr": "1e-5",
            "kl_loss_type": "low_var_kl",
            "max_tokens_per_gpu": 131072,
            "sglang_server_concurrency": 128,
            "rollout_max_prompt_len": 512,
            "rollout_max_response_len": 1536,
            "rollout_max_context_len": 2048,
        },
        "training_data": {"logical_path": "/data/train.parquet", "sha256": "a" * 64},
        "origin_hf": {"logical_path": "/checkpoints/model", "manifest_sha256": "b" * 64},
        "deployment": {
            "sources": {
                "chess_rl_miles": {"manifest_sha256": "c" * 64},
                "miles": {"manifest_sha256": "d" * 64},
            },
            "runtime": {"modal_app_id": "ap-test", "modal_image_id": "im-test"},
        },
        "initial_command_sha256": "e" * 64,
        "wandb": {
            "entity": launcher.WANDB_ENTITY,
            "project": "project",
            "group": "group",
            "run_id": "prod-test-run",
        },
    }
    run_root = tmp_path / "run"
    checkpoint = run_root / "iter_0000040"
    checkpoint.mkdir(parents=True)
    (checkpoint / launcher.CHECKPOINT_COMMIT_MARKER).write_text("committed")
    recorded_identity = {
        "run": {
            "run_name": "run",
            "model_id": launcher.CONTEXT2048_MODEL_ID,
            "num_rollout": 1500,
            "rollout_seed": 42,
            "dynamic_filter": False,
            "save_interval": 40,
            "eval_interval": 0,
            "deterministic_inference": True,
            "resume_if_available": True,
            "wandb_entity": launcher.WANDB_ENTITY,
            "wandb_project": "project",
            "wandb_group": "group",
            "wandb_run_id": "prod-test-run",
        },
        "fixed_rl_semantics": {
            "lr": "1e-5",
            "kl_loss_type": "low_var_kl",
            "kl_loss_coef": 0.001,
            "rollout_batch_size": 256,
            "samples_per_prompt": 8,
            "global_batch_size": 2048,
            "rollout_max_prompt_len": 512,
            "rollout_max_response_len": 1536,
            "rollout_max_context_len": 2048,
        },
        "training_data": identity["training_data"],
        "origin_hf": identity["origin_hf"],
        "sources": identity["deployment"]["sources"],
        "runtime": identity["deployment"]["runtime"],
        "policy_update_profile": {
            "max_tokens_per_gpu": 131072,
            "sglang_server_concurrency": 128,
            "master_parameter_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "forward_backward_dtype": "bfloat16",
            "gradient_reduction_dtype": "float32",
        },
    }
    provenance = {
        "identity": recorded_identity,
        "identity_sha256": launcher._canonical_json_sha256(recorded_identity),
        "initial_command_sha256": identity["initial_command_sha256"],
    }
    (run_root / "run_provenance.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", str(tmp_path))
    monkeypatch.setattr(
        launcher,
        "_reconcile_modal_checkpoint_root",
        lambda root: 40,
    )
    evidence = launcher._authenticated_production_recovery_checkpoint(
        run_name="run",
        launch_identity=identity,
    )
    assert evidence["state"] == "resumable"
    assert evidence["checkpoint_step"] == 40
    identity["training_data"] = {**identity["training_data"], "sha256": "f" * 64}
    with pytest.raises(RuntimeError, match="model/data identity drifted"):
        launcher._authenticated_production_recovery_checkpoint(
            run_name="run",
            launch_identity=identity,
        )


def test_durable_anchor_commit_reload_and_dict_expiry_tombstone(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", str(tmp_path))
    store = AtomicStore()
    identity = _identity()
    identity["run_root"] = str(tmp_path / "run")
    claim = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=identity,
    )["claim"]

    class Volume:
        def __init__(self):
            self.events = []

        def reload(self):
            self.events.append("reload")

        def commit(self):
            self.events.append("commit")

    volume = Volume()
    anchor = launcher._publish_production_durable_anchor(
        run_name="run",
        claim=claim,
        launch_identity=identity,
        launch_token=_token(1),
        volume=volume,
    )
    path = launcher._production_durable_anchor_path("run")
    assert volume.events == ["reload", "commit", "reload"]
    assert _token(1) not in path.read_text(encoding="utf-8")
    assert anchor["launch_token_sha256"] == (
        launcher._production_launch_token_sha256(_token(1))
    )
    assert path.exists()


def test_durable_anchor_publisher_takeover_requires_terminal_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", str(tmp_path))
    store = AtomicStore()
    identity = _identity()
    identity["run_root"] = str(tmp_path / "run")
    claim = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=identity,
    )["claim"]
    intent = launcher._production_self_hash(
        {
            "schema": "chess-rl-miles-durable-anchor-intent-v1",
            "run_name": "run",
            "claim_sha256": claim["claim_sha256"],
            "dispatcher_function_call_id": "fc-PRIOR",
        },
        hash_field="intent_sha256",
    )
    store.put(launcher._production_anchor_intent_key("run"), intent)
    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_production_call",
        lambda call_id: _terminal_evidence(call_id),
    )
    anchor = launcher._ensure_production_durable_anchor(
        store,
        run_name="run",
        claim=claim,
        launch_identity=identity,
        launch_token=_token(1),
        dispatcher_function_call_id="fc-RECOVERY",
        recovery=True,
        volume=SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )
    assert anchor["publisher_recovery"][
        "prior_dispatcher_function_call_id"
    ] == "fc-PRIOR"


def test_training_loop_refreshes_production_lease(monkeypatch, tmp_path):
    class Process:
        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise launcher.subprocess.TimeoutExpired("train", timeout)
            return 0

    clock = iter([0.0, 10.0])
    heartbeat = mock.Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        launcher,
        "_commit_new_checkpoint_if_ready",
        lambda **kwargs: kwargs["published_through_step"],
    )
    result = launcher._run_training_with_checkpoint_commits(
        ["train"],
        env={},
        cwd=tmp_path,
        run_root=tmp_path / "run",
        volume=SimpleNamespace(),
        poll_seconds=0.01,
        lease_heartbeat=heartbeat,
        lease_refresh_seconds=5.0,
    )
    assert result == (0, 0)
    heartbeat.assert_called_once_with()


def test_incremental_checkpoint_commit_waits_for_miles_staging_lock(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("40\n")
    monkeypatch.setattr(
        launcher,
        "_latest_complete_checkpoint_step",
        lambda run_root: 40,
    )
    volume = SimpleNamespace(commit=mock.Mock())
    writer_holds_lock = threading.Event()
    release_writer = threading.Event()

    def stage_checkpoint():
        lock = root.parent / (
            f".{root.name}{launcher.CHECKPOINT_VOLUME_COMMIT_LOCK_SUFFIX}"
        )
        descriptor = launcher.os.open(
            lock,
            launcher.os.O_RDWR | launcher.os.O_CREAT,
            0o644,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            staging = root / ".iter_0000041.incomplete"
            staging.mkdir()
            writer_holds_lock.set()
            assert release_writer.wait(timeout=5.0)
            staging.rmdir()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            launcher.os.close(descriptor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(stage_checkpoint)
        assert writer_holds_lock.wait(timeout=5.0)
        poller = pool.submit(
            launcher._commit_new_checkpoint_if_ready,
            run_root=root,
            volume=volume,
            published_through_step=0,
        )
        time.sleep(0.05)
        assert not poller.done()
        volume.commit.assert_not_called()
        release_writer.set()
        writer.result(timeout=5.0)
        assert poller.result(timeout=5.0) == 40
    volume.commit.assert_called_once_with()


def test_incremental_checkpoint_commit_rejects_abandoned_staging_tree(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("40\n")
    (root / ".iter_0000041.incomplete").mkdir()
    monkeypatch.setattr(
        launcher,
        "_latest_complete_checkpoint_step",
        lambda run_root: 40,
    )
    volume = SimpleNamespace(commit=mock.Mock())
    assert launcher._commit_new_checkpoint_if_ready(
        run_root=root,
        volume=volume,
        published_through_step=0,
    ) == 0
    volume.commit.assert_not_called()


def test_heartbeat_failure_kills_process_group_and_cleans_runtime(
    monkeypatch, tmp_path
):
    class Process:
        pid = 4242

        def __init__(self):
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise launcher.subprocess.TimeoutExpired("train", timeout)
            assert timeout == 30.0
            return -launcher.signal.SIGTERM

    process = Process()
    popen = mock.Mock(return_value=process)
    killpg = mock.Mock()
    cleanup = mock.Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "killpg", killpg)
    monkeypatch.setattr(
        launcher,
        "_commit_new_checkpoint_if_ready",
        lambda **kwargs: kwargs["published_through_step"],
    )
    heartbeat = mock.Mock(side_effect=RuntimeError("lease lost"))
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="lease lost"):
        launcher._run_training_with_checkpoint_commits(
            ["train"],
            env={},
            cwd=tmp_path,
            run_root=tmp_path / "run",
            volume=SimpleNamespace(),
            poll_seconds=0.01,
            lease_heartbeat=heartbeat,
            lease_refresh_seconds=5.0,
            runtime_cleanup=cleanup,
        )
    assert popen.call_args.kwargs["start_new_session"] is True
    killpg.assert_called_once_with(4242, launcher.signal.SIGTERM)
    cleanup.assert_called_once_with()
    assert process.wait_calls == 2


def test_production_identity_uses_one_deterministic_wandb_run_id(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"fixed training data")
    monkeypatch.setattr(
        launcher,
        "directory_identity",
        lambda *args, **kwargs: {
            "logical_path": str(args[0]),
            "manifest_sha256": "a" * 64,
        },
    )
    captured = []
    real_build = launcher.build_train_command

    def capture_build(**kwargs):
        captured.append(dict(kwargs))
        return real_build(**kwargs)

    monkeypatch.setattr(launcher, "build_train_command", capture_build)
    deployment = {
        "source_sha256": "b" * 64,
        "sources": {
            "chess_rl_miles": {"manifest_sha256": "c" * 64},
            "miles": {"manifest_sha256": "d" * 64},
        },
        "runtime": {"modal_app_id": "ap-test", "modal_image_id": "im-test"},
    }
    kwargs = {
        "hf_checkpoint": str(checkpoint),
        "run_name": "run",
        "num_rollout": 1500,
        "dynamic_filter": False,
        "rollout_seed": 42,
        "save_interval": 40,
        "eval_interval": 0,
        "model_id": launcher.CONTEXT2048_MODEL_ID,
        "resume_if_available": True,
        "wandb_project": "project",
        "wandb_group": "group",
        "max_tokens_per_gpu": 131072,
        "sglang_server_concurrency": 128,
        "deterministic_inference": True,
        "train_file": str(train_file),
        "train_file_sha256": launcher._sha256(train_file),
        "lr": "1e-5",
        "kl_loss_type": "low_var_kl",
        "rollout_max_prompt_len": 512,
        "rollout_max_response_len": 1536,
        "rollout_max_context_len": 2048,
        "deployment_identity": deployment,
    }
    first = launcher._production_launch_identity(**kwargs)
    second = launcher._production_launch_identity(**kwargs)
    assert first == second
    run_id = first["wandb"]["run_id"]
    assert run_id.startswith("prod")
    assert len(run_id) == 32
    assert captured[1]["wandb_run_id"] == run_id
    assert captured[3]["wandb_run_id"] == run_id


def test_durable_completion_requires_exact_target_and_round_trips(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "RAW_RL_ROOT", str(tmp_path))
    store = AtomicStore()
    identity = _identity()
    identity.update(
        {
            "run_root": str(tmp_path / "run"),
            "semantics": {"target_updates": 1500},
        }
    )
    identity["identity_sha256"] = launcher._canonical_json_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    claim = launcher._acquire_production_claim(
        store,
        run_name="run",
        launch_token=_token(1),
        launch_identity=identity,
    )["claim"]
    attempt = launcher._acquire_production_attempt(
        store,
        run_name="run",
        claim=claim,
        generation=0,
        dispatcher_function_call_id="fc-DISPATCH",
        recovery_evidence=None,
    )["attempt"]
    binding = launcher._begin_claimed_production_worker(
        store,
        run_name="run",
        launch_token=_token(1),
        expected_identity=identity,
        generation=0,
        function_call_id="fc-WORKER",
    )

    class Volume:
        def reload(self):
            return None

        def commit(self):
            return None

    volume = Volume()
    launcher._publish_production_durable_anchor(
        run_name="run",
        claim=claim,
        launch_identity=identity,
        launch_token=_token(1),
        volume=volume,
    )
    checkpoint = {
        "state": "complete",
        "run_root": str(tmp_path / "run"),
        "checkpoint_step": 1500,
        "checkpoint_marker_sha256": "e" * 64,
        "run_provenance_sha256": "f" * 64,
    }
    completion = launcher._publish_production_durable_completion(
        run_name="run",
        launch_identity=identity,
        binding=binding,
        checkpoint=checkpoint,
        target_updates=1500,
        volume=volume,
    )
    assert completion["checkpoint"] == checkpoint
    assert launcher._validate_production_durable_completion(
        run_name="run",
        launch_identity=identity,
    ) == completion
    with pytest.raises(RuntimeError, match="not exact"):
        launcher._publish_production_durable_completion(
            run_name="run",
            launch_identity=identity,
            binding={**binding, "attempt": attempt},
            checkpoint={**checkpoint, "checkpoint_step": 1499},
            target_updates=1500,
            volume=volume,
        )


def test_production_source_routes_through_dispatcher_and_authenticates_worker():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    train_branch = source[source.index('if action == "train":') :]
    assert '_deployed_function("dispatch_production_train").spawn' in train_branch
    worker_source = source[
        source.index("def train_hf(") : source.index("def _sha256(")
    ]
    assert "_begin_claimed_production_worker(" in worker_source
    assert "modal.current_function_call_id()" in worker_source


def test_production_worker_preserves_logical_training_path_for_gate_contract():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    worker_source = source[
        source.index("def train_hf(") : source.index("def _sha256(")
    ]
    assert (
        "logical_train_file = _validated_logical_file_path(" in worker_source
    )
    gate_start = worker_source.index(
        "_, production_gate_contract = _precision_gate_contract_from_inputs("
    )
    gate_end = worker_source.index(
        "precision_gate_evidence = _require_precision_resume_gate(",
        gate_start,
    )
    gate_source = worker_source[gate_start:gate_end]
    assert "train_file=logical_train_file" in gate_source
    assert "train_file=train_file" not in gate_source


def test_dispatcher_sends_logical_training_path_to_production_worker():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    dispatcher_source = source[
        source.index("def dispatch_production_train(") : source.index(
            "def _precision_gate_paths(",
            source.index("def dispatch_production_train("),
        )
    ]
    identity_start = dispatcher_source.index(
        "launch_identity = _production_launch_identity("
    )
    identity_end = dispatcher_source.index(
        "store = production_launch_claims",
        identity_start,
    )
    assert (
        "train_file=resolved_train_file"
        in dispatcher_source[identity_start:identity_end]
    )
    worker_start = dispatcher_source.index("worker_kwargs = {")
    worker_end = dispatcher_source.index(
        'call = _deployed_function("train_hf").spawn(**worker_kwargs)',
        worker_start,
    )
    worker_source = dispatcher_source[worker_start:worker_end]
    assert '"train_file": logical_train_file' in worker_source
    assert '"train_file": resolved_train_file' not in worker_source
