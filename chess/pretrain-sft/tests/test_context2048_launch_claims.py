from __future__ import annotations

import copy
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest

from modal_scripts import launch_context2048_vocab_mixing as launcher

_REAL_PUBLISH_DURABLE_ANCHOR = launcher._publish_durable_launch_anchor
_REAL_VALIDATE_DURABLE_ANCHOR = launcher._validate_durable_launch_anchor
_REAL_ENSURE_DURABLE_ANCHOR = launcher._ensure_durable_launch_anchor


@pytest.fixture(autouse=True)
def _isolate_durable_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    monkeypatch.setattr(
        launcher,
        "_validate_durable_launch_anchor",
        lambda *args, **kwargs: {"anchor_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        launcher,
        "_ensure_durable_launch_anchor",
        lambda *args, **kwargs: {"anchor_sha256": "f" * 64},
    )


class AtomicStore:
    """Thread-safe local model of Modal Dict.put(skip_if_exists=True)."""

    def __init__(self):
        import threading

        self.values: dict[str, object] = {}
        self.lock = threading.Lock()

    def get(self, key, default=None):
        with self.lock:
            return copy.deepcopy(self.values.get(key, default))

    def put(self, key, value, *, skip_if_exists=False):
        with self.lock:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = copy.deepcopy(value)
            return True

    def __getitem__(self, key):
        value = self.get(key, None)
        if value is None:
            raise KeyError(key)
        return value

    def tamper(self, key, callback):
        with self.lock:
            callback(self.values[key])


def identity(experiment: str = "vocab85_then_sft3") -> dict[str, object]:
    return {
        "schema": "test-production-launch-identity-v1",
        "experiment": experiment,
        "source_tree_sha256": launcher.SOURCE_TREE_SHA256,
        "manifest_set_hash": "a" * 64,
        "gate_sha256": {"experiment": "b" * 64},
        "output": f"/checkpoints/{experiment}",
        "wandb": {
            "entity": launcher.WANDB_ENTITY,
            "project": launcher.WANDB_PROJECT,
        },
        "runtime": {"app": "ap-test", "image": "im-test"},
        "output_roots": {"pt": f"/checkpoints/{experiment}/pt"},
        "spec": {"steps": 2},
    }


def token(number: int) -> str:
    return f"{number:064x}"


def acquire(
    store: AtomicStore,
    *,
    experiment: str,
    number: int,
    claimed_at: str = "2026-08-13T00:00:00+00:00",
):
    return launcher._acquire_launch_claim(
        store,
        experiment_key=experiment,
        launch_token=token(number),
        launch_identity=identity(experiment),
        claimed_at=claimed_at,
    )


def acquire_attempt(
    store: AtomicStore,
    *,
    experiment: str = "vocab85_then_sft3",
    number: int = 1,
    generation: int = 0,
    dispatcher: str = "fc-DISPATCHER",
):
    claim_result = acquire(store, experiment=experiment, number=number)
    claim = claim_result["claim"]
    attempt_result = launcher._acquire_launch_attempt(
        store,
        experiment_key=experiment,
        claim=claim,
        generation=generation,
        dispatcher_function_call_id=dispatcher,
        recovery_evidence=(
            None
            if generation == 0
            else {
                "schema": "test-recovery-v1",
                "previous_generation": generation - 1,
            }
        ),
        created_at="2026-08-13T00:00:00+00:00",
    )
    return claim, attempt_result["attempt"]


def _recovery_writer_process(root: str, number: int, queue) -> None:
    launcher.LOCAL_LAUNCH_RECOVERY_ROOT = launcher.Path(root)
    try:
        path = launcher._write_local_launch_recovery_record(
            experiment_key="vocab85_then_sft3",
            launch_token=token(number),
        )
        queue.put(("won", number, path.read_bytes()))
    except RuntimeError as exc:
        queue.put(("lost", number, str(exc)))


def test_atomic_claim_has_exactly_one_winner():
    store = AtomicStore()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda number: acquire(
                    store,
                    experiment="vocab85_then_sft3",
                    number=number,
                ),
                range(1, 33),
            )
        )
    assert sum(row["outcome"] == "acquired" for row in results) == 1
    assert len({row["claim"]["claim_sha256"] for row in results}) == 1


def test_different_experiment_keys_acquire_independently():
    store = AtomicStore()
    assert acquire(store, experiment="vocab81_then_sft3", number=1)[
        "outcome"
    ] == "acquired"
    assert acquire(store, experiment="mixed_sft1", number=2)[
        "outcome"
    ] == "acquired"


def _mock_dispatch_preflight(monkeypatch, launch_identity=None):
    launch_identity = launch_identity or identity()
    monkeypatch.setattr(
        launcher,
        "_production_launch_preflight",
        lambda store, experiment_key: (
            launcher.EXPERIMENTS[experiment_key],
            launch_identity,
            {"wandb_write_gate_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_authenticate_existing_completion",
        lambda experiment: None,
    )
    return launch_identity


def test_server_dispatcher_claims_attempt_spawns_and_binds_once(monkeypatch):
    store = AtomicStore()
    _mock_dispatch_preflight(monkeypatch)
    spawn = mock.Mock(return_value=SimpleNamespace(object_id="fc-WORKER123"))
    result = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=False,
        dispatcher_function_call_id="fc-DISPATCH123",
        spawn_worker=spawn,
    )
    assert result["outcome"] == "spawned"
    assert result["generation"] == 0
    spawn.assert_called_once_with("vocab85_then_sft3", token(1), 0)
    assert store.get(
        launcher._launch_execution_key("vocab85_then_sft3", 0)
    )["function_call_id"] == "fc-WORKER123"

    duplicate = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=False,
        dispatcher_function_call_id="fc-DISPATCH999",
        spawn_worker=spawn,
    )
    assert duplicate["outcome"] == "existing_claim"
    assert duplicate["spawned"] is False
    spawn.assert_called_once()


def test_same_dispatcher_replay_cannot_succeed_with_unbound_attempt(monkeypatch):
    store = AtomicStore()
    _mock_dispatch_preflight(monkeypatch)
    acquire_attempt(store, dispatcher="fc-DISPATCH123")
    spawn = mock.Mock()
    with pytest.raises(RuntimeError, match="unbound generation"):
        launcher._dispatch_production_launch(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            recovery=False,
            dispatcher_function_call_id="fc-DISPATCH123",
            spawn_worker=spawn,
        )
    spawn.assert_not_called()


def test_claim_anchor_without_attempt_replay_competes_for_generation_zero(
    monkeypatch,
):
    """A crash after the claim/anchor must not strand the initial worker."""

    store = AtomicStore()
    _mock_dispatch_preflight(monkeypatch)
    acquire(store, experiment="vocab85_then_sft3", number=1)
    spawn = mock.Mock(return_value=SimpleNamespace(object_id="fc-WORKER123"))
    result = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=False,
        dispatcher_function_call_id="fc-DISPATCHRETRY",
        spawn_worker=spawn,
    )
    assert result["outcome"] == "spawned"
    assert result["generation"] == 0
    spawn.assert_called_once_with("vocab85_then_sft3", token(1), 0)


def test_direct_worker_without_claim_is_rejected():
    with pytest.raises(RuntimeError, match="claim"):
        launcher._begin_claimed_worker(
            AtomicStore(),
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-TEST123",
        )


def test_same_call_retry_is_allowed_but_different_call_or_token_is_rejected():
    store = AtomicStore()
    acquire_attempt(store)
    first = launcher._begin_claimed_worker(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        expected_identity=identity(),
        generation=0,
        function_call_id="fc-TEST123",
    )
    retry = launcher._begin_claimed_worker(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        expected_identity=identity(),
        generation=0,
        function_call_id="fc-TEST123",
    )
    assert first["new_binding"] is True
    assert retry["new_binding"] is False
    with pytest.raises(RuntimeError, match="different Modal FunctionCall"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-OTHER456",
        )
    with pytest.raises(RuntimeError, match="token does not match"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(2),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-TEST123",
        )


def test_completion_requires_evidence_and_is_immutable_after_commit():
    store = AtomicStore()
    acquire_attempt(store)
    kwargs = {
        "store": store,
        "experiment_key": "vocab85_then_sft3",
        "launch_token": token(1),
        "expected_identity": identity(),
        "generation": 0,
        "function_call_id": "fc-TEST123",
        "stage": "sft",
        "updated_at": "2026-08-13T00:01:00+00:00",
    }
    with pytest.raises(RuntimeError, match="completion evidence"):
        launcher._update_launch_status(
            **kwargs,
            state="complete",
            detail="too-early",
        )
    completion = {"final": "/final", "export_marker_sha256": "d" * 64}
    complete = launcher._update_launch_status(
        **kwargs,
        state="complete",
        detail="validated-after-volume-commit",
        completion=completion,
    )
    assert complete["state"] == "complete"
    with pytest.raises(RuntimeError, match="status downgrade"):
        launcher._update_launch_status(
            **kwargs,
            state="running",
            detail="late-heartbeat",
        )
    assert store.get(
        launcher._launch_completion_key("vocab85_then_sft3")
    ) == complete


def test_claim_and_identity_tampering_fail_closed():
    store = AtomicStore()
    acquire_attempt(store)
    claim_key = launcher._launch_claim_key("vocab85_then_sft3")
    store.tamper(
        claim_key,
        lambda value: value["launch_identity"].__setitem__("output", "/evil"),
    )
    with pytest.raises(RuntimeError, match="self hash drifted"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-TEST123",
        )


def test_stale_claim_is_never_stolen_by_age():
    store = AtomicStore()
    first = acquire(
        store,
        experiment="vocab85_then_sft3",
        number=1,
        claimed_at="2000-01-01T00:00:00+00:00",
    )
    second = acquire(
        store,
        experiment="vocab85_then_sft3",
        number=2,
        claimed_at="2099-01-01T00:00:00+00:00",
    )
    assert first["outcome"] == "acquired"
    assert second["outcome"] == "existing_claim"
    assert second["claim"] == first["claim"]
    launcher._acquire_launch_attempt(
        store,
        experiment_key="vocab85_then_sft3",
        claim=first["claim"],
        generation=0,
        dispatcher_function_call_id="fc-DISPATCHER",
        recovery_evidence=None,
    )
    with pytest.raises(RuntimeError, match="token does not match"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(2),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-TEST123",
        )


def test_recovery_token_is_durable_before_claim_and_tamper_evident(tmp_path):
    with mock.patch.object(launcher, "LOCAL_LAUNCH_RECOVERY_ROOT", tmp_path):
        path = launcher._write_local_launch_recovery_record(
            experiment_key="vocab85_then_sft3",
            launch_token=token(7),
        )
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        record = launcher._read_local_launch_recovery_record(
            "vocab85_then_sft3"
        )
        assert record["launch_token"] == token(7)
        with pytest.raises(RuntimeError, match="already exists"):
            launcher._write_local_launch_recovery_record(
                experiment_key="vocab85_then_sft3",
                launch_token=token(8),
            )
        changed = path.read_text(encoding="utf-8").replace(token(7), token(8))
        path.write_text(changed, encoding="utf-8")
        with pytest.raises(RuntimeError, match="self hash drifted"):
            launcher._read_local_launch_recovery_record(
                "vocab85_then_sft3"
            )


def test_durable_anchor_is_committed_before_use_and_contains_no_raw_token(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        launcher,
        "_validate_durable_launch_anchor",
        _REAL_VALIDATE_DURABLE_ANCHOR,
    )
    class Volume:
        def __init__(self):
            self.events = []

        def commit(self):
            self.events.append("commit")

        def reload(self):
            self.events.append("reload")

    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    launch_identity = identity()
    claim = launcher._new_launch_claim(
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        launch_identity=launch_identity,
    )
    volume = Volume()
    anchor = _REAL_PUBLISH_DURABLE_ANCHOR(
        experiment_key="vocab85_then_sft3",
        claim=claim,
        launch_identity=launch_identity,
        launch_token=token(1),
        volume=volume,
    )
    path = launcher._durable_launch_anchor_path("vocab85_then_sft3")
    assert volume.events == ["reload", "commit", "reload"]
    assert token(1) not in path.read_text(encoding="utf-8")
    assert anchor["launch_token_sha256"] == launcher._launch_token_sha256(
        token(1)
    )
    assert _REAL_VALIDATE_DURABLE_ANCHOR(
        experiment_key="vocab85_then_sft3",
        claim=claim,
        launch_identity=launch_identity,
        launch_token=token(1),
    ) == anchor


def test_durable_anchor_blocks_fresh_claim_when_dict_entries_expire(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        launcher,
        "_validate_durable_launch_anchor",
        _REAL_VALIDATE_DURABLE_ANCHOR,
    )
    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    launch_identity = _mock_dispatch_preflight(monkeypatch)
    claim = launcher._new_launch_claim(
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        launch_identity=launch_identity,
    )
    _REAL_PUBLISH_DURABLE_ANCHOR(
        experiment_key="vocab85_then_sft3",
        claim=claim,
        launch_identity=launch_identity,
        launch_token=token(1),
        volume=SimpleNamespace(commit=lambda: None, reload=lambda: None),
    )
    with pytest.raises(RuntimeError, match="Dict claim expired or is missing"):
        launcher._dispatch_production_launch(
            AtomicStore(),
            experiment_key="vocab85_then_sft3",
            launch_token=token(2),
            recovery=False,
            dispatcher_function_call_id="fc-NEW",
            spawn_worker=mock.Mock(),
        )


def test_durable_anchor_path_survives_source_redeploy_and_blocks_fresh_claim(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    monkeypatch.setattr(
        launcher,
        "_validate_durable_launch_anchor",
        _REAL_VALIDATE_DURABLE_ANCHOR,
    )
    monkeypatch.setattr(launcher, "SOURCE_TREE_SHA256", "1" * 64)
    old_identity = identity()
    old_claim = launcher._new_launch_claim(
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        launch_identity=old_identity,
    )
    old_path = launcher._durable_launch_anchor_path("vocab85_then_sft3")
    _REAL_PUBLISH_DURABLE_ANCHOR(
        experiment_key="vocab85_then_sft3",
        claim=old_claim,
        launch_identity=old_identity,
        launch_token=token(1),
        volume=SimpleNamespace(commit=lambda: None, reload=lambda: None),
    )
    monkeypatch.setattr(launcher, "SOURCE_TREE_SHA256", "2" * 64)
    assert launcher._durable_launch_anchor_path("vocab85_then_sft3") == old_path
    _mock_dispatch_preflight(monkeypatch, identity())
    with pytest.raises(RuntimeError, match="Dict claim expired or is missing"):
        launcher._dispatch_production_launch(
            AtomicStore(),
            experiment_key="vocab85_then_sft3",
            launch_token=token(2),
            recovery=False,
            dispatcher_function_call_id="fc-NEWSOURCE",
            spawn_worker=mock.Mock(),
        )


def test_recovery_takes_over_terminal_failed_anchor_publisher(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    monkeypatch.setattr(
        launcher,
        "_validate_durable_launch_anchor",
        _REAL_VALIDATE_DURABLE_ANCHOR,
    )
    store = AtomicStore()
    launch_identity = identity()
    claim = acquire(store, experiment="vocab85_then_sft3", number=1)["claim"]
    intent = launcher._self_hash_record(
        {
            "schema": launcher.DURABLE_LAUNCH_ANCHOR_INTENT_SCHEMA,
            "experiment": "vocab85_then_sft3",
            "claim_sha256": claim["claim_sha256"],
            "dispatcher_function_call_id": "fc-PRIOR",
        },
        hash_field="intent_sha256",
    )
    store.put(
        launcher._durable_launch_anchor_intent_key("vocab85_then_sft3"),
        intent,
    )
    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        _terminal_evidence,
    )
    volume = SimpleNamespace(commit=lambda: None, reload=lambda: None)
    anchor = _REAL_ENSURE_DURABLE_ANCHOR(
        store,
        experiment_key="vocab85_then_sft3",
        claim=claim,
        launch_identity=launch_identity,
        launch_token=token(1),
        dispatcher_function_call_id="fc-RECOVERY",
        recovery=True,
        volume=volume,
    )
    assert anchor["publisher_recovery"][
        "prior_dispatcher_function_call_id"
    ] == "fc-PRIOR"
    assert anchor["publisher_recovery"]["terminal_call"][
        "function_call_id"
    ] == "fc-PRIOR"


@pytest.mark.parametrize(
    "side_effect",
    [
        TimeoutError(),
        RuntimeError("success-is-not-terminal-unsuccessful"),
        launcher.modal.exception.OutputExpiredError(),
    ],
)
def test_anchor_publisher_takeover_fails_closed_without_terminal_proof(
    tmp_path, monkeypatch, side_effect
):
    monkeypatch.setattr(launcher, "DURABLE_LAUNCH_ROOT", tmp_path / "ledger")
    store = AtomicStore()
    launch_identity = identity()
    claim = acquire(store, experiment="vocab85_then_sft3", number=1)["claim"]
    intent = launcher._self_hash_record(
        {
            "schema": launcher.DURABLE_LAUNCH_ANCHOR_INTENT_SCHEMA,
            "experiment": "vocab85_then_sft3",
            "claim_sha256": claim["claim_sha256"],
            "dispatcher_function_call_id": "fc-PRIOR",
        },
        hash_field="intent_sha256",
    )
    store.put(
        launcher._durable_launch_anchor_intent_key("vocab85_then_sft3"),
        intent,
    )

    def reject(_call_id):
        if isinstance(side_effect, RuntimeError) and side_effect.args == (
            "success-is-not-terminal-unsuccessful",
        ):
            return launcher._authoritative_terminal_call_evidence(
                mock.Mock(return_value="success"),
                function_call_id="fc-PRIOR",
            )
        return launcher._authoritative_terminal_call_evidence(
            mock.Mock(side_effect=side_effect),
            function_call_id="fc-PRIOR",
        )

    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        reject,
    )
    with pytest.raises(RuntimeError):
        _REAL_ENSURE_DURABLE_ANCHOR(
            store,
            experiment_key="vocab85_then_sft3",
            claim=claim,
            launch_identity=launch_identity,
            launch_token=token(1),
            dispatcher_function_call_id="fc-RECOVERY",
            recovery=True,
            volume=SimpleNamespace(commit=lambda: None, reload=lambda: None),
        )
    assert not launcher._durable_launch_anchor_path(
        "vocab85_then_sft3"
    ).exists()


def test_concurrent_processes_create_exactly_one_unchanged_recovery_record(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_recovery_writer_process,
            args=(str(tmp_path), number, queue),
        )
        for number in range(1, 9)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    winners = [result for result in results if result[0] == "won"]
    assert len(winners) == 1
    winning_number = winners[0][1]
    winning_bytes = winners[0][2]
    path = (
        tmp_path
        / launcher.SOURCE_TREE_SHA256
        / "vocab85_then_sft3.json"
    )
    assert path.read_bytes() == winning_bytes
    with mock.patch.object(launcher, "LOCAL_LAUNCH_RECOVERY_ROOT", tmp_path):
        assert launcher._read_local_launch_recovery_record(
            "vocab85_then_sft3"
        )["launch_token"] == token(winning_number)
    assert path.stat().st_mode & 0o777 == 0o600


def test_wandb_gate_performs_minimal_remote_write_and_exact_readback(monkeypatch):
    store = AtomicStore()
    summary: dict[str, object] = {}
    local_run = SimpleNamespace(
        summary=summary,
        log=mock.Mock(),
        finish=mock.Mock(),
    )
    run_id = launcher._wandb_gate_run_id()
    remote_run = SimpleNamespace(
        id=run_id,
        entity=launcher.WANDB_ENTITY,
        project=launcher.WANDB_PROJECT,
        path=[launcher.WANDB_ENTITY, launcher.WANDB_PROJECT, run_id],
        url="https://wandb.example/gate",
        job_type="infra-gate",
        group="",
        tags=list(launcher._wandb_gate_tags()),
        state="finished",
        summary=summary,
    )
    api = SimpleNamespace(
        viewer=SimpleNamespace(username="researcher"),
        team=mock.Mock(return_value=SimpleNamespace(name=launcher.WANDB_ENTITY)),
        run=mock.Mock(return_value=remote_run),
    )
    monkeypatch.setenv("WANDB_API_KEY", "unit-test-secret")
    runtime = {"modal_app_id": "ap-test", "modal_image_id": "im-test"}
    monkeypatch.setattr(launcher, "_modal_runtime_identity", lambda: runtime)
    monkeypatch.setattr(
        launcher,
        "_validate_recorded_runtime_identity",
        lambda value: None,
    )
    with (
        mock.patch("wandb.Api", return_value=api) as api_class,
        mock.patch("wandb.Settings", return_value=SimpleNamespace()) as settings,
        mock.patch("wandb.init", return_value=local_run) as init,
    ):
        result = launcher._publish_wandb_write_gate(store)
    assert result["decision"] == "pass"
    assert result["remote_evidence"]["path"] == (
        f"{launcher.WANDB_ENTITY}/{launcher.WANDB_PROJECT}/{run_id}"
    )
    init_kwargs = init.call_args.kwargs
    assert init_kwargs["entity"] == launcher.WANDB_ENTITY
    assert init_kwargs["project"] == launcher.WANDB_PROJECT
    assert init_kwargs["id"] == run_id
    assert init_kwargs["job_type"] == "infra-gate"
    assert "group" not in init_kwargs
    assert "infra-gate" in init_kwargs["tags"]
    local_run.log.assert_called_once_with({"infra_gate/write_marker": 1}, step=0)
    local_run.finish.assert_called_once_with(exit_code=0)
    api.run.assert_called_with(
        f"{launcher.WANDB_ENTITY}/{launcher.WANDB_PROJECT}/{run_id}"
    )
    assert summary == launcher._wandb_gate_summary()
    assert "api_key" in settings.call_args.kwargs
    assert settings.call_args.kwargs["x_disable_stats"] is True
    assert api_class.call_count >= 1
    assert api.team.call_count >= 1
    api.team.assert_called_with(launcher.WANDB_ENTITY)


def test_atomicity_gate_requires_one_shared_winner(monkeypatch):
    store = AtomicStore()
    nonce = "e" * 32
    key = f"atomicity-probe:{launcher.SOURCE_TREE_SHA256}:{nonce}"
    winner = {"nonce": nonce, "contender": 3, "function_call_id": "fc-WINNER"}
    store.put(key, winner)
    results = [
        {
            "won": index == 3,
            "proposed": winner if index == 3 else {"contender": index},
            "observed": winner,
        }
        for index in range(8)
    ]
    runtime = {
        "modal_app_name": launcher.APP_NAME,
        "modal_app_id": "ap-test",
        "modal_image_id": "im-test",
        "modal_base_image": launcher.CUDA_BASE_IMAGE,
        "modal_client_version": "1.4.2",
        "runtime_package_versions": launcher.PINNED_RUNTIME_PACKAGE_VERSIONS,
        "runtime_distribution_count": 12,
        "runtime_distribution_inventory_sha256": "f" * 64,
        "python_version": "3.11",
    }
    monkeypatch.setattr(launcher, "_modal_runtime_identity", lambda: runtime)
    marker = launcher._finalize_launch_atomicity_gate(
        store,
        nonce=nonce,
        results=results,
    )
    assert marker["decision"] == "pass"
    assert marker["winner_count"] == 1


def _terminal_evidence(call_id: str):
    get_result = mock.Mock(
        side_effect=RuntimeError(launcher.PRODUCTION_TRAINING_TERMINAL_MARKER)
    )
    evidence = launcher._authoritative_terminal_call_evidence(
        get_result,
        function_call_id=call_id,
    )
    get_result.assert_called_once_with(timeout=0)
    return evidence


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
def test_recovery_rejects_success_pending_expired_and_ambiguous_results(
    outcome, message
):
    get_result = mock.Mock(
        return_value="ok" if outcome is None else mock.DEFAULT,
        side_effect=outcome,
    )
    with pytest.raises(RuntimeError, match=message):
        launcher._authoritative_terminal_call_evidence(
            get_result,
            function_call_id="fc-TEST",
        )
    get_result.assert_called_once_with(timeout=0)


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (
            RuntimeError(launcher.PRODUCTION_TRAINING_TERMINAL_MARKER),
            "application_failure",
        ),
        (
            RuntimeError(launcher.PRODUCTION_DISPATCHER_TERMINAL_MARKER),
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
def test_recovery_accepts_only_authoritative_terminal_unsuccessful_result(
    exception, category
):
    get_result = mock.Mock(side_effect=exception)
    evidence = launcher._authoritative_terminal_call_evidence(
        get_result,
        function_call_id="fc-TEST",
    )
    assert evidence["result_category"] == category
    assert evidence["function_call_id"] == "fc-TEST"
    assert "message" not in evidence
    launcher._validate_terminal_call_evidence_record(
        evidence,
        expected_function_call_id="fc-TEST",
    )
    get_result.assert_called_once_with(timeout=0)


def test_terminal_inspector_uses_get_not_call_graph(monkeypatch):
    call = SimpleNamespace(
        get=mock.Mock(
            side_effect=RuntimeError(launcher.PRODUCTION_TRAINING_TERMINAL_MARKER)
        ),
        get_call_graph=mock.Mock(side_effect=AssertionError("must not be used")),
    )
    monkeypatch.setattr(
        launcher.modal.FunctionCall,
        "from_id",
        mock.Mock(return_value=call),
    )
    launcher._inspect_terminal_unsuccessful_function_call("fc-TEST")
    call.get.assert_called_once_with(timeout=0)
    call.get_call_graph.assert_not_called()


def test_recovery_requires_same_token_then_creates_next_generation(monkeypatch):
    store = AtomicStore()
    launch_identity = _mock_dispatch_preflight(monkeypatch)
    monkeypatch.setattr(launcher.checkpoint_volume, "reload", lambda: None)
    first_spawn = mock.Mock(return_value=SimpleNamespace(object_id="fc-WORKER0"))
    first = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=False,
        dispatcher_function_call_id="fc-DISPATCH0",
        spawn_worker=first_spawn,
    )
    assert first["generation"] == 0
    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        _terminal_evidence,
    )
    monkeypatch.setattr(
        launcher,
        "_authenticate_recovery_output",
        lambda experiment: {
            "state": "resumable",
            "has_output": True,
            "stages": {"pt": {"state": "resumable", "global_step": 123}},
            "completion": None,
        },
    )
    second_spawn = mock.Mock(return_value=SimpleNamespace(object_id="fc-WORKER1"))
    with pytest.raises(RuntimeError, match="token does not match"):
        launcher._dispatch_production_launch(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(2),
            recovery=True,
            dispatcher_function_call_id="fc-DISPATCH1",
            spawn_worker=second_spawn,
        )
    result = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=True,
        dispatcher_function_call_id="fc-DISPATCH1",
        spawn_worker=second_spawn,
    )
    assert result["outcome"] == "recovery_spawned"
    assert result["generation"] == 1
    second_spawn.assert_called_once_with("vocab85_then_sft3", token(1), 1)
    assert launch_identity == identity()


def test_two_concurrent_recoveries_create_only_one_new_attempt(monkeypatch):
    import threading

    store = AtomicStore()
    _mock_dispatch_preflight(monkeypatch)
    launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=False,
        dispatcher_function_call_id="fc-DISPATCH0",
        spawn_worker=lambda *args: SimpleNamespace(object_id="fc-WORKER0"),
    )
    barrier = threading.Barrier(2)

    def terminal(call_id):
        barrier.wait(timeout=5)
        return _terminal_evidence(call_id)

    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        terminal,
    )
    monkeypatch.setattr(launcher.checkpoint_volume, "reload", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_authenticate_recovery_output",
        lambda experiment: {
            "state": "resumable",
            "has_output": False,
            "stages": {},
            "completion": None,
        },
    )
    spawn = mock.Mock(
        side_effect=lambda *args: SimpleNamespace(object_id="fc-WORKER1")
    )

    def recover(index):
        return launcher._dispatch_production_launch(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            recovery=True,
            dispatcher_function_call_id=f"fc-DISPATCH{index}",
            spawn_worker=spawn,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(recover, (1, 2)))
    assert sum(result["spawned"] for result in results) == 1
    assert sorted(result["outcome"] for result in results) == [
        "attempt_exists",
        "recovery_spawned",
    ]
    assert spawn.call_count == 1
    claim = store.get(launcher._launch_claim_key("vocab85_then_sft3"))
    current = launcher._current_launch_attempt(
        store,
        experiment_key="vocab85_then_sft3",
        claim=claim,
    )
    assert current["generation"] == 1


def test_dispatcher_spawn_before_bind_race_cannot_start_old_generation(monkeypatch):
    store = AtomicStore()
    launch_identity = _mock_dispatch_preflight(monkeypatch)
    claim, _ = acquire_attempt(store, dispatcher="fc-DISPATCH0")
    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        _terminal_evidence,
    )
    monkeypatch.setattr(launcher.checkpoint_volume, "reload", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_authenticate_recovery_output",
        lambda experiment: {
            "state": "resumable",
            "has_output": False,
            "stages": {},
            "completion": None,
        },
    )
    recovered = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=True,
        dispatcher_function_call_id="fc-DISPATCH1",
        spawn_worker=lambda *args: SimpleNamespace(object_id="fc-WORKER1"),
    )
    assert recovered["generation"] == 1
    with pytest.raises(RuntimeError, match="is not current"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=launch_identity,
            generation=0,
            function_call_id="fc-ORPHANEDOLDWORKER",
        )
    assert claim["launch_token_sha256"] == launcher._launch_token_sha256(token(1))


def test_worker_binding_wins_inverse_recovery_race_and_blocks_advance(monkeypatch):
    store = AtomicStore()
    _mock_dispatch_preflight(monkeypatch)
    acquire_attempt(store, dispatcher="fc-DISPATCH0")
    monkeypatch.setattr(launcher.checkpoint_volume, "reload", lambda: None)
    inspected: list[str] = []

    def inspect(call_id):
        inspected.append(call_id)
        if call_id == "fc-DISPATCH0":
            return _terminal_evidence(call_id)
        return launcher._authoritative_terminal_call_evidence(
            mock.Mock(side_effect=TimeoutError()),
            function_call_id=call_id,
        )

    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        inspect,
    )
    injected = False

    def authenticate(_experiment):
        nonlocal injected
        if not injected:
            injected = True
            launcher._update_launch_status(
                store,
                experiment_key="vocab85_then_sft3",
                launch_token=token(1),
                expected_identity=identity(),
                generation=0,
                function_call_id="fc-ORPHANEDOLDWORKER",
                state="running",
                stage="pt",
                detail="worker-authenticated",
            )
        return {
            "state": "resumable",
            "has_output": False,
            "stages": {},
            "completion": None,
        }

    monkeypatch.setattr(launcher, "_authenticate_recovery_output", authenticate)
    spawn = mock.Mock()
    with pytest.raises(RuntimeError, match="still pending"):
        launcher._dispatch_production_launch(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            recovery=True,
            dispatcher_function_call_id="fc-DISPATCH1",
            spawn_worker=spawn,
        )
    spawn.assert_not_called()
    resolution = store.get(
        launcher._launch_execution_key("vocab85_then_sft3", 0)
    )
    assert resolution["kind"] == "worker"
    assert resolution["function_call_id"] == "fc-ORPHANEDOLDWORKER"
    assert store.get(
        launcher._launch_attempt_key("vocab85_then_sft3", 1)
    ) is None
    assert inspected == ["fc-DISPATCH0", "fc-ORPHANEDOLDWORKER"]


def test_recovery_close_wins_during_worker_binding_and_rejects_worker():
    store = AtomicStore()
    claim, attempt0 = acquire_attempt(store, dispatcher="fc-DISPATCH0")
    closure = launcher._recovery_closure(
        experiment_key="vocab85_then_sft3",
        claim=claim,
        attempt=attempt0,
        generation=0,
        dispatcher_function_call_id="fc-DISPATCH0",
        terminal_call=_terminal_evidence("fc-DISPATCH0"),
        checkpoint_state={
            "state": "resumable",
            "has_output": False,
            "stages": {},
            "completion": None,
        },
    )
    original_put = store.put
    injected = False

    def racing_put(key, value, *, skip_if_exists=False):
        nonlocal injected
        if (
            not injected
            and key == launcher._launch_execution_key("vocab85_then_sft3", 0)
            and value.get("kind") == "worker"
        ):
            injected = True
            assert original_put(
                launcher._launch_execution_key("vocab85_then_sft3", 0),
                closure,
                skip_if_exists=True,
            )
        return original_put(key, value, skip_if_exists=skip_if_exists)

    store.put = racing_put
    with pytest.raises(RuntimeError, match="closed for recovery"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=identity(),
            generation=0,
            function_call_id="fc-OLDWORKER",
        )
    assert injected is True


def test_recovery_closure_rejects_wrong_attempt_dispatcher():
    store = AtomicStore()
    claim, attempt0 = acquire_attempt(store, dispatcher="fc-DISPATCH0")
    checkpoint = {
        "state": "resumable",
        "has_output": False,
        "stages": {},
        "completion": None,
    }
    with pytest.raises(RuntimeError, match="dispatcher drifted"):
        launcher._recovery_closure(
            experiment_key="vocab85_then_sft3",
            claim=claim,
            attempt=attempt0,
            generation=0,
            dispatcher_function_call_id="fc-WRONGDISPATCHER",
            terminal_call=_terminal_evidence("fc-WRONGDISPATCHER"),
            checkpoint_state=checkpoint,
        )
    forged = launcher._self_hash_record(
        {
            "schema": launcher.LAUNCH_EXECUTION_SCHEMA,
            "experiment": "vocab85_then_sft3",
            "generation": 0,
            "claim_sha256": claim["claim_sha256"],
            "attempt_sha256": attempt0["attempt_sha256"],
            "launch_token_sha256": claim["launch_token_sha256"],
            "kind": "recovery_closed",
            "function_call_id": "fc-WRONGDISPATCHER",
            "terminal_call": _terminal_evidence("fc-WRONGDISPATCHER"),
            "checkpoint_state": checkpoint,
        },
        hash_field="execution_sha256",
    )
    with pytest.raises(RuntimeError, match="dispatcher drifted"):
        launcher._validate_generation_resolution(
            forged,
            experiment_key="vocab85_then_sft3",
            claim=claim,
            attempt=attempt0,
            generation=0,
        )


def test_recovery_closes_generation_before_returning_completion(monkeypatch):
    store = AtomicStore()
    launch_identity = _mock_dispatch_preflight(monkeypatch)
    acquire_attempt(store, dispatcher="fc-DISPATCH0")
    monkeypatch.setattr(
        launcher,
        "_inspect_terminal_unsuccessful_function_call",
        _terminal_evidence,
    )
    monkeypatch.setattr(launcher.checkpoint_volume, "reload", lambda: None)
    completion = {
        "schema": "context2048-authenticated-production-completion-v1",
        "final": "/checkpoints/final",
    }
    monkeypatch.setattr(
        launcher,
        "_authenticate_recovery_output",
        lambda experiment: {
            "state": "complete",
            "has_output": True,
            "stages": {"pt": {"state": "complete"}},
            "completion": completion,
        },
    )
    spawn = mock.Mock()
    result = launcher._dispatch_production_launch(
        store,
        experiment_key="vocab85_then_sft3",
        launch_token=token(1),
        recovery=True,
        dispatcher_function_call_id="fc-DISPATCH1",
        spawn_worker=spawn,
    )
    assert result["outcome"] == "authenticated_completion"
    assert result["completion"] == completion
    assert result["closed_generation"] == 0
    spawn.assert_not_called()
    resolution = store.get(
        launcher._launch_execution_key("vocab85_then_sft3", 0)
    )
    assert resolution["kind"] == "recovery_closed"
    with pytest.raises(RuntimeError, match="closed for recovery"):
        launcher._begin_claimed_worker(
            store,
            experiment_key="vocab85_then_sft3",
            launch_token=token(1),
            expected_identity=launch_identity,
            generation=0,
            function_call_id="fc-LATEOLDWORKER",
        )


def test_worker_publishes_completion_only_after_reload_and_authentication(
    monkeypatch,
):
    events: list[str] = []
    launch_identity = identity()
    monkeypatch.setattr(
        launcher.modal,
        "current_function_call_id",
        lambda: "fc-TEST123",
    )
    monkeypatch.setattr(launcher, "_launch_identity", lambda experiment: launch_identity)
    monkeypatch.setattr(
        launcher,
        "_begin_claimed_worker",
        lambda *args, **kwargs: {"claim": {}, "execution": {}},
    )

    def fake_train(experiment, *, canary, heartbeat):
        events.append("train")
        return "/final"

    monkeypatch.setattr(launcher, "_run_experiment", fake_train)
    monkeypatch.setattr(
        launcher.checkpoint_volume,
        "reload",
        lambda: events.append("reload"),
    )

    def authenticate(experiment):
        events.append("authenticate")
        return {"final": "/final", "export_marker_sha256": "d" * 64}

    monkeypatch.setattr(launcher, "_authenticate_existing_completion", authenticate)

    def status(*args, **kwargs):
        events.append(f"status:{kwargs['state']}")
        return {}

    monkeypatch.setattr(launcher, "_update_launch_status", status)
    raw_worker = launcher.run_experiment.get_raw_f()
    assert raw_worker("vocab85_then_sft3", token(1), 0) == "/final"
    assert events == [
        "reload",
        "status:running",
        "train",
        "reload",
        "authenticate",
        "status:complete",
    ]


def test_production_modal_function_timeout_is_at_most_24_hours():
    source = launcher.Path(launcher.__file__).read_text(encoding="utf-8")
    assert launcher.PRODUCTION_FUNCTION_TIMEOUT_SECONDS == 86_400
    assert "timeout=PRODUCTION_FUNCTION_TIMEOUT_SECONDS" in source
