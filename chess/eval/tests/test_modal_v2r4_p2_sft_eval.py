from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from Eval import modal_eval_v2r4_p2_sft as runner


def _json_file(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _p1_state(step: int) -> dict:
    return {
        "global_step": step,
        "manifest_cursor": step,
        "manifest_hash": runner.p2_contract.P1_METADATA_FILE_SHA256,
        "sft_loss_weight": runner.SFT_LOSS_WEIGHT,
        "world_size": 8,
        "local_batch_size": 21,
        "gradient_accumulation_steps": 1,
        "snapshot_steps": [1_000, 2_000, 4_000, 6_000, 8_000, 9_920],
        "arc_steps": [9_920],
        "data_state": {
            "schema": "interleaved-stream-state-v1",
            "cursor": step,
            "local_batch_size": 21,
            "world_size": 8,
            "manifest_hash": runner.p2_contract.P1_METADATA_FILE_SHA256,
            "selection_hash": runner.p2_contract.SELECTION_HASH,
            "sft_cache_hash": runner.p2_contract.SFT_CACHE_HASH,
            "source_manifest_hash": runner.p2_contract.SOURCE_MANIFEST_HASH,
        },
        "configured_provenance": {
            "experiment_version": runner.P1_EXPERIMENT_VERSION,
            "data_artifact_version": runner.p2_contract.DATA_ARTIFACT_VERSION,
            "sft_loss_weight": runner.SFT_LOSS_WEIGHT,
            "source_repo": runner.p2_contract.SOURCE_REPO,
            "source_revision": runner.p2_contract.SOURCE_REVISION,
            "sft_repo": runner.p2_contract.SFT_REPO,
            "sft_revision": runner.p2_contract.SFT_REVISION,
            "source_tree_sha256": runner.P1_SOURCE_TREE_SHA256,
            "source_flat_manifest_sha256": (
                runner.P1_SOURCE_FLAT_MANIFEST_SHA256
            ),
            "sft_response_normalization": (
                "strip-numeric-verify-score-pairs-normalize-whitespace-v1"
            ),
            "sft_supervised_unk_policy": "reject-supervised-unk-v1",
        },
    }


def _checkpoint_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: int = 6_000
) -> tuple[Path, dict]:
    root = tmp_path / "checkpoint"
    root.mkdir()
    _json_file(root / "config.json", dict(runner.EXPECTED_MODEL_CONFIG))
    _json_file(root / "generation_config.json", {})
    (root / "model.safetensors").write_bytes(b"weights")
    state_path = root / "interleaved_training_state.json"
    _json_file(state_path, _p1_state(step))
    tokenizer_hashes = {}
    for index, relative in enumerate(runner.TOKENIZER_FILE_SHA256):
        path = root / relative
        path.write_bytes(f"tokenizer-{index}".encode())
        tokenizer_hashes[relative] = runner._sha256_file(path)
    monkeypatch.setattr(runner, "TOKENIZER_FILE_SHA256", tokenizer_hashes)
    identities = runner._directory_identities(root)
    endpoint = runner._endpoint_checkpoint_fingerprint(root)
    candidate = {
        "candidate_id": f"test-step{step}",
        "path": root,
        "recursive_hf_identity": identities["recursive_hf_identity"],
        "directory_manifest_sha256": (
            identities["directory_manifest_sha256"]
        ),
        "endpoint_checkpoint_sha256": endpoint,
        "training_state_sha256": runner._sha256_file(state_path),
    }
    monkeypatch.setattr(runner, "CANDIDATES", {step: candidate})
    return root, candidate


def test_frozen_live_cache_selection_and_shape_identities():
    assert runner.SFT_ARTIFACT_SHA256["sft_cache/input_ids.i32"] == (
        "c8c75b6eec58c6d9943a799d04f3e054221f4e2207873b521e5b8eae548bb8a8"
    )
    assert runner.SFT_ARTIFACT_SHA256["sft_cache/labels.i32"] == (
        "7bb6b16fdd6a7fe1b1e0702f21e9535334421a5c12d074848f60f8d76d357373"
    )
    assert runner.SFT_ARTIFACT_SHA256["sft_cache/offsets.npy"] == (
        "0c6f777a79ae8f0d397f1e623724e30137fa5c89060efed1ba24e5ce48c83701"
    )
    assert runner.SELECTION_HASH == (
        "99d20a1ee7dad9ab88ab5de2dfe0df50cc9d9e076636cf41252fbb1db2ea371e"
    )
    assert runner.CACHE_SHAPE_HASH == (
        "6b8b068a1d02480d9c0a9933c19a534bb64eb15fe16e9ae7a313f4ea66c4d5c5"
    )
    assert runner.EXPECTED_SELECTED_SHAPE == {
        "num_records": 4_096,
        "total_aligned_positions": 3_560_000,
        "supervised_targets": 2_759_776,
        "ignored_positions": 800_224,
        "aligned_positions_per_row_min": 160,
        "aligned_positions_per_row_max": 1_645,
        "supervised_targets_per_row_min": 37,
        "supervised_targets_per_row_max": 1_448,
        "selected_label_payload_sha256": (
            "5ed3524fd47012774b7e2f858c5fdddf6be1e1c588c666a78779142a8d3be581"
        ),
    }


def test_evaluator_source_and_runtime_contract_are_self_consistent():
    assert runner.RUNNER_SOURCE_SHA256 == runner._sha256_file(
        Path(runner.__file__)
    )
    assert runner.CONTRACT_SOURCE_SHA256 == runner._sha256_file(
        runner.CONTRACT_LOCAL_PATH
    )
    expected_bundle = hashlib.sha256(
        runner.p2_contract.canonical_json(
            {
                "runner": runner.RUNNER_SOURCE_SHA256,
                "pure_contract": runner.CONTRACT_SOURCE_SHA256,
            }
        )
    ).hexdigest()
    assert runner.EVALUATOR_SOURCE_SHA256 == expected_bundle
    assert runner.RUNTIME_CONTRACT_SHA256 == hashlib.sha256(
        runner.p2_contract.canonical_json(runner.RUNTIME_CONTRACT)
    ).hexdigest()
    assert runner.RUNTIME_CONTRACT["gpu"] == "H200"
    assert runner.RUNTIME_CONTRACT["torch"] == "2.9.0"
    assert runner.RUNTIME_CONTRACT["transformers"] == "4.57.0"
    assert runner.RUNTIME_CONTRACT["numpy"] == "2.2.6"
    assert runner.RUNTIME_CONTRACT["chess"] == "1.11.2"
    assert runner.APP_NAME == "chess-interleave-v2r4-p2-sft-eval-v2"
    assert runner.RESULT_NAMESPACE == "v2r4_p2_sft_at_p1_20260730_v2"
    assert runner.OUTPUT_SCHEMA.endswith("-v2")
    assert runner.DEFAULT_LAUNCH_LEDGER_PATH == (
        "INTERLEAVED_V2R4_P2_SFT_V2_LAUNCH_LEDGER.json"
    )


def test_checkpoint_authentication_binds_recursive_tokenizer_and_p1_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, expected = _checkpoint_fixture(tmp_path, monkeypatch)
    candidate, identity = runner._authenticate_candidate_checkpoint(
        6_000,
        expected_recursive_hf_identity=expected["recursive_hf_identity"],
        expected_endpoint_checkpoint_sha256=(
            expected["endpoint_checkpoint_sha256"]
        ),
    )
    assert candidate["training_leg"] == "p1"
    assert candidate["has_consumed_p2"] is False
    assert identity["recursive_hf_identity"] == (
        expected["recursive_hf_identity"]
    )
    assert identity["tokenizer_file_sha256"] == (
        runner.TOKENIZER_FILE_SHA256
    )

    (root / "vocab.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="directory identity drifted"):
        runner._authenticate_candidate_checkpoint(
            6_000,
            expected_recursive_hf_identity=expected[
                "recursive_hf_identity"
            ],
            expected_endpoint_checkpoint_sha256=expected[
                "endpoint_checkpoint_sha256"
            ],
        )


def test_checkpoint_authentication_rejects_caller_identity_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, expected = _checkpoint_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="caller recursive HF identity"):
        runner._authenticate_candidate_checkpoint(
            6_000,
            expected_recursive_hf_identity="0" * 64,
            expected_endpoint_checkpoint_sha256=expected[
                "endpoint_checkpoint_sha256"
            ],
        )
    with pytest.raises(ValueError, match="caller endpoint checkpoint"):
        runner._authenticate_candidate_checkpoint(
            6_000,
            expected_recursive_hf_identity=expected[
                "recursive_hf_identity"
            ],
            expected_endpoint_checkpoint_sha256="0" * 64,
        )


def test_p1_only_provenance_rejects_p2_or_wrong_cursor():
    state = _p1_state(6_000)
    state["data_state"]["manifest_hash"] = (
        runner.p2_contract.P2_METADATA_FILE_SHA256
    )
    with pytest.raises(ValueError, match="exact P1-only data cursor"):
        runner._validate_p1_training_state(state, candidate_step=6_000)

    state = _p1_state(6_000)
    state["manifest_cursor"] = 8_000
    with pytest.raises(ValueError, match="training-state drifted"):
        runner._validate_p1_training_state(state, candidate_step=6_000)


def test_immutable_atomic_result_refuses_overwrite_and_active_writer(
    tmp_path: Path,
):
    path = tmp_path / "result.json"
    runner._immutable_json(path, {"state": "complete", "value": 1})
    runner._immutable_json(path, {"state": "complete", "value": 1})
    with pytest.raises(ValueError, match="immutable result differs"):
        runner._immutable_json(path, {"state": "complete", "value": 2})

    path.unlink()
    lock = path.with_name(f".{path.name}.immutable-lock")
    lock.write_text("busy")
    with pytest.raises(RuntimeError, match="immutable result writer"):
        runner._immutable_json(path, {"state": "complete", "value": 1})
    assert not path.exists()


def test_exclusive_launch_ledger_refuses_second_controller(tmp_path: Path):
    path = tmp_path / "launch-ledger.json"
    runner._exclusive_json(path, {"state": "intent"})
    with pytest.raises(FileExistsError):
        runner._exclusive_json(path, {"state": "duplicate"})
    assert json.loads(path.read_text()) == {"state": "intent"}


def test_direct_endpoint_contract_is_exact_and_complete():
    contract = runner.direct_call_contract(9_920)
    candidate = runner.CANDIDATES[9_920]
    assert contract == {
        "app_name": "chess-interleave-v2r4-p2-sft-eval-v2",
        "function_name": "evaluate_p2_sft_candidate",
        "kwargs": {
            "candidate_step": 9_920,
            "expected_recursive_hf_identity": (
                candidate["recursive_hf_identity"]
            ),
            "expected_endpoint_checkpoint_sha256": (
                candidate["endpoint_checkpoint_sha256"]
            ),
            "runtime_contract_sha256": runner.RUNTIME_CONTRACT_SHA256,
        },
        "expected_result_path": str(runner._result_path(9_920)),
    }
    assert "/v2r4_p2_sft_at_p1_20260730_v2/" in (
        contract["expected_result_path"]
    )
    assert set(runner.CANDIDATES) == {6_000, 8_000, 9_920}
    with pytest.raises(ValueError, match="unsupported"):
        runner.direct_call_contract(1_000)


def test_cache_aligned_loss_scores_first_and_final_positions_without_shift():
    logits = torch.full((1, 4, 5), -12.0)
    labels = torch.tensor([[1, -100, 2, 4]], dtype=torch.long)
    attention = torch.ones((1, 4), dtype=torch.long)
    logits[0, 0, 1] = 12.0
    logits[0, 1, 0] = 12.0
    logits[0, 2, 2] = 12.0
    logits[0, 3, 4] = 12.0
    nll, correct, supervised = runner._unshifted_masked_sums(
        logits, labels, attention
    )
    assert supervised == 3
    assert correct == 3
    assert nll < 1e-6
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_supervised = shifted_labels.ne(-100)
    shifted_correct = int(
        (
            shifted_logits.argmax(dim=-1).eq(shifted_labels)
            & shifted_supervised
        )
        .sum()
        .item()
    )
    assert shifted_logits.shape[1] == logits.shape[1] - 1
    assert shifted_correct == 0
    assert labels[0, -1].item() == 4
    assert logits[0, -1].argmax().item() == 4

    attention[0, 3] = 0
    with pytest.raises(ValueError, match="outside the attention mask"):
        runner._unshifted_masked_sums(logits, labels, attention)


def test_gpu_worker_has_no_caller_metrics_and_zero_retry_sum_nll_contract():
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate_p2_sft_candidate"
    )
    assert [argument.arg for argument in worker.args.args] == [
        "candidate_step",
        "expected_recursive_hf_identity",
        "expected_endpoint_checkpoint_sha256",
        "runtime_contract_sha256",
    ]
    decorator_source = "\n".join(
        ast.unparse(decorator) for decorator in worker.decorator_list
    )
    assert "gpu='H200'" in decorator_source
    assert "retries=0" in decorator_source
    worker_source = ast.get_source_segment(source, worker)
    assert worker_source is not None
    assert "AutoModelForCausalLM.from_pretrained" in worker_source
    assert "_unshifted_masked_sums" in worker_source
    assert "shift_labels" not in worker_source
    assert "shift_logits" not in worker_source
    assert "correct_supervised_tokens=total_correct" in worker_source
    assert "negative_log_likelihood_sum=total_nll" in worker_source
    assert (
        'total_supervised != int(shape["supervised_targets"])'
        in worker_source
    )
    assert "input_ids.i32" not in [
        argument.arg for argument in worker.args.args
    ]
    assert "negative_log_likelihood_sum" not in [
        argument.arg for argument in worker.args.args
    ]
    assert "import chess" in worker_source
    assert '"chess": chess.__version__' in worker_source


def test_preflight_and_launcher_enforce_all_three_absent_and_durable_ids():
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    preflight_source = ast.get_source_segment(
        source, functions["preflight_p2_sft_grid"]
    )
    dependency_source = ast.get_source_segment(
        source, functions["preflight_p2_sft_v2_runtime"]
    )
    main_source = ast.get_source_segment(source, functions["main"])
    assert preflight_source is not None
    assert dependency_source is not None
    assert main_source is not None
    assert "all_three_result_roots_absent" in preflight_source
    assert "if existing:" in preflight_source
    assert "_exclusive_json(ledger_path, ledger)" in main_source
    assert "dependency_preflight_function_call_id" in main_source
    assert "preflight_function_call_id" in main_source
    assert '"function_call_id": call.object_id' in main_source
    assert 'ledger["state"] = "launched_all"' in main_source
    assert "if candidate_step:" in main_source
    assert "import chess" in dependency_source
    assert "_validate_exact_tokenizer" in dependency_source
    dependency_decorators = "\n".join(
        ast.unparse(item)
        for item in functions[
            "preflight_p2_sft_v2_runtime"
        ].decorator_list
    )
    assert "gpu=" not in dependency_decorators


def test_gpu_image_pins_tokenizer_runtime_dependency():
    source = Path(runner.__file__).read_text()
    assert 'CHESS_VERSION = "1.11.2"' in source
    assert 'f"chess=={CHESS_VERSION}"' in source
