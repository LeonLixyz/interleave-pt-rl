import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from miles.ray.actor_group import RayTrainGroup
from miles.ray import train_actor


def test_train_group_flushes_tracking_in_all_actors():
    group = RayTrainGroup.__new__(RayTrainGroup)
    group._broadcast = AsyncMock(return_value=[])

    asyncio.run(group.finish_tracking())

    group._broadcast.assert_awaited_once_with("finish_tracking")


def test_train_actor_finish_tracking_flushes_process_manager(monkeypatch):
    finish = Mock()
    monkeypatch.setattr(train_actor, "finish_tracking", finish)

    train_actor.TrainRayActor.finish_tracking(object())

    finish.assert_called_once_with()


def test_driver_flushes_workers_before_rollout_manager_and_primary():
    train_source = Path(__file__).resolve().parents[2] / "train.py"
    tree = ast.parse(train_source.read_text())
    calls = {
        ast.unparse(node.func): node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    actor_finish = calls["actor_model.finish_tracking"]
    rollout_dispose = calls["rollout_manager.dispose.remote"]
    primary_finish = calls["finish_tracking"]

    assert actor_finish < rollout_dispose < primary_finish
