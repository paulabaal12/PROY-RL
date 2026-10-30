"""Tests de la carga de replays (viz/replay.py) y de los replays de ejemplo."""

from pathlib import Path

import pytest

from agents.baselines import FifoHeuristicPolicy
from envs.kitchen_env import KitchenEnv
from viz.record import record_episode
from viz.replay import ReplayError, compatibility_warnings, load_replay

SAMPLES = ["sample_fifo", "sample_random", "sample_edf", "sample_fifo_medium"]


def make_replay(tmp_path, seed=10000, name="r.jsonl", horizon=20, kitchen="easy"):
    env = KitchenEnv(window=2, kitchen=kitchen, horizon=horizon)
    return record_episode(env, FifoHeuristicPolicy(2), seed=seed, path=tmp_path / name)


def test_load_replay_roundtrip(tmp_path):
    replay = load_replay(make_replay(tmp_path))
    assert len(replay.steps) == 21
    assert replay.steps[0].action is None and replay.steps[0].t == 0
    assert replay.window == 2 and replay.n_stations == 3
    assert replay.horizon == 20 and replay.patience == 10


def test_load_replay_missing_file_raises_replay_error(tmp_path):
    with pytest.raises(ReplayError, match="nope.jsonl"):
        load_replay(tmp_path / "nope.jsonl")


def test_load_replay_without_header_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "step", "t": 0}\n')
    with pytest.raises(ReplayError, match="bad.jsonl"):
        load_replay(path)


def test_load_empty_replay_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ReplayError):
        load_replay(path)


def test_compatibility_warns_on_seed_mismatch(tmp_path):
    a = load_replay(make_replay(tmp_path, seed=1, name="a.jsonl"))
    b = load_replay(make_replay(tmp_path, seed=2, name="b.jsonl"))
    warnings = compatibility_warnings([a, b])
    assert len(warnings) == 1 and "seed" in warnings[0]
    assert compatibility_warnings([a, a]) == []


@pytest.mark.parametrize("name", SAMPLES)
def test_samples_exist_and_load(name):
    replay = load_replay(Path("results/replays") / f"{name}.jsonl")
    assert len(replay.steps) == 201
    assert replay.header["label"]
