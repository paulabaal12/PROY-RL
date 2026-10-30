"""Tests del grabador de replays (viz/record.py)."""

import json

from agents.baselines import FifoHeuristicPolicy
from envs.kitchen_env import KitchenEnv
from viz.record import main, record_episode


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_writes_header_reset_and_one_line_per_step(tmp_path):
    env = KitchenEnv(window=2, kitchen="easy", horizon=30)
    path = record_episode(env, FifoHeuristicPolicy(2), seed=10000, path=tmp_path / "r.jsonl",
                          meta={"label": "x"})
    lines = read_lines(path)
    assert lines[0]["kind"] == "header" and lines[0]["seed"] == 10000
    assert lines[0]["window"] == 2 and lines[0]["kitchen"] == "easy" and lines[0]["label"] == "x"
    assert lines[0]["env_params"]["horizon"] == 30
    assert lines[1]["t"] == 0 and lines[1]["action"] is None and lines[1]["reward"] == 0.0
    assert len(lines) == 2 + 30
    assert [line["t"] for line in lines[1:]] == list(range(31))


def test_recorded_info_matches_a_fresh_run(tmp_path):
    path = record_episode(KitchenEnv(window=2, kitchen="easy", horizon=40), FifoHeuristicPolicy(2),
                          seed=10000, path=tmp_path / "r.jsonl")
    steps = read_lines(path)[2:]

    env = KitchenEnv(window=2, kitchen="easy", horizon=40)
    policy = FifoHeuristicPolicy(2)
    obs, _ = env.reset(seed=10000)
    for line in steps:
        action = policy(obs)
        obs, reward, *_, info = env.step(action)
        assert line["action"] == action
        assert line["reward"] == reward
        assert line["info"]["events"] == json.loads(json.dumps(info["events"]))


def test_action_fields_are_decoded(tmp_path):
    env = KitchenEnv(window=2, kitchen="easy", horizon=40)
    path = record_episode(env, FifoHeuristicPolicy(2), seed=3, path=tmp_path / "r.jsonl")
    for line in read_lines(path)[2:]:
        assert line["action_kind"] in {"wait", "prep", "place", "taste", "serve", "discard"}
        assert env.layout.decode(line["action"]) == (line["action_kind"], line["action_index"])
        assert len(line["obs"]) == env.observation_space.shape[0]
        assert len(line["visual_cook"]) == 3


def test_cli_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "r.jsonl"
    main(["--policy", "random", "--seed", "1", "--kitchen", "medium", "--window", "2",
          "--out", str(out), "--horizon", "10"])
    lines = read_lines(out)
    assert lines[0]["policy"] == "random" and len(lines) == 12
