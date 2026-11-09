"""Tests de DQN/PPO: bootstrap en truncamiento, reproducibilidad, análisis y replays."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from agents.baselines import FifoHeuristicPolicy
from agents.rl import (
    FINISH_GATE, ModelPolicy, PotentialShaping, ScaleReward, load_policy, make_train_env,
    valid_action_mask,
)
from envs.kitchen_env import OP_DISCARD, OP_SERVE, OP_TASTE, ST_OCCUPIED, ST_VISUAL, KitchenEnv
from experiments import analyze, export_replays, report_assets
from experiments.evaluate import evaluate
from experiments.run_experiments import final_evaluation
from experiments.train import load_config, train_run
from viz.record import record_episode
from viz.replay import load_replay

WINDOW = 2
TINY = {
    "dqn": {"hyperparameters": {"learning_starts": 50, "buffer_size": 500, "batch_size": 16,
                                "target_update_interval": 50}},
    "ppo": {"n_envs": 2, "hyperparameters": {"n_steps": 32, "batch_size": 32, "n_epochs": 1}},
}


def tiny_config(algo: str) -> dict:
    base = load_config(algo, f"experiments/configs/{algo}.yaml")
    return {**base, "net_arch": [16], "log_interval": 1, "n_envs": TINY[algo].get("n_envs", base["n_envs"]),
            "hyperparameters": {**base["hyperparameters"], **TINY[algo]["hyperparameters"]}}


def train_tiny(algo, path, seed=0):
    return train_run(algo, WINDOW, seed, tiny_config(algo), path, 200, eval_every=100, eval_episodes=1,
                     model_path=path / "model.zip")


def weights(model) -> list[np.ndarray]:
    return [p.detach().numpy().copy() for p in model.policy.parameters()]


def test_truncation_is_flagged_for_bootstrap():
    env = make_train_env(WINDOW, 1)
    env.reset()
    for _ in range(199):
        _, _, done, infos = env.step(np.array([0]))
        assert not done[0]
    _, _, done, infos = env.step(np.array([0]))
    # Con esta marca, DQN y PPO hacen bootstrap en vez de tratarlo como estado final
    assert done[0] and infos[0]["TimeLimit.truncated"] and "terminal_observation" in infos[0]


def test_scale_reward_changes_reward_only():
    env = ScaleReward(KitchenEnv(window=WINDOW, kitchen="medium"), 0.1)
    raw = KitchenEnv(window=WINDOW, kitchen="medium")
    env.reset(seed=0)
    raw.reset(seed=0)
    invalid = raw.layout.station(0, 0)     # probar una estación vacía: −0.1
    assert env.step(invalid)[1] == pytest.approx(raw.step(invalid)[1] * 0.1)


def test_mask_only_allows_actions_the_env_accepts():
    for window in (1, 4):
        env = KitchenEnv(window=window, kitchen="hard")
        rng = np.random.default_rng(window)
        obs, _ = env.reset(seed=window)
        for _ in range(400):
            mask = valid_action_mask(torch.as_tensor(obs[None]), env.action_space.n)[0].numpy()
            for action in np.flatnonzero(mask):
                _, _, _, _, info = copy.deepcopy(env).step(int(action))
                assert not any(e["type"] == "invalid" for e in info["events"])
            obs, _, _, truncated, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
            if truncated:
                obs, _ = env.reset()


def test_mask_blocks_finishing_food_that_looks_raw():
    env = KitchenEnv(window=WINDOW, kitchen="medium")
    obs, _ = env.reset(seed=0)
    obs = obs.copy()
    obs[ST_OCCUPIED] = 1.0
    layout, n = env.layout, env.action_space.n
    for visual, allowed in ((FINISH_GATE - 0.01, False), (FINISH_GATE + 0.01, True)):
        obs[ST_VISUAL] = visual
        mask = valid_action_mask(torch.as_tensor(obs[None]), n)[0].numpy()
        assert mask[layout.station(0, OP_TASTE)]
        assert mask[layout.station(0, OP_SERVE)] == allowed
        assert mask[layout.station(0, OP_DISCARD)] == allowed


def test_shaping_only_adds_a_potential_difference():
    # Con gamma = 1 la suma de lo agregado es exactamente Phi(final) - Phi(inicial):
    # por eso el shaping no cambia cuál es la política óptima.
    env = PotentialShaping(KitchenEnv(window=WINDOW, kitchen="medium"), gamma=1.0)
    raw = KitchenEnv(window=WINDOW, kitchen="medium")
    env.reset(seed=3)
    raw.reset(seed=3)
    start = env.potential
    policy = FifoHeuristicPolicy(WINDOW)
    obs = raw._get_obs()
    added = 0.0
    for _ in range(120):
        action = policy(obs)
        _, shaped, _, _, _ = env.step(action)
        obs, reward, _, _, _ = raw.step(action)
        added += shaped - reward
    assert added == pytest.approx(env.potential - start)


@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_training_is_reproducible_and_logs(algo, tmp_path):
    first = train_tiny(algo, tmp_path / "a")
    second = train_tiny(algo, tmp_path / "b")
    for a, b in zip(weights(first), weights(second)):
        assert np.array_equal(a, b)
    evals = pd.read_csv(tmp_path / "a" / "evals.csv")
    assert {"step", "return", "on_time_rate"} <= set(evals.columns) and len(evals) >= 2
    assert (tmp_path / "a" / "done.json").exists() and (tmp_path / "a" / "progress.csv").exists()


@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_saved_model_matches_trained_model(algo, tmp_path):
    model = train_tiny(algo, tmp_path)
    loaded = load_policy(algo, tmp_path / "model.zip")
    a = evaluate(ModelPolicy(model), 2, env_kwargs={"window": WINDOW})["return"].to_numpy()
    b = evaluate(loaded, 2, env_kwargs={"window": WINDOW})["return"].to_numpy()
    assert np.array_equal(a, b)


def test_replay_export_is_deterministic_and_loadable(tmp_path):
    model = train_tiny("ppo", tmp_path)
    paths = []
    for name in ("one", "two"):
        env = KitchenEnv(window=WINDOW, kitchen="medium")
        paths.append(record_episode(env, ModelPolicy(model), 10_000, tmp_path / f"{name}.jsonl",
                                    meta={"policy": "ppo", "label": "PPO"}))
    assert paths[0].read_bytes() == paths[1].read_bytes()
    lines = [json.loads(x) for x in paths[0].read_text(encoding="utf-8").splitlines()]
    assert lines[0]["kind"] == "header" and {"seed", "kitchen", "window", "env_params"} <= set(lines[0])
    assert all({"t", "action", "reward", "obs", "visual_cook", "info"} <= set(x) for x in lines[1:])
    assert {"true_cook", "queue", "events"} <= set(lines[1]["info"])
    assert len(load_replay(paths[0]).steps) == 201


def _fake_final() -> pd.DataFrame:
    rows = []
    for algo, bonus in (("dqn", 0.0), ("ppo", 5.0)):
        for window in (1, 2):
            for seed in range(2):
                for split, scenario in (("validation", "random"), ("test", "random")):
                    rows.append({"algo": algo, "window": window, "train_seed": seed, "split": split,
                                 "scenario": scenario, "seed": 1, "return": bonus + window * 10.0 + seed})
    return pd.DataFrame(rows)


def test_best_window_uses_validation_only():
    final = _fake_final()
    # Si en prueba N = 1 fuera mejor, la selección no debe enterarse
    final.loc[(final["split"] == "test") & (final["window"] == 1), "return"] = 1000.0
    assert analyze.best_windows(final) == {"dqn": 2, "ppo": 2}


def test_representative_seed_is_closest_to_median():
    final = _fake_final().assign(window=4)
    final.loc[final["train_seed"] == 1, "return"] += 100
    assert export_replays.representative_seed(final, "dqn", 4) in (0, 1)


def test_bootstrap_ci_contains_mean():
    low, high = analyze.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert low <= 3.0 <= high


def test_full_pipeline_from_runs(tmp_path, monkeypatch):
    """De corridas crudas a tablas y figuras, con corridas diminutas."""
    runs = tmp_path / "runs"
    for algo in ("dqn", "ppo"):
        for window in (1, 2):
            for seed in (0, 1):
                run = runs / f"{algo}_N{window}_seed{seed}"
                model = train_run(algo, window, seed, tiny_config(algo), run, 2000, eval_every=800,
                                  eval_episodes=1)
                final_evaluation(ModelPolicy(model), window, episodes=2).assign(
                    algo=algo, window=window, train_seed=seed).to_csv(run / "final_eval.csv", index=False)
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    monkeypatch.setattr(analyze, "DEFAULT_WINDOW", 2)
    result = analyze.run_analysis(runs, tables, figures, eval_episodes=1)
    assert set(result["windows"]) == {"dqn", "ppo"}
    assert len(result["main_table"]) == 4
    for name in ("main_table", "by_window", "robustness", "n1_vs_fifo", "taste_by_context", "sample_efficiency"):
        assert (tables / f"{name}.csv").exists() and (tables / f"{name}.md").exists()
    for name in ("training_curves", "training_internals", "performance_vs_n", "behavior_vs_n", "robustness"):
        assert (figures / f"{name}.png").exists()
    comparison = report_assets.build(runs, tables, figures, tmp_path / "docs", eval_episodes=1)
    assert len(comparison) == 4 and (comparison["largo_episodio"] == 200).all()
    assert (tables / "comparison.md").exists() and (tmp_path / "docs" / "dqn_vs_ppo_reward.png").exists()
    torch.use_deterministic_algorithms(False)
