"""Tests de los baselines y del evaluador común."""

import gymnasium as gym
import numpy as np

import envs  # noqa: F401  (registra HellKitchen-v0)
from agents.baselines import HeuristicPolicy, RandomPolicy
from experiments.evaluate import evaluate


def test_heuristic_beats_random():
    # Criterio de calibración de T3 (docs/env_design.md, sección 8)
    n_actions = gym.make("HellKitchen-v0").action_space.n
    random_df = evaluate(RandomPolicy(n_actions, seed=0), n_episodes=10)
    heuristic_df = evaluate(HeuristicPolicy(), n_episodes=10)
    assert heuristic_df["return"].mean() > random_df["return"].mean() + 100
    assert heuristic_df["on_time_rate"].mean() > 0.5


def test_heuristic_never_takes_invalid_actions():
    df = evaluate(HeuristicPolicy(), n_episodes=5)
    assert (df["invalid"] == 0).all()


def test_evaluation_is_reproducible():
    a = evaluate(HeuristicPolicy(), n_episodes=3)
    b = evaluate(HeuristicPolicy(), n_episodes=3)
    assert np.allclose(a["return"], b["return"])
