"""Tests de reproducibilidad de las utilidades de seeds.
"""

import random

import gymnasium as gym
import numpy as np
import torch

from utils.seeding import seed_env, set_global_seed


def _draw_global_numbers():
    """Saca un número de cada generador global."""
    return (random.random(), np.random.rand(), torch.rand(1).item())


def test_same_seed_gives_same_numbers():
    set_global_seed(123)
    first = _draw_global_numbers()
    set_global_seed(123)
    second = _draw_global_numbers()
    assert first == second


def test_different_seeds_give_different_numbers():
    set_global_seed(1)
    first = _draw_global_numbers()
    set_global_seed(2)
    second = _draw_global_numbers()
    assert first != second


def test_same_seed_gives_same_network_init():
    # Importante para DQN/PPO: misma seed -> mismos pesos iniciales.
    set_global_seed(7)
    first = torch.nn.Linear(4, 2).weight.clone()
    set_global_seed(7)
    second = torch.nn.Linear(4, 2).weight.clone()
    assert torch.equal(first, second)


def _rollout(seed: int, n_steps: int = 50):
    """Corre una política aleatoria y devuelve la trayectoria de observaciones."""
    env = gym.make("CartPole-v1")
    obs, _ = seed_env(env, seed)
    trajectory = [obs]
    for _ in range(n_steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        trajectory.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()
            trajectory.append(obs)
    env.close()
    return np.array(trajectory)


def test_env_rollout_is_reproducible():
    assert np.array_equal(_rollout(seed=42), _rollout(seed=42))


def test_env_rollout_depends_on_seed():
    assert not np.array_equal(_rollout(seed=42), _rollout(seed=43))
