"""Tests de los baselines (docs/env_design.md, sección 4.3)."""

from agents.baselines import EdfHeuristicPolicy, FifoHeuristicPolicy, RandomPolicy
from envs.kitchen_env import KitchenEnv


def play(env: KitchenEnv, policy, seed: int) -> tuple[float, list[dict]]:
    """Corre un episodio y devuelve (retorno, todos los eventos)."""
    obs, info = env.reset(seed=seed)
    total, events = 0.0, list(info["events"])
    while True:
        obs, reward, _, truncated, info = env.step(policy(obs))
        total += reward
        events += info["events"]
        if truncated:
            return total, events


def count(events, kind):
    return sum(e["type"] == kind for e in events)


def test_fifo_never_takes_invalid_actions():
    for window in (1, 4):
        env = KitchenEnv(window=window, kitchen="random")
        for seed in range(5):
            _, events = play(env, FifoHeuristicPolicy(window), seed)
            assert count(events, "invalid") == 0


def test_edf_never_takes_invalid_actions():
    for window in (1, 4):
        env = KitchenEnv(window=window, kitchen="random")
        for seed in range(5):
            _, events = play(env, EdfHeuristicPolicy(window), seed)
            assert count(events, "invalid") == 0


def test_fifo_only_touches_slot_zero():
    env = KitchenEnv(window=4, kitchen="random")
    for seed in range(3):
        _, events = play(env, FifoHeuristicPolicy(4), seed)
        slots = {e["slot"] for e in events if e["type"] in ("prepped", "placed")}
        assert slots == {0}


def test_edf_uses_other_slots():
    env = KitchenEnv(window=4, kitchen="hard")
    slots = set()
    for seed in range(3):
        _, events = play(env, EdfHeuristicPolicy(4), seed)
        slots |= {e["slot"] for e in events if e["type"] in ("prepped", "placed")}
    assert slots - {0}


def test_fifo_without_tasting_never_tastes():
    env = KitchenEnv(kitchen="medium")
    for seed in range(3):
        _, events = play(env, FifoHeuristicPolicy(4, retaste_after=None), seed)
        assert count(events, "tasted") == 0
        assert count(events, "served_ok") + count(events, "served_late") > 0


def test_fifo_beats_random():
    # Umbral provisional: el criterio definitivo sale del Paso 0 (sección 4.2).
    env = KitchenEnv(kitchen="medium")
    random_policy = RandomPolicy(env.action_space.n, seed=0)
    fifo, rand = [], []
    for seed in range(10):
        fifo.append(play(env, FifoHeuristicPolicy(4), seed)[0])
        rand.append(play(env, random_policy, seed)[0])
    assert sum(fifo) / 10 > sum(rand) / 10 + 50
