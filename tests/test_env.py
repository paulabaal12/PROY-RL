"""Tests del ambiente HellKitchen-v0 (docs/env_design.md).

Los tests de casos controlados ponen el estado oculto a mano (por ejemplo, un
nivel de cocción exacto) y verifican que la recompensa y los eventos sean los
que dice la especificación.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401  (registra HellKitchen-v0)
from agents.baselines import HeuristicPolicy, RandomPolicy
from envs.kitchen_env import (
    OP_DISCARD, OP_SERVE, OP_START, OP_TASTE, ST_LAST_TASTE, ST_VISUAL,
    STATION_FEATURES,
    WAIT, KitchenEnv, Order, Station, action_id, classify,
)


def make_env(**kwargs) -> KitchenEnv:
    return KitchenEnv(**kwargs)


def put_dish(env: KitchenEnv, k: int, cook: float, rate: float = 0.0,
             time_left: int = 20, deadline: int = 20, dish: int = 0):
    """Coloca a mano un plato en la estación k con un nivel de cocción exacto."""
    order = Order(order_id=99, dish=dish, deadline=deadline, time_left=time_left)
    env.stations[k] = Station(order=order, cook=cook, rate=rate)


def event_types(info) -> list[str]:
    return [e["type"] for e in info["events"]]


# --- API de Gymnasium -------------------------------------------------------

def test_passes_gymnasium_env_checker():
    check_env(gym.make("HellKitchen-v0").unwrapped, skip_render_check=True)


def test_observations_stay_inside_observation_space():
    env = make_env()
    policy = RandomPolicy(env.action_space.n, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (31,)
    for _ in range(env.horizon):
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(policy(obs))


def test_truncates_at_horizon_and_never_terminates():
    env = make_env(horizon=30)
    env.reset(seed=0)
    for t in range(30):
        _, _, terminated, truncated, _ = env.step(WAIT)
        assert not terminated
        assert truncated == (t == 29)


# --- Reproducibilidad ---------------------------------------------------------

def _rollout(seed, policy_seed=0):
    env = make_env()
    policy = RandomPolicy(env.action_space.n, seed=policy_seed)
    obs, info = env.reset(seed=seed)
    observations, rewards, arrivals = [obs], [], []
    for _ in range(env.horizon):
        obs, reward, _, _, info = env.step(policy(obs))
        observations.append(obs)
        rewards.append(reward)
        arrivals += [(info["t"], e["dish"], e["deadline"]) for e in info["events"] if e["type"] == "arrived"]
    return np.array(observations), np.array(rewards), arrivals


def test_same_seed_is_deterministic():
    obs_a, rew_a, _ = _rollout(seed=5)
    obs_b, rew_b, _ = _rollout(seed=5)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(rew_a, rew_b)


def test_different_seeds_differ():
    obs_a, _, _ = _rollout(seed=5)
    obs_b, _, _ = _rollout(seed=6)
    assert not np.array_equal(obs_a, obs_b)


def test_order_stream_does_not_depend_on_actions():
    """Con la misma seed, políticas distintas reciben exactamente las mismas órdenes.

    Es lo que permite comparar DQN y PPO "en las mismas órdenes" (T8).
    Se usa una cola grande para que nunca se llene: si se llena, la orden
    se rechaza y eso sí depende de cómo jugó el agente.
    """
    def arrivals(policy):
        env = make_env(queue_size=50)
        obs, info = env.reset(seed=3)
        out = [(0, e["dish"], e["deadline"]) for e in info["events"]]
        for _ in range(env.horizon):
            obs, _, _, _, info = env.step(policy(obs))
            out += [(info["t"], e["dish"], e["deadline"]) for e in info["events"] if e["type"] == "arrived"]
        return out

    assert arrivals(RandomPolicy(9, seed=1)) == arrivals(HeuristicPolicy())


# --- Dinámica y recompensas (casos controlados) -------------------------------

def test_initial_state_has_one_order_and_empty_stations():
    env = make_env()
    _, info = env.reset(seed=0)
    assert len(info["queue"]) == 1
    assert info["stations"] == [None, None]


def test_start_moves_first_order_to_station():
    env = make_env(arrival_prob=0.0)
    _, info = env.reset(seed=0)
    first_id = info["queue"][0]["id"]
    _, reward, _, _, info = env.step(action_id(0, OP_START))
    assert reward == 0.0
    assert info["stations"][0]["id"] == first_id
    assert info["queue"] == []
    assert info["true_cook"][0] > 0  # ya se cocinó un paso


def test_serve_well_cooked_gives_reward_with_speed_bonus():
    env = make_env(arrival_prob=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 0, cook=1.15, time_left=10, deadline=20)
    _, reward, _, _, info = env.step(action_id(0, OP_SERVE))
    assert reward == pytest.approx(10.0 + 5.0 * 10 / 20)
    assert "served_ok" in event_types(info)
    assert info["stations"][0] is None


@pytest.mark.parametrize("cook, quality", [(0.7, "raw"), (1.45, "burnt")])
def test_serve_bad_food_is_penalized(cook, quality):
    env = make_env(arrival_prob=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 0, cook=cook)
    _, reward, _, _, info = env.step(action_id(0, OP_SERVE))
    assert reward == pytest.approx(-5.0)
    assert f"served_{quality}" in event_types(info)


def test_serve_uses_cook_level_before_this_step_cooking():
    # c = 1.28 está en la ventana; con r = 0.1 pasaría a 1.38 (quemado) después
    # de cocinar este paso. Servir debe evaluar el 1.28.
    env = make_env(arrival_prob=0.0, sigma_proc=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 0, cook=1.28, rate=0.1)
    _, _, _, _, info = env.step(action_id(0, OP_SERVE))
    assert "served_ok" in event_types(info)


def test_taste_costs_and_reads_true_level_with_small_noise():
    env = make_env(arrival_prob=0.0, sigma_proc=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 1, cook=0.9, rate=0.1)
    obs, reward, _, _, info = env.step(action_id(1, OP_TASTE))
    assert reward == pytest.approx(-0.2)
    true_cook = info["true_cook"][1]
    assert true_cook == pytest.approx(1.0)  # la prueba se mide después de cocinar
    assert obs[STATION_FEATURES + ST_LAST_TASTE] == pytest.approx(true_cook, abs=0.1)


def test_taste_is_much_more_precise_than_visual_observation():
    env = make_env(arrival_prob=0.0, sigma_proc=0.0)
    env.reset(seed=0)
    taste_err, visual_err = [], []
    for _ in range(300):
        env.queue = []
        put_dish(env, 0, cook=1.0, rate=0.0, time_left=50, deadline=50)
        obs, *_ = env.step(action_id(0, OP_TASTE))
        taste_err.append(obs[ST_LAST_TASTE] - 1.0)
        visual_err.append(obs[ST_VISUAL] - 1.0)
    assert np.std(taste_err) == pytest.approx(0.02, rel=0.25)
    assert np.std(visual_err) == pytest.approx(0.15, rel=0.25)


def test_discard_returns_order_to_front_of_queue():
    env = make_env(arrival_prob=0.0)
    env.reset(seed=0)
    waiting_id = env.queue[0].order_id
    put_dish(env, 0, cook=1.5, time_left=15)
    _, reward, _, _, info = env.step(action_id(0, OP_DISCARD))
    assert reward == 0.0
    assert info["stations"][0] is None
    assert [o["id"] for o in info["queue"]] == [99, waiting_id]
    assert info["queue"][0]["time_left"] == 14  # conserva su plazo (y corrió un paso)


def test_invalid_actions_act_as_wait_with_penalty():
    env = make_env(arrival_prob=0.0)
    env.reset(seed=0)
    for op in (OP_TASTE, OP_SERVE, OP_DISCARD):  # estación vacía
        _, reward, _, _, info = env.step(action_id(0, op))
        assert reward == pytest.approx(-0.1)
        assert event_types(info) == ["invalid"]
    env.queue = []
    _, reward, _, _, info = env.step(action_id(0, OP_START))  # cola vacía
    assert reward == pytest.approx(-0.1)


def test_order_expires_with_penalty_and_frees_station():
    env = make_env(arrival_prob=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 0, cook=0.5, rate=0.05, time_left=1)
    _, reward, _, _, info = env.step(WAIT)
    assert reward == pytest.approx(-5.0)
    assert "expired" in event_types(info)
    assert info["stations"][0] is None


def test_cooking_advances_while_agent_does_something_else():
    env = make_env(arrival_prob=0.0, sigma_proc=0.0)
    env.reset(seed=0)
    env.queue = []
    put_dish(env, 0, cook=0.0, rate=0.1, time_left=50, deadline=50)
    put_dish(env, 1, cook=0.0, rate=0.05, time_left=50, deadline=50)
    for _ in range(3):
        env.step(action_id(1, OP_TASTE))
    assert env.stations[0].cook == pytest.approx(0.3)
    assert env.stations[1].cook == pytest.approx(0.15)


def test_classify_thresholds():
    assert classify(0.99) == "raw"
    assert classify(1.0) == "ok"
    assert classify(1.29) == "ok"
    assert classify(1.3) == "burnt"


# --- Render ---------------------------------------------------------------

def test_ansi_render_returns_text():
    env = make_env(render_mode="ansi")
    env.reset(seed=0)
    env.step(action_id(0, OP_START))
    text = env.render()
    assert isinstance(text, str)
    assert "Estación 0" in text and "Cola" in text
