"""Tests del ambiente HellKitchen-v1 (docs/env_design.md, v2).

Los tests de casos controlados ponen el estado oculto a mano (órdenes en la
cola, un plato con un nivel de cocción exacto) y verifican que la recompensa y
los eventos sean los que dice la especificación.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401  (registra HellKitchen-v1)
from agents.baselines import RandomPolicy
from envs.kitchen_env import (
    OP_DISCARD, OP_SERVE, OP_TASTE, SL_PRESENT, SLOT_FEATURES, ST_AVAILABLE, STATION_FEATURES, KitchenEnv, Order, Station,
)

# Cocina "medium" sin llegadas ni horas pico: el estado solo cambia por lo que hace el test.
QUIET = {"arrival_prob": 0.0, "n_rushes": 0}


def make_env(**kwargs) -> KitchenEnv:
    kwargs.setdefault("kitchen", QUIET)
    env = KitchenEnv(**kwargs)
    env.reset(seed=0)
    return env


def put_order(env: KitchenEnv, dish: int = 2, prep_done: int = 0, age: int = 0,
              due: int = 30, rate: float = 0.1) -> Order:
    """Agrega a mano una orden al final de la cola."""
    order = Order(order_id=env.next_order_id, dish=dish, due=due, rate=rate, age=age,
                  prep_done=prep_done)
    env.next_order_id += 1
    env.queue.append(order)
    return order


def put_dish(env: KitchenEnv, k: int, cook: float, dish: int = 1, rate: float = 0.0,
             age: int = 0, due: int = 30) -> Order:
    """Coloca a mano un plato en la estación k con un nivel de cocción exacto."""
    order = Order(order_id=env.next_order_id, dish=dish, due=due, rate=rate, age=age,
                  prep_done=0)
    env.next_order_id += 1
    env.stations[k] = Station(order=order, cook=cook)
    return order


def event_types(info) -> list[str]:
    return [e["type"] for e in info["events"]]


def slot_base(env: KitchenEnv, i: int) -> int:
    return STATION_FEATURES * env.max_stations + SLOT_FEATURES * i


# --- API de Gymnasium -------------------------------------------------------

def test_passes_gymnasium_env_checker():
    check_env(gym.make("HellKitchen-v1").unwrapped, skip_render_check=True)


def test_spaces_scale_with_window():
    for window, n_actions, obs_size in ((1, 12, 45), (4, 18, 66), (6, 22, 80)):
        env = KitchenEnv(window=window)
        assert env.action_space.n == n_actions
        assert env.observation_space.shape == (obs_size,)


def test_observations_stay_inside_space():
    for window in (1, 4, 6):
        env = KitchenEnv(window=window, kitchen="random")
        for seed in range(3):
            policy = RandomPolicy(env.action_space.n, seed=seed)
            obs, _ = env.reset(seed=seed)
            for _ in range(env.horizon):
                assert env.observation_space.contains(obs)
                obs, *_ = env.step(policy(obs))


# --- Estado inicial, cola y ventana -----------------------------------------

def test_initial_state():
    env = make_env()
    assert all(not st.occupied for st in env.stations)
    assert len(env.queue) == 1
    assert env.queue[0].age == 0


def test_prep_advances_one_step_and_is_invalid_when_done():
    env = make_env()
    env.queue = []
    order = put_order(env, dish=2)              # carne: 1 paso de preparación
    _, reward, _, _, info = env.step(env.layout.prep(0))
    assert order.prep_done == 1
    assert "prepped" in event_types(info)
    assert info["events"][0]["slot"] == 0
    _, reward, _, _, info = env.step(env.layout.prep(0))
    assert event_types(info) == ["invalid"]
    assert reward == -0.1


def test_place_requires_prepped_and_free_station():
    env = make_env()
    env.queue = []
    order = put_order(env, dish=1, prep_done=3)     # huevo: 4 pasos de preparación
    _, _, _, _, info = env.step(env.layout.place(0))
    assert event_types(info) == ["invalid"]
    order.prep_done = 4
    _, _, _, _, info = env.step(env.layout.place(0))
    assert "placed" in event_types(info)
    assert env.queue == []
    assert env.stations[0].order is order
    assert env.stations[0].cook_steps == 1          # ya cocinó el paso en que se puso

    env2 = make_env()
    env2.queue = []
    put_dish(env2, 0, cook=0.2)
    put_dish(env2, 1, cook=0.2)                     # K = 2: todas las disponibles ocupadas
    put_order(env2, dish=2, prep_done=1)
    _, _, _, _, info = env2.step(env2.layout.place(0))
    assert event_types(info) == ["invalid"]


def test_place_uses_lowest_free_available_station():
    env = make_env()
    env.queue = []
    put_dish(env, 0, cook=0.2)
    order = put_order(env, dish=2, prep_done=1)
    env.step(env.layout.place(0))
    assert env.stations[1].order is order


def test_window_bigger_than_queue():
    env = make_env(window=4)
    env.queue = []
    put_order(env)
    obs = env._get_obs()
    assert obs[slot_base(env, 0) + SL_PRESENT] == 1.0
    for i in (1, 2, 3):
        base = slot_base(env, i)
        assert np.all(obs[base:base + SLOT_FEATURES] == 0.0)
    for action in (env.layout.prep(2), env.layout.place(3)):
        _, reward, _, _, info = env.step(action)
        assert event_types(info) == ["invalid"]
        assert reward == -0.1


def test_orders_outside_window_are_counted_in_backlog():
    env = make_env(window=1)
    env.queue = []
    for _ in range(3):
        put_order(env)
    obs = env._get_obs()
    assert obs[-1] == np.float32(2 / env.queue_max)


def test_queue_is_in_arrival_order():
    env = KitchenEnv(kitchen={"arrival_prob": 1.0, "n_rushes": 0})
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.layout.WAIT)
    ids = [o.order_id for o in env.queue]
    assert ids == sorted(ids) and len(ids) == 6


def test_station_availability_follows_kitchen():
    env = make_env()                               # K = 2 de 3
    obs = env._get_obs()
    assert [obs[k * STATION_FEATURES + ST_AVAILABLE] for k in range(3)] == [1.0, 1.0, 0.0]


# --- Determinismo -----------------------------------------------------------

def _obs_rollout(seed: int, n_steps: int = 50):
    env = KitchenEnv(kitchen="random")
    policy = RandomPolicy(env.action_space.n, seed=123)
    obs, _ = env.reset(seed=seed)
    trajectory = [obs]
    for _ in range(n_steps):
        obs, *_ = env.step(policy(obs))
        trajectory.append(obs)
    return np.array(trajectory)


def test_same_seed_is_deterministic():
    assert np.array_equal(_obs_rollout(5), _obs_rollout(5))


def test_different_seeds_differ():
    assert not np.array_equal(_obs_rollout(5), _obs_rollout(6))


def _arrivals(seed: int, policy) -> list:
    # Cola sin límite práctico: con la cola llena la orden no llega (sección 3.3),
    # y eso sí depende del agente. Lo que no debe depender es la secuencia de órdenes.
    env = KitchenEnv(kitchen="medium", horizon=60, queue_max=1000)
    obs, info = env.reset(seed=seed)
    arrived = []
    while True:
        arrived += [(e["order"]["id"], e["order"]["dish"], e["order"]["due"], e["order"]["rate"])
                    for e in info["events"] if e["type"] == "arrived"]
        obs, _, _, truncated, info = env.step(policy(obs))
        if truncated:
            return arrived


def test_order_stream_and_rates_do_not_depend_on_actions():
    env = KitchenEnv()
    random_policy = RandomPolicy(env.action_space.n, seed=0)
    for seed in range(3):
        assert _arrivals(seed, lambda obs: 0) == _arrivals(seed, random_policy)


def test_kitchen_is_drawn_per_episode():
    env = KitchenEnv(kitchen="random")
    sigmas = set()
    for seed in range(20):
        env.reset(seed=seed)
        sigmas.add(env.params.sigma_obs)
    assert len(sigmas) > 1
    env.reset(seed=3)
    first = env.params
    env.reset(seed=3)
    assert env.params == first


# --- Cocción, probar, servir y descartar (Task 3) ---------------------------

def taste(env, k):
    return env.layout.station(k, OP_TASTE)


def serve(env, k):
    return env.layout.station(k, OP_SERVE)


def discard(env, k):
    return env.layout.station(k, OP_DISCARD)


def test_cooking_advances_every_step():
    env = make_env(sigma_proc=0.0)
    put_dish(env, 0, cook=0.0, rate=0.1)
    for _ in range(3):
        env.step(env.layout.WAIT)
    assert abs(env.stations[0].cook - 0.3) < 1e-9
    assert env.stations[0].cook_steps == 3


def test_taste_costs_and_is_precise():
    env = make_env()
    env.queue = []
    for _ in range(50):
        put_dish(env, 0, cook=0.8, rate=0.0)
        _, reward, _, _, info = env.step(taste(env, 0))
        assert abs(reward - (-0.2)) < 1e-9
        assert "tasted" in event_types(info)
        st = env.stations[0]
        assert st.has_tasted and st.steps_since_taste == 0
        assert abs(st.last_taste - st.cook) < 0.1


def test_serve_on_time_gives_flat_reward():
    env = make_env()
    put_dish(env, 0, cook=1.1, age=30, due=30)
    _, reward, _, _, info = env.step(serve(env, 0))
    assert reward == 10.0
    assert "served_ok" in event_types(info)
    assert not env.stations[0].occupied


@pytest.mark.parametrize("delay, expected", [(1, 2.5), (5, 1.5), (10, 0.25)])
def test_late_reward_boundaries(delay, expected):
    env = make_env()
    put_dish(env, 0, cook=1.1, age=30 + delay, due=30)
    _, reward, _, _, info = env.step(serve(env, 0))
    assert abs(reward - expected) < 1e-9
    served = [e for e in info["events"] if e["type"] == "served_late"]
    assert served and served[0]["delay"] == delay


def test_order_expires_after_patience():
    env = make_env()
    env.queue = []
    put_order(env, age=40, due=30)                  # d = P = 10: último paso con paciencia
    _, reward, _, _, info = env.step(env.layout.WAIT)
    assert reward == -8.0
    assert "expired" in event_types(info)
    assert env.queue == []


def test_expire_while_cooking_frees_station():
    env = make_env()
    env.queue = []
    put_dish(env, 0, cook=0.5, age=40, due=30)
    _, reward, _, _, info = env.step(env.layout.WAIT)
    assert reward == -8.0
    assert not env.stations[0].occupied
    _, reward, _, _, info = env.step(serve(env, 0))
    assert event_types(info) == ["invalid"]


@pytest.mark.parametrize("cook, quality", [(0.5, "raw"), (1.4, "burnt")])
def test_serve_bad_food(cook, quality):
    env = make_env()
    put_dish(env, 0, cook=cook)
    _, reward, _, _, info = env.step(serve(env, 0))
    assert reward == -10.0
    assert f"served_{quality}" in event_types(info)


def test_serve_uses_cook_before_this_step():
    env = make_env()
    put_dish(env, 0, cook=1.29, rate=0.1)
    _, _, _, _, info = env.step(serve(env, 0))
    assert "served_ok" in event_types(info)


def test_discard_returns_order_needing_prep_again():
    env = make_env()
    env.queue = []
    order = put_dish(env, 0, cook=1.5, dish=1, rate=0.11, age=12, due=25)
    order.prep_done = 4
    later = put_order(env)                          # llegó después: va detrás
    env.step(discard(env, 0))
    assert env.queue[0] is order and env.queue[1] is later
    assert order.prep_done == 0
    assert order.due == 25 and order.rate == 0.11
    assert order.age == 13                          # envejeció el paso del discard
    assert not env.stations[0].occupied


def test_discard_with_full_queue():
    env = make_env()
    env.queue = []
    put_dish(env, 0, cook=1.5)
    for _ in range(env.queue_max):
        put_order(env)
    obs, *_ = env.step(discard(env, 0))
    assert len(env.queue) == env.queue_max + 1
    assert env.observation_space.contains(obs)


def test_actions_on_missing_stations_are_invalid():
    env = make_env(kitchen={"n_stations": 2, "arrival_prob": 0.0, "n_rushes": 0})
    for op in (OP_TASTE, OP_SERVE, OP_DISCARD):
        obs, reward, _, _, info = env.step(env.layout.station(2, op))
        assert reward == -0.1
        assert event_types(info) == ["invalid"]
    assert obs[2 * STATION_FEATURES + ST_AVAILABLE] == 0.0


def test_truncates_at_horizon_and_never_terminates():
    env = make_env(horizon=30)
    for t in range(30):
        _, _, terminated, truncated, _ = env.step(env.layout.WAIT)
        assert not terminated
        assert truncated == (t == 29)


def test_rush_raises_arrivals():
    env = KitchenEnv(kitchen={"arrival_prob": 0.1, "n_rushes": 2}, queue_max=1000)
    inside = outside = steps_in = steps_out = 0
    for seed in range(100):
        env.reset(seed=seed)
        rushes = env.params.rushes
        for t in range(env.horizon):
            _, _, _, _, info = env.step(env.layout.WAIT)
            arrived = sum(e["type"] == "arrived" for e in info["events"])
            if any(s <= t < e for s, e in rushes):
                inside, steps_in = inside + arrived, steps_in + 1
            else:
                outside, steps_out = outside + arrived, steps_out + 1
    assert inside / steps_in > 2 * outside / steps_out
