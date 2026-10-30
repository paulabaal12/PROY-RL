"""Tests de la línea de tiempo del visualizador (viz/timeline.py).

Cada test arma un episodio con estado controlado (como en tests/test_env.py),
ejecuta una lista de acciones y revisa qué animación produce cada paso.
"""

import json

from envs.kitchen_env import OP_DISCARD, OP_SERVE, OP_TASTE, KitchenEnv, Order, Station
from viz.record import _plain
from viz.replay import Replay, Step
from viz.timeline import build_timeline

# Cocina "medium" (2 estaciones) sin llegadas ni horas pico
QUIET = {"arrival_prob": 0.0, "n_rushes": 0}


def scripted(setup, actions, window=2):
    """Replay en memoria: reset, `setup(env)` para fijar el estado, luego `actions`."""
    env = KitchenEnv(window=window, kitchen=QUIET)
    env.reset(seed=0)
    env.queue.clear()
    setup(env)
    plain = lambda info: json.loads(json.dumps(_plain(info)))  # noqa: E731
    steps = [Step(0, None, None, None, 0.0, plain(env._get_info()))]
    for action in actions:
        _, reward, _, _, info = env.step(action)
        kind, index = env.layout.decode(action)
        steps.append(Step(info["t"], action, kind, index, float(reward), plain(info)))
    header = {"window": window, "max_stations": 3, "seed": 0, "kitchen": QUIET,
              "env_params": {"horizon": env.horizon, "patience": env.patience, "queue_max": 10}}
    return env, build_timeline(Replay(header, steps))


def put_order(env, dish=2, prep_done=0, age=0, due=30):
    order = Order(order_id=env.next_order_id, dish=dish, due=due, rate=0.0, age=age,
                  prep_done=prep_done)
    env.next_order_id += 1
    env.queue.append(order)
    return order


def put_dish(env, k, cook, dish=1, age=0, due=30):
    order = Order(order_id=env.next_order_id, dish=dish, due=due, rate=0.0, age=age)
    env.next_order_id += 1
    env.stations[k] = Station(order=order, cook=cook)
    return order


def act(env, k, op):
    return env.layout.station(k, op)


L = KitchenEnv(window=2).layout


def test_place_path_goes_through_prep_with_board():
    ids = []
    _, (snaps, scripts) = scripted(lambda env: ids.append(put_order(env, dish=2, prep_done=1).order_id),
                                   [L.place(0)])
    s = scripts[1]
    assert s.path == ["center", "prep", "burner0"]
    assert s.carry == "board" and s.anim == "place" and s.dish == "steak"
    assert s.focus_order == ids[0] and s.burner == 0
    assert snaps[1].burners[0].order_id == ids[0] and snaps[1].chef_at == "burner0"


def test_taste_burnt_reveals_burner_until_it_empties():
    _, (snaps, scripts) = scripted(lambda env: put_dish(env, 0, cook=1.5),
                                   [L.station(0, OP_TASTE), L.station(0, OP_DISCARD)])
    assert scripts[1].anim == "taste_burnt"
    assert snaps[1].burners[0].revealed_burnt
    assert scripts[2].anim == "discard" and scripts[2].carry_color == "burnt"
    assert scripts[2].path == ["burner0", "trash"]
    assert not snaps[2].burners[0].revealed_burnt and snaps[2].burners[0].order_id is None


def test_taste_raw_and_ok():
    _, (_, scripts) = scripted(lambda env: (put_dish(env, 0, cook=0.2), put_dish(env, 1, cook=1.1)),
                               [L.station(0, OP_TASTE), L.station(1, OP_TASTE)])
    assert scripts[1].anim == "taste_raw" and scripts[2].anim == "taste_ok"
    assert scripts[2].path == ["burner0", "burner1"]


def test_serve_late_flashes_orange_and_on_time_none():
    _, (_, scripts) = scripted(lambda env: (put_dish(env, 0, cook=1.1, age=35, due=30),
                                            put_dish(env, 1, cook=1.1)),
                               [L.station(0, OP_SERVE), L.station(1, OP_SERVE)])
    assert scripts[1].anim == "serve_late" and scripts[1].flash == "orange"
    assert scripts[1].path == ["center", "burner0", "serve"] and scripts[1].carry == "plate"
    assert scripts[2].anim == "serve_ok" and scripts[2].flash is None
    assert scripts[2].carry_color == "egg" and scripts[2].reward == 10.0


def test_serve_burnt_carries_black_and_flashes_red():
    _, (_, scripts) = scripted(lambda env: put_dish(env, 0, cook=1.5), [L.station(0, OP_SERVE)])
    assert scripts[1].anim == "serve_bad" and scripts[1].flash == "red"
    assert scripts[1].carry_color == "burnt"


def test_expired_on_burner_sets_expired_burners_and_red_wins_over_orange():
    ids = []

    def setup(env):
        put_dish(env, 0, cook=1.1, age=35, due=30)
        ids.append(put_dish(env, 1, cook=0.5, age=40, due=30).order_id)

    _, (snaps, scripts) = scripted(setup, [L.station(0, OP_SERVE)])
    assert scripts[1].anim == "serve_late"
    assert scripts[1].flash == "red"
    assert scripts[1].expired == ids and scripts[1].expired_burners == [1]
    assert snaps[1].burners[1].order_id is None


def test_wait_walks_to_center_and_invalid_stays():
    _, (_, scripts) = scripted(lambda env: put_order(env, dish=0),
                               [L.prep(0), L.WAIT, L.station(0, OP_TASTE)])
    assert scripts[1].path == ["center", "prep"] and scripts[1].anim == "prep"
    assert scripts[2].path == ["prep", "center"] and scripts[2].anim == "wait"
    assert scripts[2].focus_order is None
    assert scripts[3].anim == "invalid" and scripts[3].path == ["center"]


def test_action_on_unavailable_burner_is_invalid_shrug():
    _, (snaps, scripts) = scripted(lambda env: None, [L.station(2, OP_TASTE)])
    assert scripts[1].anim == "invalid" and scripts[1].path == ["center"]
    assert [b.available for b in snaps[0].burners] == [True, True, False]


def test_score_is_cumulative_reward():
    _, (snaps, _) = scripted(lambda env: (put_dish(env, 0, cook=1.1), put_dish(env, 1, cook=1.1)),
                             [L.station(0, OP_TASTE), L.station(0, OP_SERVE), L.WAIT])
    assert snaps[0].score == 0.0
    assert abs(snaps[1].score - (-0.2)) < 1e-9
    assert abs(snaps[3].score - 9.8) < 1e-9


def test_in_window_false_beyond_window():
    _, (snaps, _) = scripted(lambda env: (put_order(env), put_order(env)), [L.WAIT], window=1)
    assert [o.in_window for o in snaps[0].orders] == [True, False]


def test_burner_orders_listed_with_burner_index():
    _, (snaps, _) = scripted(lambda env: (put_order(env), put_dish(env, 1, cook=0.3)), [L.WAIT])
    assert [o.burner for o in snaps[0].orders] == [None, 1]
    assert snaps[0].orders[1].in_window


def test_samples_build_full_timelines():
    from pathlib import Path

    from viz.replay import load_replay
    for path in sorted(Path("results/replays").glob("sample_*.jsonl")):
        snaps, scripts = build_timeline(load_replay(path))
        assert len(snaps) == len(scripts) == 201
