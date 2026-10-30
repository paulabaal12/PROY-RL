"""Tests del evaluador común (docs/env_design.md, sección 5)."""

import numpy as np

from agents.baselines import FifoHeuristicPolicy, RandomPolicy
from envs.kitchen_config import PRESETS
from envs.kitchen_env import OP_SERVE, OP_TASTE, KitchenEnv
from experiments.evaluate import (
    TEST_BASE_SEED, VALIDATION_BASE_SEED, evaluate, evaluate_scenarios, run_episode, summarize,
)


class OraclePolicy:
    """Política de test que mira el estado oculto: prepara, pone al fuego, prueba
    una vez justo cuando el plato va a quedar listo y lo sirve."""

    def __init__(self, env: KitchenEnv):
        self.env = env

    def __call__(self, obs):
        env, layout = self.env, self.env.layout
        st = env.stations[0]
        if st.occupied:
            if st.has_tasted:
                return layout.station(0, OP_SERVE)
            if 1.05 <= st.cook + st.order.rate < 1.25:
                return layout.station(0, OP_TASTE)
            return layout.WAIT
        if env.queue:
            return layout.place(0) if env.queue[0].is_prepped else layout.prep(0)
        return layout.WAIT


def test_metrics_on_known_episode():
    env = KitchenEnv(kitchen={"arrival_prob": 0.0, "n_rushes": 0}, horizon=60, sigma_proc=0.0)
    row = run_episode(env, OraclePolicy(env), seed=0)
    assert row["served_ok"] == 1 and row["expired"] == 0
    assert row["on_time_rate"] == 1.0
    assert row["tastes_per_order"] == 1.0
    assert row["prioritization_rate"] == 0.0
    assert row["steps"] == 60
    assert np.isnan(row["mean_delay"])
    # Solo hubo una orden: después de servirla todo es esperar sin nada que preparar
    assert row["avoidable_idle_rate"] == 0.0
    assert row["idle_rate"] > 0.5
    assert row["n_stations"] == 2


def test_evaluation_is_reproducible():
    a = evaluate(FifoHeuristicPolicy(4), n_episodes=3)
    b = evaluate(FifoHeuristicPolicy(4), n_episodes=3)
    assert np.allclose(a["return"], b["return"])


def test_validation_and_test_seeds_differ():
    a = evaluate(FifoHeuristicPolicy(4), n_episodes=3, base_seed=TEST_BASE_SEED)
    b = evaluate(FifoHeuristicPolicy(4), n_episodes=3, base_seed=VALIDATION_BASE_SEED)
    assert set(a["seed"]).isdisjoint(b["seed"])
    assert not np.allclose(a["return"], b["return"])


def test_evaluate_scenarios_has_one_block_per_scenario():
    scenarios = ["random", *PRESETS]
    df = evaluate_scenarios(lambda env: RandomPolicy(env.action_space.n, seed=0), scenarios,
                            n_episodes=2)
    assert list(df["scenario"].unique()) == scenarios
    assert (df.groupby("scenario").size() == 2).all()
    single = df[df["scenario"] == "single_station"]
    assert (single["n_stations"] == 1).all()


def test_summarize_reports_mean_and_std():
    df = evaluate(FifoHeuristicPolicy(4), n_episodes=3)
    summary = summarize({"fifo": df})
    assert summary.loc[0, "policy"] == "fifo"
    assert "±" in summary.loc[0, "return"]
