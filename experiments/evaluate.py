"""Evaluador común: mide cualquier política con las mismas seeds y las mismas métricas.

Todos los agentes (baselines, DQN y PPO) se evalúan con este archivo. El episodio i
usa la seed `base_seed + i`, y como la cocina y las órdenes salen de generadores
propios del ambiente, todos los agentes enfrentan exactamente la misma cocina y la
misma secuencia de órdenes.

Hay dos conjuntos de seeds (docs/env_design.md, sección 4.5): las de validación
(para tuning y para elegir el mejor N) y las de prueba (solo para el reporte).

Una "política" es cualquier función `obs -> acción`.

Uso:
    python -m experiments.evaluate --episodes 100 --window 4
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import envs  # noqa: F401  (registra HellKitchen-v1)
from agents.baselines import FifoHeuristicPolicy, RandomPolicy
from envs.kitchen_config import DISHES, PRESETS

TEST_BASE_SEED = 10_000        # seeds de prueba: solo para los números del reporte
VALIDATION_BASE_SEED = 20_000  # seeds de validación: tuning y selección de N
TABLES_DIR = Path("results/tables")
SCENARIOS = ["random", *PRESETS]

Policy = Callable[[np.ndarray], int]

EVENT_TYPES = ("arrived", "prepped", "placed", "tasted", "served_ok", "served_late",
               "served_raw", "served_burnt", "discarded", "expired", "invalid", "waited")


def _ratio(num: float, den: float) -> float:
    return num / den if den else np.nan


def run_episode(env: gym.Env, policy: Policy, seed: int) -> dict:
    """Corre un episodio completo y devuelve sus métricas (docs/env_design.md, sección 5)."""
    obs, info = env.reset(seed=seed)
    kitchen = info["kitchen"]
    counts = dict.fromkeys(EVENT_TYPES, 0)
    delays = []
    avoidable_waits = prioritized = prep_ahead = 0
    tastes_by_dish, served_by_dish = dict.fromkeys(DISHES, 0), dict.fromkeys(DISHES, 0)
    episode_return, steps = 0.0, 0
    events = info["events"]
    done = False
    while True:
        for event in events:
            kind = event["type"]
            counts[kind] += 1
            if kind == "served_late":
                delays.append(event["delay"])
            elif kind == "waited" and event["prep_available"]:
                avoidable_waits += 1
            elif kind in ("prepped", "placed") and event["slot"] > 0:
                prioritized += 1
            if kind == "prepped" and event["all_busy"]:
                prep_ahead += 1
            if kind == "tasted":
                tastes_by_dish[event["dish"]] += 1
            elif kind.startswith("served_"):
                served_by_dish[event["order"]["dish"]] += 1
        if done:
            break
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        episode_return += reward
        steps += 1
        events = info["events"]
        done = terminated or truncated

    good = counts["served_ok"] + counts["served_late"]
    bad = counts["served_raw"] + counts["served_burnt"]
    served = good + bad
    closed = served + counts["expired"]      # órdenes que ya se resolvieron (bien o mal)
    return {
        "seed": seed,
        "return": episode_return,
        **counts,
        "on_time_rate": _ratio(counts["served_ok"], closed),
        "late_rate": _ratio(counts["served_late"], closed),
        "expired_rate": _ratio(counts["expired"], closed),
        "bad_serve_rate": _ratio(bad, served),
        "mean_delay": np.mean(delays) if delays else np.nan,
        "tastes_per_order": _ratio(counts["tasted"], served),
        **{f"tastes_per_order_{dish}": _ratio(tastes_by_dish[dish], served_by_dish[dish]) for dish in DISHES},
        "idle_rate": _ratio(counts["waited"] + counts["invalid"], steps),
        "avoidable_idle_rate": _ratio(avoidable_waits, steps),
        "prioritization_rate": _ratio(prioritized, counts["prepped"] + counts["placed"]),
        "prep_ahead_rate": _ratio(prep_ahead, counts["prepped"]),
        "steps": steps,
        "n_stations": kitchen["n_stations"],
        "arrival_prob": kitchen["arrival_prob"],
        "sigma_obs": kitchen["sigma_obs"],
        "n_rushes": len(kitchen["rushes"]),
    }


def evaluate(policy: Policy, n_episodes: int = 100, base_seed: int = TEST_BASE_SEED,
             env_kwargs: dict | None = None) -> pd.DataFrame:
    """Evalúa una política en `n_episodes` episodios. Devuelve una fila por episodio."""
    env = gym.make("HellKitchen-v1", **(env_kwargs or {}))
    rows = [run_episode(env, policy, base_seed + i) for i in range(n_episodes)]
    env.close()
    return pd.DataFrame(rows)


def evaluate_scenarios(policy_factory: Callable[[gym.Env], Policy], scenarios: list[str] = SCENARIOS,
                       n_episodes: int = 100, base_seed: int = TEST_BASE_SEED,
                       env_kwargs: dict | None = None) -> pd.DataFrame:
    """Evalúa una política en cada escenario (sección 4.6). Agrega la columna `scenario`.

    `policy_factory` recibe el ambiente y devuelve la política, para las que
    necesitan conocer sus espacios (por ejemplo, la aleatoria).
    """
    frames = []
    for scenario in scenarios:
        kwargs = {**(env_kwargs or {}), "kitchen": scenario}
        policy = policy_factory(gym.make("HellKitchen-v1", **kwargs))
        frames.append(evaluate(policy, n_episodes, base_seed, kwargs).assign(scenario=scenario))
    return pd.concat(frames, ignore_index=True)


SUMMARY_METRICS = ["return", "on_time_rate", "late_rate", "expired_rate", "bad_serve_rate",
                   "tastes_per_order", "idle_rate", "avoidable_idle_rate",
                   "prioritization_rate", "prep_ahead_rate"]


def summarize(results: dict[str, pd.DataFrame], metrics: list[str] = SUMMARY_METRICS) -> pd.DataFrame:
    """Tabla resumen (media ± desviación estándar) de varias políticas.

    Si los resultados tienen columna `scenario`, se resume por escenario.
    """
    rows = []
    for name, df in results.items():
        groups = df.groupby("scenario", sort=False) if "scenario" in df else [(None, df)]
        for scenario, group in groups:
            row = {"policy": name} if scenario is None else {"policy": name, "scenario": scenario}
            for metric in metrics:
                row[metric] = f"{group[metric].mean():.3f} ± {group[metric].std():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Tabla Markdown sin depender de `tabulate`."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main():
    parser = argparse.ArgumentParser(description="Evalúa los baselines en HellKitchen-v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=TEST_BASE_SEED)
    args = parser.parse_args()

    env_kwargs = {"window": args.window}
    results = {
        "random": evaluate_scenarios(lambda env: RandomPolicy(env.action_space.n, seed=args.seed),
                                     n_episodes=args.episodes, base_seed=args.seed,
                                     env_kwargs=env_kwargs),
        "fifo": evaluate_scenarios(lambda env: FifoHeuristicPolicy(args.window),
                                   n_episodes=args.episodes, base_seed=args.seed,
                                   env_kwargs=env_kwargs),
    }
    summary = summarize(results)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([df.assign(policy=name) for name, df in results.items()]).to_csv(
        TABLES_DIR / "baselines_episodes.csv", index=False)
    summary.to_csv(TABLES_DIR / "baselines_summary.csv", index=False)
    (TABLES_DIR / "baselines_summary.md").write_text(to_markdown(summary) + "\n", encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
