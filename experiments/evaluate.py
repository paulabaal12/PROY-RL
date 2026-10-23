"""Evaluador común: mide cualquier política con las mismas seeds y las mismas métricas.

Todos los agentes (baselines, DQN y PPO) se evalúan con este archivo. El episodio i
usa la seed `base_seed + i`, y como las órdenes salen de un generador propio del
ambiente, todos los agentes enfrentan exactamente la misma secuencia de órdenes.

Una "política" es cualquier función `obs -> acción`.

Uso:
    python -m experiments.evaluate --episodes 100
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

import envs  # noqa: F401  (registra HellKitchen-v0)
from agents.baselines import HeuristicPolicy, RandomPolicy

EVAL_BASE_SEED = 10_000  # seeds de evaluación separadas de las de entrenamiento
TABLES_DIR = Path("results/tables")


def run_episode(env: gym.Env, policy, seed: int) -> dict:
    """Corre un episodio completo y devuelve sus métricas (docs/env_design.md, sección 7)."""
    obs, info = env.reset(seed=seed)
    counts = {key: 0 for key in ("served_ok", "served_raw", "served_burnt", "expired",
                                 "tasted", "discarded", "invalid", "arrived", "started")}
    delivery_times = []
    episode_return = 0.0
    events = info["events"]
    done = False
    while True:
        for event in events:
            counts[event["type"]] += 1
            if event["type"] == "served_ok":
                delivery_times.append(event["time_used"])
        if done:
            break
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        episode_return += reward
        events = info["events"]
        done = terminated or truncated

    served = counts["served_ok"] + counts["served_raw"] + counts["served_burnt"]
    closed = served + counts["expired"]  # órdenes que ya se resolvieron (bien o mal)
    return {
        "seed": seed,
        "return": episode_return,
        **counts,
        "on_time_rate": counts["served_ok"] / closed if closed else np.nan,
        "bad_serve_rate": (served - counts["served_ok"]) / served if served else np.nan,
        "tastes_per_order": counts["tasted"] / served if served else np.nan,
        "mean_delivery_time": np.mean(delivery_times) if delivery_times else np.nan,
    }


def evaluate(policy, n_episodes: int = 100, base_seed: int = EVAL_BASE_SEED,
             env_kwargs: dict | None = None) -> pd.DataFrame:
    """Evalúa una política en `n_episodes` episodios. Devuelve una fila por episodio."""
    env = gym.make("HellKitchen-v0", **(env_kwargs or {}))
    rows = [run_episode(env, policy, base_seed + i) for i in range(n_episodes)]
    env.close()
    return pd.DataFrame(rows)


SUMMARY_METRICS = ["return", "on_time_rate", "bad_serve_rate", "tastes_per_order",
                   "mean_delivery_time", "served_ok", "expired", "invalid"]


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabla resumen (media ± desviación estándar) de varias políticas."""
    rows = []
    for name, df in results.items():
        row = {"policy": name}
        for metric in SUMMARY_METRICS:
            row[metric] = f"{df[metric].mean():.3f} ± {df[metric].std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Tabla Markdown sin depender de `tabulate`."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main():
    parser = argparse.ArgumentParser(description="Evalúa los baselines en HellKitchen-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=EVAL_BASE_SEED)
    args = parser.parse_args()

    n_actions = gym.make("HellKitchen-v0").action_space.n
    results = {
        "random": evaluate(RandomPolicy(n_actions, seed=args.seed), args.episodes, args.seed),
        "heuristic": evaluate(HeuristicPolicy(), args.episodes, args.seed),
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
