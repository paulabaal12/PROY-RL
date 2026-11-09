"""Exporta replays de los agentes entrenados para el visualizador (T8).

Para cada algoritmo toma su mejor N (elegido en validación) y, de sus 5 seeds, la
corrida con el retorno de validación más cercano a la mediana: un agente representativo,
no el más afortunado. DQN, PPO y la FIFO corren con la misma seed de prueba, así que
enfrentan la misma cocina y las mismas órdenes.

Esquema: archivos JSON Lines, descrito en viz/record.py. La línea 1 es el encabezado
(`kind="header"`, seed, kitchen, window, max_stations, policy, label, env_params); cada
línea siguiente es un paso (`kind="step"`, t, action, action_kind, action_index, reward,
obs, visual_cook, info). `info` trae el estado oculto real, la cola y los eventos.

Uso:
    python -m experiments.export_replays
"""

import argparse
from pathlib import Path

import pandas as pd

from agents.baselines import FifoHeuristicPolicy
from agents.rl import load_policy
from envs.kitchen_env import KitchenEnv
from experiments.analyze import best_windows, load_final
from viz.record import record_episode

OUT_DIR = Path("results/replays")
MODELS_DIR = Path("results/models")
SEED = 10_000                    # primera seed de prueba
SCENARIOS = ("medium", "hard")


def representative_seed(final: pd.DataFrame, algo: str, window: int) -> int:
    """La seed de entrenamiento cuyo retorno de validación está más cerca de la mediana."""
    runs = final[(final["algo"] == algo) & (final["window"] == window) & (final["split"] == "validation")]
    returns = runs.groupby("train_seed")["return"].mean()
    return int((returns - returns.median()).abs().idxmin())


def export_all(out_dir: Path = OUT_DIR, seed: int = SEED, scenarios=SCENARIOS) -> list[Path]:
    final = load_final()
    windows = best_windows(final)
    paths = []
    for scenario in scenarios:
        for algo, window in windows.items():
            train_seed = representative_seed(final, algo, window)
            policy = load_policy(algo, MODELS_DIR / f"{algo}_N{window}_seed{train_seed}.zip")
            env = KitchenEnv(window=window, kitchen=scenario)
            paths.append(record_episode(env, policy, seed, out_dir / f"{algo}_{scenario}.jsonl", meta={
                "policy": algo, "label": f"{algo.upper()} (N={window})", "train_seed": train_seed}))
        window = 4
        paths.append(record_episode(KitchenEnv(window=window, kitchen=scenario), FifoHeuristicPolicy(window),
                                    seed, out_dir / f"fifo_{scenario}.jsonl",
                                    meta={"policy": "fifo", "label": "FIFO"}))
    return paths


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Exporta replays de DQN, PPO y FIFO")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    for path in export_all(Path(args.out), args.seed):
        print(f"Replay guardado en {path}")


if __name__ == "__main__":
    main()
