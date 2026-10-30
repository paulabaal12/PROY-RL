"""Graba un episodio de HellKitchen en un archivo JSON Lines para el visualizador.

Formato: la línea 1 es el encabezado (seed, cocina, ventana, política); la línea 2
es el estado después del reset (t = 0); luego una línea por paso con la acción, la
recompensa, la observación, la lectura visual ruidosa y el `info` del ambiente.

Uso:
    python -m viz.record --policy fifo --seed 10000 --kitchen easy --window 4 \
        --out results/replays/fifo.jsonl

Para agentes entrenados, llamar `record_episode` con una función obs -> acción; el
preprocesamiento (normalización, `model.predict`) va dentro de esa función.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from agents.baselines import EdfHeuristicPolicy, FifoHeuristicPolicy, RandomPolicy
from envs.kitchen_env import KitchenEnv


def _plain(value):
    """Convierte tipos de numpy a tipos de Python para poder escribir JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def record_episode(env: KitchenEnv, policy: Callable[[np.ndarray], int], seed: int,
                   path: str | Path, meta: dict | None = None) -> Path:
    """Corre un episodio completo con `policy` y lo guarda en `path`."""
    env = getattr(env, "unwrapped", env)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "header", "seed": seed, "kitchen": env.kitchen, "window": env.window,
        "max_stations": env.max_stations, "policy": type(policy).__name__,
        "env_params": {"horizon": env.horizon, "patience": env.patience,
                       "queue_max": env.queue_max, "expire_penalty": env.expire_penalty},
        **(meta or {}),
    }
    obs, info = env.reset(seed=seed)
    lines = [header, {"kind": "step", "t": 0, "action": None, "action_kind": None,
                      "action_index": None, "reward": 0.0, "obs": obs,
                      "visual_cook": env.last_visual, "info": info}]
    done = False
    while not done:
        action = int(policy(obs))
        kind, index = env.layout.decode(action)
        obs, reward, terminated, truncated, info = env.step(action)
        lines.append({"kind": "step", "t": info["t"], "action": action, "action_kind": kind,
                      "action_index": index, "reward": float(reward), "obs": obs,
                      "visual_cook": env.last_visual, "info": info})
        done = terminated or truncated
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(_plain(line)) + "\n")
    return path


def make_policy(name: str, env: KitchenEnv, seed: int):
    if name == "random":
        return RandomPolicy(env.action_space.n, seed=seed)
    if name == "fifo":
        return FifoHeuristicPolicy(env.window)
    return EdfHeuristicPolicy(env.window)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["random", "fifo", "edf"], default="fifo")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--kitchen", default="easy", help='preset o "random"')
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--label", default=None, help="nombre para mostrar en el visualizador")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    env = KitchenEnv(window=args.window, kitchen=args.kitchen, horizon=args.horizon)
    policy = make_policy(args.policy, env, args.seed)
    meta = {"policy": args.policy}
    if args.label:
        meta["label"] = args.label
    path = record_episode(env, policy, args.seed, args.out, meta)
    print(f"Replay guardado en {path}")


if __name__ == "__main__":
    main()
