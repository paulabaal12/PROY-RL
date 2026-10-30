"""Genera replays de ejemplo para probar el visualizador.

Uso:
    python -m viz.make_samples
"""

from pathlib import Path

from envs.kitchen_env import KitchenEnv
from viz.record import make_policy, record_episode

OUT_DIR = Path("results/replays")
SEED = 10000
WINDOW = 4

# (archivo, política, cocina, nombre para mostrar)
SAMPLES = [
    ("sample_fifo", "fifo", "easy", "Naive FIFO Baseline"),
    ("sample_random", "random", "easy", "Random"),
    ("sample_edf", "edf", "easy", "EDF heuristic"),
    # Cocina con 2 estaciones: el tercer fuego aparece apagado
    ("sample_fifo_medium", "fifo", "medium", "Naive FIFO (medium kitchen)"),
]


def main():
    for name, policy_name, kitchen, label in SAMPLES:
        env = KitchenEnv(window=WINDOW, kitchen=kitchen)
        policy = make_policy(policy_name, env, SEED)
        path = record_episode(env, policy, SEED, OUT_DIR / f"{name}.jsonl",
                              meta={"policy": policy_name, "label": label})
        print(f"{label:30s} -> {path}")


if __name__ == "__main__":
    main()
