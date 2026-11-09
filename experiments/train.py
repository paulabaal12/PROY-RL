"""Entrena una corrida de DQN o PPO. Lo usan train_dqn.py, train_ppo.py y run_experiments.py.

Cada corrida escribe en su carpeta:
    config.yaml   configuración completa (seed, N, pasos, hiperparámetros)
    progress.csv  métricas de stable-baselines3: retorno y largo de episodio, loss y
                  epsilon en DQN; policy loss, value loss, entropía y KL en PPO
    evals.csv     retorno y métricas en las seeds de VALIDACIÓN durante el entrenamiento
    done.json     solo si la corrida terminó; permite retomar la grilla sin repetirla

La evaluación se hace con el evaluador común y en la escala real de recompensa.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from agents.rl import ModelPolicy, build_model, make_train_env
from experiments.evaluate import VALIDATION_BASE_SEED, evaluate
from utils.seeding import set_global_seed

CONFIG_DIR = Path("experiments/configs")
EVAL_METRICS = ["return", "on_time_rate", "late_rate", "expired_rate", "bad_serve_rate",
                "tastes_per_order", "idle_rate", "avoidable_idle_rate", "prioritization_rate",
                "prep_ahead_rate", "invalid"]


def load_config(algo: str, path: str | Path | None = None) -> dict:
    """Lee la configuración: la indicada, la ajustada en el tuning o, si no hay, la base."""
    tuned = CONFIG_DIR / f"{algo}_tuned.yaml"
    source = Path(path) if path else (tuned if tuned.exists() else CONFIG_DIR / f"{algo}.yaml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))


class PeriodicEval(BaseCallback):
    """Evalúa la política actual cada `every` pasos y guarda la curva en evals.csv."""

    def __init__(self, window: int, every: int, episodes: int, path: Path, kitchen="random"):
        super().__init__()
        self.window, self.every, self.episodes, self.path, self.kitchen = \
            window, every, episodes, path, kitchen
        self.rows: list[dict] = []
        self.next_eval = 0

    def run_eval(self):
        df = evaluate(ModelPolicy(self.model), self.episodes, VALIDATION_BASE_SEED,
                      {"window": self.window, "kitchen": self.kitchen})
        self.rows.append({"step": self.num_timesteps, **df[EVAL_METRICS].mean().to_dict()})
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.run_eval()
            self.next_eval = self.num_timesteps + self.every
        return True

    def _on_training_end(self):
        if not self.rows or self.rows[-1]["step"] != self.num_timesteps:
            self.run_eval()


def _resolve_exploration(config: dict, total_timesteps: int) -> dict:
    """Convierte `exploration_steps` en la fracción que pide stable-baselines3.

    En pasos absolutos, epsilon baja igual de rápido en el tuning y en la grilla, aunque
    duren distinto.
    """
    hyper = dict(config["hyperparameters"])
    steps = hyper.pop("exploration_steps", None)
    if steps is not None:
        hyper["exploration_fraction"] = min(1.0, steps / total_timesteps)
    return {**config, "hyperparameters": hyper}


def train_run(algo: str, window: int, seed: int, config: dict, run_dir: str | Path,
              total_timesteps: int, eval_every: int = 25_000, eval_episodes: int = 30,
              model_path: str | Path | None = None, kitchen: str | dict = "random"):
    """Entrena una corrida completa y devuelve el modelo entrenado."""
    # Un solo hilo por corrida: más rápido al correr varias a la vez, y determinista.
    torch.set_num_threads(1)
    set_global_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    full_config = {"algo": algo, "window": window, "seed": seed, "kitchen": kitchen,
                   "total_timesteps": total_timesteps, "eval_every": eval_every,
                   "eval_episodes": eval_episodes, **config}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(full_config, sort_keys=False),
                                         encoding="utf-8")

    env = make_train_env(window, config["n_envs"], kitchen, config["reward_scale"],
                         shaping=config.get("shaping"), gamma=config["gamma"])
    model = build_model(algo, env, _resolve_exploration(config, total_timesteps), seed)
    model.set_logger(configure(str(run_dir), ["csv"]))
    callback = PeriodicEval(window, eval_every, eval_episodes, run_dir / "evals.csv", kitchen)

    start = time.time()
    model.learn(total_timesteps, callback=callback, log_interval=config["log_interval"])
    elapsed = time.time() - start
    env.close()

    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
    (run_dir / "done.json").write_text(
        json.dumps({"seconds": elapsed, "steps_per_second": total_timesteps / elapsed}),
        encoding="utf-8")
    return model


def cli(algo: str, argv: list[str] | None = None):
    """Línea de comandos común de train_dqn.py y train_ppo.py."""
    import argparse
    parser = argparse.ArgumentParser(description=f"Entrena {algo.upper()} en HellKitchen-v1")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--config", default=None, help="YAML de configuración")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--group", default="manual", help="subcarpeta dentro de results/runs/")
    args = parser.parse_args(argv)

    name = f"{algo}_N{args.window}_seed{args.seed}"
    run_dir = Path("results/runs") / args.group / name
    train_run(algo, args.window, args.seed, load_config(algo, args.config), run_dir, args.timesteps,
              args.eval_every, model_path=Path("results/models") / args.group / f"{name}.zip")
    print(f"Corrida guardada en {run_dir}")
