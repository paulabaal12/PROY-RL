"""Corre el tuning y la grilla de experimentos (docs/env_design.md, secciones 4.4 y 4.5).

Etapas:
    tune   Con el N por defecto, prueba las configuraciones de configs/tuning.yaml
           (el mismo número para cada algoritmo) y guarda la mejor de cada uno en
           configs/{dqn,ppo}_tuned.yaml. Se elige con las seeds de VALIDACIÓN.
    grid   2 algoritmos x 4 valores de N x 5 seeds = 40 corridas con esa configuración.
           Cada una se evalúa al final en validación y en todos los escenarios de prueba.
    all    Ambas, en ese orden.

Las corridas ya terminadas (con done.json) se saltan, así que se puede interrumpir y
volver a lanzar el mismo comando.

Uso:
    python -m experiments.run_experiments --stage all --jobs 4
"""

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agents.rl import ModelPolicy, load_policy
from experiments.evaluate import (
    SCENARIOS, TEST_BASE_SEED, VALIDATION_BASE_SEED, evaluate, to_markdown,
)
from experiments.train import CONFIG_DIR, load_config, train_run

ALGOS = ("dqn", "ppo")
WINDOWS = (1, 2, 4, 6)
DEFAULT_WINDOW = 6            # Paso 0: cubre 89.2 % de los pasos (ver env_design.md, sección 10)
TRAIN_SEEDS = tuple(range(5))
TUNE_SEEDS = (100, 101)       # distintas de las de la grilla
RUNS_DIR = Path("results/runs")
MODELS_DIR = Path("results/models")
TABLES_DIR = Path("results/tables")
FINAL_EPISODES = 100


def final_evaluation(policy, window: int, episodes: int = FINAL_EPISODES) -> pd.DataFrame:
    """Evalúa en validación (para elegir N) y en todos los escenarios de prueba (para reportar)."""
    frames = [evaluate(policy, episodes, VALIDATION_BASE_SEED, {"window": window, "kitchen": "random"})
              .assign(split="validation", scenario="random")]
    for scenario in SCENARIOS:
        frames.append(evaluate(policy, episodes, TEST_BASE_SEED, {"window": window, "kitchen": scenario})
                      .assign(split="test", scenario=scenario))
    return pd.concat(frames, ignore_index=True)


def _execute(job: dict) -> dict:
    """Corre un trabajo en un proceso aparte. Devuelve el error en vez de tumbar la grilla."""
    name = job["name"]
    run_dir = Path(job["run_dir"])
    if (run_dir / "done.json").exists() and (not job["final_eval"] or (run_dir / "final_eval.csv").exists()):
        return {"name": name, "skipped": True}
    start = time.time()
    try:
        model = train_run(job["algo"], job["window"], job["seed"], job["config"], run_dir,
                          job["timesteps"], job["eval_every"], job["eval_episodes"],
                          model_path=job["model_path"])
        if job["final_eval"]:
            final_evaluation(ModelPolicy(model), job["window"]).assign(
                algo=job["algo"], window=job["window"], train_seed=job["seed"],
            ).to_csv(run_dir / "final_eval.csv", index=False)
    except Exception:
        return {"name": name, "error": traceback.format_exc()}
    return {"name": name, "seconds": time.time() - start}


def run_jobs(jobs: list[dict], n_workers: int) -> None:
    """Corre los trabajos en paralelo; al final avisa cuáles fallaron."""
    failed, start = [], time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_execute, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if "error" in result:
                failed.append(result)
                status = "ERROR\n" + result["error"]
            elif result.get("skipped"):
                status = "ya estaba"
            else:
                status = f"{result['seconds'] / 60:.1f} min"
            print(f"[{i}/{len(jobs)}] {result['name']}: {status} "
                  f"(total {(time.time() - start) / 60:.1f} min)", flush=True)
    if failed:
        raise SystemExit(f"Fallaron {len(failed)} corridas: {[f['name'] for f in failed]}")


def _job(group: str, name: str, algo: str, window: int, seed: int, config: dict, timesteps: int,
         eval_every: int, eval_episodes: int, final_eval: bool, model_path) -> dict:
    return {"name": f"{group}/{name}", "algo": algo, "window": window, "seed": seed,
            "config": config, "timesteps": timesteps, "eval_every": eval_every,
            "eval_episodes": eval_episodes, "final_eval": final_eval,
            "run_dir": str(RUNS_DIR / group / name),
            "model_path": None if model_path is None else str(model_path)}


def tune(timesteps: int, n_workers: int, eval_every: int) -> pd.DataFrame:
    """Prueba cada configuración con `TUNE_SEEDS` y guarda la mejor de cada algoritmo."""
    candidates = yaml.safe_load((CONFIG_DIR / "tuning.yaml").read_text(encoding="utf-8"))
    if len(candidates["dqn"]) != len(candidates["ppo"]):
        raise ValueError("El tuning debe probar el mismo número de configuraciones por algoritmo")
    jobs, configs = [], {}
    for algo in ALGOS:
        base = load_config(algo, CONFIG_DIR / f"{algo}.yaml")
        for i, overrides in enumerate(candidates[algo]):
            config = {**base, "hyperparameters": {**base["hyperparameters"], **overrides}}
            configs[(algo, i)] = config
            jobs += [_job("tuning", f"{algo}_cfg{i}_seed{seed}", algo, DEFAULT_WINDOW, seed, config,
                          timesteps, eval_every, 50, False, None) for seed in TUNE_SEEDS]
    run_jobs(jobs, n_workers)

    rows = []
    for algo in ALGOS:
        for i, overrides in enumerate(candidates[algo]):
            finals = [pd.read_csv(RUNS_DIR / "tuning" / f"{algo}_cfg{i}_seed{seed}" / "evals.csv")
                      .iloc[-1]["return"] for seed in TUNE_SEEDS]
            rows.append({"algo": algo, "config": i, "overrides": str(overrides) if overrides else "base",
                         "validation_return": float(np.mean(finals))})
    table = pd.DataFrame(rows)
    table["chosen"] = table["validation_return"] == table.groupby("algo")["validation_return"].transform("max")
    for algo in ALGOS:
        best = int(table[(table["algo"] == algo) & table["chosen"]]["config"].iloc[0])
        (CONFIG_DIR / f"{algo}_tuned.yaml").write_text(
            f"# Generado por run_experiments (etapa tune): configuración {best} de tuning.yaml\n"
            + yaml.safe_dump(configs[(algo, best)], sort_keys=False), encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "tuning.csv", index=False)
    (TABLES_DIR / "tuning.md").write_text(to_markdown(table.round(2)) + "\n", encoding="utf-8")
    print(to_markdown(table.round(2)))
    return table


def grid(timesteps: int, n_workers: int, eval_every: int) -> None:
    """Las 40 corridas: mismos pasos, misma red y mismas seeds para los dos algoritmos."""
    jobs = []
    for algo in ALGOS:
        config = load_config(algo)
        for window in WINDOWS:
            for seed in TRAIN_SEEDS:
                name = f"{algo}_N{window}_seed{seed}"
                jobs.append(_job("grid", name, algo, window, seed, config, timesteps, eval_every,
                                 30, True, MODELS_DIR / f"{name}.zip"))
    run_jobs(jobs, n_workers)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Tuning y grilla de experimentos")
    parser.add_argument("--stage", choices=["tune", "grid", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--tune-timesteps", type=int, default=None,
                        help="pasos por corrida de tuning; por defecto, los mismos de la grilla")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.stage in ("tune", "all"):
        tune(args.tune_timesteps or args.timesteps, args.jobs, args.eval_every)
    if args.stage in ("grid", "all"):
        grid(args.timesteps, args.jobs, args.eval_every)


__all__ = ["final_evaluation", "load_policy"]

if __name__ == "__main__":
    main()
