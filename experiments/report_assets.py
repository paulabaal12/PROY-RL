"""Tabla comparativa y figura DQN vs. PPO del informe, y copia de las figuras a docs/figuras.

Parte de lo que dejó `experiments.analyze`:

    python -m experiments.analyze
    python -m experiments.report_assets

La recompensa de entrenamiento es `rollout/ep_rew_mean` de stable-baselines3: la media móvil
de los últimos 100 episodios, en la escala real de la recompensa y sin shaping, porque
`Monitor` va por dentro de los demás wrappers (agents/rl.py).
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import analyze
from experiments.analyze import ALGO_COLOR, ALGO_LABEL, ALGO_MARKER, FIGURES_DIR, RUNS_DIR, TABLES_DIR
from experiments.calibrate import SURFACE, TEXT, _style
from experiments.evaluate import to_markdown

DOCS_FIGURES = Path("docs/figuras")
BIN_STEPS = 25_000
SKIP_FIGURES = {"sprites_preview.png"}      # material de diseño del visualizador


def steps_to_converge(curves: pd.DataFrame, algo: str, window: int, fraction: float = 0.9) -> float:
    """Mediana entre seeds de los pasos para lograr `fraction` de la mejora total.

    La mejora va del retorno de la primera evaluación a la media de las tres últimas.
    """
    sub = curves[(curves["algo"] == algo) & (curves["window"] == window)]
    steps = []
    for _, run in sub.groupby("train_seed"):
        run = run.sort_values("step")
        start, final = run["return"].iloc[0], run["return"].tail(3).mean()
        reached = run[run["return"] >= start + fraction * (final - start)]
        steps.append(float(reached["step"].iloc[0]) if len(reached) else np.nan)
    return float(np.nanmedian(steps)) if np.any(~np.isnan(steps)) else np.nan


def training_minutes(runs_dir: Path, algo: str, window: int) -> float:
    """Minutos promedio de entrenamiento por corrida, según done.json."""
    paths = sorted(Path(runs_dir).glob(f"{algo}_N{window}_seed*/done.json"))
    return float(np.mean([json.loads(p.read_text(encoding="utf-8"))["seconds"] for p in paths])) / 60


def comparison_table(final: pd.DataFrame, baselines: pd.DataFrame, curves: pd.DataFrame,
                     efficiency: pd.DataFrame, windows: dict[str, int], runs_dir: Path) -> pd.DataFrame:
    """Una fila por política en las seeds de prueba, cocina randomizada."""
    test = analyze.seed_means(final, "test")
    rows = []
    for algo, window in windows.items():
        group = test[(test["algo"] == algo) & (test["window"] == window)]
        eff = efficiency[(efficiency["algo"] == algo) & (efficiency["N"] == window)].iloc[0]
        rows.append({
            "politica": f"{ALGO_LABEL[algo]} (N={window})",
            "retorno_media": group["return"].mean(), "retorno_de": group["return"].std(ddof=1),
            "de_entre": "5 seeds", "tasa_a_tiempo": group["on_time_rate"].mean(),
            "largo_episodio": group["steps"].mean(),
            "pasos_superar_fifo": eff["pasos_medianos_para_superar"],
            "pasos_90pct_mejora": steps_to_converge(curves, algo, window),
            "minutos_por_corrida": training_minutes(runs_dir, algo, window),
        })
    base = baselines[baselines["scenario"] == "random"]
    for policy, label in (("fifo", "FIFO"), ("random", "Aleatoria")):
        group = base[base["policy"] == policy]
        rows.append({"politica": label, "retorno_media": group["return"].mean(),
                     "retorno_de": group["return"].std(ddof=1), "de_entre": f"{len(group)} episodios",
                     "tasa_a_tiempo": group["on_time_rate"].mean(), "largo_episodio": group["steps"].mean(),
                     "pasos_superar_fifo": np.nan, "pasos_90pct_mejora": np.nan,
                     "minutos_por_corrida": np.nan})
    return pd.DataFrame(rows).round({"retorno_media": 2, "retorno_de": 2, "tasa_a_tiempo": 3,
                                     "largo_episodio": 1, "minutos_por_corrida": 1})


def plot_reward_comparison(progress: pd.DataFrame, windows: dict[str, int], fifo_return: float,
                           figures_dir: Path):
    """Recompensa por episodio al entrenar: DQN y PPO en la misma gráfica, con su mejor N."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for algo, window in windows.items():
        sub = progress[(progress["algo"] == algo) & (progress["window"] == window)].dropna(
            subset=["rollout/ep_rew_mean", "time/total_timesteps"]).copy()
        sub["bin"] = (sub["time/total_timesteps"] // BIN_STEPS) * BIN_STEPS
        by_seed = sub.groupby(["train_seed", "bin"])["rollout/ep_rew_mean"].last().reset_index()
        stats = by_seed.groupby("bin")["rollout/ep_rew_mean"].agg(["mean", "std"]).fillna(0.0)
        x = stats.index.to_numpy() / 1e6
        ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        color=ALGO_COLOR[algo], alpha=0.15, linewidth=0)
        ax.plot(x, stats["mean"], color=ALGO_COLOR[algo], linewidth=2, marker=ALGO_MARKER[algo],
                markevery=max(len(x) // 8, 1), markersize=6, label=f"{ALGO_LABEL[algo]} (N = {window})")
    analyze._reference(ax, fifo_return, "FIFO")
    _style(ax, "Recompensa por episodio al entrenar (media móvil de 100 episodios)",
           "millones de pasos", "recompensa por episodio")
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "dqn_vs_ppo_reward.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build(runs_dir: Path = RUNS_DIR, tables_dir: Path = TABLES_DIR, figures_dir: Path = FIGURES_DIR,
          docs_figures: Path = DOCS_FIGURES, eval_episodes: int = 30) -> pd.DataFrame:
    tables_dir, figures_dir, docs_figures = Path(tables_dir), Path(figures_dir), Path(docs_figures)
    final = analyze.load_final(runs_dir)
    windows = analyze.best_windows(final)
    curves = analyze._load_curves("evals.csv", runs_dir)
    efficiency = pd.read_csv(tables_dir / "sample_efficiency.csv")
    table = comparison_table(final, analyze.load_baselines(tables_dir), curves, efficiency, windows, runs_dir)
    table.to_csv(tables_dir / "comparison.csv", index=False)
    (tables_dir / "comparison.md").write_text(to_markdown(table) + "\n", encoding="utf-8")

    progress = analyze._load_curves("progress.csv", runs_dir)
    plot_reward_comparison(progress, windows, analyze.fifo_validation_return(analyze.DEFAULT_WINDOW, eval_episodes),
                           figures_dir)
    docs_figures.mkdir(parents=True, exist_ok=True)
    for path in sorted(figures_dir.glob("*.png")):
        if path.name not in SKIP_FIGURES:
            shutil.copy2(path, docs_figures / path.name)
    return table


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Genera la tabla y las figuras del informe")
    parser.parse_args(argv)
    print(to_markdown(build()))


if __name__ == "__main__":
    main()
