"""Convierte las corridas de la grilla en las tablas y figuras del reporte (T7).

Lee solo `results/runs/grid/` (y los baselines, que calcula si no existen). Un comando
regenera todo:

    python -m experiments.analyze

La unidad estadística de un agente es la SEED de entrenamiento (5 por configuración): se
promedian sus 100 episodios y los intervalos se calculan entre seeds. El mejor N de cada
algoritmo se elige en validación y se reporta en prueba (docs/env_design.md, sección 4.5).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from agents.baselines import FifoHeuristicPolicy, RandomPolicy
from experiments.calibrate import GRID, SERIES, SURFACE, TEXT, TEXT_SECONDARY, _style
from experiments.evaluate import (
    SCENARIOS, SUMMARY_METRICS, VALIDATION_BASE_SEED, evaluate, evaluate_scenarios, to_markdown,
)

RUNS_DIR = Path("results/runs/grid")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
BASELINE_WINDOW = 4
DEFAULT_WINDOW = 6
ALGO_COLOR = {"dqn": SERIES[0], "ppo": SERIES[1]}
ALGO_MARKER = {"dqn": "o", "ppo": "s"}
ALGO_LABEL = {"dqn": "DQN", "ppo": "PPO"}
WINDOW_MARKER = ("o", "s", "^", "D")
DISH_NAMES = ("stirfry", "egg", "steak")


# --- Carga ---------------------------------------------------------------------

def load_final(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Todas las evaluaciones finales, una fila por episodio."""
    paths = sorted(Path(runs_dir).glob("*/final_eval.csv"))
    if not paths:
        raise FileNotFoundError(f"No hay corridas terminadas en {runs_dir}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _load_curves(filename: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(runs_dir).glob(f"*/{filename}")):
        if path.stat().st_size == 0:     # corrida tan corta que no alcanzó a registrar nada
            continue
        algo, n, seed = path.parent.name.split("_")
        frames.append(pd.read_csv(path).assign(algo=algo, window=int(n[1:]), train_seed=int(seed[4:])))
    return pd.concat(frames, ignore_index=True)


def load_baselines(tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """Baselines en las seeds de prueba: los del notebook 01 o, si no existen, los calcula."""
    path = Path(tables_dir) / "baselines_episodes.csv"
    if path.exists():
        return pd.read_csv(path)
    kwargs = {"window": BASELINE_WINDOW}
    frames = [
        evaluate_scenarios(lambda env: RandomPolicy(env.action_space.n, seed=0), env_kwargs=kwargs)
        .assign(policy="random"),
        evaluate_scenarios(lambda env: FifoHeuristicPolicy(BASELINE_WINDOW), env_kwargs=kwargs)
        .assign(policy="fifo"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


# --- Estadística ---------------------------------------------------------------

def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Intervalo de confianza del 95 % de la media, por bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_boot, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_means(final: pd.DataFrame, split: str = "test", scenario: str = "random") -> pd.DataFrame:
    """Una fila por corrida: la media de sus episodios en ese conjunto y escenario."""
    subset = final[(final["split"] == split) & (final["scenario"] == scenario)]
    return subset.groupby(["algo", "window", "train_seed"], as_index=False).mean(numeric_only=True)


def best_windows(final: pd.DataFrame) -> dict[str, int]:
    """El mejor N de cada algoritmo según el retorno en VALIDACIÓN."""
    runs = seed_means(final, "validation")
    means = runs.groupby(["algo", "window"])["return"].mean()
    return {algo: int(means[algo].idxmax()) for algo in sorted(runs["algo"].unique())}


def _fmt(values) -> str:
    low, high = bootstrap_ci(values)
    return f"{np.nanmean(values):.3f} [{low:.3f}, {high:.3f}]"


def _save(df: pd.DataFrame, name: str, tables_dir: Path):
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / f"{name}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")


# --- Tablas --------------------------------------------------------------------

def table_by_window(final: pd.DataFrame) -> pd.DataFrame:
    """Desempeño por algoritmo y N: en validación (para elegir) y en prueba (para reportar)."""
    val, test = seed_means(final, "validation"), seed_means(final, "test")
    rows = []
    for (algo, window), group in test.groupby(["algo", "window"]):
        v = val[(val["algo"] == algo) & (val["window"] == window)]
        rows.append({"algo": algo, "N": window, "validation_return": _fmt(v["return"]),
                     **{m: _fmt(group[m]) for m in SUMMARY_METRICS}})
    return pd.DataFrame(rows)


def table_main(final: pd.DataFrame, baselines: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    """Tabla principal: mejor DQN, mejor PPO, FIFO y aleatoria en las seeds de prueba."""
    test = seed_means(final, "test")
    rows = []
    for algo, window in windows.items():
        group = test[(test["algo"] == algo) & (test["window"] == window)]
        rows.append({"policy": f"{ALGO_LABEL[algo]} (N={window})",
                     **{m: _fmt(group[m]) for m in SUMMARY_METRICS}})
    base = baselines[baselines["scenario"] == "random"]
    for policy, label in (("fifo", "FIFO"), ("random", "Aleatoria")):
        group = base[base["policy"] == policy]
        rows.append({"policy": label, **{m: _fmt(group[m]) for m in SUMMARY_METRICS}})
    return pd.DataFrame(rows)


def table_robustness(final: pd.DataFrame, baselines: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    """Retorno medio de cada agente en cada escenario de prueba."""
    rows = []
    test = final[final["split"] == "test"]
    for algo, window in windows.items():
        sub = test[(test["algo"] == algo) & (test["window"] == window)]
        by_seed = sub.groupby(["scenario", "train_seed"])["return"].mean().groupby("scenario").mean()
        rows.append({"policy": f"{ALGO_LABEL[algo]} (N={window})", **by_seed.to_dict()})
    for policy, label in (("fifo", "FIFO"), ("random", "Aleatoria")):
        means = baselines[baselines["policy"] == policy].groupby("scenario")["return"].mean()
        rows.append({"policy": label, **means.to_dict()})
    return pd.DataFrame(rows)[["policy", *SCENARIOS]]


def table_n1_vs_fifo(final: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Agente con N = 1 (obligado a FIFO) contra la heurística FIFO."""
    test = seed_means(final, "test")
    metrics = ["return", "on_time_rate", "expired_rate", "bad_serve_rate", "tastes_per_order", "idle_rate"]
    rows = []
    for algo in sorted(test["algo"].unique()):
        group = test[(test["algo"] == algo) & (test["window"] == 1)]
        rows.append({"policy": f"{ALGO_LABEL[algo]} (N=1)", **{m: _fmt(group[m]) for m in metrics}})
    fifo = baselines[(baselines["scenario"] == "random") & (baselines["policy"] == "fifo")]
    rows.append({"policy": "FIFO", **{m: _fmt(fifo[m]) for m in metrics}})
    return pd.DataFrame(rows)


def table_taste_by_context(final: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    """Pruebas por orden según el ruido y según el plato (hipótesis H3)."""
    test = final[final["split"] == "test"]
    rows = []
    for algo, window in windows.items():
        sub = test[(test["algo"] == algo) & (test["window"] == window)]
        row = {"policy": f"{ALGO_LABEL[algo]} (N={window})"}
        for scenario, label in (("low_noise", "ruido bajo"), ("high_noise", "ruido alto")):
            row[f"pruebas/orden, {label}"] = _fmt(
                sub[sub["scenario"] == scenario].groupby("train_seed")["tastes_per_order"].mean())
        medium = sub[sub["scenario"] == "medium"]
        for dish in DISH_NAMES:
            row[f"pruebas/orden, {dish}"] = _fmt(
                medium.groupby("train_seed")[f"tastes_per_order_{dish}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fifo_validation_return(window: int, episodes: int) -> float:
    return float(evaluate(FifoHeuristicPolicy(window), episodes, VALIDATION_BASE_SEED,
                          {"window": window, "kitchen": "random"})["return"].mean())


def table_sample_efficiency(curves: pd.DataFrame, final: pd.DataFrame, eval_episodes: int = 30) -> pd.DataFrame:
    """Pasos hasta superar a la FIFO en validación, y estabilidad del retorno entre seeds."""
    test = seed_means(final, "test")
    rows = []
    for (algo, window), group in curves.groupby(["algo", "window"]):
        threshold = fifo_validation_return(window, eval_episodes)
        steps = []
        for _, run in group.groupby("train_seed"):
            beat = run[run["return"] >= threshold].sort_values("step")
            steps.append(float(beat["step"].iloc[0]) if len(beat) else np.nan)
        returns = test[(test["algo"] == algo) & (test["window"] == window)]["return"].to_numpy()
        reached = np.count_nonzero(~np.isnan(steps))
        rows.append({"algo": algo, "N": window, "umbral_FIFO_validacion": round(threshold, 2),
                     "seeds_que_superan_FIFO": f"{reached}/{len(steps)}",
                     "pasos_medianos_para_superar": np.nan if reached == 0 else int(np.nanmedian(steps)),
                     "retorno_prueba_media": round(float(returns.mean()), 2),
                     "retorno_prueba_std_entre_seeds": round(float(returns.std(ddof=1)), 2)})
    return pd.DataFrame(rows)


# --- Figuras -------------------------------------------------------------------

def _finish(fig, name: str, figures_dir: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _reference(ax, value: float, label: str):
    """Línea horizontal de referencia, con su nombre afuera del panel para no tapar los datos."""
    ax.axhline(value, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.annotate(label, (1.0, value), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", color=TEXT_SECONDARY,
                fontsize=8, annotation_clip=False)


def plot_training_curves(curves: pd.DataFrame, fifo_return: float, figures_dir: Path):
    """Retorno en validación vs. pasos: un panel por algoritmo, una línea por N."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, algo in zip(axes, sorted(curves["algo"].unique())):
        sub = curves[curves["algo"] == algo]
        for j, window in enumerate(sorted(sub["window"].unique())):
            stats = sub[sub["window"] == window].groupby("step")["return"].agg(["mean", "std"])
            x = stats.index.to_numpy() / 1e6
            ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                            color=SERIES[j], alpha=0.15, linewidth=0)
            ax.plot(x, stats["mean"], color=SERIES[j], linewidth=2, marker=WINDOW_MARKER[j],
                    markevery=max(len(x) // 8, 1), markersize=6, label=f"N = {window}")
        _reference(ax, fifo_return, "FIFO")
        _style(ax, f"{ALGO_LABEL[algo]}: retorno en validación", "millones de pasos", "retorno por episodio")
        ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    _finish(fig, "training_curves", figures_dir)


def plot_training_internals(progress: pd.DataFrame, figures_dir: Path, window: int = DEFAULT_WINDOW):
    """Curvas propias de cada algoritmo: loss y epsilon en DQN; pérdidas, entropía y KL en PPO."""
    panels = [("dqn", "train/loss", "DQN: loss"),
              ("dqn", "rollout/exploration_rate", "DQN: epsilon"),
              ("ppo", "train/policy_gradient_loss", "PPO: policy loss"),
              ("ppo", "train/value_loss", "PPO: value loss"),
              ("ppo", "train/entropy_loss", "PPO: entropía (loss)"),
              ("ppo", "train/approx_kl", "PPO: KL aproximada")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    sub = progress[progress["window"] == window]
    for ax, (algo, col, title) in zip(axes.flat, panels):
        data = sub[sub["algo"] == algo].dropna(subset=[col, "time/total_timesteps"])
        bins = (data["time/total_timesteps"] // 25_000) * 25_000
        means = data.groupby(bins)[col].mean()
        ax.plot(means.index / 1e6, means.to_numpy(), color=ALGO_COLOR[algo], linewidth=2)
        _style(ax, title, "millones de pasos", "")
    _finish(fig, "training_internals", figures_dir)


def plot_performance_vs_n(final: pd.DataFrame, baselines: pd.DataFrame, figures_dir: Path):
    """Retorno y tasa a tiempo en función de N, con las heurísticas como referencia."""
    test = seed_means(final, "test")
    base = baselines[baselines["scenario"] == "random"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in ((axes[0], "return", "retorno por episodio"),
                              (axes[1], "on_time_rate", "tasa a tiempo")):
        for algo in sorted(test["algo"].unique()):
            sub = test[test["algo"] == algo]
            xs = sorted(sub["window"].unique())
            means = [sub[sub["window"] == w][metric].mean() for w in xs]
            cis = [bootstrap_ci(sub[sub["window"] == w][metric]) for w in xs]
            err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
            ax.errorbar(xs, means, yerr=err, color=ALGO_COLOR[algo], marker=ALGO_MARKER[algo], linewidth=2,
                        markersize=7, capsize=3, label=ALGO_LABEL[algo])
        for policy, name in (("fifo", "FIFO"), ("random", "Aleatoria")):
            _reference(ax, base[base["policy"] == policy][metric].mean(), name)
        ax.set_xticks([1, 2, 4, 6])
        _style(ax, f"{label} según N (prueba, IC 95 %)", "tamaño de la ventana N", label)
        ax.legend(frameon=False, labelcolor=TEXT, loc="center right")
    _finish(fig, "performance_vs_n", figures_dir)


def plot_behavior_vs_n(final: pd.DataFrame, baselines: pd.DataFrame, figures_dir: Path):
    """Cómo juegan los agentes: pruebas, ociosidad, priorización y preparación adelantada."""
    test = seed_means(final, "test")
    fifo = baselines[(baselines["scenario"] == "random") & (baselines["policy"] == "fifo")]
    panels = [("tastes_per_order", "pruebas por orden"), ("idle_rate", "ociosidad"),
              ("prioritization_rate", "priorización"), ("prep_ahead_rate", "preparación adelantada")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        for algo in sorted(test["algo"].unique()):
            sub = test[test["algo"] == algo]
            xs = sorted(sub["window"].unique())
            means = [sub[sub["window"] == w][metric].mean() for w in xs]
            cis = [bootstrap_ci(sub[sub["window"] == w][metric]) for w in xs]
            err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
            ax.errorbar(xs, means, yerr=err, color=ALGO_COLOR[algo], marker=ALGO_MARKER[algo], linewidth=2,
                        markersize=7, capsize=3, label=ALGO_LABEL[algo])
        _reference(ax, fifo[metric].mean(), "FIFO")
        ax.set_xticks([1, 2, 4, 6])
        _style(ax, label, "tamaño de la ventana N", label)
        ax.legend(frameon=False, labelcolor=TEXT)
    _finish(fig, "behavior_vs_n", figures_dir)


def plot_robustness(robust: pd.DataFrame, figures_dir: Path):
    """Mapa de calor agente x escenario. Color divergente centrado en retorno 0."""
    data = robust.set_index("policy")
    values = data.to_numpy(dtype=float)
    limit = float(np.abs(values).max())
    cmap = LinearSegmentedColormap.from_list("polarity", [SERIES[1], "#f0efec", SERIES[0]])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(values.shape[1]), data.columns, color=TEXT_SECONDARY)
    ax.set_yticks(range(values.shape[0]), data.index, color=TEXT)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", color=TEXT, fontsize=9)
    ax.set_title("Retorno por escenario (prueba)", color=TEXT, fontsize=11, loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, length=0)
    fig.gca()
    _finish(fig, "robustness", figures_dir)


# --- Principal -----------------------------------------------------------------

def run_analysis(runs_dir: Path = RUNS_DIR, tables_dir: Path = TABLES_DIR,
                 figures_dir: Path = FIGURES_DIR, eval_episodes: int = 30) -> dict:
    tables_dir, figures_dir = Path(tables_dir), Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    final = load_final(runs_dir)
    baselines = load_baselines(tables_dir)
    curves = _load_curves("evals.csv", runs_dir)
    progress = _load_curves("progress.csv", runs_dir)
    windows = best_windows(final)

    robust = table_robustness(final, baselines, windows)
    outputs = {
        "by_window": table_by_window(final),
        "main_table": table_main(final, baselines, windows),
        "robustness": robust.round(2),
        "n1_vs_fifo": table_n1_vs_fifo(final, baselines),
        "taste_by_context": table_taste_by_context(final, windows),
        "sample_efficiency": table_sample_efficiency(curves, final, eval_episodes),
    }
    for name, table in outputs.items():
        _save(table, name, tables_dir)

    plot_training_curves(curves, fifo_validation_return(DEFAULT_WINDOW, eval_episodes), figures_dir)
    plot_training_internals(progress, figures_dir)
    plot_performance_vs_n(final, baselines, figures_dir)
    plot_behavior_vs_n(final, baselines, figures_dir)
    plot_robustness(robust, figures_dir)
    return {"windows": windows, **outputs}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Genera las tablas y figuras del reporte")
    parser.parse_args(argv)
    result = run_analysis()
    print("Mejor N (validación):", result["windows"])
    print(to_markdown(result["main_table"]))


if __name__ == "__main__":
    main()
