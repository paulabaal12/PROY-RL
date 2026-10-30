"""Paso 0: calibración del ambiente sin entrenar (docs/env_design.md, sección 4.2).

Fija con datos los valores marcados como "(propuesta)" en el diseño, usando solo
políticas escritas a mano. Cada criterio de la sección 4.2 es una función que
devuelve un DataFrame (una fila por episodio), y cada una tiene su figura.

Todas las corridas usan las seeds de VALIDACIÓN: las de prueba quedan intactas
para el reporte.

Uso:
    python -m experiments.calibrate --episodes 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.baselines import EdfHeuristicPolicy, FifoHeuristicPolicy, RandomPolicy
from envs.kitchen_env import KitchenEnv
from experiments.evaluate import (
    SCENARIOS, VALIDATION_BASE_SEED, evaluate, evaluate_scenarios, to_markdown,
)

FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")

# Paleta categórica en orden fijo y colores de texto (skill dataviz, paleta de referencia)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEVER = "nunca"   # etiqueta de "no probar nunca" en el barrido


# --- 1 y 2. Costo de probar y rango de ruido visual -------------------------

def taste_sweep(taste_costs=(0.0, 0.1, 0.2, 0.5, 1.0), sigmas=(0.05, 0.15, 0.25),
                retaste=(None, 1, 2, 3, 4, 6, 8), n_episodes: int = 50,
                window: int = 4) -> pd.DataFrame:
    """Heurística FIFO que prueba cada k pasos, para cada costo de probar y σ_obs.

    El costo de probar no cambia lo que hace la heurística, solo su recompensa.
    Por eso cada (σ, k) se corre una sola vez con costo 0 y el retorno con costo c
    se calcula exacto como `retorno − c · pruebas`.
    """
    frames = []
    for sigma in sigmas:
        for k in retaste:
            df = evaluate(FifoHeuristicPolicy(window, retaste_after=k), n_episodes,
                          VALIDATION_BASE_SEED,
                          {"window": window, "kitchen": {"sigma_obs": sigma}, "taste_cost": 0.0})
            for cost in taste_costs:
                frames.append(df.assign(taste_cost=cost, sigma_obs=sigma,
                                        retaste_after=NEVER if k is None else k,
                                        **{"return": df["return"] - cost * df["tasted"]}))
    return pd.concat(frames, ignore_index=True)


def best_retaste(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Por cada (costo, σ), la frecuencia de prueba con mayor retorno medio.

    `interior` es True si la mejor no es un extremo ("nunca" ni k = 1): ahí probar
    es una decisión real y no una regla obvia.
    """
    means = (sweep_df.groupby(["taste_cost", "sigma_obs", "retaste_after"], sort=False)["return"]
             .mean().reset_index())
    rows = []
    for (cost, sigma), group in means.groupby(["taste_cost", "sigma_obs"], sort=False):
        best = group.loc[group["return"].idxmax()]
        rows.append({"taste_cost": cost, "sigma_obs": sigma, "best_retaste": best["retaste_after"],
                     "best_return": best["return"],
                     "interior": best["retaste_after"] not in (NEVER, 1)})
    return pd.DataFrame(rows)


# --- 3. N por defecto -------------------------------------------------------

def queue_lengths(n_episodes: int = 100, window: int = 6) -> pd.DataFrame:
    """Largo de la cola (órdenes que esperan, sin contar las del fuego) en cada paso.

    Corre la heurística FIFO en la cocina randomizada. Devuelve una fila por paso.
    """
    env = KitchenEnv(window=window, kitchen="random")
    policy = FifoHeuristicPolicy(window)
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=VALIDATION_BASE_SEED + i)
        for t in range(env.horizon):
            obs, _, _, _, info = env.step(policy(obs))
            rows.append({"episode": i, "t": t, "queue_length": len(info["queue"])})
    return pd.DataFrame(rows)


def default_window(lengths_df: pd.DataFrame, quantile: float = 0.9) -> int:
    """El menor N tal que la cola cabe en la ventana al menos en `quantile` de los pasos."""
    lengths = lengths_df["queue_length"].to_numpy()
    for n in range(1, int(lengths.max(initial=1)) + 1):
        if np.mean(lengths <= n) >= quantile:
            return n
    return 1


# --- 4. Que priorizar importe -----------------------------------------------

def prioritization_check(window: int = 6, n_episodes: int = 100) -> pd.DataFrame:
    """FIFO (solo la orden más antigua) vs. EDF (elige por hora de entrega)."""
    kwargs = {"window": window}
    frames = [
        evaluate_scenarios(lambda env: FifoHeuristicPolicy(window), ["random", "hard"],
                           n_episodes, VALIDATION_BASE_SEED, kwargs).assign(policy="fifo"),
        evaluate_scenarios(lambda env: EdfHeuristicPolicy(window), ["random", "hard"],
                           n_episodes, VALIDATION_BASE_SEED, kwargs).assign(policy="edf"),
    ]
    return pd.concat(frames, ignore_index=True)


# --- 5. Dificultad ------------------------------------------------------------

def difficulty_check(window: int = 4, n_episodes: int = 100) -> pd.DataFrame:
    """Aleatoria vs. FIFO en todos los escenarios de evaluación."""
    kwargs = {"window": window}
    frames = [
        evaluate_scenarios(lambda env: RandomPolicy(env.action_space.n, seed=0), SCENARIOS,
                           n_episodes, VALIDATION_BASE_SEED, kwargs).assign(policy="random"),
        evaluate_scenarios(lambda env: FifoHeuristicPolicy(window), SCENARIOS,
                           n_episodes, VALIDATION_BASE_SEED, kwargs).assign(policy="fifo"),
    ]
    return pd.concat(frames, ignore_index=True)


# --- Figuras ------------------------------------------------------------------

def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _save(fig, name: str, out_dir: Path | None):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"calibration_{name}.png", dpi=150, facecolor=SURFACE)
    return fig


def _grouped_bars(ax, means: pd.DataFrame, stds: pd.DataFrame):
    """Barras agrupadas: una categoría por fila, una serie por columna."""
    n_series = len(means.columns)
    width = 0.8 / n_series
    x = np.arange(len(means.index))
    for j, series in enumerate(means.columns):
        ax.bar(x + (j - (n_series - 1) / 2) * width, means[series], width * 0.92,
               yerr=stds[series], color=SERIES[j], label=series, error_kw={"ecolor": TEXT_SECONDARY,
                                                                          "linewidth": 1})
    ax.set_xticks(x, means.index)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=TEXT)


def plot_taste_sweep(sweep_df: pd.DataFrame, out_dir: Path | None = FIGURES_DIR):
    """Retorno medio vs. frecuencia de prueba: un panel por costo, una línea por σ_obs."""
    costs = list(dict.fromkeys(sweep_df["taste_cost"]))
    order = list(dict.fromkeys(sweep_df["retaste_after"]))
    fig, axes = plt.subplots(1, len(costs), figsize=(3.2 * len(costs), 3.4), sharey=True,
                             squeeze=False)
    for ax, cost in zip(axes[0], costs):
        sub = sweep_df[sweep_df["taste_cost"] == cost]
        for j, sigma in enumerate(dict.fromkeys(sub["sigma_obs"])):
            means = (sub[sub["sigma_obs"] == sigma].groupby("retaste_after", sort=False)["return"]
                     .mean().reindex(order))
            ax.plot([str(k) for k in order], means.to_numpy(), color=SERIES[j], linewidth=2,
                    marker="o", markersize=5, label=f"σ_obs = {sigma}")
        _style(ax, f"Costo de probar = {cost}", "probar cada k pasos", "retorno medio")
    axes[0][-1].legend(frameon=False, labelcolor=TEXT)
    return _save(fig, "taste_sweep", out_dir)


def plot_queue_lengths(lengths_df: pd.DataFrame, quantile: float = 0.9,
                       out_dir: Path | None = FIGURES_DIR):
    """Distribución del largo de la cola, con el N por defecto marcado."""
    n_default = default_window(lengths_df, quantile)
    counts = lengths_df["queue_length"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(counts.index, counts.to_numpy(), width=0.9, color=SERIES[0])
    ax.axvline(n_default + 0.5, color=TEXT, linewidth=1, linestyle="--")
    ax.annotate(f"N por defecto = {n_default}\n(cubre ≥ {quantile:.0%} de los pasos)",
                (n_default + 0.6, counts.max() * 0.9), color=TEXT, fontsize=9)
    _style(ax, "Largo de la cola por paso (FIFO, cocina randomizada)",
           "órdenes esperando", "fracción de pasos")
    return _save(fig, "queue_lengths", out_dir)


def plot_prioritization(prio_df: pd.DataFrame, out_dir: Path | None = FIGURES_DIR):
    """Retorno de FIFO vs. EDF por escenario (media ± desviación estándar)."""
    grouped = prio_df.groupby(["scenario", "policy"], sort=False)["return"]
    means, stds = grouped.mean().unstack(), grouped.std().unstack()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _grouped_bars(ax, means, stds)
    _style(ax, "¿Priorizar importa? FIFO vs. EDF", "escenario", "retorno por episodio")
    return _save(fig, "prioritization", out_dir)


def plot_difficulty(diff_df: pd.DataFrame, out_dir: Path | None = FIGURES_DIR):
    """Retorno y tasa a tiempo de aleatoria vs. FIFO, en dos paneles (sin doble eje)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, metric, label in ((axes[0], "return", "retorno por episodio"),
                              (axes[1], "on_time_rate", "tasa a tiempo")):
        grouped = diff_df.groupby(["scenario", "policy"], sort=False)[metric]
        _grouped_bars(ax, grouped.mean().unstack(), grouped.std().unstack())
        _style(ax, f"Aleatoria vs. FIFO: {label}", "", label)
    axes[1].set_xlabel("escenario", color=TEXT_SECONDARY)
    return _save(fig, "difficulty", out_dir)


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paso 0: calibración sin entrenamiento")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    sweep = taste_sweep(n_episodes=args.episodes)
    best = best_retaste(sweep)
    lengths = queue_lengths(n_episodes=args.episodes)
    prio = prioritization_check(n_episodes=args.episodes)
    diff = difficulty_check(n_episodes=args.episodes)

    for name, df in (("taste_sweep", sweep), ("best_retaste", best), ("queue_lengths", lengths),
                     ("prioritization", prio), ("difficulty", diff)):
        df.to_csv(TABLES_DIR / f"calibration_{name}.csv", index=False)
    plot_taste_sweep(sweep)
    plot_queue_lengths(lengths)
    plot_prioritization(prio)
    plot_difficulty(diff)

    print("Mejor frecuencia de prueba por (costo, σ_obs):")
    print(to_markdown(best))
    print(f"\nN por defecto (cuantil 0.9): {default_window(lengths)}")
    for name, df in (("FIFO vs. EDF", prio), ("Aleatoria vs. FIFO", diff)):
        summary = df.groupby(["scenario", "policy"], sort=False)[["return", "on_time_rate"]].mean()
        print(f"\n{name}:\n{summary.round(3)}")


if __name__ == "__main__":
    main()
