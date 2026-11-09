"""Tests del Paso 0: calibración sin entrenamiento (docs/env_design.md, sección 4.2)."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from experiments import calibrate  # noqa: E402


def test_taste_sweep_has_one_row_block_per_combo():
    df = calibrate.taste_sweep(taste_costs=(0.0, 0.5), sigmas=(0.05, 0.25),
                               retaste=(None, 2), n_episodes=2)
    assert len(df) == 2 * 2 * 2 * 2
    assert set(df["retaste_after"]) == {"nunca", 2}
    assert (df.groupby(["taste_cost", "sigma_obs", "retaste_after"]).size() == 2).all()


def test_taste_cost_only_shifts_return_by_tastes():
    df = calibrate.taste_sweep(taste_costs=(0.0, 0.5), sigmas=(0.15,), retaste=(2,), n_episodes=2)
    free = df[df["taste_cost"] == 0.0].reset_index(drop=True)
    paid = df[df["taste_cost"] == 0.5].reset_index(drop=True)
    assert ((free["return"] - paid["return"] - 0.5 * free["tasted"]).abs() < 1e-9).all()


def test_best_retaste_flags_interior():
    rows = [(0.2, 0.15, k, r) for k, r in (("nunca", 1.0), (1, 2.0), (3, 5.0), (6, 4.0))]
    rows += [(1.0, 0.15, k, r) for k, r in (("nunca", 9.0), (1, 2.0), (3, 5.0), (6, 4.0))]
    df = pd.DataFrame(rows, columns=["taste_cost", "sigma_obs", "retaste_after", "return"])
    best = calibrate.best_retaste(df).set_index("taste_cost")
    assert best.loc[0.2, "best_retaste"] == 3 and best.loc[0.2, "interior"]
    assert best.loc[1.0, "best_retaste"] == "nunca" and not best.loc[1.0, "interior"]


def test_default_window_is_quantile():
    lengths = pd.DataFrame({"queue_length": [1] * 90 + [5] * 10})
    assert calibrate.default_window(lengths, quantile=0.9) == 1
    lengths = pd.DataFrame({"queue_length": [1] * 80 + [5] * 20})
    assert calibrate.default_window(lengths, quantile=0.9) == 5
    assert calibrate.default_window(pd.DataFrame({"queue_length": [0] * 10})) == 1


def test_checks_run_and_return_frames(tmp_path):
    lengths = calibrate.queue_lengths(n_episodes=2)
    assert len(lengths) == 2 * 200
    prio = calibrate.prioritization_check(n_episodes=2)
    assert set(prio["policy"]) == {"fifo", "edf", "triage"}
    assert set(prio["scenario"]) == {"random", "hard"}
    diff = calibrate.difficulty_check(n_episodes=2)
    assert set(diff["policy"]) == {"random", "fifo"}

    sweep = calibrate.taste_sweep(taste_costs=(0.2,), sigmas=(0.15,), retaste=(None, 2),
                                  n_episodes=2)
    figures = [
        calibrate.plot_taste_sweep(sweep, out_dir=tmp_path),
        calibrate.plot_queue_lengths(lengths, out_dir=tmp_path),
        calibrate.plot_prioritization(prio, out_dir=tmp_path),
        calibrate.plot_difficulty(diff, out_dir=tmp_path),
    ]
    assert all(fig is not None for fig in figures)
    assert len(list(tmp_path.glob("calibration_*.png"))) == 4
