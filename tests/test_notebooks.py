"""Los notebooks de cada fase se ejecutan de punta a punta (con pocos episodios).

Son lentos, así que llevan el marcador `slow`: corren con `pytest -m slow`.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = ["00_calibracion.ipynb", "01_baselines.ipynb"]


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes(name, tmp_path, monkeypatch):
    # Pocos episodios, y las tablas y figuras van a una carpeta temporal para no
    # pisar los resultados reales en results/.
    monkeypatch.setenv("KITCHEN_NB_EPISODES", "3")
    monkeypatch.setenv("KITCHEN_NB_OUTPUT_DIR", str(tmp_path))
    nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    assert list(tmp_path.rglob("*.png")), "el notebook no guardó ninguna figura"
