"""Tests del arranque de los notebooks en Colab (notebooks/_setup.py).

No se puede correr Colab aquí: se simula `google.colab` y se registran los
comandos que `setup` mandaría a correr, sin ejecutarlos.
"""

import sys
import types

import pytest

sys.path.insert(0, "notebooks")
import _setup  # noqa: E402


@pytest.fixture
def fake_colab(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    monkeypatch.setattr(_setup, "COLAB_ROOT", tmp_path / "PROY-RL")
    monkeypatch.setattr(_setup.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(_setup.os, "chdir", lambda path: None)
    return calls


def test_colab_clones_the_requested_branch(fake_colab):
    _setup.setup(branch="kitchen-v2")
    clone = fake_colab[0]
    assert clone[:2] == ["git", "clone"] and "--branch" in clone and "kitchen-v2" in clone


def test_colab_updates_an_existing_clone(fake_colab):
    _setup.COLAB_ROOT.mkdir()
    _setup.setup(branch="main")
    git_cmds = [c for c in fake_colab if c[0] == "git"]
    assert not any("clone" in c for c in git_cmds)
    assert any("checkout" in c for c in git_cmds) and any("pull" in c for c in git_cmds)


def test_colab_installs_only_what_colab_lacks(fake_colab):
    # Repo ya clonado, con el requirements.txt real (de ahí sale la versión fijada)
    _setup.COLAB_ROOT.mkdir()
    (_setup.COLAB_ROOT / "requirements.txt").write_text(
        open("requirements.txt", encoding="utf-8").read(), encoding="utf-8")
    _setup.setup()
    pip = [c for c in fake_colab if "pip" in c]
    assert len(pip) == 1
    packages = [arg for arg in pip[0] if "==" in arg]
    assert packages == ["gymnasium==1.4.0"]
