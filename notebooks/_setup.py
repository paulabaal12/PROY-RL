"""Arranque común de los notebooks: funciona igual en local y en Google Colab.

En Colab clona el repositorio en la rama pedida (o la actualiza si ya estaba
clonado) e instala solo lo que Colab no trae; en local busca la raíz del repo
subiendo desde la carpeta actual. En ambos casos deja la raíz como directorio
de trabajo y la agrega a `sys.path`, para poder importar `envs`, `agents` y
`experiments`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/paulabaal12/PROY-RL.git"
COLAB_ROOT = Path("/content/PROY-RL")
# Colab ya trae numpy, pandas, matplotlib, torch y Jupyter. Reinstalarlos con
# otras versiones dentro de un kernel ya iniciado puede romperlo, así que solo
# se instala lo que falta, con la versión fijada en requirements.txt.
COLAB_MISSING = ("gymnasium",)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _pinned(requirements: Path, names: tuple[str, ...]) -> list[str]:
    """Las líneas `paquete==versión` de requirements.txt para los paquetes pedidos."""
    lines = requirements.read_text(encoding="utf-8").splitlines() if requirements.exists() else []
    pins = [line.strip() for line in lines
            if re.split(r"[=<>]", line.strip())[0].lower() in names]
    return pins or list(names)


def setup(branch: str = "main", repo_url: str = REPO_URL) -> Path:
    """Prepara el entorno y devuelve la raíz del repositorio."""
    if in_colab():
        root = COLAB_ROOT
        if not root.exists():
            subprocess.run(["git", "clone", "--branch", branch, repo_url, str(root)], check=True)
        else:
            # Ya clonado (por ejemplo, al re-ejecutar): traer la versión más nueva de la rama
            subprocess.run(["git", "-C", str(root), "fetch", "origin", branch], check=True)
            subprocess.run(["git", "-C", str(root), "checkout", branch], check=True)
            subprocess.run(["git", "-C", str(root), "pull", "origin", branch], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        *_pinned(root / "requirements.txt", COLAB_MISSING)], check=True)
    else:
        root = Path.cwd().resolve()
        while not (root / "envs").is_dir():
            if root.parent == root:
                raise RuntimeError("No se encontró la raíz del repositorio (carpeta envs/)")
            root = root.parent
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
