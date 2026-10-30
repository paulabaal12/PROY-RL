"""Arranque común de los notebooks: funciona igual en local y en Google Colab.

En Colab clona el repositorio (si hace falta) e instala las dependencias; en
local busca la raíz del repo subiendo desde la carpeta actual. En ambos casos
deja la raíz como directorio de trabajo y la agrega a `sys.path`, para poder
importar `envs`, `agents` y `experiments`.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/paulabaal12/PROY-RL.git"
COLAB_ROOT = Path("/content/PROY-RL")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup(repo_url: str = REPO_URL) -> Path:
    """Prepara el entorno y devuelve la raíz del repositorio."""
    if in_colab():
        root = COLAB_ROOT
        if not root.exists():
            subprocess.run(["git", "clone", repo_url, str(root)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(root / "requirements.txt")], check=True)
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
