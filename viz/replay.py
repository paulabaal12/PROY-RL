"""Carga de replays grabados con viz.record."""

import json
from dataclasses import dataclass
from pathlib import Path


class ReplayError(ValueError):
    """El archivo de replay no existe o no tiene el formato esperado."""


@dataclass(frozen=True)
class Step:
    t: int
    action: int | None
    kind: str | None      # wait, prep, place, taste, serve, discard (None en el reset)
    index: int | None     # lugar de la ventana o estación
    reward: float
    info: dict
    visual: list | None = None   # lectura visual ruidosa de cada estación (lo que ve el agente)


@dataclass(frozen=True)
class Replay:
    header: dict
    steps: list[Step]     # steps[0] es el estado después del reset

    @property
    def window(self) -> int:
        return self.header["window"]

    @property
    def n_stations(self) -> int:
        return self.steps[0].info["kitchen"]["n_stations"]

    @property
    def max_stations(self) -> int:
        return self.header.get("max_stations", 3)

    @property
    def patience(self) -> int:
        return self.header["env_params"]["patience"]

    @property
    def horizon(self) -> int:
        return self.header["env_params"]["horizon"]


def load_replay(path: str | Path) -> Replay:
    path = Path(path)
    if not path.is_file():
        raise ReplayError(f"No existe el replay {path}")
    try:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except json.JSONDecodeError as e:
        raise ReplayError(f"{path} no es JSON Lines válido: {e}") from e
    if not lines or lines[0].get("kind") != "header":
        raise ReplayError(f"{path} no empieza con un encabezado de replay")
    steps = [Step(t=d["t"], action=d["action"], kind=d["action_kind"], index=d["action_index"],
                  reward=d["reward"], info=d["info"], visual=d.get("visual_cook")) for d in lines[1:]]
    if not steps:
        raise ReplayError(f"{path} no tiene pasos")
    return Replay(header=lines[0], steps=steps)


def compatibility_warnings(replays: list[Replay]) -> list[str]:
    """Avisos para los paneles que no comparten seed o cocina con el primero."""
    warnings = []
    ref = replays[0].header
    for i, replay in enumerate(replays[1:], start=1):
        for key in ("seed", "kitchen"):
            if replay.header.get(key) != ref.get(key):
                warnings.append(f"Panel {i + 1}: {key} = {replay.header.get(key)!r}, "
                                f"distinto del panel 1 ({ref.get(key)!r}); la comparación no es justa")
    return warnings
