"""Configuración de la cocina: platos, randomización de dominio y escenarios fijos.

Las referencias "(sección X)" apuntan a docs/env_design.md.
"""

from dataclasses import dataclass

import numpy as np

# --- Platos (sección 3.1) ---------------------------------------------------
# Índices: 0 = vegetales salteados, 1 = huevo, 2 = carne (steak).
DISHES = ("stirfry", "egg", "steak")
DISH_PREP = {0: 7, 1: 4, 2: 1}                                  # pasos de preparación (fijos)
DISH_RATE_RANGE = {0: (0.14, 0.20), 1: (0.08, 0.14), 2: (0.04, 0.08)}  # r ~ U(lo, hi)
DISH_DUE_RANGE = {0: (25, 35), 1: (20, 30), 2: (35, 50)}        # tau_due ~ U{lo, ..., hi}

# Calidad según el nivel real de cocción c, igual para todos los platos
RAW_BELOW = 1.0     # c < 1.0          -> crudo
BURNT_FROM = 1.3    # c >= 1.3         -> quemado; en medio -> bien cocido


def classify(cook: float) -> str:
    """Calidad de la comida según su nivel real de cocción."""
    if cook < RAW_BELOW:
        return "raw"
    if cook >= BURNT_FROM:
        return "burnt"
    return "ok"


# --- Randomización de dominio (sección 3.8) ---------------------------------
RUSH_LENGTH = 20     # pasos que dura una hora pico
RUSH_FACTOR = 2.5    # la probabilidad de llegada se multiplica por esto durante la pico

DEFAULT_RANGES = {
    "n_stations": (2, 3),          # enteros inclusive
    "arrival_prob": (0.06, 0.16),
    "sigma_obs": (0.05, 0.25),
    "n_rushes": (0, 2),            # enteros inclusive
}

# Escenarios fijos de evaluación (sección 4.6)
_MEDIUM = {"n_stations": 2, "arrival_prob": 0.10, "sigma_obs": 0.15, "n_rushes": 1}
PRESETS = {
    "easy": {"n_stations": 3, "arrival_prob": 0.06, "sigma_obs": 0.05, "n_rushes": 0},
    "medium": dict(_MEDIUM),
    "hard": {"n_stations": 2, "arrival_prob": 0.16, "sigma_obs": 0.25, "n_rushes": 2},
    "low_noise": {**_MEDIUM, "sigma_obs": 0.05},
    "high_noise": {**_MEDIUM, "sigma_obs": 0.25},
    "single_station": {**_MEDIUM, "n_stations": 1},
}


@dataclass(frozen=True)
class KitchenParams:
    """Cocina sorteada para un episodio."""

    n_stations: int
    arrival_prob: float
    sigma_obs: float
    rushes: tuple[tuple[int, int], ...]   # pares (inicio, fin exclusivo)

    def arrival_prob_at(self, t: int) -> float:
        if any(start <= t < end for start, end in self.rushes):
            return min(1.0, RUSH_FACTOR * self.arrival_prob)
        return self.arrival_prob

    def to_dict(self) -> dict:
        return {"n_stations": self.n_stations, "arrival_prob": self.arrival_prob,
                "sigma_obs": self.sigma_obs, "rushes": [list(r) for r in self.rushes]}


def sample_kitchen(rng: np.random.Generator, horizon: int, spec: str | dict = "random",
                   ranges: dict = DEFAULT_RANGES) -> KitchenParams:
    """Sortea la cocina de un episodio.

    `spec` es "random" (todo se sortea con `ranges`), el nombre de un preset o un
    dict que sobreescribe claves de "medium". Siempre se sacan los mismos números
    de `rng` y luego se aplica el preset, para que el resto del episodio (que usa
    otros generadores) no dependa del escenario elegido.
    """
    drawn = {
        "n_stations": int(rng.integers(ranges["n_stations"][0], ranges["n_stations"][1] + 1)),
        "arrival_prob": float(rng.uniform(*ranges["arrival_prob"])),
        "sigma_obs": float(rng.uniform(*ranges["sigma_obs"])),
        "n_rushes": int(rng.integers(ranges["n_rushes"][0], ranges["n_rushes"][1] + 1)),
    }
    # Una posible pico por mitad del episodio, así nunca se traslapan.
    half = horizon // 2
    starts = [int(rng.integers(j * half, (j + 1) * half - RUSH_LENGTH + 1)) for j in range(2)]
    single_half = int(rng.integers(2))   # con una sola pico, en qué mitad cae

    if spec == "random":
        values = drawn
    elif isinstance(spec, str):
        values = PRESETS[spec]
    else:
        values = {**_MEDIUM, **spec}

    n_rushes = values["n_rushes"]
    chosen = starts if n_rushes == 2 else ([starts[single_half]] if n_rushes == 1 else [])
    return KitchenParams(
        n_stations=values["n_stations"],
        arrival_prob=values["arrival_prob"],
        sigma_obs=values["sigma_obs"],
        rushes=tuple((s, s + RUSH_LENGTH) for s in chosen),
    )
