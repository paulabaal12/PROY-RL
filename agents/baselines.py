"""Políticas de referencia (baselines) para comparar contra DQN y PPO.

Ambas deciden SOLO con la observación (igual que los agentes de RL), nunca con
`info`, que contiene el estado oculto.
"""

import numpy as np

from envs.kitchen_env import (
    BURNT_FROM, OP_DISCARD, OP_SERVE, OP_START, OP_TASTE, RAW_BELOW,
    SINCE_TASTE_CAP, ST_HAS_TASTED, ST_LAST_TASTE, ST_OCCUPIED, ST_SINCE_TASTE,
    ST_VISUAL, STATION_FEATURES, WAIT, action_id,
)


class RandomPolicy:
    """Elige una acción uniforme al azar. Es la cota inferior de desempeño."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


class HeuristicPolicy:
    """Política "sensata" escrita a mano (docs/env_design.md, sección 8).

    Prioridades en cada paso:
      1. Si una estación tiene una prueba recién hecha: servir si está en la
         ventana, descartar si está quemada.
      2. Si hay una estación libre y órdenes en cola: empezar.
      3. Si una estación "se ve" casi lista y no se ha probado hace poco: probar.
      4. Si nada aplica: esperar.
    """

    def __init__(self, n_stations: int = 2, taste_threshold: float = 0.85,
                 retaste_after: int = 2, serve_margin: float = 0.02):
        self.n_stations = n_stations
        self.taste_threshold = taste_threshold
        self.retaste_after = retaste_after
        # Margen para no servir justo en el borde de la ventana, donde el ruido
        # de la prueba (sigma_taste) podría engañar.
        self.serve_margin = serve_margin

    def __call__(self, obs: np.ndarray) -> int:
        stations = [obs[k * STATION_FEATURES:(k + 1) * STATION_FEATURES]
                    for k in range(self.n_stations)]
        queue_has_order = obs[STATION_FEATURES * self.n_stations] > 0.5

        # 1. Actuar sobre pruebas recientes (steps_since_taste == 0 -> la prueba
        #    mide exactamente el c que se evaluará al servir en este paso).
        for k, st in enumerate(stations):
            if st[ST_OCCUPIED] < 0.5 or st[ST_HAS_TASTED] < 0.5:
                continue
            fresh = st[ST_SINCE_TASTE] * SINCE_TASTE_CAP < 0.5
            if not fresh:
                continue
            taste = st[ST_LAST_TASTE]
            if taste >= BURNT_FROM - self.serve_margin:
                return action_id(k, OP_DISCARD)
            if taste >= RAW_BELOW + self.serve_margin:
                return action_id(k, OP_SERVE)

        # 2. Empezar órdenes en estaciones libres
        if queue_has_order:
            for k, st in enumerate(stations):
                if st[ST_OCCUPIED] < 0.5:
                    return action_id(k, OP_START)

        # 3. Probar lo que parece casi listo
        for k, st in enumerate(stations):
            if st[ST_OCCUPIED] < 0.5 or st[ST_VISUAL] < self.taste_threshold:
                continue
            since = st[ST_SINCE_TASTE] * SINCE_TASTE_CAP
            if st[ST_HAS_TASTED] < 0.5 or since >= self.retaste_after:
                return action_id(k, OP_TASTE)

        return WAIT


__all__ = ["RandomPolicy", "HeuristicPolicy"]
