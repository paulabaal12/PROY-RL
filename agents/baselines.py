"""Políticas de referencia (baselines) para comparar contra DQN y PPO.

Deciden SOLO con la observación (igual que los agentes de RL), nunca con
`info`, que contiene el estado oculto.
"""

import numpy as np

from envs.kitchen_config import BURNT_FROM, RAW_BELOW
from envs.kitchen_env import (
    OP_DISCARD, OP_SERVE, OP_TASTE, SINCE_TASTE_CAP, SL_PREP_PROGRESS, SL_PRESENT,
    SL_TIME_TO_DUE, SLOT_FEATURES, ST_AVAILABLE, ST_HAS_TASTED, ST_LAST_TASTE, ST_OCCUPIED,
    ST_SINCE_TASTE, ST_VISUAL, STATION_FEATURES, ActionLayout,
)


class RandomPolicy:
    """Elige una acción uniforme al azar. Es la cota inferior de desempeño."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


class FifoHeuristicPolicy:
    """Chef ingenuo escrito a mano: atiende solo la orden más antigua (lugar 0).

    Prioridades en cada paso:
      1. Si una estación tiene una prueba recién hecha: servir si está en la
         ventana, descartar si está quemada.
      2. Si no prueba nunca (`retaste_after=None`): servir cuando la lectura
         visual diga "listo" y descartar cuando diga "quemado".
      3. Probar una estación que "se ve" casi lista y no se probó hace poco.
      4. Poner al fuego la orden 0 si ya está preparada y hay estación libre.
      5. Preparar la orden 0 si le falta.
      6. Si nada aplica: esperar.

    Probar va antes que preparar: si no, el chef seguiría picando mientras un
    plato rápido (salteado) se quema.
    """

    def __init__(self, window: int, max_stations: int = 3, taste_threshold: float = 0.85,
                 retaste_after: int | None = 2, serve_margin: float = 0.02):
        self.window = window
        self.max_stations = max_stations
        self.layout = ActionLayout(window, max_stations)
        self.taste_threshold = taste_threshold
        self.retaste_after = retaste_after
        # Margen para no servir justo en el borde de la ventana, donde el ruido
        # de la prueba (sigma_taste) podría engañar.
        self.serve_margin = serve_margin

    # Reglas de estación (1 a 3), compartidas con la heurística EDF
    def _station_action(self, stations) -> int | None:
        for k, st in enumerate(stations):
            if st[ST_OCCUPIED] < 0.5 or st[ST_HAS_TASTED] < 0.5:
                continue
            if st[ST_SINCE_TASTE] * SINCE_TASTE_CAP >= 0.5:   # la prueba no es de este paso
                continue
            taste = st[ST_LAST_TASTE]
            if taste >= BURNT_FROM - self.serve_margin:
                return self.layout.station(k, OP_DISCARD)
            if taste >= RAW_BELOW + self.serve_margin:
                return self.layout.station(k, OP_SERVE)

        for k, st in enumerate(stations):
            if st[ST_OCCUPIED] < 0.5:
                continue
            visual = st[ST_VISUAL]
            if self.retaste_after is None:
                if visual >= BURNT_FROM:
                    return self.layout.station(k, OP_DISCARD)
                if visual >= RAW_BELOW + self.serve_margin:
                    return self.layout.station(k, OP_SERVE)
                continue
            since = st[ST_SINCE_TASTE] * SINCE_TASTE_CAP
            if visual >= self.taste_threshold and (
                    st[ST_HAS_TASTED] < 0.5 or since >= self.retaste_after):
                return self.layout.station(k, OP_TASTE)
        return None

    def _split(self, obs: np.ndarray):
        stations = [obs[k * STATION_FEATURES:(k + 1) * STATION_FEATURES]
                    for k in range(self.max_stations)]
        wbase = STATION_FEATURES * self.max_stations
        slots = [obs[wbase + i * SLOT_FEATURES: wbase + (i + 1) * SLOT_FEATURES]
                 for i in range(self.window)]
        free_station = any(st[ST_AVAILABLE] > 0.5 and st[ST_OCCUPIED] < 0.5 for st in stations)
        return stations, slots, free_station

    def __call__(self, obs: np.ndarray) -> int:
        stations, slots, free_station = self._split(obs)
        action = self._station_action(stations)
        if action is not None:
            return action
        first = slots[0]
        if first[SL_PRESENT] > 0.5:
            if first[SL_PREP_PROGRESS] >= 1.0:
                if free_station:
                    return self.layout.place(0)
            else:
                return self.layout.prep(0)
        return self.layout.WAIT


class EdfHeuristicPolicy(FifoHeuristicPolicy):
    """Heurística que SÍ prioriza: hora de entrega más próxima primero (EDF).

    Solo se usa en el Paso 0 (docs/env_design.md, sección 4.2, criterio 4) para
    verificar que poder elegir la orden vale algo. No es un baseline del reporte.

    Mismas reglas de estación que la FIFO, pero entre todos los lugares de la
    ventana: primero pone al fuego la orden preparada con menos tiempo para su
    hora de entrega; si no puede, adelanta la preparación de la orden con menos
    tiempo, aunque las estaciones estén ocupadas.
    """

    def __call__(self, obs: np.ndarray) -> int:
        stations, slots, free_station = self._split(obs)
        action = self._station_action(stations)
        if action is not None:
            return action
        present = [i for i, sl in enumerate(slots) if sl[SL_PRESENT] > 0.5]
        by_due = sorted(present, key=lambda i: slots[i][SL_TIME_TO_DUE])
        if free_station:
            for i in by_due:
                if slots[i][SL_PREP_PROGRESS] >= 1.0:
                    return self.layout.place(i)
        for i in by_due:
            if slots[i][SL_PREP_PROGRESS] < 1.0:
                return self.layout.prep(i)
        return self.layout.WAIT


__all__ = ["RandomPolicy", "FifoHeuristicPolicy", "EdfHeuristicPolicy"]
