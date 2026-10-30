"""Políticas de referencia (baselines) para comparar contra DQN y PPO.

Deciden SOLO con la observación (igual que los agentes de RL), nunca con
`info`, que contiene el estado oculto.
"""

import numpy as np


class RandomPolicy:
    """Elige una acción uniforme al azar. Es la cota inferior de desempeño."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


__all__ = ["RandomPolicy"]
