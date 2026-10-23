"""Utilidades de seeds para que los experimentos sean reproducibles.

Hay tres fuentes de aleatoriedad independientes que hay que fijar:
1. Los generadores globales de Python (`random`), NumPy y PyTorch, que usan
   los agentes (inicialización de redes, epsilon-greedy, muestreo de minibatches...).
2. El generador interno del ambiente Gymnasium (`env.np_random`), que se fija
   pasando `seed` a `env.reset()`. Es el que decide la estocasticidad de la cocina.
3. Los espacios de acciones/observaciones (`space.sample()`), que tienen su propio
   generador. La política aleatoria de baseline depende de él.
"""

import random

import gymnasium as gym
import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Fija las seeds de `random`, NumPy y PyTorch.

    Con `deterministic_torch=True` PyTorch lanza un error si alguna operación no
    tiene implementación determinista, en vez de dar resultados distintos en
    silencio. En CPU (nuestro caso) prácticamente todas las operaciones lo son.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def seed_env(env: gym.Env, seed: int):
    """Fija las seeds del ambiente y de sus espacios, y devuelve el `reset` inicial.

    Solo hace falta pasar la seed en el primer `reset`: los siguientes episodios
    continúan la misma secuencia del generador, así que toda la corrida queda
    determinada por esta seed.
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env.reset(seed=seed)
