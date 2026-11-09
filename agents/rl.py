"""Infraestructura común de DQN y PPO, sobre stable-baselines3.

Ambos algoritmos comparten la misma red, el mismo preprocesamiento (ninguno, porque
la observación ya viene normalizada) y el mismo escalado de recompensa. Así, las
diferencias que se midan vienen del algoritmo y no de la implementación
(docs/env_design.md, sección 4.4).

Sobre el truncamiento (sección 3.9): el ambiente no tiene estados terminales y corta
a los H pasos con `truncated=True`. `DummyVecEnv` lo marca como
`TimeLimit.truncated`, y tanto el replay buffer de DQN como las ventajas de PPO hacen
bootstrap con el valor del último estado en ese caso.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from envs.kitchen_config import BURNT_FROM, RAW_BELOW
from envs.kitchen_env import (
    SL_PREP_PROGRESS, SL_PRESENT, SLOT_FEATURES, ST_AVAILABLE, ST_OCCUPIED, ST_VISUAL,
    STATION_FEATURES, KitchenEnv,
)

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"dqn": DQN, "ppo": PPO}
MASKED_VALUE = -1e8     # valor o logit de una acción inválida: nunca se elige
FINISH_GATE = 0.6       # lectura visual mínima para poder servir o descartar


class ScaleReward(gym.RewardWrapper):
    """Multiplica la recompensa, para que los valores queden cerca de 1 al entrenar.

    Solo se usa para entrenar. `Monitor` va por dentro, de modo que los retornos
    registrados están en la escala real, y el evaluador común usa el ambiente sin esto.
    """

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class PotentialShaping(gym.Wrapper):
    """Reward shaping basado en potencial: F = gamma * Phi(s') - Phi(s) (Ng et al., 1999).

    Sin esto ningún agente aprende. Al explorar, casi todo lo que se sirve sale crudo
    (-10), que es peor que dejar expirar la orden (-8), y el agente aprende a no servir
    nunca. El potencial premia el progreso de cada orden y, por ser una diferencia de
    potenciales, NO cambia cuál es la política óptima.

    Phi suma, por orden: la fracción preparada (hasta `prep`), estar en el fuego
    (`placed`) y el avance de la cocción hacia lo listo (hasta `cook`). Si se quema, ese
    avance se pierde. Usa el nivel real de cocción, pero solo como señal de
    entrenamiento: el agente nunca lo observa y la evaluación usa la recompensa original.
    """

    BURN_SPAN = 0.3      # niveles de cocción en que se pierde el avance una vez quemado

    def __init__(self, env: gym.Env, gamma: float, prep: float = 1.0, placed: float = 1.0,
                 cook: float = 2.0):
        super().__init__(env)
        self.gamma, self.w_prep, self.w_placed, self.w_cook = gamma, prep, placed, cook
        self.potential = 0.0

    def _phi(self) -> float:
        env = self.env.unwrapped
        total = sum(self.w_prep * min(o.prep_done / o.prep_required, 1.0) for o in env.queue)
        for station in env.stations:
            if station.occupied:
                cook = station.cook
                if cook < RAW_BELOW:
                    progress = cook / RAW_BELOW
                else:
                    progress = 1.0 - max(cook - BURNT_FROM, 0.0) / self.BURN_SPAN
                total += self.w_prep + self.w_placed + self.w_cook * max(progress, 0.0)
        return total

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.potential = self._phi()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = self._phi()
        reward += self.gamma * potential - self.potential
        self.potential = potential
        return obs, reward, terminated, truncated, info


def make_train_env(window: int, n_envs: int = 1, kitchen: str | dict = "random",
                   reward_scale: float = 0.1, env_kwargs: dict | None = None,
                   shaping: dict | None = None, gamma: float = 0.99) -> DummyVecEnv:
    """Ambiente vectorizado de entrenamiento.

    `Monitor` va por dentro de los demás wrappers: los retornos que registra son los reales.
    """
    def factory():
        env = Monitor(KitchenEnv(window=window, kitchen=kitchen, **(env_kwargs or {})))
        if shaping is not None:
            env = PotentialShaping(env, gamma, **shaping)
        return ScaleReward(env, reward_scale)
    return DummyVecEnv([factory for _ in range(n_envs)])


def valid_action_mask(obs: torch.Tensor, n_actions: int, max_stations: int = 3) -> torch.Tensor:
    """Qué acciones son válidas (bool, una fila por observación), deducido SOLO de la observación.

    Sigue las reglas de env_design.md, sección 3.4: `wait` siempre; `prep_i` si la orden
    existe y le falta preparación; `place_i` si está preparada y hay estación libre; probar,
    servir y descartar si la estación está ocupada.
    """
    window = (n_actions - 1 - 3 * max_stations) // 2
    stations = obs[:, :STATION_FEATURES * max_stations].reshape(-1, max_stations, STATION_FEATURES)
    occupied = stations[..., ST_OCCUPIED] > 0.5
    free_station = ((stations[..., ST_AVAILABLE] > 0.5) & ~occupied).any(dim=1, keepdim=True)
    start = STATION_FEATURES * max_stations
    slots = obs[:, start:start + SLOT_FEATURES * window].reshape(-1, window, SLOT_FEATURES)
    present = slots[..., SL_PRESENT] > 0.5
    prepped = slots[..., SL_PREP_PROGRESS] >= 1.0
    wait = torch.ones((obs.shape[0], 1), dtype=torch.bool, device=obs.device)
    # Taste, serve, discard. No se sirve ni se bota lo que a simple vista sigue claramente crudo.
    finishable = occupied & (stations[..., ST_VISUAL] >= FINISH_GATE)
    station_ops = torch.stack([occupied, finishable, finishable], dim=2).reshape(obs.shape[0], -1)
    return torch.cat([wait, present & ~prepped, present & prepped & free_station, station_ops], dim=1)


class MaskedQNetwork(QNetwork):
    """Red Q que nunca elige ni usa como objetivo una acción inválida."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q_values = super().forward(obs)
        return q_values.masked_fill(~valid_action_mask(obs, q_values.shape[1]), MASKED_VALUE)


class MaskedDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


class MaskedDQN(DQN):
    """DQN cuya exploración aleatoria también se limita a las acciones válidas."""

    def _random_valid(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation.reshape(-1, *self.observation_space.shape))
        masks = valid_action_mask(obs, int(self.action_space.n)).numpy()
        return np.array([np.random.choice(np.flatnonzero(mask)) for mask in masks])

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self._random_valid(observation)
            return (action if observation.ndim > 1 else action[0]), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action = self._random_valid(self._last_obs)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)


class MaskedActorCritic(ActorCriticPolicy):
    """Política de PPO cuya distribución solo da probabilidad a las acciones válidas."""

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            action_logits=logits.masked_fill(~self._mask, MASKED_VALUE))

    def _remember_mask(self, obs: torch.Tensor) -> None:
        self._mask = valid_action_mask(obs, int(self.action_space.n))

    def forward(self, obs, deterministic=False):
        self._remember_mask(obs)
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._remember_mask(obs)
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._remember_mask(obs)
        return super().get_distribution(obs)


def build_model(algo: str, env, config: dict, seed: int, device: str = "cpu") -> BaseAlgorithm:
    """Crea el modelo con la red y los hiperparámetros de `config`."""
    arch = list(config["net_arch"])
    policy_kwargs = {"activation_fn": torch.nn.ReLU,
                     "net_arch": arch if algo == "dqn" else {"pi": arch, "vf": arch}}
    masked = config.get("action_mask", False)
    cls = (MaskedDQN if algo == "dqn" else PPO) if masked else ALGORITHMS[algo]
    policy = "MlpPolicy"
    if masked:
        policy = MaskedDQNPolicy if algo == "dqn" else MaskedActorCritic
    return cls(policy, env, gamma=config["gamma"], seed=seed, device=device,
               policy_kwargs=policy_kwargs, verbose=0, **config["hyperparameters"])


class ModelPolicy:
    """Adapta un modelo entrenado a la interfaz `obs -> acción` del evaluador común.

    Es determinista en ambos algoritmos: argmax de Q en DQN y moda de la política en PPO.
    """

    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def __call__(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_policy(algo: str, path: str | Path) -> ModelPolicy:
    """Carga un modelo guardado y lo devuelve listo para el evaluador o el visualizador."""
    return ModelPolicy(ALGORITHMS[algo].load(str(path), device="cpu"))
