from gymnasium.envs.registration import register

from envs.kitchen_env import KitchenEnv

# Permite crear el ambiente con gym.make("HellKitchen-v1", **kwargs).
# v1 corresponde al diseño v2 (docs/env_design.md); cambian los espacios, por eso cambia el id.
# Sin max_episode_steps: el truncamiento a `horizon` pasos lo hace el propio ambiente.
register(id="HellKitchen-v1", entry_point="envs.kitchen_env:KitchenEnv")

__all__ = ["KitchenEnv"]
