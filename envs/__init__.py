from gymnasium.envs.registration import register

from envs.kitchen_env import KitchenEnv

# Permite crear el ambiente con gym.make("HellKitchen-v0", **kwargs).
# Sin max_episode_steps: el truncamiento a `horizon` pasos lo hace el propio ambiente.
register(id="HellKitchen-v0", entry_point="envs.kitchen_env:KitchenEnv")

__all__ = ["KitchenEnv"]
