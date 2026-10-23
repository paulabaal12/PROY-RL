"""Muestra un episodio paso a paso en ASCII.

Uso:
    python -m experiments.demo_episode --policy heuristic --seed 0 --steps 40
"""

import argparse

import gymnasium as gym

import envs  # noqa: F401  (registra HellKitchen-v0)
from agents.baselines import HeuristicPolicy, RandomPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="por defecto, el episodio completo")
    args = parser.parse_args()

    env = gym.make("HellKitchen-v0", render_mode="ansi")
    policy = (HeuristicPolicy() if args.policy == "heuristic"
              else RandomPolicy(env.action_space.n, seed=args.seed))

    obs, _ = env.reset(seed=args.seed)
    print(env.render())
    total, t = 0.0, 0
    while args.steps is None or t < args.steps:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        t += 1
        print(env.render())
        print(f"Recompensa: {reward:+.2f} | acumulada: {total:+.2f}\n")
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
