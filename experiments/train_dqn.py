"""Entrena DQN en HellKitchen-v1.

Uso:
    python -m experiments.train_dqn --window 4 --seed 0 --timesteps 1000000
"""

from experiments.train import cli

if __name__ == "__main__":
    cli("dqn")
