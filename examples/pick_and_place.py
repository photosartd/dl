import gymnasium as gym

from dl.rl import GreedyTrainer
from dl.rl.agents.td3 import TD3
from dl.rl.utils.config import TD3Config
from dl.rl.utils.networks import AlgorithmFactory


def main():
    env = gym.make("FetchPickAndPlace-v2", render_mode="human")
    config = TD3Config(
        name=AlgorithmFactory.TD3_alg,
        networks={
            "actor"
        }
    )
    agent = TD3(config)
    trainer = GreedyTrainer()


if __name__ == '__main__':
    main()
