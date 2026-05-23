import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import SAC
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        reward_type="dense",
    )

    # TODO: add randomization wrapper here
    # Define a wider mass range if UDR is selected, otherwise keep it constant (1.0 kg)
    mass_range = (0.5, 3.0) if args.sampling_strategy == "udr" else (1.0, 1.0)
    env = RandomizationWrapper(env, mass_range=mass_range, mode=args.sampling_strategy)

    # TODO: create model and train it
    # Using MultiInputPolicy and optimized hyperparameters for robotic manipulation
    model = SAC(
        "MultiInputPolicy", 
        env, 
        learning_rate=1e-3,  # Faster learning for the dense reward structure
        batch_size=1024,     # High batch size ensures stability in continuous control tasks
        gamma=0.95,          # Lower gamma makes the agent focus on immediate task completion
        verbose=1
    )
    model.learn(total_timesteps=args.timesteps)

    save_name = f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: model.save(save_name)
    model.save(save_name)


if __name__ == "__main__":
    main()