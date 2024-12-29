import argparse

import gymnasium as gym
from scripts.finetune import (  # Update as per your wrapper imports
    FrameStackWrapper,
    ImageOnlyWrapper,
)
from stable_baselines3 import PPO

from skill_based.utils import make_env


def evaluate_model(env, model_path, num_episodes=10, render=False):
    # Load the saved PPO model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    total_reward = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        print(f"Episode {episode + 1}: Reward = {episode_reward}")
        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment to evaluate"
    )
    parser.add_argument(
        "--model_path", required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="number of episodes for evaluation"
    )
    parser.add_argument(
        "--render", action="store_true", help="render the environment during evaluation"
    )

    args = parser.parse_args()

    # Create the environment
    env = FrameStackWrapper(
        ImageOnlyWrapper(make_env(args.env, render_mode="human")), n_frames=5
    )

    # Evaluate the model
    evaluate_model(
        env, args.model_path, num_episodes=args.num_episodes, render=args.render
    )
