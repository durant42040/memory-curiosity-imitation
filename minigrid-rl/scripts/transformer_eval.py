import argparse
import json
import os
import random

import numpy as np
import torch
import utils
from bc_train import TransformerPolicy  # Import your Transformer model architecture
from utils import device

import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True, help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--model", required=True, help="path to the trained BC model weights (REQUIRED)"
)
parser.add_argument(
    "--render", default=True, help="render the environment (default: True)"
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--episodes", type=int, default=10, help="number of episodes to visualize"
)
parser.add_argument(
    "--pause",
    type=float,
    default=0.1,
    help="pause duration between actions (default: 0.1)",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.1,
    help="epsilon for random action selection (default: 0.1)",
)

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
if args.render:
    env = utils.make_env(args.env, args.seed, render_mode="human")
else:
    env = utils.make_env(args.env, args.seed)

print("Environment loaded\n")

# Initialize and load Transformer model
model = TransformerPolicy(action_dim=env.action_space.n)
state_dict = torch.load(args.model, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Transformer model loaded")

# Create a window to view the environment
if args.render:
    env.render()

total_rewards = []
for episode in range(args.episodes):
    obs, _ = env.reset()
    # Instead of fixed history, maintain a growing list of observations
    history = [obs["image"]]
    episode_reward = 0
    step_count = 0

    while True:
        if args.render:
            env.render()

        # Prepare the observation history for the model
        # Stack all historical observations
        state = np.stack(history)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action from Transformer model
        if step_count < 10:
            action = np.random.randint(0, 3)
            print(f"Random action: {action}")
        else:
            with torch.no_grad():
                if random.uniform(0, 1) < args.epsilon:
                    action = random.randint(0, env.action_space.n - 1)
                    print(f"Random action (epsilon): {action}")
                else:
                    # Use the model's get_action method which handles the transformation
                    action = model.get_action(state)
                    print(f"Model action: {action}")

        # Take the action in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        episode_reward += reward

        # Add new observation to history (growing history)
        history.append(obs["image"])

        done = terminated or truncated
        if done:
            print(f"Episode {episode + 1} finished with reward {episode_reward}")
            total_rewards.append(episode_reward)
            break

# Print final statistics
avg_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
print(f"\nEvaluation over {args.episodes} episodes:")
print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
