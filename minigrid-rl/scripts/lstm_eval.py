import argparse
import json
import os
import random

import numpy as np
import torch
import utils
from lstm import PolicyNetwork
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
    "--render",
    default=False,
    help="render the environment (default: True)",
    action="store_true",
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--history_size",
    type=int,
    default=5,
    help="number of observations to stack for input (default: 5)",
)
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
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="maximum number of steps per episode (default: 1000)",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="number of rounds",
)

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
if args.render:
    env = utils.make_env(args.env, args.seed, render_mode="human", rounds=args.rounds)
else:
    env = utils.make_env(args.env, args.seed, render_mode=None, rounds=args.rounds)


print("Environment loaded\n")

# Initialize and load BC model
model = PolicyNetwork(
    action_dim=env.action_space.n,
    input_dim=3 * 7 * 7,
    hidden_dim=128,
    lstm_hidden_dim=256,
)
state_dict = torch.load(args.model, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("BC model loaded")


# Create a window to view the environment
if args.render:
    env.render()

total_rewards = []
for episode in range(args.episodes):
    obs, _ = env.reset()
    history = [obs["image"]]

    episode_reward = 0
    step_count = 0

    while True:
        if args.render:
            env.render()

        # Prepare the observation history for the model
        state = torch.FloatTensor(history).unsqueeze(0).to(device)
        seq_length = torch.tensor(state.size(1)).unsqueeze(0).to(device)

        # Get action from BC model
        with torch.no_grad():
            logits = model(state, seq_length)
            action = torch.argmax(logits).item()

        # Take the action in the environment
        obs, reward, terminated, truncated, _ = env.step(action)

        step_count += 1
        episode_reward += reward

        # Update history with new observation
        history.append(obs["image"])

        done = terminated or truncated or step_count >= args.max_steps
        if done:
            print(f"Episode {episode + 1} finished with reward {episode_reward}")
            total_rewards.append(episode_reward)
            break

# Print final statistics
avg_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
print(f"\nEvaluation over {args.episodes} episodes:")
print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
