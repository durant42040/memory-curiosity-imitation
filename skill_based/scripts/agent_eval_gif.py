import argparse
import json
import os
import random
from PIL import Image
import io

import gymnasium as gym
from scripts.finetune import (  # Update as per your wrapper imports
    FrameStackWrapper,
    ImageOnlyWrapper,
)

import numpy as np
import torch
import utils
from agents import BCAgent, ExploreAgent, RouterAgent
from utils import device
from stable_baselines3 import PPO

import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

# Parse arguments
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def update(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, value)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True, help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--map_size", type=int, default=11, help="size of the environment (default: 11)"
)
parser.add_argument(
    "--agent", required=True, help="path to the trained BC model weights (REQUIRED)", choices=["bc", "explore", "router", "finetune", "fake"]
)

parser.add_argument(
    "--policy_ckpt_path",
    type=str,
    default=None,
    help="path to the trained agent's policy weights (REQUIRED)",
)

parser.add_argument(
    "--encoder",
    default="lstm",
    help="type of encoder to use for trajectory encoding (default: lstm)",
)
parser.add_argument(
    "--save_traj_path",
    default="traj_data",
    type=str,
)
parser.add_argument(
    "--exp_alg",
    type=str,
    default="ppo",
    help="exploration algorithm to use (default: ppo)",
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
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
)
parser.add_argument(
    "--gif_path",
    type=str,
    default="episode_recordings",
    help="directory to save GIF recordings",
)

args = parser.parse_args()
args.policy_ckpt_path = f"ckpt/memory_{args.encoder}_{args.map_size}/best.pth"

# Create output directory if it doesn't exist
os.makedirs(args.gif_path, exist_ok=True)

# Set seed for all randomness sources
utils.seed(args.seed)

print(f"Device: {device}\n")

# Load environment with rgb_array render mode for GIF creation
env = utils.make_env(args.env, args.seed, render_mode="rgb_array", rounds=args.rounds)
print("Environment loaded\n")

# Initialize and load agents
config = Config(**vars(args))
custom_params = {
    "exp_name": "bc_inference"
}
config.update(custom_params)

encodingAgent = BCAgent(config, env, device)

if args.agent == "bc":
    agent = BCAgent(config, env, device)
elif args.agent == "explore":
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=False,
        use_memory=True,
        use_text=False
    )
    print("explore agent loaded!")
elif args.agent == "router":
    agent = RouterAgent(config, env, device)
elif args.agent == "finetune":
    agent = PPO.load(args.policy_ckpt_path)
    env = FrameStackWrapper(ImageOnlyWrapper(env), n_frames=5)
elif args.agent == "fake":
    pass
else:
    raise ValueError("Invalid agent type")

embeddings = []
labels = []
total_rewards = []

for episode in range(args.episodes):
    obs, _ = env.reset()
    history = [obs["image"]]
    frames = []  # Store frames for GIF creation

    episode_reward = 0
    step_count = 0

    while True:
        # Get RGB array from environment
        frame = env.render()
        # if frame is None:  # Sometimes render returns None before a step is fully complete
        #     frame = np.zeros((env.observation_space.shape[:2] + (3,)), dtype=np.uint8)
        frames.append(Image.fromarray(frame))

        # Prepare the observation history for the model
        state = torch.FloatTensor(history).unsqueeze(0).to(device)
        seq_length = torch.tensor(state.size(1)).unsqueeze(0).to(device)
        
        if args.encoder == "lstm":
            embedding = encodingAgent.policy_net(state, seq_length, return_embedding=True)
        elif args.encoder == "transformer":
            embedding = encodingAgent.policy_net(state, return_embedding=True)

        # Get action from agent
        with torch.no_grad():
            if args.agent == "finetune":
                action, _ = agent.predict(state)
            elif args.agent == "explore":
                action = agent.get_action(obs)
            elif args.agent == "fake":
                action = torch.randint(0, 3, (1,), device='cuda').item()
            elif args.agent == "bc":
                if agent.encoder == "lstm":
                    logits = agent.policy_net(state, seq_length)
                    action = torch.argmax(logits).item()
                elif agent.encoder == "transformer":
                    logits = agent.policy_net(state)
                    action = torch.argmax(logits).item()
            
            embedding_np = embedding.cpu().numpy()
            embeddings.append(embedding_np)
            labels.append(1 if args.agent == "bc" else 0)

        # Take action in environment
        obs, reward, terminated, truncated, _ = env.step(action)

        step_count += 1
        episode_reward += reward

        # Update history with new observation
        history.append(obs["image"])

        done = terminated or truncated or step_count >= args.max_steps
        if done:
            print(f"Episode {episode + 1} finished with reward {episode_reward} using {step_count} steps")
            # Save episode as GIF
            gif_file = os.path.join(args.gif_path, f"episode_{episode + 1}.gif")
            valid_frames = [frame for frame in frames if frame is not None]
            if valid_frames:
                valid_frames[0].save(
                    gif_file,
                    save_all=True,
                    append_images=valid_frames[1:],
                    duration=100,  # milliseconds
                    loop=0
                )
            print(f"Saved episode recording to {gif_file}")
            total_rewards.append(episode_reward)
            break

# Save trajectory data if specified
if args.save_traj_path is not None:
    embeddings = np.array(embeddings).squeeze(1)
    labels = np.array(labels)
    traj_numpy_path = os.path.join(args.save_traj_path, f"{args.agent}_{args.encoder}_traj_{args.map_size}.npy")
    np.save(traj_numpy_path, {"embeddings": embeddings, "labels": labels})

# Print final statistics
avg_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
print(f"\nEvaluation over {args.episodes} episodes:")
print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"GIF recordings saved in: {args.gif_path}")