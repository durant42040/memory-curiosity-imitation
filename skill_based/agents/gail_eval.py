import numpy as np
import gymnasium as gym
import torch
import sys
sys.path.append("../../imitations/src")
sys.path.append("../../")
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from torch import nn
from tqdm import tqdm
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import DiscRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, OneHotPartialObsWrapper
import utils
from utils import device
from agents.models import LSTMPolicy
SEED = 42

class TransposeImage(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))

class FloatRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), terminated, truncated, info

class LSTMFrameStackWrapper(ObservationWrapper):
    def __init__(self, env, ckpt_path):
        super().__init__(env)
        self.frames = []
        self.lstm_encoder = LSTMPolicy(
            action_dim=env.action_space.n,
            input_dim=3 * 7 * 7,
            hidden_dim=128,
            lstm_hidden_dim=256,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_encoder.to(device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.lstm_encoder.load_state_dict(state_dict)
        self.lstm_encoder.eval()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = []
        self.frames.append(obs)
        frames_tensor = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in self.frames]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.lstm_encoder(frames_tensor, torch.tensor([len(self.frames)]).to(self.device), return_embedding = True)

        return embedding.squeeze(0).cpu(), None

    def observation(self, obs):
        self.frames.append(obs)
        frames_tensor = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in self.frames]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.lstm_encoder(frames_tensor, torch.tensor([len(self.frames)]).to(self.device), return_embedding = True)
        # Stack frames along the channel dimension
        return embedding.squeeze(0).cpu()



parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", default="MiniGrid-MemoryS7-v0", help="name of the environment to be run (REQUIRED)"
)
parser.add_argument("--model_path", default="./gail_result/gail_trained_ppo.zip", help="path to the trained PPO model")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes to evaluate")
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument("--seed", type=int, default=400, help="random seed")
args = parser.parse_args()

lstm_ckpt_path = "../ckpt/memory_lstm/best.pth"

if args.render:
    env = utils.make_env(args.env, args.seed, render_mode="human")
else:
    env = utils.make_env(args.env, args.seed)

env = ImgObsWrapper(env)
env = TransposeImage(env)
env = LSTMFrameStackWrapper(env, lstm_ckpt_path)
model = PPO.load(args.model_path)

episode_rewards = []
for episode in range(args.episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        if args.render:
            env.render()
        
        # Get action from the model
        action, _ = model.predict(obs, deterministic=True)
        # Execute action
        obs, reward, terminated, truncated, _ = env.step(action)
        # print(obs)
        done = terminated or truncated
        episode_reward += reward
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Calculate and print statistics
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"\nEvaluation over {args.episodes} episodes:")
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


