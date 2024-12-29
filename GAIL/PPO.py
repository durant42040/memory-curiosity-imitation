import argparse
import numpy as np
from stable_baselines3 import PPO
import gym
import sys
sys.path.append('/home/hjko/Projects/DIFO-on-POMDP/')
sys.path.append('/home/hjko/Projects/DIFO-on-POMDP/minigrid-rl')
import utils
from envs.minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, OneHotPartialObsWrapper
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
import envs.minigrid

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="name of the environment")
parser.add_argument("--model_path", required=True, help="path to the trained PPO model")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes to evaluate")
parser.add_argument("--render", action="store_true", help="render the environment")
parser.add_argument("--seed", type=int, default=400, help="random seed")
parser.add_argument("--full_obs", action="store_true", help="use full observation space")
parser.add_argument("--onehot_obs", action="store_true", help="use partial observation space")
args = parser.parse_args()

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


# Set seed
utils.seed(args.seed)

# Create environment
if args.render:
    env = utils.make_env(args.env, args.seed, render_mode="human")
else:
    env = utils.make_env(args.env, args.seed)

# Load the trained PPO model
if args.full_obs:
    env = FullyObsWrapper(env)
if args.onehot_obs:
    env = OneHotPartialObsWrapper(env)
env = ImgObsWrapper(env)
env = TransposeImage(env)
model = PPO.load(args.model_path)

# Evaluate the model
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
