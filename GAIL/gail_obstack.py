import numpy as np
import gymnasium as gym
import torch
import sys
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper
from torch import nn
from tqdm import tqdm
from collections import deque
sys.path.append('/home/hjko/Projects/DIFO-on-POMDP/')
sys.path.append('/home/hjko/Projects/DIFO-on-POMDP/minigrid-rl')
sys.path.append('/home/hjko/Projects/DIFO-on-POMDP/imitation/src')
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet, TileRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, OneHotPartialObsWrapper
import utils
from utils import device
SEED = 42

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        print(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_obs = torch.zeros((1, observation_space.shape[0], observation_space.shape[1], observation_space.shape[2]))
            x = sample_obs
            n_flatten = self.cnn(x).shape[1]
            print(n_flatten)
            self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        
        # Pass through CNN and linear layers
        x = self.cnn(x)
        x = self.linear(x)
        
        return x

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

class FrameStackWrapper(ObservationWrapper):
    def __init__(self, env, n_frames=5):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space to handle stacked frames
        obs_shape = env.observation_space.shape
        # New shape will be (channels * n_frames, height, width)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * n_frames, obs_shape[1], obs_shape[2]),
            dtype=np.uint8
        )
        print("frame_stack", self.observation_space.shape)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(np.zeros_like(obs))
        return self.observation(obs), info

    def observation(self, obs):
        self.frames.append(obs)
        # Stack frames along the channel dimension
        return np.concatenate(list(self.frames), axis=0)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", default="MiniGrid-Empty-16x16-v0", help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--model", default="MiniGrid-Empty-16x16-v0_ppo_seed1_24-11-30-16-49-37", help="name of the trained model (REQUIRED)"
)
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="select the action with highest probability (default: False)",
)
parser.add_argument(
    "--memory", action="store_true", default=False, help="add a LSTM to the model"
)
parser.add_argument(
    "--text", action="store_true", default=False, help="add a GRU to the model"
)
parser.add_argument(
    "--episodes", type=int, default=10000, help="number of episodes to visualize"
)
parser.add_argument(
    "--full_obs", action="store_true", default=False, help="use fully observable wrapper for the environment"
)
parser.add_argument(
    "--onehot_obs", action="store_true", default=False, help="use onehot partial observation wrapper for the environment"
)
parser.add_argument(
    "--n_frames", type=int, default=5, help="number of frames to stack"
)
args = parser.parse_args()

utils.seed(SEED)
env = utils.make_env(args.env, SEED)
if args.full_obs:
    env = FullyObsWrapper(env)
partial_env = utils.make_env(args.env, SEED)
if args.onehot_obs:
    partial_env = OneHotPartialObsWrapper(partial_env)
env.reset()
partial_env.reset()
print("Environment loaded\n")

model_dir = "../minigrid-rl/storage/"
model_dir = os.path.join(model_dir, args.model)
print(model_dir)
print(env.observation_space)
print(env.action_space)
expert = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    argmax=args.argmax,
    use_memory=args.memory,
    use_text=args.text,
)
print("Expert loaded\n")

trajectories = []
for episode in tqdm(range(args.episodes), desc="Collecting trajectories"):
    obs, _ = env.reset()
    partial_obs, _ = partial_env.reset()
    observations = []
    actions = []
    infos = []
    dones = []
    done = False
    action = 2
    obs_stack = deque(maxlen=args.n_frames)
    initial_obs = types.maybe_unwrap_dictobs(np.transpose(partial_obs['image'], (2,0,1)))
    for _ in range(args.n_frames):
        obs_stack.append(np.zeros_like(initial_obs))

    while True:
        current_obs = types.maybe_unwrap_dictobs(np.transpose(partial_obs['image'], (2,0,1)))
        obs_stack.append(current_obs)
        pre_action = action
        pre_obs = obs_stack
        stacked_obs = np.concatenate(list(obs_stack), axis=0)
        observations.append(stacked_obs)
        action = expert.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        partial_obs, _, _, _, _ = partial_env.step(action)
        done = terminated | truncated
        actions.append(int(action))
        infos.append(None)
        dones.append(done)
        expert.analyze_feedback(reward, done)

        if done:
            for _ in range(20):
                d_obs = []
                d_acts =[]
                d_infos = []
                d_obs.append(stacked_obs)
                current_obs = types.maybe_unwrap_dictobs(np.transpose(partial_obs['image'], (2,0,1)))
                obs_stack.append(current_obs)
                stacked_obs = np.concatenate(list(obs_stack), axis=0)
                d_obs.append(stacked_obs)
                d_acts.append(pre_action)
                d_infos.append(None)
                trajectory = types.Trajectory(
                    obs = np.array(d_obs),
                    acts = np.array(d_acts),
                    infos = np.array(d_infos),
                    terminal = True
                )
                trajectories.append(trajectory)
            observations.append(stacked_obs)
            break
    
    observations = np.array(observations)
    actions = np.array(actions)
    dones = np.array(dones)
    infos = np.array(infos)

    trajectory = types.Trajectory(
        obs = observations,
        acts = actions,
        infos = infos,
        terminal = True
    )
    trajectories.append(trajectory)

print("Trajectories generated", f"len: {len(trajectories)}")

# rollouts is a list of traj, to get the observations, rollouts[0].obs to attach the array that store all the obs in the traj.

p_env = make_vec_env(
    args.env,
    rng=np.random.default_rng(SEED),
    n_envs=4,
    post_wrappers=[
        lambda env, _: ImgObsWrapper(env),
        lambda env, _: TransposeImage(env),
        lambda env, _: FrameStackWrapper(env, n_frames=args.n_frames),
        lambda env, _: FloatRewardWrapper(env),
        lambda env, _: RolloutInfoWrapper(env)
    ],
    # vec_env_cls=gym.vector.SyncVectorEnv
)
print("p_env:",p_env.observation_space)
p_env.reset()
learner = PPO(
    env=p_env,
    policy=CnnPolicy,
    batch_size=256,
    ent_coef=0.01,
    learning_rate=1e-4,
    gamma=0.95,
    n_epochs=10,
    seed=SEED,
    policy_kwargs=policy_kwargs,
    verbose=1
)

optimizer_params = {
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.999)
}
reward_net = TileRewardNet(
    observation_space=p_env.observation_space,
    action_space=p_env.action_space,
    num_actions=7,
    num_color_types=6,
    num_states=3,
    num_views=7,
    num_object_types=11 
)
gail_trainer = GAIL(
    demonstrations=trajectories,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=p_env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)

gail_trainer.train(800000)

# Save the trained PPO model
save_path = os.path.join("./result", "gail_trained_ppo")
learner.save(save_path)
print(f"Trained PPO model saved to {save_path}")


