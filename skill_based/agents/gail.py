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
args = parser.parse_args()

lstm_ckpt_path = "../ckpt/memory_lstm/best.pth"
traj_data_path = "../traj_data/real_traj.npy"
traj_data=np.load(traj_data_path, allow_pickle=True)
print("traj_data", traj_data.shape)

trajectories = []
for observation in traj_data:
    observations = [observation[0].squeeze(0), observation[0].squeeze(0)]
    actions=[0]
    infos = [None]
    trajectory = types.Trajectory(
        obs = np.array(observations),
        acts = np.array(actions),
        infos = np.array(infos),
        terminal = True
    )
    trajectories.append(trajectory) 

print("Trajectories generated", f"len: {len(trajectories)}")

p_env = make_vec_env(
    args.env,
    rng=np.random.default_rng(SEED),
    n_envs=4,
    post_wrappers=[
        lambda env, _: ImgObsWrapper(env),
        lambda env, _: TransposeImage(env),
        lambda env, _: LSTMFrameStackWrapper(env, lstm_ckpt_path),
        lambda env, _: FloatRewardWrapper(env),
        lambda env, _: RolloutInfoWrapper(env)
    ],
)
p_env.reset()
learner = PPO(
    env=p_env,
    policy=MlpPolicy,
    batch_size=256,
    ent_coef=0.01,
    learning_rate=3e-4,
    gamma=1,
    n_epochs=20,
    seed=SEED,
    verbose=1
)

optimizer_params = {
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.999)
}


reward_net =DiscRewardNet(
    observation_space=p_env.observation_space,
    action_space=p_env.action_space
)


gail_trainer = GAIL(
    demonstrations=trajectories,
    demo_batch_size=45,
    gen_replay_buffer_capacity=60,
    n_disc_updates_per_round=5,
    venv=p_env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    **{'disc_opt_kwargs':optimizer_params}
)

gail_trainer.train(80000)

# Save the trained PPO model
save_path = os.path.join("./gail_result", "gail_trained_ppo")
learner.save(save_path)
print(f"Trained PPO model saved to {save_path}")

