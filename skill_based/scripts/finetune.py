import argparse

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium import ObservationWrapper, Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from wandb.integration.sb3 import WandbCallback

import envs.minigrid
from skill_based.utils import make_env


class FrameStackWrapper(Wrapper):
    def __init__(self, env, n_frames=10):
        super(FrameStackWrapper, self).__init__(env)
        self.n_frames = n_frames
        self.frames = np.zeros(
            (n_frames,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames,) + env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Unpack tuple
        self.frames[:] = 0
        self.frames[-1] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs[0] if isinstance(obs, tuple) else obs
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs
        return self.frames.copy(), reward, done, truncated, info


class ImageOnlyWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ImageOnlyWrapper, self).__init__(env)
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


class IdentityFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(IdentityFeatureExtractor, self).__init__(observation_space, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class CustomPolicyNetwork(nn.Module):
    def __init__(self, features_dim: int = 128):
        super(CustomPolicyNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.fc_net = nn.Sequential(
            nn.Linear(1024, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, env.action_space.n),
        )
        self.value_net = nn.Sequential(
            nn.Linear(1024, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
        )
        self.latent_dim_pi = 7
        self.latent_dim_vf = 1

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc_net(x), self.value_net(x)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features)[0]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features)[1]


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=IdentityFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomPolicyNetwork(features_dim=128)

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]


def evaluate(env, model, num_episodes=10):
    total_reward = 0
    max_steps = 245
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done and step < max_steps:
            step += 1
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            episode_reward += reward

    return total_reward / num_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run"
    )
    parser.add_argument("--ckpt_path", help="path to the pre-trained model")
    parser.add_argument(
        "--name", required=True, help="path to save the model checkpoints"
    )
    parser.add_argument("--history_size", type=int, default=5)

    args = parser.parse_args()

    run = wandb.init(
        project="memory",
        name=args.name,
        sync_tensorboard=True,
    )

    env = FrameStackWrapper(ImageOnlyWrapper(make_env(args.env)), n_frames=10)
    obs = env.reset()

    model = PPO(
        policy=CustomActorCriticPolicy,
        env=env,
        verbose=1,
        batch_size=64,
        learning_rate=1e-4,
        tensorboard_log=args.name,
    )

    if args.ckpt_path:
        model.policy.mlp_extractor.load_state_dict(
            torch.load(args.ckpt_path), strict=False
        )
        print("Loaded BC Policy")

    model.learn(
        total_timesteps=200000,
        reset_num_timesteps=False,
        callback=WandbCallback(
            gradient_save_freq=100,
            verbose=2,
        ),
    )

    model.save(args.name)
    run.finish()
