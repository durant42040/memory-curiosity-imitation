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
sys.path.append('/Users/hjko/Project/2024RL/DIFO-on-POMDP/')
sys.path.append('/Users/hjko/Project/2024RL/DIFO-on-POMDP/minigrid-rl')
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet, CnnRewardNet, NormalizedRewardNet, TileRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
import utils
from utils import device
import wandb  # for logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
SEED = 42

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        num_object_types: int = 11,
        num_color_types: int = 6,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        
        # Embedding dimensions
        self.object_dim = 5
        self.color_dim = 3
        total_embedding_dim = self.object_dim + self.color_dim
        
        # Embedding layers
        self.object_embedding = nn.Embedding(num_object_types, self.object_dim)
        self.color_embedding = nn.Embedding(num_color_types, self.color_dim)
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(total_embedding_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create sample input
            sample_obs = torch.zeros((1, 3, observation_space.shape[1], observation_space.shape[2]))
            object_idx = sample_obs[:, 0, :, :].long()
            color_idx = sample_obs[:, 1, :, :].long()
            
            # Get embeddings
            object_emb = self.object_embedding(object_idx)
            color_emb = self.color_embedding(color_idx)
            print(object_emb.shape, color_emb.shape)
            
            # Prepare for CNN
            object_emb = object_emb.permute(0, 3, 1, 2)
            color_emb = color_emb.permute(0, 3, 1, 2)
            x = torch.cat((object_emb, color_emb), dim=1)
            print(x.shape)
            
            n_flatten = self.cnn(x).shape[1]
            print(n_flatten)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract object and color indices
        object_idx = observations[:, 0, :, :].long()
        color_idx = observations[:, 1, :, :].long()
        
        # Get embeddings
        object_emb = self.object_embedding(object_idx)
        color_emb = self.color_embedding(color_idx)
        
        # Prepare for CNN (permute to [batch, channels, height, width])
        object_emb = object_emb.permute(0, 3, 1, 2)
        color_emb = color_emb.permute(0, 3, 1, 2)
        
        # Concatenate embeddings along channel dimension
        x = torch.cat((object_emb, color_emb), dim=1)
        
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
    "--episodes", type=int, default=100, help="number of episodes to visualize"
)
parser.add_argument(
    "--full_obs", action="store_true", default=False, help="use fully observable wrapper for the environment"
)
args = parser.parse_args()

utils.seed(SEED)
env = utils.make_env(args.env, SEED)
if args.full_obs:
    env = FullyObsWrapper(env)
partial_env = utils.make_env(args.env, SEED)
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
    while True:
        action = expert.get_action(obs)
        wrapped_obs = types.maybe_unwrap_dictobs(np.transpose(obs['image'], (2,0,1)))
        observations.append(wrapped_obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        partial_obs,  _, _, _, _ = partial_env.step(action)
        done = terminated | truncated
        # Append the current step's data to the trajectory
        actions.append(int(action))
        infos.append(None)
        expert.analyze_feedback(reward, done)

        if done:
            wrapped_obs = types.maybe_unwrap_dictobs(np.transpose(obs['image'], (2,0,1)))
            observations.append(wrapped_obs)
            break
    
    observations = np.array(observations)
    actions = np.array(actions)
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
        lambda env, _: FullyObsWrapper(env),
        lambda env, _: ImgObsWrapper(env),
        lambda env, _: TransposeImage(env),
        lambda env, _: FloatRewardWrapper(env),
        lambda env, _: RolloutInfoWrapper(env)
    ],
)
p_env.reset()
learner = PPO(
    env=p_env,
    policy=CnnPolicy,
    batch_size=256,
    ent_coef=0.01,
    learning_rate=3e-4,
    gamma=0.95,
    n_epochs=10,
    seed=SEED,
    policy_kwargs=policy_kwargs,
    verbose=1
)
learner.learn(total_timesteps=200000)
save_path = os.path.join("./result", "ppo_trained")
learner.save(save_path)
print(f"Trained PPO model saved to {save_path}")
reward_net = BasicRewardNet(
    observation_space=p_env.observation_space,
    action_space=p_env.action_space,
    normalize_input_layer=RunningNorm,
)
optimizer_params = {
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.999)
}

# reward_net = CnnRewardNet(
#     observation_space=p_env.observation_space,
#     action_space=p_env.action_space,
#     stride = 1, 
#     kernel_size = 1,
# )
# normalize_reward_net = NormalizedRewardNet(
#     base=reward_net,
#     normalize_output_layer=RunningNorm  # or any other normalization layer
# )

reward_net = TileRewardNet(
    observation_space=p_env.observation_space,
    action_space=p_env.action_space,
    num_actions=7,
    num_color_types=6,
    num_states=3,
    num_views=7,
    num_object_types=11 
)

normalize_reward_net = NormalizedRewardNet(
    base=reward_net,
    normalize_output_layer=RunningNorm  # or any other normalization layer
)

class GAILCallback(BaseCallback):
    """Custom callback for logging GAIL training metrics."""
    
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -float('inf')
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes
            )
            
            # Log metrics
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "train/timesteps": self.num_timesteps,
            })
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("best_model")
                
        return True

# Initialize wandb
wandb.init(
    project="gail-minigrid",
    config={
        "env_name": args.env,
        "seed": SEED,
        "batch_size": 256,
        "n_disc_updates_per_round": 16,
        "learning_rate": 3e-4,
        "total_timesteps": 800000,
    }
)

# Create evaluation environment
eval_env = make_vec_env(
    args.env,
    rng=np.random.default_rng(SEED + 1),  # Different seed for eval
    n_envs=1,
    post_wrappers=[
        lambda env, _: FullyObsWrapper(env),
        lambda env, _: ImgObsWrapper(env),
        lambda env, _: TransposeImage(env),
        lambda env, _: FloatRewardWrapper(env),
        lambda env, _: RolloutInfoWrapper(env)
    ],
)

# Create callback
gail_callback = GAILCallback(eval_env=eval_env, eval_freq=5000)

# Modify GAIL trainer initialization to include logging
class LoggingGAIL(GAIL):
    def train_disc(self, *args, **kwargs):
        disc_metrics = super().train_disc(*args, **kwargs)
        wandb.log({
            "discriminator/loss": disc_metrics["disc_loss"],
            "discriminator/acc_expert": disc_metrics["disc_acc_expert"],
            "discriminator/acc_gen": disc_metrics["disc_acc_gen"],
            "train/timesteps": self._global_step,
        })
        return disc_metrics

# Create GAIL trainer with logging
gail_trainer = LoggingGAIL(
    demonstrations=trajectories,
    demo_batch_size=256,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=p_env,
    gen_algo=learner,
    reward_net=normalize_reward_net,
    allow_variable_horizon=True,
    custom_logger=wandb,  # Add wandb logger
    **{'disc_opt_kwargs':optimizer_params}
)

# Train with callback
gail_trainer.train(
    total_timesteps=800000,
    callback=gail_callback
)

# Close wandb run
wandb.finish()

# Save the trained PPO model
save_path = os.path.join("./result", "gail_trained_ppo")
learner.save(save_path)
print(f"Trained PPO model saved to {save_path}")

