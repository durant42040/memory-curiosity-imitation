import argparse
import glob
import json
import os

import einops
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from bc import BehavioralCloning
from scripts.lstm import RobustLSTMBehavioralCloning
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import make_env, seed
from envs.minigrid.wrappers import FullyObsWrapper

import envs.minigrid


# Configuration block for training
class Config:
    def __init__(self, args):
        self.env_name = args.env
        self.epochs = args.epochs
        self.history_size = args.history_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.seed = args.seed
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.name = args.name
        self.save_interval = args.save_interval
        self.robust_max_steps = args.robust_max_steps
        self.train_data_size = args.train_data_size
        self.val_data_size = args.val_data_size
        self.collect_data_interval = args.collect_data_interval
        self.train_data_dir = args.train_data_dir
        self.val_data_dir = args.val_data_dir


class TransformerPolicy(nn.Module):
    def __init__(
        self, action_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1
    ):
        super(TransformerPolicy, self).__init__()

        # Image embedding layers
        self.image_embedding = nn.Linear(
            7 * 7 * 3, hidden_dim
        )  # Flatten and project each image
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )  # Learnable position embedding

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape and flatten images: (batch, seq, height, width, channels) -> (batch, seq, height*width*channels)
        x = einops.rearrange(x, "b s h w c -> b s (h w c)")

        # Project each flattened image to hidden dimension
        x = self.image_embedding(x)  # (batch, seq, hidden_dim)

        # Add positional embeddings
        pos_emb = self.pos_embedding.expand(batch_size, seq_len, -1)
        x = x + pos_emb

        # Pass through transformer
        x = self.transformer(x)

        # Use the last token's representation for prediction
        x = x[:, -1]  # (batch, hidden_dim)

        # Generate action logits
        return self.fc(x)

    def get_action(self, x):
        device = next(self.parameters()).device
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits).item()


class TransformerExpertDataset(Dataset):
    def __init__(self, data_dir):
        """
        Modified to store entire trajectories without fixed history size
        """
        self.trajectories = []
        self.actions = []

        traj_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        print(f"Found {len(traj_files)} trajectory files.")

        for traj_file in tqdm(traj_files, desc="Loading trajectories"):
            with open(traj_file, "r") as f:
                trajectory = [json.loads(line) for line in f]

            # Store full trajectory
            images = [step["partial_image"] for step in trajectory]
            actions = [step["action"] for step in trajectory]

            # For each position in trajectory, store all previous observations
            for i in range(len(images)):
                history = np.stack(
                    images[: i + 1], axis=0
                )  # Include all previous frames
                self.trajectories.append(history)
                self.actions.append(actions[i])

        # Convert to tensors (don't stack trajectories as they have different lengths)
        self.actions = torch.LongTensor(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.trajectories[idx]), self.actions[idx]


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    trajectories, actions = zip(*batch)

    # Get max sequence length in this batch
    max_len = trajectories[0].shape[0]

    # Pad sequences to max length
    padded_trajectories = []
    for traj in trajectories:
        pad_len = max_len - traj.shape[0]
        if pad_len > 0:
            # Pad with copies of the first frame
            padding = traj[0:1].repeat(pad_len, 1, 1, 1)
            traj = torch.cat([padding, traj], dim=0)
        padded_trajectories.append(traj)

    return torch.stack(padded_trajectories), torch.tensor(actions)


class TransformerBehavioralCloning:
    def __init__(self, env, config):
        self.env = env
        self.device = config.device

        self.action_dim = env.action_space.n
        self.policy = TransformerPolicy(
            action_dim=self.action_dim,
            hidden_dim=256,  # You can adjust these hyperparameters
            num_heads=8,
            num_layers=3,
            dropout=0.1,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Note: No history_size needed anymore since we're using full trajectories
        self.dataset = TransformerExpertDataset(config.data_dir)
        self.name = config.name
        print(f"Dataset size: {len(self.dataset)}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,  # Use custom collate function for variable-length sequences
        )

        self.save_interval = config.save_interval

    def train(self, num_epochs):
        best_accuracy = 0
        for epoch in range(num_epochs):
            self.policy.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for states, actions in tqdm(
                self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                # Move everything to device
                states = states.to(
                    self.device
                )  # shape: (batch, seq_len, height, width, channels)
                actions = actions.to(self.device)

                # Forward pass
                logits = self.policy(states)
                loss = self.criterion(logits, actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
                total_loss += loss.item()

            # Calculate epoch metrics
            epoch_accuracy = correct_predictions / total_predictions
            epoch_loss = total_loss / len(self.dataloader)
            print(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}"
            )

            if (epoch + 1) % self.save_interval == 0:
                torch.save(self.policy.state_dict(), f"{self.name}/{epoch + 1}.pth")
                print("Saved model.")

            # Save best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(self.policy.state_dict(), f"{self.name}/best.pth")
                print("Saved best model.")

    def evaluate(self, num_episodes=10, render=False):
        self.policy.eval()
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_states = []  # Keep track of all states in episode

            while not done:
                if render:
                    self.env.render()

                # Add current state to history
                episode_states.append(state)
                # Convert episode_states to correct format (seq_len, height, width, channels)
                states_tensor = torch.FloatTensor(np.stack(episode_states, axis=0))

                # Get action from policy
                action = self.policy.get_action(states_tensor)

                # Take step in environment
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward: {avg_reward:.2f}")
        return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment (REQUIRED)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--eval_epochs", type=int, default=10, help="number of eval epochs"
    )
    parser.add_argument(
        "--history_size", type=int, default=5, help="number of observations to stack"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cuda", help="device to train on")
    parser.add_argument("--name", type=str, default="bc", help="name of the experiment")
    parser.add_argument("--encoding", type=str, default="stack", help="encoding type")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--robust_max_steps", type=int, default=5)
    parser.add_argument("--train_data_size", type=int, default=5)
    parser.add_argument("--val_data_size", type=int, default=1)
    parser.add_argument("--collect_data_interval", type=int, default=50)
    parser.add_argument("--train_data_dir", type=str, default="expert_data/train_data")
    parser.add_argument("--val_data_dir", type=str, default="expert_data/val_data")
    args = parser.parse_args()

    seed(args.seed)
    config = Config(args)
    if args.render:
        env = make_env(config.env_name, seed=config.seed, rounds=args.rounds, render_mode="human")
    else:
        env = make_env(config.env_name, seed=config.seed, rounds=args.rounds)
    env = FullyObsWrapper(env)
    partial_env = make_env(config.env_name, seed=config.seed, rounds=args.rounds)

    # if args.encoding == "stack":
    #     bc_agent = BehavioralCloning(env, config)
    # elif args.encoding == "lstm":
    #     bc_agent = LSTMBehavioralCloning(env, config)
    # elif args.encoding == "transformer":
    #     bc_agent = TransformerBehavioralCloning(env, config)

    bc_agent = RobustLSTMBehavioralCloning(env, partial_env, config)

    if not os.path.exists(config.name):
        os.makedirs(config.name)

    # if os.path.exists(f"{config.name}/best.pth"):
    #     bc_agent.policy.load_state_dict(torch.load(f"{config.name}/best.pth"))
    #     print("Loaded pre-trained model.")

    bc_agent.train(config.epochs)
    


if __name__ == "__main__":
    main()
