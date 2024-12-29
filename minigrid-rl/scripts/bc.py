import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def create_history_stack(images, history_size, i):
    """
    Function to create a history stack of size `history_size`.
    Args:
        images: List of image frames.
        history_size: Number of frames to include in the stack.
        i: Current index in the image list.
    Returns:
        A numpy array representing the stacked history.
    """
    if i + 1 < history_size:
        history = [images[0] for _ in range(history_size - 1 - i)] + images[: i + 1]
    else:
        history = images[i - history_size + 1 : i + 1]
    return np.stack(history, axis=0)


class ExpertDataset(Dataset):
    def __init__(self, data_dir, history_size=5):
        """
        Args:
            data_dir: Directory containing trajectory JSON files
            history_size: Number of steps to include in the history
        """
        self.history_size = history_size
        self.states = []
        self.actions = []

        traj_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        print(f"Found {len(traj_files)} trajectory files.")

        for traj_file in tqdm(traj_files, desc="Loading trajectories"):
            with open(traj_file, "r") as f:
                trajectory = [json.loads(line) for line in f]

            images = [step["observation"]["image"] for step in trajectory]
            actions = [step["action"] for step in trajectory]

            for i in range(len(images)):
                history = create_history_stack(images, history_size, i)
                self.states.append(history)
                self.actions.append(actions[i])

        self.states = torch.FloatTensor(self.states)
        self.actions = torch.LongTensor(self.actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        # self.conv_net = nn.Sequential(
        #     nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        #     nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        # )
        # self.fc_net = nn.Sequential(
        #     nn.Linear(64 * 1 * 1 * 1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, action_dim),
        # )

        # for history size 5, 10
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
            nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample further
        )

        # for history size 1
        # self.conv_net = nn.Sequential(
        #     nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        # )

        self.fc_net = nn.Sequential(
            nn.Linear(512, hidden_dim),  # history size 5
            # nn.Linear(1024, hidden_dim),  # history size 10
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_net(x)

    def get_action(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits).item()


class BehavioralCloning:
    def __init__(self, env, config):
        self.env = env
        self.device = config.device
        self.history_size = config.history_size

        self.action_dim = env.action_space.n
        self.policy = PolicyNetwork(self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.name = config.name

        self.train_dataset = ExpertDataset(
            config.train_data_dir, history_size=self.history_size
        )
        self.val_dataset = ExpertDataset(
            config.val_data_dir, history_size=self.history_size
        )

        self.save_interval = config.save_interval

        print(f"train dataset size: {len(self.train_dataset)}")
        print(f"val dataset size: {len(self.val_dataset)}")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def train(self, num_epochs):
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            self.policy.train()
            total_loss, correct_predictions, total_predictions = 0, 0, 0

            for states, actions in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                states, actions = states.to(self.device), actions.to(self.device)
                logits = self.policy(states)
                loss = self.criterion(logits, actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=1)

                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
                total_loss += loss.item()

            epoch_accuracy = correct_predictions / total_predictions
            epoch_loss = total_loss / len(self.train_dataloader)
            print(
                f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}"
            )

            # Validation
            val_accuracy, val_loss = self.validate()
            success_rate = (val_accuracy - 0.9) / 0.1
            print(
                f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Success Rate = {success_rate:.4f}"
            )
            wandb.log({"Success Rate": success_rate})

            if (epoch + 1) % self.save_interval == 0:
                torch.save(self.policy.state_dict(), f"{self.name}/{epoch + 1}.pth")
                print("Saved model.")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.policy.state_dict(), f"{self.name}/best.pth")
                print("Saved best model.")

    def validate(self):
        self.policy.eval()
        total_loss, correct_predictions, total_predictions = 0, 0, 0

        with torch.no_grad():
            for states, actions in self.val_dataloader:
                states, actions = states.to(self.device), actions.to(self.device)

                logits = self.policy(states)
                loss = self.criterion(logits, actions)

                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
                total_loss += loss.item()

        val_accuracy = correct_predictions / total_predictions
        val_loss = total_loss / len(self.val_dataloader)
        return val_accuracy, val_loss

    def evaluate(self, num_episodes=10, render=False):
        self.policy.eval()
        total_rewards = []
        for _ in range(num_episodes):
            state, done, episode_reward = self.env.reset(), False, 0
            while not done:
                if render:
                    self.env.render()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.policy.get_action(state_tensor)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        return np.mean(total_rewards)
