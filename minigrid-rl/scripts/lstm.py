import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing trajectory JSON files
        """
        self.states = []
        self.actions = []
        self.seq_lengths = []

        traj_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        print(f"Found {len(traj_files)} trajectory files.")

        for traj_file in tqdm(traj_files, desc="Loading trajectories"):
            with open(traj_file, "r") as f:
                trajectory = [json.loads(line) for line in f]

            images = [step["partial_image"] for step in trajectory]
            actions = [step["action"] for step in trajectory]

            for i in range(len(images)):
                self.states.append(images[: i + 1])
                self.seq_lengths.append(i + 1)
                self.actions.append(actions[i])

        self.states = pad_sequence(
            [torch.tensor(state, dtype=torch.float32) for state in self.states],
            batch_first=True,
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.seq_lengths[idx]


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, input_dim, hidden_dim=128, lstm_hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = LSTM(input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)

        self.fc_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x, seq_lengths):
        # Reshape the input: (batch, seq_length, feature_dim)
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size, seq_length, -1)  # Flatten spatial dimensions

        # Sort sequences by length in descending order for LSTM
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        # Pack the padded sequence
        packed_x = pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # Pass through the LSTM
        packed_out, (h_n, c_n) = self.lstm(packed_x)

        # Restore the original order of the batch
        _, unperm_idx = perm_idx.sort(0)

        # Use the last hidden state of the LSTM
        encoded = h_n[-1]  # Hidden state from the last LSTM layer
        encoded = encoded[unperm_idx]  # Restore original order for consistency

        # Pass through the fully connected layers
        return self.fc_net(encoded)

    def get_action(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits).item()


class LSTMBehavioralCloning:
    def __init__(self, env, config):
        self.env = env
        self.device = config.device

        self.action_dim = env.action_space.n
        self.policy = PolicyNetwork(
            action_dim=env.action_space.n,
            input_dim=3 * 7 * 7,
            hidden_dim=128,
            lstm_hidden_dim=256,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.name = config.name

        self.train_dataset = ExpertDataset(config.train_data_dir)
        self.val_dataset = ExpertDataset(config.val_data_dir)
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

            for states, actions, seq_length in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                states, actions, seq_length = (
                    states.to(self.device),
                    actions.to(self.device),
                    seq_length.to(self.device),
                )
                logits = self.policy(states, seq_length)
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
            for states, actions, seq_length in self.val_dataloader:
                states, actions, seq_length = (
                    states.to(self.device),
                    actions.to(self.device),
                    seq_length.to(self.device),
                )

                logits = self.policy(states, seq_length)
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
