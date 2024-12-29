import glob
import json
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences for the transformer.
    We will pad trajectories to the length of the longest sequence in the batch.
    Padding is done by repeating the first frame (as in the original code).
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    trajectories, actions = zip(*batch)

    # Get max sequence length in this batch
    max_len = trajectories[0].shape[0]

    # Pad sequences to max length at the start (front-padding)
    padded_trajectories = []
    for traj in trajectories:
        pad_len = max_len - traj.shape[0]
        if pad_len > 0:
            # Pad with copies of the first frame
            padding = traj[0:1].repeat(pad_len, 1, 1, 1)
            traj = torch.cat([padding, traj], dim=0)
        padded_trajectories.append(traj)

    return torch.stack(padded_trajectories), torch.tensor(actions)


class TransformerExpertDataset(Dataset):
    """
    This dataset class will load entire trajectories from JSON files.
    Each trajectory file contains multiple steps. We store all prefix subsequences
    (from the start of the trajectory up to the current step).

    For each step i in a trajectory, we store [images[:i+1]] and the action at step i.
    The 'images' here are assumed to be partial observations ("partial_image")
    as per the transformer's approach in bc_train.py.
    """

    def __init__(self, data_dir):
        self.trajectories = []
        self.actions = []

        if data_dir is None or not os.path.exists(data_dir):
            # If no data_dir provided or doesn't exist, dataset is empty.
            print(
                "No data directory provided or directory does not exist. Empty dataset."
            )
            return

        traj_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        print(f"Found {len(traj_files)} trajectory files in {data_dir}.")

        for traj_file in tqdm(traj_files, desc=f"Loading trajectories from {data_dir}"):
            with open(traj_file, "r") as f:
                trajectory = [json.loads(line) for line in f]

            # Extract partial images and actions
            # Each step is expected to have a "partial_image" and "action" field
            images = [step["observation"]["image"] for step in trajectory]
            actions = [step["action"] for step in trajectory]

            # For each position in trajectory, store all previous observations
            for i in range(len(images)):
                history = np.stack(images[: i + 1], axis=0)
                self.trajectories.append(history)
                self.actions.append(actions[i])

        # Convert actions to a tensor
        self.actions = torch.LongTensor(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.trajectories[idx]), self.actions[idx]


class TransformerPolicy(nn.Module):
    def __init__(
        self, action_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1
    ):
        super(TransformerPolicy, self).__init__()

        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Input: (batch, 3, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the spatial dimensions (64 * 7 * 7)
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
        )
        self.image_embedding = nn.Linear(7 * 7 * 3, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )  # Positional embedding

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
        batch_size, seq_len, h, w, c = x.shape

        # Reshape input for convolutional layers: (batch * seq, 3, 7, 7)
        x = einops.rearrange(x, "b s h w c -> (b s) c h w")

        # Extract features
        x = self.feature_extractor(x)  # (batch * seq, hidden_dim)
        
        # Reshape for Transformer input: (batch, seq, hidden_dim)
        x = einops.rearrange(x, "(b s) d -> b s d", b=batch_size, s=seq_len)

        # Reshape and flatten images: (batch, seq, height, width, channels) -> (batch, seq, height*width*channels)
        # x = einops.rearrange(x, "b s h w c -> b s (h w c)")

        # # Project each flattened image to hidden dimension
        # x = self.image_embedding(x) 
        # Add positional embeddings
        pos_emb = self.pos_embedding.expand(batch_size, seq_len, -1)
        x = x + pos_emb

        # Pass through transformer
        x = self.transformer(x)

        # Take the last token representation for classification
        x = x[:, -1]  # (batch, hidden_dim)

        return self.fc(x)

    def get_action(self, x):
        """
        x: (seq_len, height, width, channels)
        Evaluate a single sequence and return the best action.
        """
        device = next(self.parameters()).device
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits).item()


class TransformerBehavioralCloning:
    """
    This class handles the training, validation, and evaluation of a Transformer-based
    behavioral cloning agent, in a similar style as LSTMBehavioralCloning in lstm.py.

    It uses TransformerPolicy for the model, and TransformerExpertDataset for the data.
    """

    def __init__(self, env, config):
        self.env = env
        self.device = config.device
        self.action_dim = env.action_space.n
        self.name = config.name
        self.save_interval = config.save_interval

        # Initialize the policy
        # self.policy = TransformerPolicy(
        #     action_dim=self.action_dim,
        #     hidden_dim=256,   # Can be tuned
        #     num_heads=4,      # Can be tuned
        #     num_layers=2,     # Can be tuned
        #     dropout=0.1,      # Can be tuned
        # ).to(self.device)
        self.policy = TransformerPolicy(
            action_dim=self.action_dim,
            hidden_dim=64,  # Reduced hidden dimension
            num_heads=2,  # Reduced number of attention heads
            num_layers=1,  # Single Transformer layer
            dropout=0.1,  # Keep dropout for regularization
        ).to(self.device)

        # Initialize optimizer and loss
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.criterion = nn.CrossEntropyLoss()

        # Load training dataset
        self.train_dataset = TransformerExpertDataset(config.train_data_dir)
        print(f"Training dataset size: {len(self.train_dataset)}")

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Load validation dataset if provided
        self.val_dataset = None
        self.val_dataloader = None
        if config.val_data_dir is not None and os.path.exists(config.val_data_dir):
            self.val_dataset = TransformerExpertDataset(config.val_data_dir)
            print(f"Validation dataset size: {len(self.val_dataset)}")
            if len(self.val_dataset) > 0:
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn,
                )
        else:
            print("No validation data provided or directory not found.")

    def train(self, num_epochs):
        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            self.policy.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for states, actions in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
            ):
                states = states.to(self.device)
                actions = actions.to(self.device)

                # Forward pass
                logits = self.policy(states)
                loss = self.criterion(logits, actions)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accuracy
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == actions).sum().item()
                total_predictions += actions.size(0)
                total_loss += loss.item()
            self.scheduler.step()

            train_accuracy = correct_predictions / total_predictions
            train_loss = total_loss / len(self.train_dataloader)
            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}"
            )

            # Validation step if we have val data
            if self.val_dataloader is not None:
                val_accuracy, val_loss = self.validate()
                success_rate = (val_accuracy - 0.9) / 0.1
                print(
                    f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Success Rate = {success_rate:.4f}"
                )
                wandb.log({"Success Rate": success_rate})
            else:
                # If no validation data, just skip validation
                val_accuracy = train_accuracy
                val_loss = train_loss

            # Save model periodically
            if (epoch + 1) % self.save_interval == 0:
                torch.save(self.policy.state_dict(), f"{self.name}/{epoch+1}.pth")
                print("Saved model checkpoint.")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.policy.state_dict(), f"{self.name}/best.pth")
                print("Saved best model.")

    def validate(self):
        self.policy.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for states, actions in self.val_dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                logits = self.policy(states)
                loss = self.criterion(logits, actions)

                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == actions).sum().item()
                total_predictions += actions.size(0)
                total_loss += loss.item()

        val_accuracy = correct_predictions / total_predictions
        val_loss = total_loss / len(self.val_dataloader)
        return val_accuracy, val_loss

    def evaluate(self, num_episodes=10, render=False):
        """
        Run the agent in the environment for a given number of episodes and report the average reward.
        This is similar to the LSTMBehavioralCloning evaluate method.
        """
        self.policy.eval()
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_states = []

            while not done:
                if render:
                    self.env.render()

                # Append current state
                episode_states.append(state)
                # Convert to tensor
                states_tensor = torch.FloatTensor(np.stack(episode_states, axis=0)).to(
                    self.device
                )

                with torch.no_grad():
                    logits = self.policy(states_tensor.unsqueeze(0))
                    action = torch.argmax(logits).item()

                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward: {avg_reward:.2f}")
        return avg_reward
