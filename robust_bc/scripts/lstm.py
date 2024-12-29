import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import utils


class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing trajectory JSON files
        """
        self.states_arr = []
        self.actions_arr = []
        self.seq_lengths = []

        traj_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        print(f"Found {len(traj_files)} trajectory files.")

        for traj_file in tqdm(traj_files, desc="Loading trajectories"):
            with open(traj_file, "r") as f:
                trajectory = [json.loads(line) for line in f]

            images = [step["partial_image"] for step in trajectory]
            actions = [step["action"] for step in trajectory]

            for i in range(len(images)):
                self.states_arr.append(images[: i + 1])
                self.seq_lengths.append(i + 1)
                self.actions_arr.append(actions[i])

        self.states = pad_sequence(
            [torch.tensor(state, dtype=torch.float32) for state in self.states_arr],
            batch_first=True,
        )
        self.actions = torch.LongTensor(self.actions_arr)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.seq_lengths[idx]
    
    def add_item(self, state, action, seq_length):
        self.states_arr.append(state)
        self.actions_arr.append(action)
        self.seq_lengths.append(seq_length)
        self.states_arr.pop(0)
        self.actions_arr.pop(0)
        self.seq_lengths.pop(0)
        self.states = torch.FloatTensor(pad_sequence(
            [torch.tensor(state, dtype=torch.float32) for state in self.states_arr],
            batch_first=True,
        ))
        self.actions = torch.LongTensor(self.actions_arr)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, lstm_hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = LSTM(input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)

        self.action_fc_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.expert_fc_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, seq_lengths, get_expert=False):
        if (get_expert):
            with torch.no_grad():
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
            return self.expert_fc_net(encoded)
        else:
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
            return self.action_fc_net(encoded)

    def get_action(self, x):
        seq_length = len(x)
        with torch.no_grad():
            logits = self.forward(torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.Tensor([seq_length]).long())
            return torch.argmax(logits).item()
        
    def get_expert(self, x):
        seq_length = len(x)
        with torch.no_grad():
            logits = self.forward(torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.Tensor([seq_length]).long())
            return torch.argmax(logits).item()

class RobustLSTMBehavioralCloning:
    def __init__(self, env, partial_env, config, robust_env_discount=0.9):
        self.env = env
        self.partial_env = partial_env
        self.device = config.device

        self.action_dim = env.unwrapped.get_action_space()
        self.policy = PolicyNetwork(
            action_dim=env.unwrapped.get_action_space(),
            input_dim=3 * 7 * 7,
            hidden_dim=128,
            lstm_hidden_dim=256,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.name = config.name

        self.train_data_size = config.train_data_size
        self.val_data_size = config.val_data_size
        self.collect_data_interval = config.collect_data_interval

        env_unwrapped = env.unwrapped
        self.expert = utils.ValueIteration(env_unwrapped, discount_factor=0.9)
        self.robust_expert = utils.ValueIteration(env_unwrapped, discount_factor=0.9, robust=True)
        self.robust_max_steps = config.robust_max_steps
        self.robust_env_discount = robust_env_discount

        self.train_dataset = ExpertDataset(config.train_data_dir)
        self.val_dataset = ExpertDataset(config.val_data_dir)
        self.train_dataloader = None
        self.val_dataloader = None
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
        self.save_interval = config.save_interval

        self.batch_size = config.batch_size

    def train(self, num_epochs):
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            if epoch + 1 > 20 and (epoch + 1) % self.collect_data_interval == 0:
                self.evaluate()
                self.collect_data()
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
            print(
                f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
            )

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

    def evaluate(self, num_episodes=5):
        self.policy.eval()
        self.env.unwrapped.render_mode = 'human'
        total_rewards = []
        state_rewards = np.zeros_like(self.env.unwrapped.state_rewards)
        for _ in range(num_episodes):
            episode_reward = 0
            _, _ = self.env.unwrapped.reset()
            partial_obs, _ = self.partial_env.reset()
            partial_obs_history = []
            state_history = []
            done = False
            while not done:
                partial_obs = partial_obs['image']
                partial_obs_history.append(partial_obs.copy())
                with torch.no_grad():
                    action = self.policy.get_action(partial_obs_history)
                _, _, _, _, _  = self.env.step(action)
                state_history.append(self.env.unwrapped.agent_information_to_state())
                partial_obs, reward, terminated, truncated, _ = self.partial_env.step(action)
                done = terminated | truncated
                episode_reward += reward
            total_rewards.append(episode_reward)
            episode_reward -= 0.1
            if (episode_reward < 0):
                for state_index in state_history:
                    state_rewards[state_index] += episode_reward
            # self.env.unwrapped.render_mode = None
        state_rewards /= num_episodes
        # state_rewards += self.robust_env_discount * self.env.unwrapped.state_rewards
        # state_rewards /= (1 + self.robust_env_discount)
        state_rewards += self.env.unwrapped.state_rewards
        self.env.unwrapped.set_state_rewards(state_rewards)
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        # for i in range(self.env.unwrapped.width):
        #     for j in range(self.env.unwrapped.height):
        #         print(action_rewards[(i * self.env.unwrapped.width + j) * 3 + 0], end=' ')
        #     print()
        # for i in range(7 ** 2 * 4 * 3 ** 2):
        #     if action_rewards[i] > 0:
        #         print(i, action_rewards[i])
        # print(action_rewards.sum())
        self.env.unwrapped.render_mode = None
        return np.mean(total_rewards)
    
    def collect_data(self):
        # self.env.unwrapped.render_mode = 'human'
        for traj_count in range(self.train_data_size + self.val_data_size):
            print(traj_count, end='\r')
            partial_obs_history, action_history = [], []
            obs, _ = self.env.reset()
            prev_partial_obs, _ = self.partial_env.reset()
            step_count = 0
            self.expert.run()
            self.robust_expert.run()
            while True:
                # if step_count < 0:
                if step_count < 5:
                    action = self.robust_expert.get_action(obs)
                    step_count += 1
                else:
                    action = self.expert.get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                partial_obs,  _, _, _, _ = self.partial_env.step(action)
                done = terminated | truncated

                # Append the current step's data to the trajectory
                partial_obs_history.append(prev_partial_obs['image'].copy())
                action_history.append(action)
                
                self.expert.analyze_feedback(reward, done)

                prev_partial_obs = partial_obs

                if done:
                    if traj_count < self.train_data_size:
                        for i in range(len(action_history)):
                            self.train_dataset.add_item(partial_obs_history[:i+1], action_history[i], i+1)
                    else:
                        for i in range(len(action_history)):
                            self.val_dataset.add_item(partial_obs_history[:i+1], action_history[i], i+1)
                    break
            self.env.unwrapped.render_mode = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
                        