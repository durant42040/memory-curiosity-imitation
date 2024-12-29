import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LSTMPolicy(nn.Module):
    def __init__(self, action_dim, input_dim, hidden_dim=128, lstm_hidden_dim=256):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = LSTM(input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)

        self.fc_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x, seq_lengths, return_embedding=False):
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
        if return_embedding:
            return encoded
        # Pass through the fully connected layers
        return self.fc_net(encoded)

    def get_action(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits).item()
