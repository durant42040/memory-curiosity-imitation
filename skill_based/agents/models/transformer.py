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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import make_env, seed

import envs.minigrid


# class TransformerPolicy(nn.Module):
#     """
#     A Transformer-based policy network that encodes a sequence of observations
#     and outputs an action probability distribution. Similar to what was done inline
#     in bc_train.py, but moved here for modularity.
#     """

#     def __init__(self, action_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1):
#         super(TransformerPolicy, self).__init__()

#         # Image embedding layers: Flatten each frame (7*7*3)
#         self.image_embedding = nn.Linear(7 * 7 * 3, hidden_dim)
#         # Learnable positional embedding for each token
#         self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output layers
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#         )

#     def forward(self, x, return_embedding=False):
#         """
#         x: (batch, seq_len, height, width, channels)
#         """
#         batch_size, seq_len, h, w, c = x.shape

#         # Flatten spatial dimensions
#         x = einops.rearrange(x, "b s h w c -> b s (h w c)")
#         # Embed each frame
#         x = self.image_embedding(x)  # (batch, seq, hidden_dim)

#         # Add positional embeddings
#         pos_emb = self.pos_embedding.expand(batch_size, seq_len, -1)
#         x = x + pos_emb

#         # Pass through transformer
#         x = self.transformer(x)

#         # Take the last token representation for classification
#         x = x[:, -1]  # (batch, hidden_dim)

#         if return_embedding:
#             return x
#         # Output logits
#         logits = self.fc(x) 
#         return logits

#     def get_action(self, x):
#         """
#         x: (seq_len, height, width, channels)
#         Evaluate a single sequence and return the best action.
#         """
#         device = next(self.parameters()).device
#         x = torch.FloatTensor(x).unsqueeze(0).to(device)
#         with torch.no_grad():
#             logits = self.forward(x)
#             return torch.argmax(logits).item()


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

    def forward(self, x, return_embedding=False):
        batch_size, seq_len, h, w, c = x.shape

        # Reshape input for convolutional layers: (batch * seq, 3, 7, 7)
        x = einops.rearrange(x, "b s h w c -> (b s) c h w")

        # Extract features
        x = self.feature_extractor(x)  # (batch * seq, hidden_dim)

        # Reshape for Transformer input: (batch, seq, hidden_dim)
        x = einops.rearrange(x, "(b s) d -> b s d", b=batch_size, s=seq_len)

        # Reshape and flatten images: (batch, seq, height, width, channels) -> (batch, seq, height*width*channels)
        # x = einops.rearrange(x, "b s h w c -> b s (h w c)")

        # Project each flattened image to hidden dimension
        # x = self.image_embedding(x)
        # Add positional embeddings
        pos_emb = self.pos_embedding.expand(batch_size, seq_len, -1)
        x = x + pos_emb

        # Pass through transformer
        x = self.transformer(x)

        # Take the last token representation for classification
        x = x[:, -1]  # (batch, hidden_dim)
        if return_embedding:
            return x
        logits = self.fc(x)

        return logits

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