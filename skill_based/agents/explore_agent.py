from agents.models import Discriminator
import argparse
import datetime
import sys
import time
import tensorboardX
import torch_ac
import utils
from model import ACModel

import torch

from utils import device

import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

from agents.models import LSTMPolicy, Discriminator, TransformerPolicy


class GAILPPOAlgo(torch_ac.PPOAlgo):
    def __init__(self, envs, acmodel, device,
                 frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence,
                 optim_eps, clip_eps, epochs, batch_size, preprocess_obss, disc_ckpt, disc_reg=False, encoder=None, policy_ckpt_path=None):
        
        super().__init__(envs, acmodel, device, frames_per_proc, discount, lr, gae_lambda,
                        entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                        optim_eps, clip_eps, epochs, batch_size, preprocess_obss)
        
        self.encoder_name = encoder
        self.discriminator = None
        if encoder == "lstm":
            self.encoder = LSTMPolicy(
                action_dim=envs[0].action_space.n,
                input_dim=3 * 7 * 7,
                hidden_dim=128,
                lstm_hidden_dim=256,
            )
            self.discriminator = Discriminator(input_dim = 256 ,regularized=disc_reg)
        elif encoder == "transformer":
            self.encoder = TransformerPolicy(
                action_dim=envs[0].action_space.n,
                hidden_dim=64,  # Reduced hidden dimension
                num_heads=2,  # Reduced number of attention heads
                num_layers=1,  # Single Transformer layer
                dropout=0.1,  # Keep dropout for regularization
            ).to(self.device)
            self.discriminator = Discriminator(input_dim = 64,regularized=disc_reg)
        self.discriminator.load_state_dict(torch.load(disc_ckpt))
        self.discriminator.to(device)
        print(f"Discriminator loaded from {disc_ckpt}")
        state_dict = torch.load(policy_ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.to(device)
        self.obs_history = [[] for _ in range(len(envs))]  # History for each env
        self.envs = envs
    def collect_experiences(self):
        """Collects experiences and overrides reward computation"""
        exps, logs = super().collect_experiences()
        # print("print(dir(exps.obs[0]))")
        # print(dir(exps.obs[0]))
        # print("First obs keys:", exps.obs[0].keys())  # See what keys are in the observation
        # print("First obs['image'] type:", type(exps.obs[0].image))
        # print("First obs['image'] shape:", exps.obs[0].image.shape)
        # print("History type:", type(self.obs_history[0]))
        # print("History length:", len(self.obs_history[0]))
        # if len(self.obs_history[0]) > 0:
        #     print("History element type:", type(self.obs_history[0][0]))
        
        total_reward = 0
        total_score = 0
        for i in range(len(exps)):
            env_id = i % len(self.envs)
            
            # Update history with new observation using .image for DictList
            obs_img = exps.obs[i].image  # Using attribute access
            
            # Permute the image to (3, 7, 7) as expected by the LSTM
            # print(f"original shape: {obs_img.shape}")
            # obs_img = obs_img.permute(2, 0, 1)
            # print(f"permute shape: {obs_img.shape}")
            
            # Add to history
            self.obs_history[env_id].append(obs_img)
            
            if len(self.obs_history[env_id]) > 0:
                # Stack the history tensors directly without converting to FloatTensor
                history_tensor = torch.stack(self.obs_history[env_id])
                history_tensor = history_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
                # print(f"history_tensor shape: {history_tensor.shape}")
                # Get embedding
                with torch.no_grad():
                    if self.encoder_name == "lstm":
                        embedding = self.encoder(history_tensor, 
                                                torch.tensor([len(self.obs_history[env_id])]).to(self.device), 
                                                return_embedding=True)
                    elif self.encoder_name == "transformer":
                        embedding = self.encoder(history_tensor, return_embedding=True)
                # Compute reward using discriminator
                raw_score = self.discriminator(embedding)
                disc_reward = torch.log(raw_score)
                total_score += raw_score.item()
                total_reward += disc_reward.item()
                exps.reward[i] = disc_reward.item()
            
            # Clear history if episode done
            if i == len(exps) - 1:
                print(f"Mean disc score:{total_score/len(exps)}")
                print(f"Mean reward: {total_reward/len(exps)}")
                self.obs_history[env_id] = []
        
        return exps, logs
    

class ExploreAgent:
    def __init__(self, config, env, device):
        self.algo = None