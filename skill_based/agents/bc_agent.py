from agents.models import LSTMPolicy, TransformerPolicy
import torch
from utils import device


class BCAgent:
    def __init__(self, config, env, device):
        self.device = device
        self.encoder = config.encoder
        self.env = env
        if self.encoder == "lstm":
            self.policy_net = LSTMPolicy(
                action_dim=self.env.action_space.n,
                input_dim=3 * 7 * 7,
                hidden_dim=128,
                lstm_hidden_dim=256,
            )
        elif self.encoder == "transformer":
            print("loading transformer policy")
            self.policy_net = TransformerPolicy(
                action_dim=self.env.action_space.n,
                hidden_dim=64,  # Reduced hidden dimension
                num_heads=2,  # Reduced number of attention heads
                num_layers=1,  # Single Transformer layer
                dropout=0.1,  # Keep dropout for regularization
            ).to(self.device)
        else:
            print("no such encoder")
        state_dict = torch.load(config.policy_ckpt_path, map_location=self.device)
        print(f"Loading policy network from {config.policy_ckpt_path}")
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(self.device)
        self.policy_net.eval()
        print("policy network loaded")
