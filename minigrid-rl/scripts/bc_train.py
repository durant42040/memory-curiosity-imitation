import argparse
import os

import torch
import wandb
from bc import BehavioralCloning
from lstm import LSTMBehavioralCloning
from transformer import TransformerBehavioralCloning
from utils import make_env, seed

import envs.minigrid


# Configuration block for training
class Config:
    def __init__(self, args):
        self.env_name = args.env
        self.train_data_dir = args.train_data_dir
        self.val_data_dir = args.val_data_dir
        self.epochs = args.epochs
        self.history_size = args.history_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.seed = args.seed
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.name = args.name
        self.save_interval = args.save_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment (REQUIRED)"
    )
    parser.add_argument(
        "--train_data_dir",
        required=True,
        help="directory of expert trajectories (REQUIRED)",
    )
    parser.add_argument("--val_data_dir", help="directory of expert trajectories")
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
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
    parser.add_argument("--save_interval", type=int, default=50)
    args = parser.parse_args()

    seed(args.seed)
    config = Config(args)
    env = make_env(config.env_name, seed=config.seed, rounds=args.rounds)

    wandb.init(
        project="memory",
        name=args.name,
    )

    if args.encoding == "stack":
        bc_agent = BehavioralCloning(env, config)
    elif args.encoding == "lstm":
        bc_agent = LSTMBehavioralCloning(env, config)
    elif args.encoding == "transformer":
        bc_agent = TransformerBehavioralCloning(env, config)

    if not os.path.exists(config.name):
        os.makedirs(config.name)

    # if os.path.exists(f"{config.name}/best.pth"):
    #     bc_agent.policy.load_state_dict(torch.load(f"{config.name}/best.pth"))
    #     print("Loaded pre-trained model.")

    bc_agent.train(config.epochs)


if __name__ == "__main__":
    main()
