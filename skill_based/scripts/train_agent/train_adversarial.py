import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from scripts.train_agent.train_discriminator import DiscriminatorTrainer
from scripts.train_agent.train_explore import ExploreTrainer
from agents import BCAgent, ExploreAgent, RouterAgent
import utils
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Full adversarial training pipeline")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--map_size", type=int, default=11, help="Size of the environment")
    parser.add_argument("--exp_name", type=str, default="adversarial_training", help="Experiment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for trajectory collection")
    parser.add_argument("--output_dir", type=str, default="scripts/train_agent/advs_output", help="Output directory")
    parser.add_argument("--disc_epochs", type=int, default=10, help="Number of epochs for discriminator training")
    parser.add_argument("--traj_dir", type=str, default="traj_data", help="Directory for storing trajectories")
    parser.add_argument("--encoder", type=str, default="lstm", help="Encoder type for trajectory encoding")
    parser.add_argument("--eval_max_steps", type=int, default=20, help="Maximum number of steps per episode")
    parser.add_argument("--GAIL_iters", type=int, default=5, help="Number of GAIL iterations")
    parser.add_argument("--ppo_updates", type=int, default=15, help="Number of PPO updates per GAIL iteration")
    parser.add_argument("--regularize_disc", action="store_true", help="Regularize discriminator with dropout and label smoothing")
    parser.add_argument("--policy_ckpt_path", type=str, default=None, help="Path to the trained agent's policy weights")
    return parser.parse_args()

def create_directories(base_dir, exp_name):
    """Create necessary directories for storing artifacts."""
    base_dir = os.path.join(base_dir, exp_name)
    dirs = {
        'traj': os.path.join(base_dir, 'trajectories'),
        'disc': os.path.join(base_dir, 'discriminator'),
        'rl': os.path.join(base_dir, 'rl_agent')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

class Config:
    """Configuration class to match the expected format"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)





def train_discriminator(args, dirs):
    """Train the discriminator using collected trajectories."""
    print("\n=== Stage 1: Training Discriminator ===")
    
    if not os.listdir(dirs['disc']):
        print(f"first time training discriminator")
        trainer = DiscriminatorTrainer(
            bc_traj=os.path.join(args.traj_dir, f'bc_{args.encoder}_traj_{args.map_size}.npy'),
            fake_traj=os.path.join(args.traj_dir, f'fake_{args.encoder}_traj_{args.map_size}.npy'),
            regularized=args.regularize_disc
        )
    else:
        trainer = DiscriminatorTrainer(
            bc_traj=os.path.join(args.traj_dir, f'bc_{args.encoder}_traj_{args.map_size}.npy'),
            fake_traj=os.path.join(args.traj_dir, f'fake_{args.encoder}_traj_{args.map_size}.npy'),
            ckpt=os.path.join(dirs['disc'], 'model.pt'),
            regularized=args.regularize_disc
        )
    
    iter_loss = trainer.train(dirs['disc'], num_epochs=args.disc_epochs)
    disc_path = os.path.join(dirs['disc'], "model.pt")
    print(f"Discriminator saved to {disc_path}")
    return disc_path, iter_loss

def train_rl_agent(args, dirs, disc_path):
    """Train the RL agent using the frozen discriminator."""
    print("\n=== Stage 2: Training RL Agent ===")
    

    rl_args = argparse.Namespace(
        algo="ppo",  # Required argument
        env=args.env,  # Required argument
        disc_path=disc_path,  # Required argument
        model=args.exp_name,  # Default: None
        seed=1,  # Default: 1
        log_interval=1,  # Default: 1
        save_interval=5,  # Default: 10
        update_per_gail_iter=args.ppo_updates,
        procs=16,  # Default: 16
        frames=10**7,  # Adjust as needed
        epochs=4,  # Default: 4
        batch_size=256,  # Default: 256
        frames_per_proc=None,  # Default: None
        discount=0.99,  # Default: 0.99
        lr=0.001,  # Default: 0.001
        gae_lambda=0.95,  # Default: 0.95
        entropy_coef=0.05,  # Default: 0.01
        value_loss_coef=0.5,  # Default: 0.5
        max_grad_norm=0.5,  # Default: 0.5
        optim_eps=1e-8,  # Default: 1e-8
        clip_eps=0.2,  # Default: 0.2
        recurrence=4,  # Default: 1
        text=False,  # Default: False
        full_obs=False,  # Default: False
        regularize_disc=args.regularize_disc,
        encoder=args.encoder,
        policy_ckpt_path= f"ckpt/memory_{args.encoder}_{args.map_size}/best.pth"
    )
    
    trainer = ExploreTrainer(rl_args)
    trainer.train()
def collect_trajectories(args, dirs, render=False):
    """Collect trajectories using the explore agent."""
    print("\n=== Stage 3: Collecting Trajectories ===")
    
    # Set up environment
    if render:
        env = utils.make_env(args.env, args.seed, render_mode="human")
    else:
        env = utils.make_env(args.env, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize explore agent
    model_dir = utils.get_model_dir(args.exp_name)  # You might want to specify a specific model path
    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=False,
        use_memory=True,
        use_text=False
    )
    
    # Initialize BC agent for getting embeddings
    config = Config(
        encoder = args.encoder,
        policy_ckpt_path= f"ckpt/memory_{args.encoder}_{args.map_size}/best.pth",
    )
    encoding_agent = BCAgent(config, env, device)
    
    # Collect trajectories
    embeddings = []
    labels = []
    total_rewards = []
    
    if render:
        env.render()
        num_episodes = 5
    else:
        num_episodes = args.episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        history = [obs["image"]]
        episode_reward = 0
        step_count = 0
        
        while True:
            # Get embedding from BC agent
            state = torch.FloatTensor(history).unsqueeze(0).to(device)
            seq_length = torch.tensor(state.size(1)).unsqueeze(0).to(device)
            if args.encoder == "lstm":
                embedding = encoding_agent.policy_net(state, seq_length, return_embedding=True)
            elif args.encoder == "transformer":
                embedding = encoding_agent.policy_net(state, return_embedding=True)
            
            # Get action from explore agent
            with torch.no_grad():
                action = agent.get_action(obs)
                
                # Store embedding and label
                embeddings.append(embedding.cpu().numpy())
                labels.append(0)  # 0 for explore agent trajectories
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            episode_reward += reward
            history.append(obs["image"])
            
            if terminated or truncated or step_count >= args.eval_max_steps:
                print(f"Episode {episode + 1} finished with reward {episode_reward}")
                total_rewards.append(episode_reward)
                break
    
    # Save trajectories
    embeddings = np.array(embeddings).squeeze(1)
    labels = np.array(labels)
    traj_path = os.path.join(dirs['traj'], "explore_traj.npy")
    np.save(traj_path, {"embeddings": embeddings, "labels": labels})
    
    print(f"Trajectories saved to {traj_path}")
    return traj_path

def evaluate_agent(args, dirs):
    collect_trajectories(args, dirs, render=True)

def main():
    args = parse_args()
    print("Starting adversarial training pipeline...")
    

    wandb.init(
        project = "adversarial-training",
        name=args.exp_name,
        config=args.__dict__
    )
    # Create directory structure
    dirs = create_directories(args.output_dir, args.exp_name)
    
    # Stage 1: Collect trajectories
    
    
    # Stage 2: Train discriminator
    for i in range(args.GAIL_iters):
        disc_path, disc_iter_loss = train_discriminator(args, dirs)
        wandb.log({"Iteration": i, "Discriminator Loss": disc_iter_loss})
        train_rl_agent(args, dirs, disc_path)
        evaluate_agent(args, dirs)
        explore_traj_path = collect_trajectories(args, dirs)

    # Stage 3: Train RL agent
    
    
    print("\nAdversarial training pipeline completed!")
    wandb.finish()

if __name__ == "__main__":
    main()