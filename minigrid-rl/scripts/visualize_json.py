import json
import argparse
import time
import os
import numpy as np
import utils
from envs.minigrid.wrappers import FullyObsWrapper

def load_trajectory(filepath):
    """Load trajectory from JSON file."""
    trajectory = []
    with open(filepath, 'r') as f:
        for line in f:
            trajectory.append(json.loads(line))
    return trajectory

def replay_trajectory(env, trajectory, render=True, pause=0.5):
    """Replay a trajectory in the environment."""
    total_reward = 0
    
    # Reset environment
    obs, _ = env.reset()
    
    if render:
        env.render()
        time.sleep(pause)
    
    # Step through each action in the trajectory
    for step in trajectory:
        action = step['action']
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if render:
            env.render()
            time.sleep(pause)
            
        if terminated or truncated:
            break
    
    return total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="name of the environment")
    parser.add_argument("--trajectory_dir", required=True, help="path to directory with trajectory JSON files")
    parser.add_argument("--pause", type=float, default=0, help="pause between steps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    
    # Set seed
    utils.seed(args.seed)
    
    # Create environment
    env = utils.make_env(args.env, args.seed, render_mode="human")
    
    # Process each trajectory file in the directory
    traj_files = [f for f in os.listdir(args.trajectory_dir) if f.endswith('.json')]
    if not traj_files:
        print(f"No trajectory files found in directory: {args.trajectory_dir}")
        return
    
    total_rewards = []
    for traj_file in traj_files:
        traj_path = os.path.join(args.trajectory_dir, traj_file)
        print(f"Loading trajectory from {traj_path}")
        trajectory = load_trajectory(traj_path)
        print(f"Loaded trajectory with {len(trajectory)} steps")
        
        # Replay trajectory
        total_reward = replay_trajectory(env, trajectory, render=True, pause=args.pause)
        print(f"Trajectory {traj_file} complete. Total reward: {total_reward}")
        total_rewards.append(total_reward)
    
    print(f"All trajectories processed. Total rewards: {total_rewards}")
    env.close()

if __name__ == "__main__":
    main()
