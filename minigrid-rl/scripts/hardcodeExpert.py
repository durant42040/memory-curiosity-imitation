import argparse
import json
import os
import numpy as np
import gymnasium as gym
import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

class HardcodeExpertAgent:
    def __init__(self, env):
        self.env = env
        self.current_path = []
        self.next_action = 0
        
    def plan_path(self):
        """Plan optimal path based on object locations"""
        env = self.env.unwrapped
        
        # Get starting room object type and position
        start_room_pos = None
        start_room_type = None
        for i in range(1, 4):
            for j in [env.height // 2 - 1, env.height // 2 + 1]:
                cell = env.grid.get(i, j)
                if cell is not None and cell.type in ['key', 'ball']:
                    start_room_pos = (i, j)
                    start_room_type = cell.type
                    break
        
        # Get target position based on matching object
        hallway_end = env.width - 3  # Based on env implementation
        pos0 = (hallway_end + 1, env.height // 2 - 2)
        pos1 = (hallway_end + 1, env.height // 2 + 2)
        
        # Check which end position has the matching object
        cell0 = env.grid.get(*pos0)
        if cell0.type == start_room_type:
            target_pos = (pos0[0], pos0[1] + 1)
        else:
            target_pos = (pos1[0], pos1[1] - 1)
            
        # Generate sequence of actions to reach target
        path = []
        current_pos = env.agent_pos
        current_dir = env.agent_dir
        
        # First move to the hallway
        while current_pos[0] <= hallway_end:
            path.append(2)  # forward
            current_pos = (current_pos[0] + 1, current_pos[1])
            
        # Turn to face the correct direction
        if target_pos[1] < env.height // 2:  # Need to go up
            path.append(0)  # left turn
            for _ in range((env.height // 2) - target_pos[1]):
                path.append(2)  # forward
        else:  # Need to go down
            path.append(1)  # right turn
            for _ in range(target_pos[1] - (env.height // 2)):
                path.append(2)  # forward
                
        self.current_path = path
        self.next_action = 0

    def get_action(self):
        """Return next action in the planned path"""
        if self.next_action >= len(self.current_path):
            return 6  # done
        action = self.current_path[self.next_action]
        self.next_action += 1
        return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='MiniGrid-MemoryS7-v0', 
                      help='Environment name')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save trajectories')
    parser.add_argument('--set', type=str, default=None, )
    
    args = parser.parse_args()
    
    # Create environment
    if args.render:
        env = gym.make(args.env, render_mode="human")
    else:
        env = gym.make(args.env, render_mode=None)
    
    # Create expert agent
    expert = HardcodeExpertAgent(env)
    
    # Create save directory if needed
    save_dir = f"./expert_data/{args.env}"
    save_dir = save_dir + f"-{args.set}" if args.set is not None else save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Run episodes
    for episode in range(args.episodes):
        obs, _ = env.reset()
        trajectory = []
        expert.plan_path()
        
        done = False
        while not done:
            if args.render:
                env.render()
                
            # Get expert action
            action = expert.get_action()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Save trajectory data
            trajectory.append({
                "observation": {
                    "image": obs["image"].tolist(),
                    "direction": int(obs["direction"]),
                    "mission": str(obs["mission"])
                },
                "action": int(action),
                "reward": float(reward)
            })
            
            obs = next_obs
            
        # Save trajectory
        trajectory_path = os.path.join(save_dir, f"episode_{episode}.json")
        with open(trajectory_path, "w") as f:
            for step in trajectory:
                f.write(json.dumps(step, separators=(",", ":")) + "\n")
        print(f"Saved trajectory to {trajectory_path}")
            
        print(f"Episode {episode + 1}/{args.episodes} completed with reward {reward}")
    
    env.close()

if __name__ == "__main__":
    main()