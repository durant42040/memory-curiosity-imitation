import gym
import numpy as np
from PPO import PPO

def train_ppo(env, max_episodes=1000, max_steps=200, 
              batch_size=32):
    # Create environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize PPO agent
    agent = PPO(state_dim=state_dim, 
                action_dim=action_dim,
                hidden_dim=64,
                lr=3e-4,
                gamma=0.99)

    # Training loop
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # Storage for batch updates
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        next_states = []

        # Episode loop
        for step in range(max_steps):
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            episode_reward += reward
            state = next_state

            # Update PPO if we have enough samples
            if len(states) >= batch_size:
                agent.update(
                    states=np.array(states),
                    actions=np.array(actions),
                    old_log_probs=np.array(log_probs),
                    rewards=np.array(rewards),
                    dones=np.array(dones),
                    next_states=np.array(next_states)
                )
                # Clear storage
                states, actions, log_probs = [], [], []
                rewards, dones, next_states = [], [], []

            if done or truncated:
                break

        # Print episode statistics
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
