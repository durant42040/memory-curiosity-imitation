import torch
import numpy as np
import envs.minigrid
import gymnasium as gym
from collections import deque
import utils
from model import ACModel

from .other import device


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(
        self,
        obs_space,
        action_space,
        model_dir,
        argmax=False,
        num_envs=1,
        use_memory=False,
        use_text=False,
    ):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(
            obs_space, action_space, use_memory=use_memory, use_text=use_text
        )
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_size, device=device
            )

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(
                1
            )
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

class ValueIteration():
    def __init__(self, env, discount_factor: float = 1.0):
        self.env = env
        self.discount_factor = discount_factor
        self.threshold = 1e-4
        self.values = np.zeros(env.get_state_space())
        self.policy = np.zeros(env.get_state_space(), dtype=int)
        self.assert_state_agent_information_transformation()

    def policy_evaluation(self):
        while True:
            env_cloned = self.env.clone()
            env_cloned.render_mode = None
            env_cloned.dp = True
            delta = 0
            old_values = self.values.copy()
            for state in range(env_cloned.get_state_space()):
                best_q_value = float("-inf")
                pos_state = env_cloned.get_pos_state(state)
                if (env_cloned.grid.grid[pos_state] != None and env_cloned.grid.grid[pos_state].type == 'wall'):
                    self.values[state] = None
                    continue
                env_cloned.dp_reset()
                for action in range(env_cloned.get_action_space()):
                    env_tmp = env_cloned.clone()
                    env_tmp.state_to_agent_information(state)
                    _, reward, terminated, truncated, _ = env_tmp.step(action)
                    next_state = env_tmp.agent_information_to_state()
                    end = terminated or truncated
                    q_value = reward if end else reward + self.discount_factor * old_values[next_state]
                    best_q_value = max(best_q_value, q_value)
                self.values[state] = best_q_value
                delta = max(delta, abs(old_values[state] - self.values[state]))
            if delta < self.threshold:
                break

    def policy_improvement(self):
        env_cloned = self.env.clone()
        env_cloned.render_mode = None
        env_cloned.dp = True
        for state in range(env_cloned.get_state_space()):
            best_action = None
            best_q_value = float("-inf")
            pos_state = env_cloned.get_pos_state(state)
            if (env_cloned.grid.grid[pos_state] != None and env_cloned.grid.grid[pos_state].type == 'wall'):
                continue
            env_cloned.dp_reset()
            for action in range(env_cloned.get_action_space()):
                env_tmp = env_cloned.clone()
                env_tmp.state_to_agent_information(state)
                _, reward, terminated, truncated, _ = env_tmp.step(action)
                next_state = env_tmp.agent_information_to_state()
                end = terminated or truncated
                q_value = reward if end else reward + self.discount_factor * self.values[next_state]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            self.policy[state] = best_action

    def run(self) -> None:
        self.values = np.zeros(self.env.get_state_space())
        self.policy = np.zeros(self.env.get_state_space(), dtype=int)
        self.policy_evaluation()
        self.policy_improvement()
    
    def get_action(self, obss) -> int:
        state = self.env.agent_information_to_state()
        return self.policy[state]

    def analyze_feedback(self, rewards, dones):
        pass

    def assert_state_agent_information_transformation(self):
        env_cloned = self.env.clone()
        for state in range(env_cloned.get_state_space()):
            env_cloned.state_to_agent_information(state)
            # print(state, env_cloned.agent_information_to_state())
            assert env_cloned.agent_information_to_state() == state
