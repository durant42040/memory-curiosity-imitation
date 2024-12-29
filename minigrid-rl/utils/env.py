import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None, rounds=1):
    env = gym.make(env_key, render_mode=render_mode, rounds=rounds)
    env.reset(seed=seed)
    return env
