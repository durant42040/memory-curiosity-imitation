import argparse
import json
import os
import sys

import numpy
import utils
from utils import device

import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

from scripts.hardcodeExpert import HardcodeExpertAgent

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True, help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--model", default=None, help="name of the trained model (REQUIRED)"
)

parser.add_argument(
    "--render",
    action="store_true",
    default=False,
    help="render the environment (default: False)",
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--shift",
    type=int,
    default=0,
    help="number of times the environment is reset at the beginning (default: 0)",
)
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="select the action with highest probability (default: False)",
)
parser.add_argument(
    "--pause",
    type=float,
    default=0.1,
    help="pause duration between two consequent actions of the agent (default: 0.1)",
)
parser.add_argument(
    "--gif", type=str, default=None, help="store output as gif with the given filename"
)
parser.add_argument(
    "--episodes", type=int, default=1000000, help="number of episodes to visualize"
)
parser.add_argument(
    "--memory", action="store_true", default=False, help="add a LSTM to the model"
)
parser.add_argument(
    "--text", action="store_true", default=False, help="add a GRU to the model"
)

parser.add_argument(
    "--full_obs", action="store_true", default=False, help="use full observation"
)
parser.add_argument(
    "--dp", action="store_true", default=False, help="use dynamic programming agent"
)
parser.add_argument(
    "--save_dir", type=str, default=None, help="directory to save expert data"
)
parser.add_argument(
    "--rounds", type=int, default=1, help="number of rounds per trajectory"
)
parser.add_argument(
    "--hardcode", action="store_true", default=False, help="use hardcode expert agent"
)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

if args.render:
    env = utils.make_env(args.env, args.seed, render_mode="human", rounds=args.rounds)
else:
    env = utils.make_env(args.env, args.seed, render_mode=None, rounds=args.rounds)
if args.full_obs:
    env = FullyObsWrapper(env)
partial_env = utils.make_env(args.env, args.seed, render_mode=None, rounds=args.rounds)

for _ in range(args.shift):
    env.reset()
    partial_env.reset()
print("Environment loaded\n")

# Load agent

DISCOUNT_FACTOR = 0.9

if args.dp:
    agent = utils.ValueIteration(env.unwrapped, discount_factor=DISCOUNT_FACTOR)
elif args.hardcode:
    agent = HardcodeExpertAgent(partial_env)
else:
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=args.argmax,
        use_memory=args.memory,
        use_text=args.text,
    )
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
if args.render == True:
    env.render()

if args.save_dir is not None:
    SAVE_DIR = f"./expert_data/{args.save_dir}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
for episode in range(args.episodes):
    trajectory = []  # List to store trajectory data for the episode
    prev_obs, _ = env.reset()
    prev_partial_obs, _ = partial_env.reset()
    if args.hardcode:
        agent.plan_path()
    while True:
        if args.dp:
            agent.run()
        done = False
        while True:
            if args.render == True:
                env.render()
            if args.gif:
                frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
            if args.hardcode:
                action = agent.get_action()
                # print(f"hardcode action: {action}")
            else:
                action = agent.get_action(prev_obs)
            # if args.full_obs:
            #     print(obs['image'].transpose(2, 1, 0))
            # else:
            #     print(obs['image'].transpose(2, 0, 1))
            obs, reward, terminated, truncated, _ = env.step(action)
            # print((env.unwrapped.agent_pos == numpy.array([4, 3])).all())
            # print(env.unwrapped.grid.grid[1 + (env.unwrapped.height // 2 - 1) * env.unwrapped.width].type)
            partial_obs, _, _, _, _ = partial_env.step(action)
            done = terminated or truncated

            # Append the current step's data to the trajectory
            trajectory.append(
                {
                    "partial_obs": {
                        "image": prev_partial_obs["image"].tolist(),
                    },
                    "observation": {
                        "image": prev_obs["image"].tolist(),  # Convert NumPy array to list
                        "direction": int(
                            prev_obs["direction"]
                        ),  # Convert NumPy int64 to Python int
                        "mission": str(prev_obs["mission"]),  # Ensure mission is a string
                    },
                    "action": int(action),  # Convert NumPy int64 to Python int
                    "reward": float(reward),  # Convert NumPy float64 to Python float
                }
            )
            if not args.hardcode:
                agent.analyze_feedback(reward, done)

            prev_obs = obs
            prev_partial_obs = partial_obs

            if done:
                # Save the trajectory to a file with custom formatting
                if args.save_dir is not None:
                    file_path = os.path.join(SAVE_DIR, f"episode_{episode}.json")
                    with open(file_path, "w") as f:
                        for step in trajectory:
                            f.write(
                                json.dumps(
                                    {
                                        "direction": step["observation"]["direction"],
                                        "action": step["action"],
                                        "reward": step["reward"],
                                        "mission": step["observation"]["mission"],
                                        "partial_image": step["partial_obs"]["image"],
                                        "full_image": step["observation"]["image"],
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                    print(f"Saved trajectory to {file_path}")
                break

            if (terminated == None):
                env.unwrapped.round_reset()
                partial_env.unwrapped.round_reset()
                break
        if done:
            break


if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
