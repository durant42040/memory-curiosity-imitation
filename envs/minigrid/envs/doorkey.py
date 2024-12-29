from __future__ import annotations

from envs.minigrid.core.grid import Grid
from envs.minigrid.core.mission import MissionSpace
from envs.minigrid.core.world_object import Door, Goal, Key
from envs.minigrid.minigrid_env import MiniGridEnv


class DoorKeyEnv(MiniGridEnv):
    """
    ## Description

    This environment has a key that the agent must pick up in order to unlock a
    door and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ## Mission Space

    "use the key to open the door and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3-dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-DoorKey-5x5-v0`
    - `MiniGrid-DoorKey-6x6-v0`
    - `MiniGrid-DoorKey-8x8-v0`
    - `MiniGrid-DoorKey-16x16-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 100
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        # splitIdx = self._rand_int(2, width - 2)
        splitIdx = 4
        self.grid.vert_wall(splitIdx, 0)
        self.splitIdx = splitIdx

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        # doorIdx = self._rand_int(1, height - 2)
        doorIdx = 3
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)
        self.door_pos = (splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.key_pos = self.place_obj(
            obj=Key("yellow"), top=(0, 0), size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def get_state_space(self):
        return self.grid_size * 24

    def get_pos_state(self, state):
        return state % (self.grid_size)

    def agent_information_to_state(self):
        return (
            self.agent_pos[0]
            + self.agent_pos[1] * self.width
            + self.agent_dir * self.grid_size
            + (self.carrying != None) * self.grid_size * 4
            + (self.grid.get(*self.door_pos).is_locked == False)
            * self.grid_size
            * 4
            * 2
            + self.grid.get(*self.door_pos).is_open * self.grid_size * 4 * 2
        )

    def state_to_agent_information(self, state):
        grid_size = self.grid_size
        pos_state = self.get_pos_state(state)
        self.agent_dir = state // (grid_size) % 4
        self.agent_pos = [pos_state % self.width, pos_state // self.width]
        self.carrying = Key("yellow") if state // (grid_size * 4) % 2 == 1 else None
        self.grid.get(*self.door_pos).is_locked = (
            True if state // (grid_size * 4 * 2) == 0 else False
        )
        self.grid.get(*self.door_pos).is_open = (
            True if state // (grid_size * 4 * 2) == 2 else False
        )
        # 0: locked, 1: unlocked, unopened, 2: unlocked, opened
        if self.carrying != None:
            self.grid.set(self.key_pos[0], self.key_pos[1], None)

    def dp_reset(self):
        self.step_count = 0
        for i in range(self.width):
            for j in range(self.height):
                if i == self.key_pos[0] and j == self.key_pos[1]:
                    self.grid.set(i, j, Key("yellow"))
                elif i == self.door_pos[0] and j == self.door_pos[1]:
                    self.grid.set(i, j, Door("yellow", is_locked=True))
                else:
                    if (
                        self.grid.grid[i + j * self.width] != None
                        and self.grid.grid[i + j * self.width].type != "wall"
                        and self.grid.grid[i + j * self.width].type != "goal"
                    ):
                        self.grid.set(i, j, None)
