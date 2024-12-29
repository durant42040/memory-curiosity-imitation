from agents import BCAgent, ExploreAgent
from agents.models import Discriminator


class RouterAgent:
    def __init__(self, config, env, device):
        self.discriminator = Discriminator()
        self.bc_agent = BCAgent(config, env, device)
        self.explore_agent = ExploreAgent(config, env, device)

        # pseudocode
        # obs = env(action)
        # if self.discriminator(obs) == True:
        #     action = self.bc_agent(obs)
        # else:
        #     action = self.explore_agent(obs)

        print(f"env:{config.env}")
