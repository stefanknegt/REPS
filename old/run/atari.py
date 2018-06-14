import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent import Agent
from environments.atari import AtariBreakout
from policies.rand import RandomPolicy
from policies.normal import NormalPolicy
from models.mlp import MLP

environment = AtariBreakout()
# policy_model = RandomPolicy(0, 3, is_discrete=True)
policy_model = NormalPolicy([128, 64, 1], 1.)
value_model = MLP([128, 64, 1])

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps(exp_render=True)
