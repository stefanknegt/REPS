import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.agent import Agent
from environments.atari import AtariBreakout
from policies.rand import RandomPolicy
from models.mlp import MLP

"""
Berakout Actions: 0 NOP, 1 BTN, 2 R, 3 L
"""
environment = AtariBreakout()
policy_model = RandomPolicy(0, 3, is_discrete=True)
value_model = MLP([128, 64, 1])

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps()
