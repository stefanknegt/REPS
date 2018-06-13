import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from agents.agent import Agent
from environments.lqr import LQR
from models.simple import Simple
from policies.rand import RandomPolicy
from policies.normal import NormalPolicy
from models.mlp import MLP
from utils.data import SARSDataset

import gym
from torch.nn import functional as F



environment = gym.make("InvertedDoublePendulum-v2")
policy_model_normal = NormalPolicy([11,15,5,1], 1, activation=F.tanh)
policy_model = policy_model_normal
value_model = MLP([11,9,5,1])

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
# environment.seed(42)

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps(10, exp_timesteps=1000, exp_episodes=10)

state_space = np.arange(-2, 2.1, 0.1)
action_space = np.arange(-2, 2.1, 0.1)

# loop over state action pairs
observations = []
for state in state_space:
    for action in action_space:
        environment.state = state
        new_state, reward, _, _ = environment.step(action)
        observations.append({
            'prev_state': state,
            'action': action,
            'reward': reward,
            'new_state':new_state})
observations = SARSDataset(observations)

# calculate weights
prev_states = observations[:][0]
actions = observations[:][1]
rewards = observations[:][2]
new_states = observations[:][3]

weights = agent.calc_weights(prev_states, new_states, rewards)
sc = plt.scatter(prev_states.data, actions.data, c=weights.data)
plt.colorbar(sc)
plt.show()

observations = []
for state in state_space:
    for action in action_space:
        mu = agent.policy_model.get_mu(state)
        sigma = agent.policy_model.mu_net.eta
        from scipy.stats import norm
        c = norm.pdf(action, )
