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
from utils.data import SARSDataset

import gym
from torch.nn import functional as F

random.seed(42)

environment = gym.make("MountainCarContinuous-v0")

policy_model_normal = NormalPolicy(input_size=2)
policy_model = policy_model_normal
value_model = Simple(input_size=2)

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps(10,exp_timesteps=2000, exp_episodes=10)

state_space = np.arange(-2, 2.1, 0.1)
action_space = np.arange(-2, 2.1, 0.1)

# loop over state action pairs
observations = []
for state in state_space:
    for action in action_space:
        environment.state = state
        new_state, reward = environment.step(action)
        observations.append({
            'prev_state': state,
            'action': action,
            'reward': reward,
            'new_state':new_state})
observations = SARSDataset(observations)

# calculate weights
prev_states = observations[:][:,0].view(len(observations), 1)
actions = observations[:][:,1].view(len(observations), 1)
rewards = observations[:][:,2].view(len(observations), 1)
new_states = observations[:][:,3].view(len(observations), 1)

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
