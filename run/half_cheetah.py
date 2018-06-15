import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from policies.normal import NormalPolicy
from values.linear import LinearValue
from controller import Controller
from utils.check_env import environment_check

import torch
import gym

name = 'HalfCheetah-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = NormalPolicy(state_dim, action_dim, hidden_dim, mu_range=0.4, sigma_range=1, lr=0.01)
value_model = LinearValue(state_dim, hidden_dim, value_range=1, eta=0.1, epsilon=0.5, lr=0.01)

model = Controller(name, policy_model, value_model, action_min, action_max, verbose=True)
model.train(iterations=5, batch_size=100,
            exp_episodes=100, exp_timesteps=5000, exp_reset_prob=.1, exp_history=5, exp_render=True)

policy_model.save('run/' + name + '.pth')
