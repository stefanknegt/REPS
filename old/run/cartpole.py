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



environment = LQR(-1,1)
policy_model_normal = NormalPolicy([1,20,45,1], [4 for i in range(1)], activation=F.tanh)
policy_model = policy_model_normal
value_model = MLP([1,20,45,1])

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
#environment.seed(42)

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps(100, exp_timesteps=1000, exp_episodes=10, exp_render=False, val_epsilon=0.1, pol_lr=1e-2)





