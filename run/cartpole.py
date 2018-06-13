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



environment = gym.make("HalfCheetah-v2")
policy_model_normal = NormalPolicy([17,200,40,6], [1 for i in range(6)], activation=F.tanh)
policy_model = policy_model_normal
value_model = MLP([17,200,40,1])

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
environment.seed(42)

agent = Agent(environment, policy_model, value_model, verbose=True)
agent.run_reps(20, exp_timesteps=100, exp_episodes=100, exp_render=True, val_epsilon=0.1)





