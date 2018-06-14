from values.value_model import Value
from policies.policy_model import Policy
from controller import Controller
from utils.check_env import environment_check
import torch
import gym
from torch.autograd import Variable

name = 'HalfCheetah-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = Policy(state_dim, action_dim, hidden_dim, mu_range=0.4, sigma_range=1, lr=0.01)
value_model = Value(state_dim, hidden_dim, value_range=1, eta=0.1, epsilon=0.5, lr=0.01)

model = Controller(name, policy_model, value_model, action_min, action_max)
model.train(iterations=5000, batch_size=100, trajectories=10, epochs=5, history_size=5, verbose=1)

policy_model.save('test.pth')
