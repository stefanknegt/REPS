import sys, os
import torch.nn.functional as F
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Swimmer-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
value_model = MLPValue([state_dim, 128, 64, 32, 1], learning_rate=1e-4, activation=F.tanh, epsilon=0.1, eta=10)
policy_model = MLPNormalPolicy([state_dim, 128, 64, 32, 2], sigma=2, learning_rate=1e-4, act_bound=1, activation=F.tanh)

prefix = "try"

policy_model = torch.load('../run/results/' + name + '_' + prefix +'_policy_best.pth')
model = Controller(name, policy_model, value_model, reset_prob=0.002, history_depth=1, verbose=True, cuda=True)
print(model.evaluate(20, True))

