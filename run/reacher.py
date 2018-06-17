import sys, os
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Pendulum-v0'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = MLPNormalPolicy([state_dim, 15, 45, 1], sigma=4, learning_rate=1e-3, act_bound=2, activation=F.tanh)
value_model = MLPValue([state_dim, 15, 45, 1], learning_rate=1e-3, activation=F.tanh)

model = Controller(name, policy_model, value_model, reset_prob=0.02, history_depth=1, verbose=True,)
model.train(exp_episodes=5, exp_timesteps=1000, val_epochs=100, batch_size=64)

policy_model.save('../run/' + name + '.pth')