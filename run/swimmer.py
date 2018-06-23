import sys, os
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Swimmer-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

policy_model = MLPNormalPolicy([state_dim, 128, 64, action_dim], sigma=2, learning_rate=1e-4, act_bound=1, activation=F.tanh)
value_model = MLPValue([state_dim, 128, 64, 1], learning_rate=1e-3, activation=F.tanh, epsilon=0.25, eta=10)

model = Controller(name, policy_model, value_model, reset_prob=0.002, history_depth=1, verbose=True, cuda=True)
model.set_seeds(41)
model.train(iterations=200, exp_timesteps=40000, val_epochs=500, pol_epochs=300, batch_size=128, pickle_name='128a', val_ratio=0.2, eval_episodes=25, render_step=5)