import sys, os
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'HalfCheetah-v2'

state_dim, action_dim, action_min, action_max = environment_check(name)

policy_model = MLPNormalPolicy([state_dim, 20, action_dim], sigma=2, learning_rate=1e-3, act_bound=2, activation=F.relu, init_value=0.4)
value_model = MLPValue([state_dim, 20, 1], learning_rate=5e-4, epsilon=0.1, activation=F.relu)

model = Controller(name, policy_model, value_model, reset_prob=0.02, history_depth=2, verbose=True,)
model.set_seeds(41)
model.train(iterations=30, exp_episodes=50, exp_timesteps=100, val_epochs=200, pol_epochs=200, batch_size=64, pickle_name='v0.1')

policy_model.save('../run/' + name + '.pth')
