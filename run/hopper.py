import sys, os
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Hopper-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = MLPNormalPolicy([state_dim, 300, 400, 3], sigma=2, learning_rate=1e-3, act_bound=1, activation=F.tanh)
value_model = MLPValue([state_dim, 300, 400, 1], learning_rate=1e-3, activation=F.tanh, epsilon=0.1, eta=50)

model = Controller(name, policy_model, value_model, reset_prob=0.005, history_depth=2, verbose=True, cuda=True)
model.set_seeds(41)
model.train(iterations=100, exp_episodes=25, exp_timesteps=1000, val_epochs=500, pol_epochs=500, batch_size=128)

policy_model.save('../run/' + name + '.pth')