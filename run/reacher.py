import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from controller import Controller
from policies.normal_mlp import MLPNormalPolicy
from values.mlp import MLPValue
from utils.check_env import environment_check

name = 'Reacher-v2'
state_dim, action_dim, action_min, action_max = environment_check(name)

hidden_dim = 20
policy_model = MLPNormalPolicy([state_dim, 15, 45, 2], sigma=4, learning_rate=1e-4, act_bound=2)
value_model = MLPValue([state_dim, 15, 45, 2], learning_rate=1e-4)

model = Controller(name, policy_model, value_model, action_min, action_max, verbose=True)
model.train(iterations=30, batch_size=100,
            exp_episodes=100, exp_timesteps=100, val_epochs=1000,  exp_history=5, exp_render=True, exp_reset_prob=0)

policy_model.save('../run/' + name + '.pth')